from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from .gaps import GapRepairOutput
from .models import (
    DeepMeshFusionConfig,
    GeometryIssue,
    GeometryQualityScores,
    GeometryValidationResult,
    GeometryValidationSummary,
)
from .reconstruction import ArchitectureMeshOutput, DeepMeshFusionReconstructionService


@dataclass
class GeometryValidationOutput:
    mesh: ArchitectureMeshOutput
    issues: List[GeometryIssue]
    summary: GeometryValidationSummary
    vertex_quality: np.ndarray
    review_points: np.ndarray
    review_severity: np.ndarray


class GeometryValidationService:
    """Evidence-aware topology, surface, and architectural mesh validation."""

    SEVERITY_CODES = {"info": 0, "warning": 1, "error": 2, "critical": 3}

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config
        self.reconstruction = DeepMeshFusionReconstructionService(config)

    def validate(self, gap_output: GapRepairOutput) -> GeometryValidationOutput:
        mesh = gap_output.mesh
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("Geometry validation requires a non-empty triangle mesh")

        tri = vertices[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        area = twice_area * 0.5
        face_normals = np.divide(cross, twice_area[:, None], out=np.zeros_like(cross), where=twice_area[:, None] > 1e-15)
        repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
        minimum_area = self.config.validation_degenerate_area_ratio * self.config.effective_architecture_grid_size() ** 2
        degenerate = repeated | ~np.isfinite(area) | (area <= minimum_area)

        lengths = np.stack((
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ), axis=1)
        longest = lengths.max(axis=1)
        aspect = np.divide(longest ** 2, np.maximum(twice_area, 1e-15))
        stretched = (~degenerate) & (aspect > self.config.validation_max_triangle_aspect_ratio)

        edge_faces = self._edge_faces(faces)
        boundary_edges = [edge for edge, owners in edge_faces.items() if len(owners) == 1]
        nonmanifold_edges = [edge for edge, owners in edge_faces.items() if len(owners) > 2]
        boundary_components = self._edge_components(boundary_edges)
        boundary_loops = sum(all(degree == 2 for degree in component.values()) for component in boundary_components)
        classified_boundaries = (
            gap_output.summary.intentional_opening_count
            + gap_output.summary.exterior_boundary_count
            + gap_output.summary.unresolved_gap_count
        )
        unclassified_holes = max(0, boundary_loops - classified_boundaries)

        face_components = self._face_components(faces, edge_faces)
        tiny_limit = max(4, int(np.ceil(len(faces) * 0.001)))
        tiny_components = [component for component in face_components if len(component) <= tiny_limit]

        bad_normal_vertices, flipped_edges, discontinuity_edges = self._normal_checks(
            mesh, faces, face_normals, edge_faces, degenerate
        )
        intersections, pairs_tested, intersection_complete = self._self_intersections(
            vertices, faces, degenerate, mesh.surface_ids
        )
        architectural_score, architectural_findings = self._architectural_consistency(mesh, faces)

        unresolved_area = sum(gap.area for gap in gap_output.gaps if gap.review_required)
        total_area = float(np.sum(area[~degenerate]))
        completeness = 1.0 - unresolved_area / max(total_area + unresolved_area, 1e-12)
        completeness -= min(0.25, unclassified_holes * 0.02)
        completeness = float(np.clip(completeness, 0.0, 1.0))
        confidence = self._area_weighted_confidence(mesh.confidence, faces, area, degenerate)

        face_count = max(len(faces), 1)
        edge_count = max(len(edge_faces), 1)
        surface_penalty = (
            4.0 * np.count_nonzero(degenerate) / face_count
            + 1.5 * np.count_nonzero(stretched) / face_count
            + 2.0 * len(bad_normal_vertices) / max(len(vertices), 1)
            + 8.0 * len(intersections) / face_count
            + 3.0 * len(discontinuity_edges) / edge_count
        )
        surface_quality = float(np.clip(1.0 - surface_penalty, 0.0, 1.0))
        topology_penalty = (
            8.0 * len(nonmanifold_edges) / edge_count
            + 8.0 * len(intersections) / face_count
            + 0.08 * len(tiny_components)
            + 0.02 * unclassified_holes
        )
        topology_score = float(np.clip(1.0 - topology_penalty, 0.0, 1.0))
        consistency = float(np.clip(0.55 * topology_score + 0.45 * architectural_score, 0.0, 1.0))
        overall = float(0.30 * completeness + 0.25 * surface_quality + 0.25 * consistency + 0.20 * confidence)
        scores = GeometryQualityScores(
            completeness=completeness,
            surface_quality=surface_quality,
            consistency=consistency,
            confidence=confidence,
            architectural_consistency=architectural_score,
            overall=overall,
        )

        issue_data: List[Tuple[str, str, str, np.ndarray, Set[int], str]] = []
        centers = tri.mean(axis=1)
        self._add_issue(issue_data, "degenerate-triangles", "error", np.flatnonzero(degenerate), centers, mesh, "Zero-area, repeated-index, or invalid triangles")
        self._add_issue(issue_data, "stretched-triangles", "warning", np.flatnonzero(stretched), centers, mesh, "Triangles exceed the configured aspect-ratio limit")
        self._add_point_issue(issue_data, "bad-normals", "error", bad_normal_vertices, vertices, mesh, "Missing, invalid, or poorly aligned vertex normals")
        self._add_edge_issue(issue_data, "non-manifold-geometry", "critical", nonmanifold_edges, vertices, mesh, "Edges shared by more than two faces")
        self._add_edge_issue(issue_data, "surface-discontinuities", "warning", discontinuity_edges, vertices, mesh, "Unexpected sharp changes within one reconstructed surface")
        if flipped_edges:
            self._add_edge_issue(issue_data, "inconsistent-face-orientation", "error", flipped_edges, vertices, mesh, "Adjacent faces have opposing orientation")
        if intersections:
            locations = np.asarray([(centers[a] + centers[b]) * 0.5 for a, b in intersections])
            surfaces = {int(mesh.surface_ids[faces[index, 0]]) for pair in intersections for index in pair}
            issue_data.append(("self-intersections", "critical", "Intersecting non-adjacent triangle pairs", locations, surfaces, "face-pairs"))
        if not intersection_complete:
            issue_data.append(("self-intersection-audit-limit", "warning", "Self-intersection candidate budget was exhausted", np.empty((0, 3)), set(), "audit"))
        if len(face_components) > 1:
            locations = np.asarray([centers[component].mean(axis=0) for component in face_components[1:]]) if len(face_components) > 1 else np.empty((0, 3))
            issue_data.append(("disconnected-components", "warning", "Mesh contains multiple disconnected components", locations, set(), "components"))
        if unclassified_holes:
            locations = np.asarray([vertices[list(component)].mean(axis=0) for component in boundary_components[:unclassified_holes]])
            issue_data.append(("unclassified-holes", "error", "Closed mesh boundaries are not explained by openings or tracked gaps", locations, set(), "loops"))
        unresolved = [gap for gap in gap_output.gaps if gap.review_required]
        if unresolved:
            locations = np.asarray([(np.asarray(gap.bounds_min) + np.asarray(gap.bounds_max)) * 0.5 for gap in unresolved])
            issue_data.append(("unresolved-gaps", "error", "Evidence was insufficient to repair these gaps", locations, {gap.surface_id for gap in unresolved}, "gaps"))
        for category, message, locations, surface_ids in architectural_findings:
            issue_data.append((category, "error", message, locations, surface_ids, "architectural-checks"))

        issues, review_points, review_severity = self._materialize_issues(issue_data)
        review_regions = self._review_region_count(review_points)
        critical = any(issue.severity == "critical" for issue in issues)
        ready = bool(
            not critical
            and intersection_complete
            and gap_output.summary.unresolved_gap_count == 0
            and scores.completeness >= self.config.validation_min_completeness
            and scores.surface_quality >= self.config.validation_min_surface_quality
            and scores.consistency >= self.config.validation_min_consistency
            and scores.confidence >= self.config.validation_min_confidence
            and scores.overall >= self.config.validation_min_overall_quality
        )
        vertex_quality = self._vertex_quality(mesh, faces, degenerate, stretched, bad_normal_vertices, nonmanifold_edges, discontinuity_edges, intersections)
        summary = GeometryValidationSummary(
            vertex_count=len(vertices), face_count=len(faces), surface_area=total_area,
            boundary_edge_count=len(boundary_edges), boundary_loop_count=boundary_loops,
            unclassified_hole_count=unclassified_holes, nonmanifold_edge_count=len(nonmanifold_edges),
            self_intersection_count=len(intersections), self_intersection_pairs_tested=pairs_tested,
            self_intersection_audit_complete=intersection_complete,
            degenerate_triangle_count=int(np.count_nonzero(degenerate)),
            stretched_triangle_count=int(np.count_nonzero(stretched)), bad_normal_count=len(bad_normal_vertices),
            surface_discontinuity_count=len(discontinuity_edges), disconnected_component_count=len(face_components),
            tiny_component_count=len(tiny_components), unresolved_gap_count=gap_output.summary.unresolved_gap_count,
            review_region_count=review_regions, ready_for_appearance_processing=ready, scores=scores,
        )
        return GeometryValidationOutput(mesh, issues, summary, vertex_quality, review_points, review_severity)

    def export(self, output: GeometryValidationOutput, mesh_path: str, report_path: str, quality_map_path: str) -> GeometryValidationResult:
        mesh_target, report_target, quality_target = Path(mesh_path), Path(report_path), Path(quality_map_path)
        self.reconstruction._write_mesh_ply(output.mesh, mesh_target)
        payload = {
            "schema_version": 1,
            "readiness_thresholds": {
                "completeness": self.config.validation_min_completeness,
                "surface_quality": self.config.validation_min_surface_quality,
                "consistency": self.config.validation_min_consistency,
                "confidence": self.config.validation_min_confidence,
                "overall": self.config.validation_min_overall_quality,
            },
            "summary": asdict(output.summary),
            "issues": [asdict(issue) for issue in output.issues],
        }
        report_target.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_target.with_suffix(report_target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(report_target)
        self._write_quality_ply(output, quality_target)
        return GeometryValidationResult(str(mesh_target), str(report_target), str(quality_target), output.summary)

    @staticmethod
    def _edge_faces(faces) -> Dict[Tuple[int, int], List[int]]:
        result: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for face_index, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                result[tuple(sorted((int(a), int(b))))].append(face_index)
        return result

    @staticmethod
    def _edge_components(edges):
        graph = defaultdict(set)
        for a, b in edges:
            graph[a].add(b); graph[b].add(a)
        components, unseen = [], set(graph)
        while unseen:
            start = unseen.pop(); queue = [start]; vertices = {start}
            while queue:
                current = queue.pop()
                for neighbor in graph[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor); vertices.add(neighbor); queue.append(neighbor)
            components.append({vertex: len(graph[vertex]) for vertex in vertices})
        return components

    @staticmethod
    def _face_components(faces, edge_faces):
        adjacency = [set() for _ in range(len(faces))]
        for owners in edge_faces.values():
            for owner in owners:
                adjacency[owner].update(other for other in owners if other != owner)
        unseen, result = set(range(len(faces))), []
        while unseen:
            start = unseen.pop(); queue = [start]; component = [start]
            while queue:
                for neighbor in adjacency[queue.pop()]:
                    if neighbor in unseen:
                        unseen.remove(neighbor); queue.append(neighbor); component.append(neighbor)
            result.append(np.asarray(component, dtype=np.int64))
        result.sort(key=len, reverse=True)
        return result

    def _normal_checks(self, mesh, faces, face_normals, edge_faces, degenerate):
        normals = np.asarray(mesh.normals, dtype=float)
        lengths = np.linalg.norm(normals, axis=1) if len(normals) == len(mesh.vertices) else np.zeros(len(mesh.vertices))
        expected = np.zeros((len(mesh.vertices), 3), dtype=float)
        for face_index, face in enumerate(faces):
            if not degenerate[face_index]:
                expected[face] += face_normals[face_index]
        expected_length = np.linalg.norm(expected, axis=1)
        valid = (lengths > 1e-12) & np.isfinite(normals).all(axis=1) & (expected_length > 1e-12)
        alignment = np.zeros(len(mesh.vertices))
        alignment[valid] = np.sum(normals[valid] * expected[valid], axis=1) / (lengths[valid] * expected_length[valid])
        bad = np.flatnonzero(~valid | (alignment < self.config.validation_normal_alignment))
        flipped, discontinuity = [], []
        threshold = np.cos(np.deg2rad(self.config.validation_discontinuity_angle_degrees))
        for edge, owners in edge_faces.items():
            if len(owners) != 2 or np.any(degenerate[owners]):
                continue
            dot = float(np.dot(face_normals[owners[0]], face_normals[owners[1]]))
            if dot < -self.config.validation_normal_alignment:
                flipped.append(edge)
            first_surface = int(mesh.surface_ids[faces[owners[0], 0]])
            second_surface = int(mesh.surface_ids[faces[owners[1], 0]])
            if first_surface >= 0 and first_surface == second_surface and abs(dot) < threshold:
                discontinuity.append(edge)
        return bad, flipped, discontinuity

    def _self_intersections(self, vertices, faces, degenerate, surface_ids):
        cell = max(self.config.effective_architecture_grid_size() * 2.0, 1e-6)
        buckets = defaultdict(list)
        oversized = []
        tri = vertices[faces]
        for index in np.flatnonzero(~degenerate):
            low = np.floor(tri[index].min(axis=0) / cell).astype(int)
            high = np.floor(tri[index].max(axis=0) / cell).astype(int)
            spans = high - low + 1
            if int(np.prod(spans)) > 4096:
                oversized.append(int(index))
                continue
            for i in range(low[0], high[0] + 1):
                for j in range(low[1], high[1] + 1):
                    for k in range(low[2], high[2] + 1):
                        buckets[(i, j, k)].append(int(index))
        seen, intersections, tested = set(), [], 0
        limit = self.config.validation_self_intersection_pair_limit
        for members in buckets.values():
            for offset, first in enumerate(members):
                for second in members[offset + 1:]:
                    pair = (min(first, second), max(first, second))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if np.intersect1d(faces[first], faces[second]).size:
                        continue
                    first_surface = int(surface_ids[faces[first, 0]])
                    second_surface = int(surface_ids[faces[second, 0]])
                    if first_surface >= 0 and second_surface >= 0 and first_surface != second_surface:
                        continue
                    tested += 1
                    if tested > limit:
                        return intersections, limit, False
                    if self._triangles_intersect(tri[first], tri[second]):
                        intersections.append(pair)
        valid_faces = np.flatnonzero(~degenerate)
        for first in oversized:
            for second in valid_faces:
                second = int(second)
                pair = (min(first, second), max(first, second))
                if first == second or pair in seen or np.intersect1d(faces[first], faces[second]).size:
                    continue
                seen.add(pair)
                first_surface = int(surface_ids[faces[first, 0]])
                second_surface = int(surface_ids[faces[second, 0]])
                if first_surface >= 0 and second_surface >= 0 and first_surface != second_surface:
                    continue
                if np.any(tri[first].max(axis=0) < tri[second].min(axis=0)) or np.any(tri[second].max(axis=0) < tri[first].min(axis=0)):
                    continue
                tested += 1
                if tested > limit:
                    return intersections, limit, False
                if self._triangles_intersect(tri[first], tri[second]):
                    intersections.append(pair)
        return intersections, tested, True

    @staticmethod
    def _triangles_intersect(first, second):
        eps = 1e-9
        if np.any(first.max(axis=0) < second.min(axis=0) - eps) or np.any(second.max(axis=0) < first.min(axis=0) - eps):
            return False
        for triangle, other in ((first, second), (second, first)):
            for index in range(3):
                if GeometryValidationService._segment_triangle(triangle[index], triangle[(index + 1) % 3], other, eps):
                    return True
        n1 = np.cross(first[1] - first[0], first[2] - first[0])
        n2 = np.cross(second[1] - second[0], second[2] - second[0])
        if np.linalg.norm(np.cross(n1, n2)) <= eps * max(np.linalg.norm(n1) * np.linalg.norm(n2), 1.0):
            if abs(float(np.dot(n1, second[0] - first[0]))) <= eps * max(np.linalg.norm(n1), 1.0):
                return GeometryValidationService._coplanar_overlap(first, second, n1, eps)
        return False

    @staticmethod
    def _segment_triangle(start, end, triangle, eps):
        direction = end - start
        edge1, edge2 = triangle[1] - triangle[0], triangle[2] - triangle[0]
        p = np.cross(direction, edge2); determinant = float(np.dot(edge1, p))
        if abs(determinant) <= eps:
            return False
        inverse = 1.0 / determinant; tvec = start - triangle[0]
        u = float(np.dot(tvec, p) * inverse)
        if u < -eps or u > 1 + eps: return False
        q = np.cross(tvec, edge1); v = float(np.dot(direction, q) * inverse)
        if v < -eps or u + v > 1 + eps: return False
        distance = float(np.dot(edge2, q) * inverse)
        return eps < distance < 1.0 - eps

    @staticmethod
    def _coplanar_overlap(first, second, normal, eps):
        axis = int(np.argmax(np.abs(normal)))
        a, b = np.delete(first, axis, axis=1), np.delete(second, axis, axis=1)
        def orient(p, q, r):
            first, second = q - p, r - p
            return float(first[0] * second[1] - first[1] * second[0])
        def segments(p1, p2, q1, q2):
            o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
            return o1 * o2 < -eps and o3 * o4 < -eps
        for i in range(3):
            for j in range(3):
                if segments(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]): return True
        def inside(point, triangle):
            values = [orient(triangle[i], triangle[(i + 1) % 3], point) for i in range(3)]
            return (all(value >= -eps for value in values) or all(value <= eps for value in values)) and any(abs(value) > eps for value in values)
        return inside(a[0], b) or inside(b[0], a)

    def _architectural_consistency(self, mesh, faces):
        findings = []
        penalties = 0.0
        up = np.zeros(3); up[{"x": 0, "y": 1, "z": 2}[self.config.architecture_up_axis]] = 1.0
        floors, ceilings, walls = [], [], []
        for surface_id, plane in enumerate(mesh.planes):
            indices = np.flatnonzero(mesh.surface_ids == surface_id)
            if not len(indices): continue
            points = mesh.vertices[indices]
            equation = np.asarray(plane.equation)
            residual = np.abs(points @ equation[:3] + equation[3])
            tolerance = self.config.effective_architecture_plane_distance() * 2.0
            if float(np.percentile(residual, 95)) > tolerance:
                penalties += 0.12
                findings.append(("plane-residual", f"{plane.plane_id} vertices depart from its supporting plane", np.asarray([points.mean(axis=0)]), {surface_id}))
            vertical = abs(float(np.dot(np.asarray(plane.normal), up)))
            expected = (plane.classification in {"floor", "ceiling"} and vertical >= self.config.architecture_orientation_threshold) or (plane.classification == "wall" and vertical <= 1.0 - self.config.architecture_orientation_threshold)
            if plane.classification in {"wall", "floor", "ceiling"} and not expected:
                penalties += 0.15
                findings.append(("plane-orientation", f"{plane.plane_id} orientation conflicts with its architectural class", np.asarray([plane.centroid]), {surface_id}))
            if plane.classification == "floor": floors.append(plane)
            if plane.classification == "ceiling": ceilings.append(plane)
            if plane.classification == "wall": walls.append(plane)
            for opening in plane.openings:
                face_indices = np.flatnonzero(np.all(mesh.surface_ids[faces] == surface_id, axis=1))
                if not len(face_indices): continue
                centers = mesh.vertices[faces[face_indices]].mean(axis=1)
                corners = np.asarray(opening.corners); centroid = np.asarray(plane.centroid)
                uv_open = np.column_stack(((corners-centroid) @ np.asarray(plane.basis_u), (corners-centroid) @ np.asarray(plane.basis_v)))
                uv_face = np.column_stack(((centers-centroid) @ np.asarray(plane.basis_u), (centers-centroid) @ np.asarray(plane.basis_v)))
                bridged = (uv_face[:, 0] > uv_open[:, 0].min()) & (uv_face[:, 0] < uv_open[:, 0].max()) & (uv_face[:, 1] > uv_open[:, 1].min()) & (uv_face[:, 1] < uv_open[:, 1].max())
                if np.any(bridged):
                    penalties += 0.12
                    findings.append(("opening-occlusion", f"{opening.opening_id} is bridged by reconstructed faces", np.asarray([centers[bridged].mean(axis=0)]), {surface_id}))
        if floors and ceilings:
            floor_height = max(float(np.dot(np.asarray(item.centroid), up)) for item in floors)
            ceiling_height = min(float(np.dot(np.asarray(item.centroid), up)) for item in ceilings)
            if floor_height >= ceiling_height:
                penalties += 0.35
                findings.append(("floor-ceiling-order", "Floor is not below ceiling along the configured up axis", np.asarray([(np.asarray(floors[0].centroid)+np.asarray(ceilings[0].centroid))*0.5]), set()))
        for i, first in enumerate(walls):
            for second in walls[i + 1:]:
                dot = abs(float(np.dot(np.asarray(first.normal), np.asarray(second.normal))))
                if min(dot, abs(1.0 - dot)) > 0.22:
                    penalties += 0.01
        return float(np.clip(1.0 - penalties, 0.0, 1.0)), findings

    @staticmethod
    def _area_weighted_confidence(confidence, faces, area, degenerate):
        face_confidence = np.mean(np.asarray(confidence)[faces], axis=1)
        valid_area = np.where(degenerate, 0.0, area)
        return float(np.clip(np.sum(face_confidence * valid_area) / max(np.sum(valid_area), 1e-12), 0.0, 1.0))

    @staticmethod
    def _add_issue(data, category, severity, indices, locations, mesh, message):
        if len(indices):
            surfaces = {int(mesh.surface_ids[mesh.faces[index, 0]]) for index in indices}
            data.append((category, severity, message, locations[indices], surfaces, "faces"))

    @staticmethod
    def _add_point_issue(data, category, severity, indices, vertices, mesh, message):
        if len(indices): data.append((category, severity, message, vertices[indices], {int(mesh.surface_ids[index]) for index in indices}, "vertices"))

    @staticmethod
    def _add_edge_issue(data, category, severity, edges, vertices, mesh, message):
        if edges:
            locations = np.asarray([(vertices[a] + vertices[b]) * 0.5 for a, b in edges])
            surfaces = {int(mesh.surface_ids[index]) for edge in edges for index in edge}
            data.append((category, severity, message, locations, surfaces, "edges"))

    def _materialize_issues(self, data):
        issues, points, codes = [], [], []
        for number, (category, severity, message, locations, surface_ids, _unit) in enumerate(data, 1):
            locations = np.asarray(locations, dtype=float).reshape((-1, 3)) if np.asarray(locations).size else np.empty((0, 3))
            issues.append(GeometryIssue(
                issue_id=f"issue-{number:04d}", category=category, severity=severity,
                count=max(1, len(locations)), message=message, review_required=severity != "info",
                sample_locations=locations[:20].tolist(), surface_ids=sorted(surface_ids),
            ))
            points.extend(locations.tolist()); codes.extend([self.SEVERITY_CODES[severity]] * len(locations))
        return issues, np.asarray(points, dtype=float).reshape((-1, 3)), np.asarray(codes, dtype=np.uint8)

    def _review_region_count(self, points):
        if not len(points): return 0
        cell = self.config.effective_validation_review_cell_size()
        return len({tuple(index) for index in np.floor(points / cell).astype(int)})

    @staticmethod
    def _vertex_quality(mesh, faces, degenerate, stretched, bad_normals, nonmanifold, discontinuity, intersections):
        quality = np.asarray(mesh.confidence, dtype=float).copy()
        for indices, penalty in ((faces[degenerate], 0.8), (faces[stretched], 0.35)):
            if len(indices): quality[np.unique(indices)] -= penalty
        if len(bad_normals): quality[bad_normals] -= 0.5
        for edges, penalty in ((nonmanifold, 0.8), (discontinuity, 0.3)):
            if edges: quality[np.unique(np.asarray(edges))] -= penalty
        if intersections:
            affected = np.unique(faces[np.asarray(intersections).ravel()]); quality[affected] -= 0.8
        return np.clip(quality, 0.0, 1.0)

    @staticmethod
    def _write_quality_ply(output, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        quality = output.vertex_quality
        colors = np.column_stack((np.clip(2.0 * (1.0 - quality), 0, 1), np.clip(2.0 * quality, 0, 1), np.zeros(len(quality))))
        rgb = np.rint(colors * 255).astype(np.uint8)
        with path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(output.mesh.vertices)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("property float quality_score\nproperty float confidence\n")
            handle.write(f"element face {len(output.mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            for point, color, score, confidence in zip(output.mesh.vertices, rgb, quality, output.mesh.confidence):
                handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {color[0]} {color[1]} {color[2]} {score:.6f} {confidence:.6f}\n")
            for face in output.mesh.faces:
                handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


GeometryQualityValidationService = GeometryValidationService
