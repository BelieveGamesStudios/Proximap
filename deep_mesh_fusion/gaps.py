from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy import ndimage

from .fusion import ConsensusFusionOutput
from .models import (
    ArchitectureReconstructionSummary,
    DeepMeshFusionConfig,
    GapRegion,
    GapRepairResult,
    GapRepairSummary,
    SpatialEvidenceMap,
)
from .reconstruction import ArchitectureMeshOutput, DeepMeshFusionReconstructionService


@dataclass
class GapRepairOutput:
    mesh: ArchitectureMeshOutput
    gaps: List[GapRegion]
    summary: GapRepairSummary
    review_points: np.ndarray
    review_codes: np.ndarray


class EvidenceBasedGapRepairService:
    """Repairs only bounded, evidence-supported gaps and preserves uncertainty otherwise."""

    REVIEW_CODES = {"repaired": 0, "manual-review": 1, "intentional-opening": 2, "exterior-boundary": 3}

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config
        self.reconstruction = DeepMeshFusionReconstructionService(config)

    def recover(
        self,
        architecture: ArchitectureMeshOutput,
        fused: ConsensusFusionOutput,
        evidence_map: Optional[SpatialEvidenceMap] = None,
    ) -> GapRepairOutput:
        grid = self.config.effective_architecture_grid_size()
        vertices = architecture.vertices.copy()
        faces = architecture.faces.copy()
        colors = architecture.colors.copy()
        confidence = architecture.confidence.copy()
        class_codes = architecture.class_codes.copy()
        surface_ids = architecture.surface_ids.copy()
        added_vertices, added_faces, added_colors, added_confidence, added_classes, added_surfaces = [], [], [], [], [], []
        gaps: List[GapRegion] = []
        review_points, review_codes = [], []
        evidence_lookup = {
            tuple(region.grid_index): region for region in evidence_map.regions
        } if evidence_map is not None else {}

        for surface_id, plane in enumerate(architecture.planes):
            result = self._planar_gaps(
                surface_id, plane, vertices, faces, colors, confidence, class_codes, surface_ids,
                fused, evidence_map, evidence_lookup, grid
            )
            gaps.extend(result["gaps"])
            review_points.extend(result["review_points"])
            review_codes.extend(result["review_codes"])
            for patch in result["patches"]:
                offset = len(vertices) + sum(len(part) for part in added_vertices)
                added_vertices.append(patch["vertices"])
                added_faces.append(patch["faces"] + offset)
                added_colors.append(patch["colors"])
                added_confidence.append(patch["confidence"])
                added_classes.append(patch["class_codes"])
                added_surfaces.append(patch["surface_ids"])

        complex_result = self._complex_gaps(
            vertices, faces, colors, confidence, class_codes, surface_ids,
            len(vertices) + sum(len(part) for part in added_vertices),
        )
        gaps.extend(complex_result["gaps"])
        review_points.extend(complex_result["review_points"])
        review_codes.extend(complex_result["review_codes"])
        for patch in complex_result["patches"]:
            offset = len(vertices) + sum(len(part) for part in added_vertices)
            added_vertices.append(patch["vertices"])
            added_faces.append(patch["faces"] + offset)
            added_colors.append(patch["colors"])
            added_confidence.append(patch["confidence"])
            added_classes.append(patch["class_codes"])
            added_surfaces.append(patch["surface_ids"])

        if added_vertices:
            vertices = np.vstack([vertices] + added_vertices)
            faces = np.vstack([faces] + added_faces)
            colors = np.vstack([colors] + added_colors)
            confidence = np.concatenate([confidence] + added_confidence)
            class_codes = np.concatenate([class_codes] + added_classes)
            surface_ids = np.concatenate([surface_ids] + added_surfaces)
            vertices, faces, colors, confidence, class_codes, surface_ids = self.reconstruction._merge_vertices(
                vertices, faces, colors, confidence, class_codes, surface_ids
            )
        normals = self.reconstruction._vertex_normals(vertices, faces)
        boundary, nonmanifold, components, area = self.reconstruction._mesh_metrics(vertices, faces)
        updated_summary = asdict(architecture.summary)
        updated_summary.update({
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "boundary_edge_count": boundary,
            "nonmanifold_edge_count": nonmanifold,
            "connected_component_count": components,
            "surface_area": area,
        })
        mesh = ArchitectureMeshOutput(
            vertices=vertices,
            faces=faces,
            normals=normals,
            colors=colors,
            confidence=confidence,
            class_codes=class_codes,
            surface_ids=surface_ids,
            planes=architecture.planes,
            edges=architecture.edges,
            corners=architecture.corners,
            summary=ArchitectureReconstructionSummary(**updated_summary),
        )
        repaired = [gap for gap in gaps if gap.decision == "repair"]
        summary = GapRepairSummary(
            detected_gap_count=len(gaps),
            repaired_gap_count=len(repaired),
            observed_geometry_repair_count=sum(gap.classification == "observed-geometry" for gap in repaired),
            planar_continuation_count=sum(gap.classification == "planar-continuation" for gap in repaired),
            surface_interpolation_count=sum(gap.classification == "surface-interpolation" for gap in repaired),
            intentional_opening_count=sum(gap.classification == "intentional-opening" for gap in gaps),
            exterior_boundary_count=sum(gap.classification == "exterior-boundary" for gap in gaps),
            unresolved_gap_count=sum(gap.review_required for gap in gaps),
            added_vertex_count=max(0, len(vertices) - len(architecture.vertices)),
            added_face_count=max(0, len(faces) - len(architecture.faces)),
            final_vertex_count=len(vertices),
            final_face_count=len(faces),
        )
        return GapRepairOutput(
            mesh=mesh,
            gaps=gaps,
            summary=summary,
            review_points=np.asarray(review_points, dtype=float) if review_points else np.empty((0, 3), dtype=float),
            review_codes=np.asarray(review_codes, dtype=np.uint8) if review_codes else np.empty(0, dtype=np.uint8),
        )

    def export(self, output: GapRepairOutput, mesh_path: str, report_path: str, review_path: str) -> GapRepairResult:
        self.reconstruction._write_mesh_ply(output.mesh, Path(mesh_path))
        payload = {
            "schema_version": 1,
            "hierarchy": ["observed-geometry", "confident-inference", "manual-review"],
            "review_codes": self.REVIEW_CODES,
            "summary": asdict(output.summary),
            "gaps": [asdict(gap) for gap in output.gaps],
        }
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(target)
        self._write_review_ply(output, Path(review_path))
        return GapRepairResult(str(Path(mesh_path)), str(target), str(Path(review_path)), output.summary)

    def _planar_gaps(
        self, surface_id, plane, vertices, faces, vertex_colors, vertex_confidence, class_codes,
        surface_ids, fused, evidence_map, evidence_lookup, grid
    ):
        u0, u1, v0, v1 = plane.projected_bounds
        width = max(1, int(round((u1 - u0) / grid)))
        height = max(1, int(round((v1 - v0) / grid)))
        occupied = np.zeros((width, height), dtype=bool)
        plane_faces = faces[np.all(surface_ids[faces] == surface_id, axis=1)]
        if len(plane_faces):
            centers = vertices[plane_faces].mean(axis=1)
            local = centers - np.asarray(plane.centroid)
            coords = np.column_stack((local @ np.asarray(plane.basis_u), local @ np.asarray(plane.basis_v)))
            indices = np.floor((coords - [u0, v0]) / grid).astype(int)
            valid = (indices[:, 0] >= 0) & (indices[:, 0] < width) & (indices[:, 1] >= 0) & (indices[:, 1] < height)
            occupied[indices[valid, 0], indices[valid, 1]] = True
        labels, count = ndimage.label(~occupied, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        gaps, patches, review_points, review_codes = [], [], [], []
        plane_vertex_mask = surface_ids == surface_id
        boundary_confidence = float(np.mean(vertex_confidence[plane_vertex_mask])) if np.any(plane_vertex_mask) else plane.confidence
        normal = np.asarray(plane.normal)
        centroid = np.asarray(plane.centroid)
        basis_u, basis_v = np.asarray(plane.basis_u), np.asarray(plane.basis_v)
        fused_distance = np.abs(np.asarray(fused.points) @ normal + plane.equation[3])
        fused_near = fused_distance <= self.config.effective_architecture_plane_distance()
        fused_local = np.asarray(fused.points)[fused_near] - centroid
        fused_coords = np.column_stack((fused_local @ basis_u, fused_local @ basis_v)) if len(fused_local) else np.empty((0, 2))
        fused_indices = np.floor((fused_coords - [u0, v0]) / grid).astype(int) if len(fused_coords) else np.empty((0, 2), dtype=int)
        fused_source_indices = np.flatnonzero(fused_near)

        for label_id in range(1, count + 1):
            cells = np.argwhere(labels == label_id)
            if not len(cells):
                continue
            imin, jmin = cells.min(axis=0)
            imax, jmax = cells.max(axis=0)
            touches_border = imin == 0 or jmin == 0 or imax == width - 1 or jmax == height - 1
            area = float(len(cells) * grid * grid)
            perimeter_edges = 0
            cell_set = {tuple(cell) for cell in cells}
            for i, j in cell_set:
                perimeter_edges += sum((i + di, j + dj) not in cell_set for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            perimeter = float(perimeter_edges * grid)
            cell_centers_uv = np.column_stack((u0 + (cells[:, 0] + 0.5) * grid, v0 + (cells[:, 1] + 0.5) * grid))
            opening_overlap = 0.0
            for opening in plane.openings:
                corners = np.asarray(opening.corners) - centroid
                opening_uv = np.column_stack((corners @ basis_u, corners @ basis_v))
                inside = (
                    (cell_centers_uv[:, 0] >= opening_uv[:, 0].min())
                    & (cell_centers_uv[:, 0] <= opening_uv[:, 0].max())
                    & (cell_centers_uv[:, 1] >= opening_uv[:, 1].min())
                    & (cell_centers_uv[:, 1] <= opening_uv[:, 1].max())
                )
                opening_overlap = max(opening_overlap, float(np.mean(inside)))
            observed_mask = np.asarray([tuple(index) in cell_set for index in fused_indices], dtype=bool)
            observed_indices = fused_source_indices[observed_mask]
            observed_count = int(len(observed_indices))
            evidence_count, evidence_confidence = self._gap_evidence(
                cells, u0, v0, grid, centroid, basis_u, basis_v, evidence_map, evidence_lookup
            )
            repair_confidence = 0.0
            repaired_faces = 0
            if opening_overlap >= self.config.gap_intentional_opening_overlap:
                classification, decision, reason = "intentional-opening", "preserve", "Matches a detected doorway or window"
                review_code = self.REVIEW_CODES["intentional-opening"]
            elif touches_border:
                classification, decision, reason = "exterior-boundary", "preserve", "Unbounded surface edge; insufficient enclosure for repair"
                review_code = self.REVIEW_CODES["exterior-boundary"]
            elif observed_count:
                classification, decision, reason = "observed-geometry", "repair", "Fused observations exist inside the mesh gap"
                repair_confidence = float(np.clip(np.mean(fused.confidence[observed_indices]) - self.config.gap_confidence_penalty * 0.25, 0.0, 1.0))
                review_code = self.REVIEW_CODES["repaired"]
            elif (
                area <= self.config.gap_max_planar_area
                and plane.confidence >= self.config.gap_min_plane_confidence
                and boundary_confidence >= self.config.gap_min_boundary_confidence
            ):
                classification, decision, reason = "planar-continuation", "repair", "Closed high-confidence planar boundary supports continuation"
                size_penalty = self.config.gap_confidence_penalty * area / self.config.gap_max_planar_area
                repair_confidence = float(np.clip(min(plane.confidence, boundary_confidence, evidence_confidence or 1.0) - size_penalty, 0.0, 1.0))
                review_code = self.REVIEW_CODES["repaired"]
            else:
                classification, decision, reason = "uncertain", "manual-review", "Boundary, confidence, or size evidence is insufficient"
                repair_confidence = float(min(plane.confidence, boundary_confidence, evidence_confidence))
                review_code = self.REVIEW_CODES["manual-review"]
            world_centers = centroid + cell_centers_uv[:, :1] * basis_u + cell_centers_uv[:, 1:] * basis_v
            bounds_min, bounds_max = world_centers.min(axis=0), world_centers.max(axis=0)
            gap_id = f"gap-{len(gaps) + 1:04d}-surface-{surface_id}"
            if decision == "repair":
                existing = np.flatnonzero(plane_vertex_mask)
                plane_color = np.mean(vertex_colors[existing], axis=0) if len(existing) else np.asarray([0.65, 0.65, 0.65])
                plane_class = int(class_codes[existing[0]]) if len(existing) else 4
                patch = self._planar_patch(
                    cells, u0, v0, grid, centroid, basis_u, basis_v,
                    plane_color, repair_confidence, surface_id, plane_class,
                )
                patches.append(patch)
                repaired_faces = len(patch["faces"])
            gaps.append(GapRegion(
                gap_id=gap_id,
                surface_id=surface_id,
                plane_id=plane.plane_id,
                classification=classification,
                decision=decision,
                area=area,
                perimeter=perimeter,
                boundary_confidence=boundary_confidence,
                repair_confidence=repair_confidence,
                observed_point_count=observed_count,
                evidence_observation_count=evidence_count,
                bounds_min=bounds_min.tolist(),
                bounds_max=bounds_max.tolist(),
                repaired_face_count=repaired_faces,
                review_required=decision == "manual-review",
                reason=reason,
            ))
            review_points.append(np.mean(world_centers, axis=0))
            review_codes.append(review_code)
        return {"gaps": gaps, "patches": patches, "review_points": review_points, "review_codes": review_codes}

    def _planar_patch(self, cells, u0, v0, grid, centroid, basis_u, basis_v, color, confidence, surface_id, class_code):
        vertex_map, vertices, faces = {}, [], []

        def vertex(i, j):
            key = (int(i), int(j))
            if key not in vertex_map:
                vertex_map[key] = len(vertices)
                vertices.append(centroid + (u0 + i * grid) * basis_u + (v0 + j * grid) * basis_v)
            return vertex_map[key]

        for i, j in cells:
            a, b, c, d = vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)
            faces.extend(((a, b, c), (a, c, d)))
        vertices = np.asarray(vertices)
        return {
            "vertices": vertices,
            "faces": np.asarray(faces, dtype=np.int64),
            "colors": np.tile(color, (len(vertices), 1)),
            "confidence": np.full(len(vertices), confidence),
            "class_codes": np.full(len(vertices), class_code, dtype=np.uint8),
            "surface_ids": np.full(len(vertices), surface_id, dtype=np.int32),
        }

    def _complex_gaps(self, vertices, faces, colors, confidence, class_codes, surface_ids, _offset):
        edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique[counts == 1]
        boundary_edges = boundary_edges[(surface_ids[boundary_edges[:, 0]] == -1) & (surface_ids[boundary_edges[:, 1]] == -1)]
        adjacency = {}
        for left, right in boundary_edges:
            adjacency.setdefault(int(left), set()).add(int(right))
            adjacency.setdefault(int(right), set()).add(int(left))
        visited, gaps, patches, review_points, review_codes = set(), [], [], [], []
        for start in adjacency:
            if start in visited:
                continue
            component, stack = set(), [start]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(adjacency[current] - component)
            visited.update(component)
            if len(component) < 3 or any(len(adjacency[index] & component) != 2 for index in component):
                continue
            loop = [start]
            previous, current = None, start
            while True:
                candidates = [item for item in adjacency[current] if item != previous]
                next_vertex = candidates[0]
                if next_vertex == start:
                    break
                loop.append(next_vertex)
                previous, current = current, next_vertex
                if len(loop) > len(component):
                    break
            loop_points = vertices[loop]
            perimeter = float(np.sum(np.linalg.norm(np.roll(loop_points, -1, axis=0) - loop_points, axis=1)))
            centered = loop_points - loop_points.mean(axis=0)
            _values, vectors = np.linalg.eigh(np.cov(centered.T))
            basis = vectors[:, 1:]
            projected = centered @ basis
            area = float(abs(np.sum(projected[:, 0] * np.roll(projected[:, 1], -1) - projected[:, 1] * np.roll(projected[:, 0], -1))) * 0.5)
            boundary_conf = float(np.mean(confidence[loop]))
            repair = (
                perimeter <= self.config.gap_max_complex_perimeter
                and area <= self.config.gap_max_complex_area
                and boundary_conf >= self.config.gap_min_boundary_confidence
            )
            center = loop_points.mean(axis=0)
            if repair:
                repair_confidence = float(np.clip(boundary_conf - self.config.gap_confidence_penalty, 0.0, 1.0))
                patch_vertices = center[None, :]
                center_index = 0
                patch_faces = []
                # Include loop vertices in this patch so final vertex merging reconnects the fan.
                patch_vertices = np.vstack((patch_vertices, loop_points))
                for index in range(len(loop)):
                    patch_faces.append((center_index, index + 1, (index + 1) % len(loop) + 1))
                patches.append({
                    "vertices": patch_vertices,
                    "faces": np.asarray(patch_faces, dtype=np.int64),
                    "colors": np.vstack((np.mean(colors[loop], axis=0), colors[loop])),
                    "confidence": np.r_[repair_confidence, confidence[loop]],
                    "class_codes": np.full(len(patch_vertices), class_codes[loop[0]], dtype=np.uint8),
                    "surface_ids": np.full(len(patch_vertices), -1, dtype=np.int32),
                })
                classification, decision, reason = "surface-interpolation", "repair", "Small closed high-confidence residual-surface boundary"
                review_code = self.REVIEW_CODES["repaired"]
                repaired_faces = len(patch_faces)
            else:
                repair_confidence = 0.0
                classification, decision, reason = "uncertain", "manual-review", "Residual boundary exceeds conservative interpolation limits"
                review_code = self.REVIEW_CODES["manual-review"]
                repaired_faces = 0
            gaps.append(GapRegion(
                gap_id=f"gap-complex-{len(gaps) + 1:04d}", surface_id=-1, plane_id=None,
                classification=classification, decision=decision, area=area, perimeter=perimeter,
                boundary_confidence=boundary_conf, repair_confidence=repair_confidence,
                observed_point_count=0, evidence_observation_count=0,
                bounds_min=loop_points.min(axis=0).tolist(), bounds_max=loop_points.max(axis=0).tolist(),
                repaired_face_count=repaired_faces, review_required=not repair, reason=reason,
            ))
            review_points.append(center)
            review_codes.append(review_code)
        return {"gaps": gaps, "patches": patches, "review_points": review_points, "review_codes": review_codes}

    @staticmethod
    def _gap_evidence(cells, u0, v0, grid, centroid, basis_u, basis_v, evidence_map, lookup):
        if evidence_map is None:
            return 0, 0.0
        counts, confidence = [], []
        origin = np.asarray(evidence_map.grid_origin)
        for i, j in cells:
            world = centroid + (u0 + (i + 0.5) * grid) * basis_u + (v0 + (j + 0.5) * grid) * basis_v
            key = tuple(np.floor((world - origin) / evidence_map.cell_size).astype(int))
            region = lookup.get(key)
            if region is not None:
                counts.append(region.observation_count)
                confidence.append(region.confidence)
        return (max(counts, default=0), float(np.mean(confidence)) if confidence else 0.0)

    @staticmethod
    def _write_review_ply(output, path):
        colors = {
            0: (32, 220, 80),
            1: (255, 48, 48),
            2: (40, 120, 255),
            3: (160, 160, 160),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii") as handle:
            handle.write("ply\nformat ascii 1.0\ncomment Proximap gap repair review markers\n")
            handle.write(f"element vertex {len(output.review_points)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar review_code\nend_header\n")
            for point, code in zip(output.review_points, output.review_codes):
                red, green, blue = colors[int(code)]
                handle.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {red} {green} {blue} {code}\n")


GapRepairService = EvidenceBasedGapRepairService
