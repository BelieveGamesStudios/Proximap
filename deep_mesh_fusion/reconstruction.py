from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .fusion import ConsensusFusionOutput
from .models import (
    ArchitecturalCorner,
    ArchitecturalEdge,
    ArchitecturalOpening,
    ArchitecturalPlane,
    ArchitectureReconstructionResult,
    ArchitectureReconstructionSummary,
    DeepMeshFusionConfig,
)


@dataclass
class ArchitectureMeshOutput:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    class_codes: np.ndarray
    surface_ids: np.ndarray
    planes: List[ArchitecturalPlane]
    edges: List[ArchitecturalEdge]
    corners: List[ArchitecturalCorner]
    summary: ArchitectureReconstructionSummary


class DeepMeshFusionReconstructionService:
    """Hybrid planar/edge-aware architecture reconstruction with a general residual path."""

    CLASS_CODES = {
        "wall": 1,
        "floor": 2,
        "ceiling": 3,
        "large-planar-surface": 4,
        "complex": 5,
    }

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config
        self.last_complex_report = {
            "method": self.config.complex_reconstruction_method,
            "requested_backend": self.config.complex_reconstruction_backend,
            "backend": None,
            "fallback_reason": None,
            "input_point_count": 0,
            "cleaned_point_count": 0,
            "rejected_point_count": 0,
            "point_component_count": 0,
            "support_trimmed_vertex_count": 0,
            "removed_mesh_component_count": 0,
            "generated_vertex_count": 0,
            "generated_face_count": 0,
            "watertight": False,
        }

    def reconstruct(self, fused: ConsensusFusionOutput) -> ArchitectureMeshOutput:
        import open3d as o3d

        points = np.asarray(fused.points, dtype=np.float64)
        if len(points) < 3:
            raise ValueError("Architecture reconstruction requires at least three fused points")
        if hasattr(o3d.utility, "random"):
            o3d.utility.random.seed(self.config.random_seed)
        internal_planes, residual_indices = self._detect_planes(points, fused.confidence)
        self._classify_planes(internal_planes, points)

        vertex_parts, face_parts, color_parts, confidence_parts, class_parts, surface_parts = [], [], [], [], [], []
        plane_models = []
        vertex_offset = 0
        for surface_index, plane in enumerate(internal_planes):
            mesh = self._mesh_plane(plane, points, fused.colors, fused.confidence, surface_index)
            plane_models.append(mesh["model"])
            if len(mesh["vertices"]):
                vertex_parts.append(mesh["vertices"])
                face_parts.append(mesh["faces"] + vertex_offset)
                color_parts.append(mesh["colors"])
                confidence_parts.append(mesh["confidence"])
                class_parts.append(mesh["class_codes"])
                surface_parts.append(np.full(len(mesh["vertices"]), surface_index, dtype=np.int32))
                vertex_offset += len(mesh["vertices"])

        complex_mesh = self._mesh_complex(
            points[residual_indices], fused.colors[residual_indices],
            fused.confidence[residual_indices], fused.normals[residual_indices],
        )
        if complex_mesh is not None:
            vertex_parts.append(complex_mesh["vertices"])
            face_parts.append(complex_mesh["faces"] + vertex_offset)
            color_parts.append(complex_mesh["colors"])
            confidence_parts.append(complex_mesh["confidence"])
            class_parts.append(np.full(len(complex_mesh["vertices"]), self.CLASS_CODES["complex"], dtype=np.uint8))
            surface_parts.append(np.full(len(complex_mesh["vertices"]), -1, dtype=np.int32))

        if not vertex_parts or not face_parts:
            raise ValueError("Architecture reconstruction could not generate mesh faces")
        vertices = np.vstack(vertex_parts)
        faces = np.vstack(face_parts).astype(np.int64)
        colors = np.vstack(color_parts)
        confidence = np.concatenate(confidence_parts)
        class_codes = np.concatenate(class_parts)
        surface_ids = np.concatenate(surface_parts)
        vertices, faces, colors, confidence, class_codes, surface_ids = self._merge_vertices(
            vertices, faces, colors, confidence, class_codes, surface_ids
        )
        normals = self._vertex_normals(vertices, faces)
        edges = self._architectural_edges(internal_planes)
        corners = self._architectural_corners(internal_planes)
        boundary_edges, nonmanifold_edges, components, area = self._mesh_metrics(vertices, faces)
        openings = [opening for plane in plane_models for opening in plane.openings]
        summary = ArchitectureReconstructionSummary(
            input_point_count=len(points),
            plane_count=len(plane_models),
            wall_count=sum(plane.classification == "wall" for plane in plane_models),
            floor_count=sum(plane.classification == "floor" for plane in plane_models),
            ceiling_count=sum(plane.classification == "ceiling" for plane in plane_models),
            doorway_count=sum(opening.classification == "doorway" for opening in openings),
            window_count=sum(opening.classification == "window" for opening in openings),
            edge_count=len(edges),
            corner_count=len(corners),
            complex_point_count=len(residual_indices),
            vertex_count=len(vertices),
            face_count=len(faces),
            boundary_edge_count=boundary_edges,
            nonmanifold_edge_count=nonmanifold_edges,
            connected_component_count=components,
            surface_area=area,
        )
        return ArchitectureMeshOutput(
            vertices=vertices,
            faces=faces,
            normals=normals,
            colors=colors,
            confidence=confidence,
            class_codes=class_codes,
            surface_ids=surface_ids,
            planes=plane_models,
            edges=edges,
            corners=corners,
            summary=summary,
        )

    def export(self, output: ArchitectureMeshOutput, mesh_path: str, report_path: str) -> ArchitectureReconstructionResult:
        self._write_mesh_ply(output, Path(mesh_path))
        payload = {
            "schema_version": 1,
            "strategy": "architecture-aware-planar+screened-poisson",
            "up_axis": self.config.architecture_up_axis,
            "class_codes": self.CLASS_CODES,
            "summary": asdict(output.summary),
            "complex_reconstruction": self.last_complex_report,
            "planes": [asdict(item) for item in output.planes],
            "edges": [asdict(item) for item in output.edges],
            "corners": [asdict(item) for item in output.corners],
        }
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(target)
        return ArchitectureReconstructionResult(str(Path(mesh_path)), str(target), output.summary)

    def _detect_planes(self, points, confidence):
        import open3d as o3d

        remaining = np.arange(len(points), dtype=np.int64)
        planes = []
        minimum = max(20, min(self.config.architecture_plane_min_points, int(np.ceil(len(points) * self.config.architecture_plane_min_ratio))))
        distance = self.config.effective_architecture_plane_distance()
        while len(remaining) >= minimum and len(planes) < self.config.architecture_max_planes:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[remaining]))
            equation, local_inliers = cloud.segment_plane(distance_threshold=distance, ransac_n=3, num_iterations=1200)
            if len(local_inliers) < minimum:
                break
            global_inliers = remaining[np.asarray(local_inliers, dtype=np.int64)]
            normal = np.asarray(equation[:3], dtype=float)
            length = max(float(np.linalg.norm(normal)), 1e-12)
            normal /= length
            d = float(equation[3] / length)
            inlier_points = points[global_inliers]
            centroid = inlier_points.mean(axis=0)
            centroid -= (float(np.dot(normal, centroid)) + d) * normal
            basis_u, basis_v = self._plane_basis(normal)
            coordinates = np.column_stack(((inlier_points - centroid) @ basis_u, (inlier_points - centroid) @ basis_v))
            planes.append({
                "plane_id": f"plane-{len(planes) + 1:03d}",
                "equation": np.r_[normal, d],
                "normal": normal,
                "centroid": centroid,
                "basis_u": basis_u,
                "basis_v": basis_v,
                "coordinates": coordinates,
                "indices": global_inliers,
                "confidence": float(np.mean(confidence[global_inliers])),
                "bounds_min": inlier_points.min(axis=0),
                "bounds_max": inlier_points.max(axis=0),
            })
            keep = np.ones(len(remaining), dtype=bool)
            keep[np.asarray(local_inliers, dtype=np.int64)] = False
            remaining = remaining[keep]
        return planes, remaining

    def _classify_planes(self, planes, all_points):
        up = self._up_vector()
        horizontal = []
        threshold = self.config.architecture_orientation_threshold
        for plane in planes:
            alignment = abs(float(np.dot(plane["normal"], up)))
            if alignment >= threshold:
                horizontal.append(plane)
                plane["classification"] = "large-planar-surface"
            elif alignment <= np.sqrt(max(0.0, 1.0 - threshold ** 2)):
                plane["classification"] = "wall"
            else:
                plane["classification"] = "large-planar-surface"
        if horizontal:
            ordered = sorted(horizontal, key=lambda item: float(np.dot(item["centroid"], up)))
            scene_midpoint = float(np.dot((all_points.min(axis=0) + all_points.max(axis=0)) * 0.5, up))
            if len(ordered) == 1:
                ordered[0]["classification"] = "floor" if float(np.dot(ordered[0]["centroid"], up)) <= scene_midpoint else "ceiling"
            else:
                ordered[0]["classification"] = "floor"
                ordered[-1]["classification"] = "ceiling"

    def _mesh_plane(self, plane, points, source_colors, source_confidence, surface_index):
        grid = self.config.effective_architecture_grid_size()
        coords = plane["coordinates"]
        lower = np.floor(coords.min(axis=0) / grid) * grid
        cell_indices = np.floor((coords - lower) / grid).astype(np.int64)
        shape = tuple((cell_indices.max(axis=0) + 1).tolist())
        occupied = np.zeros(shape, dtype=bool)
        occupied[cell_indices[:, 0], cell_indices[:, 1]] = True
        if self.config.architecture_grid_closing_iterations:
            occupied = ndimage.binary_closing(
                occupied,
                structure=np.ones((3, 3), dtype=bool),
                iterations=self.config.architecture_grid_closing_iterations,
                border_value=1,
            )
        openings, opening_mask = self._detect_openings(plane, occupied, lower, grid)
        occupied[opening_mask] = False
        vertex_map = {}
        vertices, faces = [], []

        def vertex(i, j):
            key = (int(i), int(j))
            if key not in vertex_map:
                world = plane["centroid"] + (lower[0] + i * grid) * plane["basis_u"] + (lower[1] + j * grid) * plane["basis_v"]
                vertex_map[key] = len(vertices)
                vertices.append(world)
            return vertex_map[key]

        for i, j in np.argwhere(occupied):
            a, b, c, d = vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)
            faces.extend(((a, b, c), (a, c, d)))
        vertices = np.asarray(vertices, dtype=float) if vertices else np.empty((0, 3), dtype=float)
        faces = np.asarray(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)
        inlier_colors = source_colors[plane["indices"]]
        mean_color = np.mean(inlier_colors, axis=0) if len(inlier_colors) else np.asarray([0.65, 0.65, 0.65])
        projected_bounds = [float(lower[0]), float(lower[0] + shape[0] * grid), float(lower[1]), float(lower[1] + shape[1] * grid)]
        area = float(np.sum(occupied) * grid * grid)
        model = ArchitecturalPlane(
            plane_id=plane["plane_id"],
            classification=plane["classification"],
            equation=plane["equation"].tolist(),
            normal=plane["normal"].tolist(),
            centroid=plane["centroid"].tolist(),
            basis_u=plane["basis_u"].tolist(),
            basis_v=plane["basis_v"].tolist(),
            projected_bounds=projected_bounds,
            bounds_min=plane["bounds_min"].tolist(),
            bounds_max=plane["bounds_max"].tolist(),
            inlier_point_count=len(plane["indices"]),
            area=area,
            confidence=plane["confidence"],
            openings=openings,
        )
        return {
            "vertices": vertices,
            "faces": faces,
            "colors": np.tile(mean_color, (len(vertices), 1)),
            "confidence": np.full(len(vertices), plane["confidence"], dtype=float),
            "class_codes": np.full(len(vertices), self.CLASS_CODES[plane["classification"]], dtype=np.uint8),
            "model": model,
        }

    def _detect_openings(self, plane, occupied, lower, grid):
        mask = np.zeros_like(occupied, dtype=bool)
        if plane["classification"] != "wall" or min(occupied.shape) < 3:
            return [], mask
        empty = ~occupied
        labels, count = ndimage.label(empty, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        openings = []
        for label_id in range(1, count + 1):
            cells = np.argwhere(labels == label_id)
            if not len(cells):
                continue
            imin, jmin = cells.min(axis=0)
            imax, jmax = cells.max(axis=0)
            width = float((imax - imin + 1) * grid)
            height = float((jmax - jmin + 1) * grid)
            area = float(len(cells) * grid * grid)
            touches_left = imin == 0
            touches_right = imax == occupied.shape[0] - 1
            touches_top = jmax == occupied.shape[1] - 1
            bottom_height = float(lower[1] + jmin * grid)
            plane_bottom = float(lower[1])
            doorway = bottom_height <= plane_bottom + self.config.doorway_floor_tolerance and not (touches_left or touches_right or touches_top)
            window = not (touches_left or touches_right or jmin == 0 or touches_top)
            if width < self.config.opening_min_width or height < self.config.opening_min_height or area < self.config.opening_min_area:
                continue
            if not doorway and not window:
                continue
            classification = "doorway" if doorway else "window"
            u0, u1 = lower[0] + imin * grid, lower[0] + (imax + 1) * grid
            v0, v1 = lower[1] + jmin * grid, lower[1] + (jmax + 1) * grid
            corners = [
                plane["centroid"] + u * plane["basis_u"] + v * plane["basis_v"]
                for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
            ]
            openings.append(ArchitecturalOpening(
                opening_id=f"{plane['plane_id']}-opening-{len(openings) + 1:02d}",
                plane_id=plane["plane_id"],
                classification=classification,
                width=width,
                height=height,
                area=area,
                confidence=float(np.clip(plane["confidence"] * min(1.0, area / (self.config.opening_min_area * 2.0)), 0.0, 1.0)),
                corners=[corner.tolist() for corner in corners],
            ))
            mask[labels == label_id] = True
        return openings, mask

    def _mesh_complex(self, points, colors, confidence, normals=None):
        import open3d as o3d

        points = np.asarray(points, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64) if normals is not None else None
        input_count = len(points)
        self.last_complex_report = {
            "method": self.config.complex_reconstruction_method,
            "requested_backend": self.config.complex_reconstruction_backend,
            "backend": None,
            "fallback_reason": None,
            "input_point_count": input_count,
            "cleaned_point_count": 0,
            "rejected_point_count": input_count,
            "point_component_count": 0,
            "support_trimmed_vertex_count": 0,
            "removed_mesh_component_count": 0,
            "generated_vertex_count": 0,
            "generated_face_count": 0,
            "watertight": False,
        }
        if self.config.complex_reconstruction_method == "none" or input_count < self.config.complex_reconstruction_min_points:
            return None

        finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
        supported = finite & (confidence >= self.config.complex_min_confidence)
        source_indices = np.flatnonzero(supported)
        if len(source_indices) < self.config.complex_reconstruction_min_points:
            return None

        # Fused points are already locally selected; this second pass removes
        # quantized duplicates and isolated survivors before implicit meshing.
        cell = self.config.effective_fusion_cell_size()
        quantized = np.floor(points[source_indices] / max(cell * 0.5, 1e-12)).astype(np.int64)
        best_by_cell = {}
        for local_index, key_values in enumerate(quantized):
            key = tuple(int(value) for value in key_values)
            original_index = int(source_indices[local_index])
            incumbent = best_by_cell.get(key)
            if incumbent is None or confidence[original_index] > confidence[incumbent]:
                best_by_cell[key] = original_index
        source_indices = np.asarray(sorted(best_by_cell.values()), dtype=np.int64)
        if len(source_indices) < self.config.complex_reconstruction_min_points:
            return None

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[source_indices]))
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors[source_indices], 0.0, 1.0))
        neighbors = min(self.config.complex_outlier_neighbors, len(source_indices) - 1)
        if neighbors >= 2:
            _filtered, retained = cloud.remove_statistical_outlier(
                nb_neighbors=neighbors, std_ratio=self.config.complex_outlier_std_ratio
            )
            retained = np.asarray(retained, dtype=np.int64)
            if len(retained) >= self.config.complex_reconstruction_min_points:
                source_indices = source_indices[retained]
                cloud = cloud.select_by_index(retained.tolist())

        # Treat disconnected residual islands independently. This prevents a
        # global normal-propagation pass from flipping one object to match an
        # unrelated surface and drops tiny point islands before Poisson can
        # inflate them into closed bubbles.
        component_radius = cell * self.config.complex_component_radius_multiplier
        labels = np.asarray(cloud.cluster_dbscan(
            eps=component_radius,
            min_points=3,
            print_progress=False,
        ), dtype=np.int32)
        component_ids, component_counts = np.unique(labels[labels >= 0], return_counts=True)
        accepted_components = component_ids[component_counts >= self.config.complex_min_component_points]
        keep = np.isin(labels, accepted_components)
        if np.count_nonzero(keep) >= self.config.complex_reconstruction_min_points:
            retained = np.flatnonzero(keep)
            source_indices = source_indices[retained]
            cloud = cloud.select_by_index(retained.tolist())
            labels = labels[retained]
        else:
            labels = np.zeros(len(source_indices), dtype=np.int32)
        self.last_complex_report["point_component_count"] = int(len(np.unique(labels[labels >= 0])))

        cleaned_count = len(source_indices)
        self.last_complex_report["cleaned_point_count"] = cleaned_count
        self.last_complex_report["rejected_point_count"] = input_count - cleaned_count
        if cleaned_count < self.config.complex_reconstruction_min_points:
            return None

        radius = cell * self.config.complex_normal_radius_multiplier
        supplied = normals[source_indices] if normals is not None and len(normals) == input_count else None
        self._orient_component_normals(cloud, labels, supplied, radius)

        mesh = None
        densities = np.empty(0, dtype=float)
        if self.config.complex_reconstruction_backend == "pymeshlab":
            mesh = self._pymeshlab_screened_poisson(cloud)
            if mesh is None:
                self.last_complex_report["fallback_reason"] = "PyMeshLab worker unavailable or reconstruction failed"
        if mesh is None:
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    cloud,
                    depth=self.config.complex_poisson_depth,
                    scale=self.config.complex_poisson_scale,
                    linear_fit=True,
                )
                self.last_complex_report["backend"] = "open3d-screened-poisson"
            except (RuntimeError, ValueError):
                return None
        else:
            self.last_complex_report["backend"] = "pymeshlab-screened-poisson"
        densities = np.asarray(densities, dtype=float)
        quantile = self.config.complex_poisson_density_quantile
        if quantile > 0 and len(densities):
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, quantile))

        # Screened Poisson is implicit and will otherwise close or inflate
        # unsupported space. Require every generated vertex to remain near a
        # sufficient number of observed fused points.
        support_tree = cKDTree(points[source_indices])
        generated = np.asarray(mesh.vertices)
        if len(generated):
            support_radius = cell * self.config.complex_support_distance_multiplier
            distances, _nearest = support_tree.query(generated, k=1)
            try:
                support_counts = support_tree.query_ball_point(generated, support_radius, return_length=True)
            except TypeError:
                support_counts = np.asarray([len(item) for item in support_tree.query_ball_point(generated, support_radius)])
            unsupported = (distances > support_radius) | (np.asarray(support_counts) < self.config.complex_min_support_neighbors)
            self.last_complex_report["support_trimmed_vertex_count"] = int(np.count_nonzero(unsupported))
            if np.any(unsupported):
                mesh.remove_vertices_by_mask(unsupported)
        mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices(); mesh.remove_unreferenced_vertices()

        # Keep meaningful objects while discarding small floating shells made
        # from residual samples. Always preserve the largest component.
        if len(mesh.triangles):
            triangle_labels, triangle_counts, _areas = mesh.cluster_connected_triangles()
            triangle_labels = np.asarray(triangle_labels, dtype=np.int64)
            triangle_counts = np.asarray(triangle_counts, dtype=np.int64)
            largest = int(np.argmax(triangle_counts)) if len(triangle_counts) else -1
            small = np.asarray([
                label != largest and triangle_counts[label] < self.config.complex_min_component_faces
                for label in triangle_labels
            ], dtype=bool)
            removed_labels = set(triangle_labels[small].tolist())
            self.last_complex_report["removed_mesh_component_count"] = len(removed_labels)
            if np.any(small):
                mesh.remove_triangles_by_mask(small)
                mesh.remove_unreferenced_vertices()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        if not len(vertices) or not len(faces):
            return None
        distances, nearest = cKDTree(points[source_indices]).query(vertices, k=1)
        nearest_source = source_indices[nearest]
        generated_confidence = confidence[nearest_source] * np.exp(-distances / max(cell * 2.0, 1e-12))
        edge_counts = {}
        for face in faces:
            for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = (int(min(left, right)), int(max(left, right)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        watertight = bool(edge_counts and all(count == 2 for count in edge_counts.values()))
        self.last_complex_report.update({
            "generated_vertex_count": len(vertices),
            "generated_face_count": len(faces),
            "watertight": watertight,
        })
        return {
            "vertices": vertices,
            "faces": faces,
            "colors": colors[nearest_source],
            "confidence": np.clip(generated_confidence, 0.0, 1.0),
        }

    def _orient_component_normals(self, cloud, labels, supplied_normals, radius):
        """Estimate and orient normals independently for disconnected surfaces."""
        import open3d as o3d

        points = np.asarray(cloud.points)
        oriented = np.zeros_like(points)
        component_labels = np.unique(labels[labels >= 0]) if len(labels) else np.asarray([0])
        if not len(component_labels):
            component_labels = np.asarray([0])
            labels = np.zeros(len(points), dtype=np.int32)
        for component in component_labels:
            indices = np.flatnonzero(labels == component)
            if len(indices) < 3:
                continue
            part = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[indices]))
            chosen = supplied_normals[indices] if supplied_normals is not None else None
            valid = np.zeros(len(indices), dtype=bool)
            if chosen is not None:
                chosen = np.asarray(chosen, dtype=float).copy()
                lengths = np.linalg.norm(chosen, axis=1)
                valid = np.isfinite(chosen).all(axis=1) & (lengths > 1e-8)
            if chosen is None or np.mean(valid) < 0.8:
                part.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=max(10, self.config.complex_normal_neighbors * 2)
                ))
            else:
                chosen[valid] /= lengths[valid, None]
                part.normals = o3d.utility.Vector3dVector(chosen)
            try:
                part.orient_normals_consistent_tangent_plane(
                    min(self.config.complex_normal_neighbors, len(indices) - 1)
                )
            except RuntimeError:
                part.normalize_normals()
            oriented[indices] = np.asarray(part.normals)
        missing = np.linalg.norm(oriented, axis=1) < 1e-8
        if np.any(missing):
            fallback = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            fallback.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60))
            oriented[missing] = np.asarray(fallback.normals)[missing]
        cloud.normals = o3d.utility.Vector3dVector(oriented)

    def _pymeshlab_screened_poisson(self, cloud):
        """Run the bundled ABI-matched PyMeshLab worker and load its mesh."""
        import open3d as o3d
        from mesh_cleanup import PyMeshLabWorkerBackend

        backend = PyMeshLabWorkerBackend()
        if not backend.is_available():
            return None
        try:
            with tempfile.TemporaryDirectory(prefix="proximap-pymeshlab-") as temporary:
                input_path = Path(temporary) / "oriented_points.ply"
                output_path = Path(temporary) / "screened_poisson.ply"
                if not o3d.io.write_point_cloud(str(input_path), cloud, write_ascii=False):
                    return None
                ok = backend.screened_poisson(
                    str(input_path), str(output_path),
                    depth=self.config.complex_poisson_depth,
                    scale=self.config.complex_poisson_scale,
                    normal_neighbors=self.config.complex_normal_neighbors,
                    min_component_faces=self.config.complex_min_component_faces,
                )
                if not ok or not output_path.is_file():
                    return None
                mesh = o3d.io.read_triangle_mesh(str(output_path))
                if not len(mesh.vertices) or not len(mesh.triangles):
                    return None
                return mesh
        except (OSError, RuntimeError, ValueError):
            return None

    def _architectural_edges(self, planes):
        edges = []
        minimum_length = self.config.effective_architecture_grid_size()
        for left, right in combinations(planes, 2):
            direction = np.cross(left["normal"], right["normal"])
            magnitude = float(np.linalg.norm(direction))
            if magnitude < 0.15:
                continue
            direction /= magnitude
            n1, n2 = left["normal"], right["normal"]
            d1, d2 = left["equation"][3], right["equation"][3]
            line_point = (d2 * np.cross(n1, direction) + d1 * np.cross(direction, n2)) / max(magnitude, 1e-12)
            left_points = left["centroid"] + left["coordinates"][:, :1] * left["basis_u"] + left["coordinates"][:, 1:] * left["basis_v"]
            right_points = right["centroid"] + right["coordinates"][:, :1] * right["basis_u"] + right["coordinates"][:, 1:] * right["basis_v"]
            start_t = max(float(np.min(left_points @ direction)), float(np.min(right_points @ direction)))
            end_t = min(float(np.max(left_points @ direction)), float(np.max(right_points @ direction)))
            if end_t - start_t < minimum_length:
                continue
            base_t = float(np.dot(line_point, direction))
            start = line_point + (start_t - base_t) * direction
            end = line_point + (end_t - base_t) * direction
            classes = {left["classification"], right["classification"]}
            classification = "wall-corner" if classes == {"wall"} else "architectural-edge"
            edges.append(ArchitecturalEdge(
                edge_id=f"edge-{len(edges) + 1:03d}",
                plane_ids=[left["plane_id"], right["plane_id"]],
                classification=classification,
                start=start.tolist(),
                end=end.tolist(),
                length=float(np.linalg.norm(end - start)),
            ))
        return edges

    def _architectural_corners(self, planes):
        corners = []
        tolerance = self.config.effective_architecture_grid_size() * 2.0
        for group in combinations(planes, 3):
            matrix = np.vstack([item["normal"] for item in group])
            if abs(float(np.linalg.det(matrix))) < 0.12:
                continue
            point = np.linalg.solve(matrix, -np.asarray([item["equation"][3] for item in group]))
            if not all(np.all(point >= item["bounds_min"] - tolerance) and np.all(point <= item["bounds_max"] + tolerance) for item in group):
                continue
            if any(np.linalg.norm(point - np.asarray(existing.position)) < tolerance for existing in corners):
                continue
            corners.append(ArchitecturalCorner(
                corner_id=f"corner-{len(corners) + 1:03d}",
                plane_ids=[item["plane_id"] for item in group],
                position=point.tolist(),
                confidence=float(np.mean([item["confidence"] for item in group])),
            ))
        return corners

    def _plane_basis(self, normal):
        up = self._up_vector()
        projected_up = up - np.dot(up, normal) * normal
        if np.linalg.norm(projected_up) > 0.2:
            basis_v = projected_up / np.linalg.norm(projected_up)
            basis_u = np.cross(basis_v, normal)
            basis_u /= max(float(np.linalg.norm(basis_u)), 1e-12)
            return basis_u, basis_v
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, normal))) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        basis_u = np.cross(reference, normal)
        basis_u /= max(float(np.linalg.norm(basis_u)), 1e-12)
        basis_v = np.cross(normal, basis_u)
        return basis_u, basis_v

    def _up_vector(self):
        result = np.zeros(3)
        result[{"x": 0, "y": 1, "z": 2}[self.config.architecture_up_axis]] = 1.0
        return result

    def _merge_vertices(self, vertices, faces, colors, confidence, class_codes, surface_ids):
        tolerance = self.config.effective_architecture_grid_size() * self.config.mesh_merge_tolerance_multiplier
        quantized = np.rint(vertices / max(tolerance, 1e-12)).astype(np.int64)
        mapping = np.empty(len(vertices), dtype=np.int64)
        lookup = {}
        new_vertices, new_colors, new_confidence, new_classes, new_surfaces, counts = [], [], [], [], [], []
        for index, key_values in enumerate(quantized):
            key = tuple(int(value) for value in key_values)
            target = lookup.get(key)
            if target is None:
                target = len(new_vertices)
                lookup[key] = target
                new_vertices.append(vertices[index].copy())
                new_colors.append(colors[index].copy())
                new_confidence.append(float(confidence[index]))
                new_classes.append(int(class_codes[index]))
                new_surfaces.append(int(surface_ids[index]))
                counts.append(1)
            else:
                count = counts[target]
                new_vertices[target] = (new_vertices[target] * count + vertices[index]) / (count + 1)
                new_colors[target] = (new_colors[target] * count + colors[index]) / (count + 1)
                new_confidence[target] = (new_confidence[target] * count + float(confidence[index])) / (count + 1)
                counts[target] += 1
            mapping[index] = target
        mapped_faces = mapping[faces]
        valid = (mapped_faces[:, 0] != mapped_faces[:, 1]) & (mapped_faces[:, 1] != mapped_faces[:, 2]) & (mapped_faces[:, 0] != mapped_faces[:, 2])
        mapped_faces = mapped_faces[valid]
        if len(mapped_faces):
            _, unique_indices = np.unique(np.sort(mapped_faces, axis=1), axis=0, return_index=True)
            mapped_faces = mapped_faces[np.sort(unique_indices)]
        return (
            np.asarray(new_vertices), mapped_faces, np.asarray(new_colors), np.asarray(new_confidence),
            np.asarray(new_classes, dtype=np.uint8), np.asarray(new_surfaces, dtype=np.int32),
        )

    @staticmethod
    def _vertex_normals(vertices, faces):
        normals = np.zeros_like(vertices)
        face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, 1e-12)
        return normals

    @staticmethod
    def _mesh_metrics(vertices, faces):
        edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        _unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = int(np.sum(counts == 1))
        nonmanifold = int(np.sum(counts > 2))
        parent = list(range(len(vertices)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for face in faces:
            root = find(int(face[0]))
            for vertex in face[1:]:
                other = find(int(vertex))
                if root != other:
                    parent[other] = root
        used = np.unique(faces)
        components = len({find(int(index)) for index in used})
        cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        area = float(np.sum(np.linalg.norm(cross, axis=1)) * 0.5)
        return boundary, nonmanifold, components, area

    @staticmethod
    def _write_mesh_ply(output, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.clip(np.rint(output.colors * 255.0), 0, 255).astype(np.uint8)
        with path.open("w", encoding="ascii") as handle:
            handle.write("ply\nformat ascii 1.0\ncomment Proximap architecture-aware reconstructed mesh\n")
            handle.write(f"element vertex {len(output.vertices)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property float nx\nproperty float ny\nproperty float nz\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("property float confidence\nproperty uchar architecture_class\nproperty int surface_id\n")
            handle.write(f"element face {len(output.faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            for index, point in enumerate(output.vertices):
                normal = output.normals[index]
                handle.write(
                    f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                    f"{normal[0]:.7g} {normal[1]:.7g} {normal[2]:.7g} "
                    f"{rgb[index, 0]} {rgb[index, 1]} {rgb[index, 2]} "
                    f"{output.confidence[index]:.6f} {output.class_codes[index]} {output.surface_ids[index]}\n"
                )
            for face in output.faces:
                handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")
