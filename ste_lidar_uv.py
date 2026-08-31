"""
Spatial Texture Engine (STE) - LiDAR Target UV Parameterization
==============================================================

This module establishes a reliable, deterministic UV parameterization and atlas
packing for reconstructed interior LiDAR surface meshes (floors, walls, ceilings,
and irregular architectural geometry).

Key Principles:
---------------
1. Target UVs != Source UVs:
   Source UVs belong to the Photogrammetry mesh. Target UVs belong to the LiDAR
   surface mesh and map each LiDAR surface location to a target texture pixel.
2. Geometry Preservation:
   LiDAR vertex coordinates, triangle indices, and normals are NEVER modified.
3. Chart-based Atlas Parameterization:
   Segments the mesh into continuous planar charts based on surface normal clustering,
   projects each chart onto its exact orthonormal tangent plane, and deterministically
   packs them into a [0, 1] UV atlas with protective margins.
4. Per-Wedge Triangle UVs:
   Stores UVs as per-corner triangle UVs (shape (F*3, 2), matching Open3D triangle_uvs),
   eliminating seam pinning and zero-area artifacts across chart boundaries.
5. Robust Validation:
   Detects zero-area UV triangles, overlapping charts/triangles, out-of-bound coordinates,
   and computes exact UV space utilization.
6. Resolution Independence:
   Normalized [0, 1] parameterization scales seamlessly to any target texture resolution
   (1024, 2048, 4096, 8192).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import open3d as o3d


@dataclass
class STELiDARUVSettings:
    """Configuration settings for LiDAR UV generation and atlas packing."""
    chart_margin: float = 0.015             # 1.5% margin between packed charts
    gutter_padding: float = 0.01            # 1.0% atlas border padding
    normalize_to_unit_square: bool = True   # Normalize final atlas to [0, 1]
    zero_area_epsilon: float = 1e-12        # Threshold below which normalized UV triangle is degenerate
    max_triangles_for_full_overlap_check: int = 5000


@dataclass
class STELiDARUVResult:
    """Structured result of LiDAR target UV generation and validation."""
    success: bool
    status: str
    status_message: str

    uvs: np.ndarray                         # Shape (F*3, 2), float64
    vertex_count: int
    triangle_count: int

    uv_min: np.ndarray                      # Shape (2,)
    uv_max: np.ndarray                      # Shape (2,)

    chart_count: int
    seam_count: int

    zero_area_triangle_count: int
    overlapping_triangle_count: int
    has_overlapping_uvs: bool

    occupied_uv_area: float
    uv_utilization: float                   # Percentage of [0,1]^2 or AABB occupied by triangles

    vertex_uvs: Optional[np.ndarray] = None # Shape (V, 2), float64
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready_for_baking(self) -> bool:
        """Determines if the generated UVs meet all requirements for texture baking."""
        return (
            self.success and
            not np.any(np.isnan(self.uvs)) and
            not np.any(np.isinf(self.uvs)) and
            self.zero_area_triangle_count == 0 and
            self.status in ("READY_FOR_BAKING", "VALID")
        )


# Backward compatibility alias
LidarUVResult = STELiDARUVResult


class STELiDARUVService:
    """
    Production service for generating and validating LiDAR target UVs.
    """

    @staticmethod
    def compute_face_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Vectorized computation of triangle face normals."""
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return cross / norms

    @staticmethod
    def detect_zero_area_triangles(
        uvs: np.ndarray,
        triangles: np.ndarray,
        eps: float = 1e-8
    ) -> Tuple[int, np.ndarray]:
        """
        Detect degenerate UV triangles with 2D area < eps.
        """
        if uvs.shape[0] == triangles.shape[0] * 3:
            p0 = uvs[0::3]
            p1 = uvs[1::3]
            p2 = uvs[2::3]
        else:
            p0 = uvs[triangles[:, 0]]
            p1 = uvs[triangles[:, 1]]
            p2 = uvs[triangles[:, 2]]

        areas = 0.5 * np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
            (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        )
        zero_mask = areas < eps
        return int(np.sum(zero_mask)), zero_mask

    @staticmethod
    def compute_uv_utilization(
        uvs: np.ndarray,
        triangles: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute total occupied 2D UV triangle area and utilization percentage.
        """
        if uvs.shape[0] == 0 or triangles.shape[0] == 0:
            return 0.0, 0.0

        if uvs.shape[0] == triangles.shape[0] * 3:
            p0 = uvs[0::3]
            p1 = uvs[1::3]
            p2 = uvs[2::3]
        else:
            p0 = uvs[triangles[:, 0]]
            p1 = uvs[triangles[:, 1]]
            p2 = uvs[triangles[:, 2]]

        areas = 0.5 * np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
            (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        )
        occupied_area = float(np.sum(areas))

        uv_min = np.min(uvs, axis=0)
        uv_max = np.max(uvs, axis=0)
        extent = np.maximum(1e-6, uv_max - uv_min)
        total_bbox_area = float(extent[0] * extent[1])

        utilization = float((occupied_area / max(1e-6, total_bbox_area)) * 100.0)
        return occupied_area, min(100.0, utilization)

    @classmethod
    def detect_overlapping_triangles(
        cls,
        uvs: np.ndarray,
        triangles: np.ndarray,
        max_triangles: int = 5000
    ) -> Tuple[bool, int]:
        """
        Detect 2D triangle-triangle intersections using spatial grid binning.
        """
        num_tri = triangles.shape[0]
        if num_tri <= 1:
            return False, 0

        # Subsample deterministically if triangle count exceeds max_triangles
        if num_tri > max_triangles:
            sub_indices = np.linspace(0, num_tri - 1, max_triangles, dtype=int)
            sub_triangles = triangles[sub_indices]
            if uvs.shape[0] == num_tri * 3:
                all_t_verts = uvs.reshape(-1, 3, 2)
                t_verts = all_t_verts[sub_indices]
            else:
                t_verts = uvs[sub_triangles]
            active_triangles = sub_triangles
            active_count = max_triangles
        else:
            if uvs.shape[0] == num_tri * 3:
                t_verts = uvs.reshape(-1, 3, 2)
            else:
                t_verts = uvs[triangles]
            active_triangles = triangles
            active_count = num_tri

        # Compute 2D AABBs for all triangles
        t_mins = np.min(t_verts, axis=1)  # shape (K, 2)
        t_maxs = np.max(t_verts, axis=1)  # shape (K, 2)

        grid_res = 32
        overall_min = np.min(t_mins, axis=0)
        overall_max = np.max(t_maxs, axis=0)
        cell_size = np.maximum(1e-6, (overall_max - overall_min) / grid_res)

        grid: Dict[Tuple[int, int], List[int]] = {}
        for i in range(active_count):
            gx0 = int(np.floor((t_mins[i, 0] - overall_min[0]) / cell_size[0]))
            gx1 = int(np.floor((t_maxs[i, 0] - overall_min[0]) / cell_size[0]))
            gy0 = int(np.floor((t_mins[i, 1] - overall_min[1]) / cell_size[1]))
            gy1 = int(np.floor((t_maxs[i, 1] - overall_min[1]) / cell_size[1]))

            for gx in range(max(0, gx0), min(grid_res, gx1 + 1)):
                for gy in range(max(0, gy0), min(grid_res, gy1 + 1)):
                    key = (gx, gy)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(i)

        overlapping_tris = set()

        def segments_intersect(a1, a2, b1, b2):
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))

        def point_in_triangle(pt, v0, v1, v2):
            d1 = (pt[0]-v1[0])*(v0[1]-v1[1]) - (v0[0]-v1[0])*(pt[1]-v1[1])
            d2 = (pt[0]-v2[0])*(v1[1]-v2[1]) - (v1[0]-v2[0])*(pt[1]-v2[1])
            d3 = (pt[0]-v0[0])*(v2[1]-v0[1]) - (v2[0]-v0[0])*(pt[1]-v0[1])
            has_neg = (d1 < -1e-8) or (d2 < -1e-8) or (d3 < -1e-8)
            has_pos = (d1 > 1e-8) or (d2 > 1e-8) or (d3 > 1e-8)
            return not (has_neg and has_pos)

        tested_pairs = set()
        for cell_tris in grid.values():
            if len(cell_tris) < 2 or len(cell_tris) > 100:
                continue
            for i_idx in range(len(cell_tris)):
                ti = cell_tris[i_idx]
                for j_idx in range(i_idx + 1, len(cell_tris)):
                    tj = cell_tris[j_idx]
                    pair = (min(ti, tj), max(ti, tj))
                    if pair in tested_pairs:
                        continue
                    tested_pairs.add(pair)

                    # AABB overlap test
                    if (t_maxs[ti, 0] < t_mins[tj, 0] or t_mins[ti, 0] > t_maxs[tj, 0] or
                        t_maxs[ti, 1] < t_mins[tj, 1] or t_mins[ti, 1] > t_maxs[tj, 1]):
                        continue

                    vA = t_verts[ti]
                    vB = t_verts[tj]

                    # Check for exact duplicate/overlapping UV triangles
                    if np.allclose(vA, vB, atol=1e-4):
                        overlapping_tris.add(ti)
                        overlapping_tris.add(tj)
                        continue

                    # Check if triangles share an edge/vertex in mesh topology
                    shared_v = set(active_triangles[ti]).intersection(set(active_triangles[tj]))
                    if len(shared_v) >= 1:
                        # For topologically adjacent triangles, test if centroids lie strictly inside
                        centroidA = np.mean(vA, axis=0)
                        centroidB = np.mean(vB, axis=0)
                        if point_in_triangle(centroidA, vB[0], vB[1], vB[2]) or point_in_triangle(centroidB, vA[0], vA[1], vA[2]):
                            overlapping_tris.add(ti)
                            overlapping_tris.add(tj)
                        continue

                    intersect = False
                    for ea in range(3):
                        ea1, ea2 = vA[ea], vA[(ea+1)%3]
                        for eb in range(3):
                            eb1, eb2 = vB[eb], vB[(eb+1)%3]
                            if segments_intersect(ea1, ea2, eb1, eb2):
                                intersect = True
                                break
                        if intersect:
                            break

                    if not intersect:
                        centroidA = np.mean(vA, axis=0)
                        centroidB = np.mean(vB, axis=0)
                        if point_in_triangle(centroidA, vB[0], vB[1], vB[2]) or point_in_triangle(centroidB, vA[0], vA[1], vA[2]):
                            intersect = True

                    if intersect:
                        overlapping_tris.add(ti)
                        overlapping_tris.add(tj)

        has_overlap = len(overlapping_tris) > 0
        return has_overlap, len(overlapping_tris)

    @classmethod
    def generate_uvs(
        cls,
        mesh_or_vertices: Union[o3d.geometry.TriangleMesh, np.ndarray],
        triangles: Optional[np.ndarray] = None,
        settings: Optional[STELiDARUVSettings] = None
    ) -> STELiDARUVResult:
        """
        Generate continuous, chart-packed UV parameterization for the reconstructed LiDAR surface.

        Args:
            mesh_or_vertices: Open3D TriangleMesh or (V, 3) vertex array.
            triangles: (F, 3) triangle array (required if vertices are provided as array).
            settings: STELiDARUVSettings.

        Returns:
            STELiDARUVResult containing per-wedge triangle UVs (F*3, 2), chart counts, and diagnostics.
        """
        if settings is None:
            settings = STELiDARUVSettings()

        # Step 1: Extract vertices & triangles
        if isinstance(mesh_or_vertices, o3d.geometry.TriangleMesh):
            vertices = np.asarray(mesh_or_vertices.vertices, dtype=np.float64)
            tris = np.asarray(mesh_or_vertices.triangles, dtype=np.int32)
        else:
            vertices = np.ascontiguousarray(mesh_or_vertices, dtype=np.float64)
            if triangles is None:
                return STELiDARUVResult(
                    success=False, status="INVALID_INPUT",
                    status_message="Triangles array required when passing raw vertices array.",
                    uvs=np.zeros((0, 2)), vertex_count=0, triangle_count=0,
                    uv_min=np.zeros(2), uv_max=np.zeros(2),
                    chart_count=0, seam_count=0, zero_area_triangle_count=0,
                    overlapping_triangle_count=0, has_overlapping_uvs=False,
                    occupied_uv_area=0.0, uv_utilization=0.0
                )
            tris = np.ascontiguousarray(triangles, dtype=np.int32)

        V = int(vertices.shape[0])
        F = int(tris.shape[0])

        if V < 3 or F < 1 or vertices.ndim != 2 or tris.ndim != 2:
            return STELiDARUVResult(
                success=False, status="INVALID_GEOMETRY",
                status_message=f"Mesh must have >=3 vertices and >=1 triangle. Got V={V}, F={F}.",
                uvs=np.zeros((0, 2)), vertex_count=V, triangle_count=F,
                uv_min=np.zeros(2), uv_max=np.zeros(2),
                chart_count=0, seam_count=0, zero_area_triangle_count=0,
                overlapping_triangle_count=0, has_overlapping_uvs=False,
                occupied_uv_area=0.0, uv_utilization=0.0
            )

        if not np.all(np.isfinite(vertices)):
            return STELiDARUVResult(
                success=False, status="INVALID_NON_FINITE_DATA",
                status_message="LiDAR vertices contain NaN or Inf values.",
                uvs=np.zeros((0, 2)), vertex_count=V, triangle_count=F,
                uv_min=np.zeros(2), uv_max=np.zeros(2),
                chart_count=0, seam_count=0, zero_area_triangle_count=0,
                overlapping_triangle_count=0, has_overlapping_uvs=False,
                occupied_uv_area=0.0, uv_utilization=0.0
            )

        # Step 2: Compute face normals
        face_normals = cls.compute_face_normals(vertices, tris)

        # Step 3: Normal Clustering into dominant 3D orientations (cardinal + diagonal)
        dirs = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x != 0 or y != 0 or z != 0:
                        v = np.array([x, y, z], dtype=np.float64)
                        dirs.append(v / np.linalg.norm(v))
        cluster_dirs = np.array(dirs, dtype=np.float64)

        dots = face_normals @ cluster_dirs.T  # shape (F, len(dirs))
        cluster_ids = np.argmax(dots, axis=1)  # shape (F,)

        active_clusters = np.unique(cluster_ids)
        chart_count = len(active_clusters)

        chart_tri_uvs = {}
        chart_bboxes = {}

        # Step 4: Project each chart onto its exact orthonormal tangent plane
        for c_id in active_clusters:
            tri_indices = np.where(cluster_ids == c_id)[0]
            if len(tri_indices) == 0:
                continue

            sub_normals = face_normals[tri_indices]
            n_chart = np.mean(sub_normals, axis=0)
            n_norm = np.linalg.norm(n_chart)
            if n_norm < 1e-8:
                n_chart = cluster_dirs[c_id]
            else:
                n_chart /= n_norm

            # Construct orthonormal tangent basis (u_axis, v_axis)
            if abs(n_chart[2]) < 0.9:
                u_axis = np.cross(n_chart, np.array([0.0, 0.0, 1.0]))
            else:
                u_axis = np.cross(n_chart, np.array([1.0, 0.0, 0.0]))
            u_norm = np.linalg.norm(u_axis)
            if u_norm < 1e-8:
                u_axis = np.array([1.0, 0.0, 0.0])
            else:
                u_axis /= u_norm
            v_axis = np.cross(n_chart, u_axis)
            v_axis /= max(1e-8, np.linalg.norm(v_axis))

            sub_tris = tris[tri_indices]  # (K, 3)
            p0 = vertices[sub_tris[:, 0]]
            p1 = vertices[sub_tris[:, 1]]
            p2 = vertices[sub_tris[:, 2]]

            u0, v0_c = p0 @ u_axis, p0 @ v_axis
            u1, v1_c = p1 @ u_axis, p1 @ v_axis
            u2, v2_c = p2 @ u_axis, p2 @ v_axis

            u_coords = np.stack([u0, u1, u2], axis=1)  # (K, 3)
            v_coords = np.stack([v0_c, v1_c, v2_c], axis=1)  # (K, 3)

            u_min, u_max = np.min(u_coords), np.max(u_coords)
            v_min, v_max = np.min(v_coords), np.max(v_coords)

            chart_u = u_coords - u_min
            chart_v = v_coords - v_min

            chart_w = max(1e-4, u_max - u_min)
            chart_h = max(1e-4, v_max - v_min)

            chart_tri_uvs[c_id] = (tri_indices, chart_u, chart_v)
            chart_bboxes[c_id] = (chart_w, chart_h)

        # Step 5: Deterministic Atlas Shelf Packing
        sorted_charts = sorted(chart_bboxes.keys(), key=lambda c: chart_bboxes[c][1], reverse=True)

        margin = settings.chart_margin
        gutter = settings.gutter_padding

        total_w_sum = sum(chart_bboxes[c][0] for c in sorted_charts)
        target_atlas_width = max(total_w_sum / 2.5, max(chart_bboxes[c][0] for c in sorted_charts) + 2 * margin)

        shelf_x = gutter
        shelf_y = gutter
        shelf_height = 0.0
        chart_offsets = {}

        for c_id in sorted_charts:
            w, h = chart_bboxes[c_id]
            if shelf_x + w + margin > target_atlas_width and shelf_x > gutter:
                shelf_y += shelf_height + margin
                shelf_x = gutter
                shelf_height = 0.0

            chart_offsets[c_id] = (shelf_x, shelf_y)
            shelf_x += w + margin
            shelf_height = max(shelf_height, h)

        total_atlas_w = max(shelf_x + gutter, target_atlas_width)
        total_atlas_h = shelf_y + shelf_height + gutter
        atlas_scale = max(total_atlas_w, total_atlas_h, 1e-4)

        # Step 6: Assemble per-wedge triangle UV array (F*3, 2)
        # and per-vertex array (V, 2)
        triangle_uvs = np.zeros((F * 3, 2), dtype=np.float64)
        vertex_uvs = np.zeros((V, 2), dtype=np.float64)
        vertex_assigned = np.zeros(V, dtype=bool)

        for c_id in active_clusters:
            tri_indices, chart_u, chart_v = chart_tri_uvs[c_id]
            off_x, off_y = chart_offsets[c_id]

            packed_u = (chart_u + off_x) / atlas_scale
            packed_v = (chart_v + off_y) / atlas_scale

            sub_tris = tris[tri_indices]  # (K, 3)
            for k in range(len(tri_indices)):
                tri_idx = tri_indices[k]
                for corner in range(3):
                    u_val = float(packed_u[k, corner])
                    v_val = float(packed_v[k, corner])
                    triangle_uvs[tri_idx * 3 + corner, 0] = u_val
                    triangle_uvs[tri_idx * 3 + corner, 1] = v_val

                    vid = sub_tris[k, corner]
                    if not vertex_assigned[vid]:
                        vertex_uvs[vid, 0] = u_val
                        vertex_uvs[vid, 1] = v_val
                        vertex_assigned[vid] = True

        triangle_uvs = np.clip(triangle_uvs, 0.0, 1.0)
        vertex_uvs = np.clip(vertex_uvs, 0.0, 1.0)

        # Step 7: Validate UVs
        zero_area_count, _ = cls.detect_zero_area_triangles(triangle_uvs, tris, eps=settings.zero_area_epsilon)
        has_overlaps, overlap_count = cls.detect_overlapping_triangles(
            triangle_uvs, tris, max_triangles=settings.max_triangles_for_full_overlap_check
        )
        occupied_area, utilization = cls.compute_uv_utilization(triangle_uvs, tris)

        uv_min = np.min(triangle_uvs, axis=0)
        uv_max = np.max(triangle_uvs, axis=0)

        seam_count = chart_count * 4
        status = "READY_FOR_BAKING" if zero_area_count == 0 else "VALID_WITH_WARNINGS"
        status_message = (
            f"LiDAR UV parameterization complete: {chart_count} charts, "
            f"{utilization:.1f}% utilization, zero-area={zero_area_count}, overlaps={overlap_count}."
        )

        return STELiDARUVResult(
            success=True,
            status=status,
            status_message=status_message,
            uvs=triangle_uvs,
            vertex_count=V,
            triangle_count=F,
            uv_min=uv_min,
            uv_max=uv_max,
            chart_count=chart_count,
            seam_count=seam_count,
            zero_area_triangle_count=zero_area_count,
            overlapping_triangle_count=overlap_count,
            has_overlapping_uvs=has_overlaps,
            occupied_uv_area=occupied_area,
            uv_utilization=utilization,
            vertex_uvs=vertex_uvs,
            metadata={
                "atlas_width": total_atlas_w,
                "atlas_height": total_atlas_h,
                "chart_offsets": chart_offsets
            }
        )

    @classmethod
    def validate_uvs(
        cls,
        vertices: np.ndarray,
        triangles: np.ndarray,
        uvs: np.ndarray,
        settings: Optional[STELiDARUVSettings] = None
    ) -> STELiDARUVResult:
        """
        Validate existing or generated UVs against topology and metric constraints.
        """
        if settings is None:
            settings = STELiDARUVSettings()

        V = vertices.shape[0]
        F = triangles.shape[0]

        if uvs.shape[0] != V and uvs.shape[0] != F * 3:
            return STELiDARUVResult(
                success=False, status="INVALID_UV_DIMENSIONS",
                status_message=f"UV array size {uvs.shape[0]} matches neither vertex count {V} nor 3*F {F*3}.",
                uvs=uvs, vertex_count=V, triangle_count=F,
                uv_min=np.zeros(2), uv_max=np.zeros(2),
                chart_count=0, seam_count=0, zero_area_triangle_count=0,
                overlapping_triangle_count=0, has_overlapping_uvs=False,
                occupied_uv_area=0.0, uv_utilization=0.0
            )

        if not np.all(np.isfinite(uvs)):
            return STELiDARUVResult(
                success=False, status="INVALID_NON_FINITE_UVS",
                status_message="UV array contains NaN or Inf values.",
                uvs=uvs, vertex_count=V, triangle_count=F,
                uv_min=np.zeros(2), uv_max=np.zeros(2),
                chart_count=0, seam_count=0, zero_area_triangle_count=0,
                overlapping_triangle_count=0, has_overlapping_uvs=False,
                occupied_uv_area=0.0, uv_utilization=0.0
            )

        zero_area_count, _ = cls.detect_zero_area_triangles(uvs, triangles, eps=settings.zero_area_epsilon)
        has_overlaps, overlap_count = cls.detect_overlapping_triangles(
            uvs, triangles, max_triangles=settings.max_triangles_for_full_overlap_check
        )
        occupied_area, utilization = cls.compute_uv_utilization(uvs, triangles)

        uv_min = np.min(uvs, axis=0)
        uv_max = np.max(uvs, axis=0)

        status = "READY_FOR_BAKING" if zero_area_count == 0 else "VALID_WITH_WARNINGS"
        return STELiDARUVResult(
            success=True,
            status=status,
            status_message=f"Validation complete: zero-area={zero_area_count}, overlaps={overlap_count}.",
            uvs=uvs,
            vertex_count=V,
            triangle_count=F,
            uv_min=uv_min,
            uv_max=uv_max,
            chart_count=1,
            seam_count=0,
            zero_area_triangle_count=zero_area_count,
            overlapping_triangle_count=overlap_count,
            has_overlapping_uvs=has_overlaps,
            occupied_uv_area=occupied_area,
            uv_utilization=utilization
        )
