"""
Spatial Texture Engine (STE) - Texture Projection Foundation
============================================================

This module provides the geometric correspondence and UV projection foundation
for the new Spatial Texture Engine (STE).

Given an aligned LiDAR surface and a photogrammetry textured mesh, this service
determines the exact surface-based correspondence:
    LiDAR point -> Aligned 3D point -> Closest Photogrammetry Surface Point
                -> Corresponding Triangle -> Barycentric Coordinates (w0, w1, w2)
                -> Interpolated Photogrammetry UV (u, v)

Key Principles:
---------------
1. Direction: Transforms LiDAR points into Photogrammetry space via accepted alignment:
       P_aligned = s * R * P_lidar + t
2. Surface-based: Uses Open3D RaycastingScene to find closest point ON THE SURFACE (not just vertices).
3. Barycentric interpolation: Calculates (w0, w1, w2) such that P = w0*V0 + w1*V1 + w2*V2 with w0+w1+w2=1.
4. UV mapping: Interpolates source UV = w0*UV0 + w1*UV1 + w2*UV2.
5. Non-destructive: Never modifies source LiDAR geometry, photogrammetry geometry, UVs, or textures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import open3d as o3d

from ste_alignment import STEAlignmentResult


@dataclass
class STETextureProjectionSettings:
    """
    Configuration parameters for texture projection correspondence queries.
    """
    max_correspondence_distance: float = 0.05    # Maximum allowed surface distance (m, e.g. 5cm)
    clamp_barycentric: bool = True               # Clamp barycentric weights to [0, 1]
    normalize_barycentric: bool = True           # Ensure w0 + w1 + w2 == 1.0
    min_coverage_for_ready: float = 0.85         # Minimum coverage ratio (85%) to be READY_FOR_BAKING
    max_median_distance_for_ready: float = 0.05  # Maximum median distance (5cm) for readiness


@dataclass
class STETextureProjectionResult:
    """
    Result of texture projection correspondence calculation.
    """
    success: bool
    status: str
    status_message: str

    total_samples: int
    valid_samples: int
    invalid_samples: int
    coverage_ratio: float                        # valid_samples / total_samples (0.0 to 1.0)

    distances: np.ndarray                        # Shape (N,), float64: Euclidean distance to surface
    valid_mask: np.ndarray                       # Shape (N,), bool: distance <= max_correspondence_distance
    triangle_ids: np.ndarray                     # Shape (N,), int32: Photogrammetry triangle ID
    barycentric_coordinates: np.ndarray          # Shape (N, 3), float64: (w0, w1, w2)
    source_uvs: np.ndarray                       # Shape (N, 2), float64: Interpolated (u, v)

    # Diagnostic statistics
    min_distance: float = 0.0
    mean_distance: float = 0.0
    median_distance: float = 0.0
    p95_distance: float = 0.0
    max_distance: float = 0.0
    uv_min: np.ndarray = field(default_factory=lambda: np.zeros(2))
    uv_max: np.ndarray = field(default_factory=lambda: np.ones(2))

    aligned_points: Optional[np.ndarray] = None  # Shape (N, 3), float64
    closest_points: Optional[np.ndarray] = None  # Shape (N, 3), float64

    @property
    def is_ready_for_baking(self) -> bool:
        """Indicates whether correspondence coverage and surface closeness are sufficient for baking."""
        return self.success and self.status == "READY_FOR_BAKING"


class STETextureProjectionService:
    """
    Production service for geometric correspondence and UV projection.
    """

    @staticmethod
    def compute_barycentric_coordinates(
        points: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        clamp: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Vectorized computation of barycentric coordinates (w0, w1, w2) for points on triangles:
            P = w0 * V0 + w1 * V1 + w2 * V2

        Args:
            points: (N, 3) closest points on triangles.
            v0, v1, v2: (N, 3) triangle vertices.
            clamp: Clamp coordinates to [0, 1] to avoid out-of-triangle floating-point precision artifacts.
            normalize: Re-normalize so sum(w) == 1.0.

        Returns:
            (N, 3) float64 array of barycentric weights [w0, w1, w2].
        """
        v0_vec = v1 - v0  # shape (N, 3)
        v1_vec = v2 - v0  # shape (N, 3)
        v2_vec = points - v0  # shape (N, 3)

        d00 = np.sum(v0_vec * v0_vec, axis=1)  # shape (N,)
        d01 = np.sum(v0_vec * v1_vec, axis=1)  # shape (N,)
        d11 = np.sum(v1_vec * v1_vec, axis=1)  # shape (N,)
        d20 = np.sum(v2_vec * v0_vec, axis=1)  # shape (N,)
        d21 = np.sum(v2_vec * v1_vec, axis=1)  # shape (N,)

        denom = (d00 * d11) - (d01 * d01)  # shape (N,)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

        w1 = (d11 * d20 - d01 * d21) / denom
        w2 = (d00 * d21 - d01 * d20) / denom
        w0 = 1.0 - w1 - w2

        bary = np.stack([w0, w1, w2], axis=1)  # shape (N, 3)

        if clamp:
            bary = np.clip(bary, 0.0, 1.0)
        if normalize:
            row_sums = np.sum(bary, axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
            bary = bary / row_sums

        return bary

    @classmethod
    def project(
        cls,
        lidar_surface_points: np.ndarray,
        photogrammetry_vertices: np.ndarray,
        photogrammetry_triangles: np.ndarray,
        photogrammetry_uvs: np.ndarray,
        alignment_result: STEAlignmentResult,
        settings: Optional[STETextureProjectionSettings] = None
    ) -> STETextureProjectionResult:
        """
        Execute surface-based geometric correspondence and UV projection.

        Args:
            lidar_surface_points: (N, 3) source LiDAR surface points / vertices in LiDAR space.
            photogrammetry_vertices: (V, 3) Photogrammetry mesh vertices.
            photogrammetry_triangles: (F, 3) Photogrammetry triangle index matrix.
            photogrammetry_uvs: (F*3, 2) or (F, 3, 2) or (V, 2) Photogrammetry UV coordinates.
            alignment_result: Valid STEAlignmentResult defining P_photo = s * R * P_lidar + t.
            settings: STETextureProjectionSettings configuration.

        Returns:
            STETextureProjectionResult containing per-sample closest triangle, barycentrics, UVs, and metrics.
        """
        if settings is None:
            settings = STETextureProjectionSettings()

        # Step 1: Validate prerequisites
        if alignment_result is None or not alignment_result.success:
            return STETextureProjectionResult(
                success=False,
                status="INVALID_ALIGNMENT",
                status_message="Valid alignment result is required for texture projection.",
                total_samples=0, valid_samples=0, invalid_samples=0, coverage_ratio=0.0,
                distances=np.zeros(0), valid_mask=np.zeros(0, dtype=bool),
                triangle_ids=np.zeros(0, dtype=np.int32),
                barycentric_coordinates=np.zeros((0, 3)), source_uvs=np.zeros((0, 2))
            )

        pts_lidar = np.ascontiguousarray(lidar_surface_points, dtype=np.float64)
        verts_photo = np.ascontiguousarray(photogrammetry_vertices, dtype=np.float64)
        tris_photo = np.ascontiguousarray(photogrammetry_triangles, dtype=np.int32)
        uvs_photo = np.ascontiguousarray(photogrammetry_uvs, dtype=np.float64)

        if pts_lidar.ndim != 2 or pts_lidar.shape[1] != 3 or pts_lidar.shape[0] == 0:
            return STETextureProjectionResult(
                success=False,
                status="INVALID_LIDAR_POINTS",
                status_message=f"LiDAR points array must be (N, 3) with N > 0. Got {pts_lidar.shape}.",
                total_samples=0, valid_samples=0, invalid_samples=0, coverage_ratio=0.0,
                distances=np.zeros(0), valid_mask=np.zeros(0, dtype=bool),
                triangle_ids=np.zeros(0, dtype=np.int32),
                barycentric_coordinates=np.zeros((0, 3)), source_uvs=np.zeros((0, 2))
            )

        if verts_photo.shape[0] < 3 or tris_photo.shape[0] < 1:
            return STETextureProjectionResult(
                success=False,
                status="INVALID_PHOTOGRAMMETRY_MESH",
                status_message="Photogrammetry mesh must have at least 3 vertices and 1 triangle.",
                total_samples=0, valid_samples=0, invalid_samples=0, coverage_ratio=0.0,
                distances=np.zeros(0), valid_mask=np.zeros(0, dtype=bool),
                triangle_ids=np.zeros(0, dtype=np.int32),
                barycentric_coordinates=np.zeros((0, 3)), source_uvs=np.zeros((0, 2))
            )

        total_samples = int(pts_lidar.shape[0])

        # Step 2: Transform LiDAR surface points into Photogrammetry coordinate space
        # P_aligned = s * R * P_lidar + t (non-destructive working copy)
        pts_aligned = alignment_result.apply(pts_lidar)

        # Step 3: Build Open3D RaycastingScene for fast closest-surface query
        mesh_legacy = o3d.geometry.TriangleMesh()
        mesh_legacy.vertices = o3d.utility.Vector3dVector(verts_photo)
        mesh_legacy.triangles = o3d.utility.Vector3iVector(tris_photo)

        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(t_mesh)

        # Step 4: Perform vectorized closest point query
        query_tensor = o3d.core.Tensor(pts_aligned, dtype=o3d.core.Dtype.Float32)
        closest_dict = scene.compute_closest_points(query_tensor)

        closest_pts = closest_dict['points'].numpy().astype(np.float64)       # (N, 3)
        primitive_ids = closest_dict['primitive_ids'].numpy().astype(np.int32) # (N,)

        # Clamp primitive_ids in valid range
        num_triangles = tris_photo.shape[0]
        primitive_ids = np.clip(primitive_ids, 0, num_triangles - 1)

        # Step 5: Compute Euclidean surface distances
        dist_vectors = pts_aligned - closest_pts
        distances = np.linalg.norm(dist_vectors, axis=1)  # (N,)

        # Step 6: Compute Barycentric coordinates
        # Retrieve vertices for each corresponding triangle
        matched_tris = tris_photo[primitive_ids]  # shape (N, 3)
        v0 = verts_photo[matched_tris[:, 0]]      # shape (N, 3)
        v1 = verts_photo[matched_tris[:, 1]]      # shape (N, 3)
        v2 = verts_photo[matched_tris[:, 2]]      # shape (N, 3)

        bary_coords = cls.compute_barycentric_coordinates(
            points=closest_pts,
            v0=v0, v1=v1, v2=v2,
            clamp=settings.clamp_barycentric,
            normalize=settings.normalize_barycentric
        )  # shape (N, 3)

        # Step 7: Interpolate Photogrammetry UV coordinates
        # Determine UV layout:
        # Layout A: per-triangle UVs (num_triangles * 3, 2) or (num_triangles, 3, 2)
        # Layout B: per-vertex UVs (num_vertices, 2)
        if uvs_photo.ndim == 2 and uvs_photo.shape[0] == (num_triangles * 3):
            uv0 = uvs_photo[primitive_ids * 3 + 0]  # shape (N, 2)
            uv1 = uvs_photo[primitive_ids * 3 + 1]  # shape (N, 2)
            uv2 = uvs_photo[primitive_ids * 3 + 2]  # shape (N, 2)
        elif uvs_photo.ndim == 3 and uvs_photo.shape[0] == num_triangles and uvs_photo.shape[1] == 3:
            uv0 = uvs_photo[primitive_ids, 0]
            uv1 = uvs_photo[primitive_ids, 1]
            uv2 = uvs_photo[primitive_ids, 2]
        elif uvs_photo.ndim == 2 and uvs_photo.shape[0] == verts_photo.shape[0]:
            uv0 = uvs_photo[matched_tris[:, 0]]
            uv1 = uvs_photo[matched_tris[:, 1]]
            uv2 = uvs_photo[matched_tris[:, 2]]
        else:
            # Fallback UV generation based on normalized XY
            uv0 = v0[:, :2]
            uv1 = v1[:, :2]
            uv2 = v2[:, :2]

        # Barycentric UV interpolation: UV = w0*UV0 + w1*UV1 + w2*UV2
        w0 = bary_coords[:, 0:1]
        w1 = bary_coords[:, 1:2]
        w2 = bary_coords[:, 2:3]
        interpolated_uvs = (w0 * uv0) + (w1 * uv1) + (w2 * uv2)  # shape (N, 2)

        # Step 8: Validate distance threshold & coverage
        valid_mask = distances <= settings.max_correspondence_distance
        valid_count = int(np.sum(valid_mask))
        invalid_count = total_samples - valid_count
        coverage_ratio = float(valid_count / total_samples) if total_samples > 0 else 0.0

        # Step 9: Diagnostic statistics
        min_dist = float(np.min(distances))
        mean_dist = float(np.mean(distances))
        median_dist = float(np.median(distances))
        p95_dist = float(np.percentile(distances, 95))
        max_dist = float(np.max(distances))

        uv_min = np.min(interpolated_uvs, axis=0)
        uv_max = np.max(interpolated_uvs, axis=0)

        # Step 10: Determine readiness status
        if (coverage_ratio >= settings.min_coverage_for_ready and
            median_dist <= settings.max_median_distance_for_ready):
            status = "READY_FOR_BAKING"
            status_message = f"Texture projection verified ({coverage_ratio*100.0:.1f}% coverage, {median_dist*100.0:.2f} cm median dist). Ready for texture baking."
        else:
            status = "NOT_READY"
            status_message = f"Texture projection coverage ({coverage_ratio*100.0:.1f}%) or median distance ({median_dist*100.0:.2f} cm) outside baking tolerance."

        return STETextureProjectionResult(
            success=True,
            status=status,
            status_message=status_message,
            total_samples=total_samples,
            valid_samples=valid_count,
            invalid_samples=invalid_count,
            coverage_ratio=coverage_ratio,
            distances=distances,
            valid_mask=valid_mask,
            triangle_ids=primitive_ids,
            barycentric_coordinates=bary_coords,
            source_uvs=interpolated_uvs,
            min_distance=min_dist,
            mean_distance=mean_dist,
            median_distance=median_dist,
            p95_distance=p95_dist,
            max_distance=max_dist,
            uv_min=uv_min,
            uv_max=uv_max,
            aligned_points=pts_aligned,
            closest_points=closest_pts
        )

    @classmethod
    def project_mesh(
        cls,
        lidar_mesh: o3d.geometry.TriangleMesh,
        photogrammetry_mesh: o3d.geometry.TriangleMesh,
        alignment_result: STEAlignmentResult,
        settings: Optional[STETextureProjectionSettings] = None
    ) -> STETextureProjectionResult:
        """
        Convenience method to project vertices of a LiDAR mesh onto a Photogrammetry TriangleMesh.
        """
        lidar_pts = np.asarray(lidar_mesh.vertices)
        photo_verts = np.asarray(photogrammetry_mesh.vertices)
        photo_tris = np.asarray(photogrammetry_mesh.triangles)
        photo_uvs = np.asarray(photogrammetry_mesh.triangle_uvs)

        if photo_uvs.shape[0] == 0 and hasattr(photogrammetry_mesh, 'vertex_uvs'):
            photo_uvs = np.asarray(photogrammetry_mesh.vertex_uvs)

        return cls.project(
            lidar_surface_points=lidar_pts,
            photogrammetry_vertices=photo_verts,
            photogrammetry_triangles=photo_tris,
            photogrammetry_uvs=photo_uvs,
            alignment_result=alignment_result,
            settings=settings
        )
