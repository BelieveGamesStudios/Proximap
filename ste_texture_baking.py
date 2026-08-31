"""
Spatial Texture Engine (STE) - Production Texture Baking Engine
==============================================================

This module implements the production-quality texture baking engine for the
Spatial Texture Engine (STE).

It rasterizes the reconstructed LiDAR surface in target UV space, queries geometric
surface correspondence against the photogrammetry textured mesh via Open3D Embree
RaycastingScene, samples source textures with bilinear interpolation, applies
Euclidean distance transform seam dilation/padding, and produces a derived LiDAR
surface mesh referencing the baked texture.

Key Features:
-------------
1. Surface-wide UV Rasterization:
   Operates on the continuous LiDAR surface in target UV space rather than isolated vertices.
2. Open3D Embree Raycasting Acceleration:
   Batched closest-surface queries against the source photogrammetry mesh.
3. Bilinear Texture Filtering:
   High-quality interpolation on source texture maps.
4. Non-Destructive Alignment & Geometry:
   Derived LiDAR mesh vertices remain in native LiDAR coordinates (unbaked transform).
   Original LiDAR and photogrammetry assets are never mutated.
5. Euclidean Seam Padding:
   Smooth, deterministic boundary dilation preventing dark GPU filtering artifacts.
6. Asynchronous Background Worker:
   STETextureBakingWorker (QThread) for responsive UI execution.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
import os
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import open3d as o3d

try:
    from PySide6.QtCore import QThread, Signal
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False
    class QThread:
        def __init__(self, *args, **kwargs): pass
        def start(self): self.run()
        def isRunning(self): return False
    def Signal(*args, **kwargs):
        class _Signal:
            def connect(self, fn): pass
            def emit(self, *a, **k): pass
        return _Signal()

from ste_alignment import STEAlignmentResult
from ste_lidar_uv import STELiDARUVService, STELiDARUVResult


@dataclass
class TextureBakeResult:
    """Structured result of texture baking process."""
    success: bool
    status: str
    status_message: str

    output_mesh: Optional[o3d.geometry.TriangleMesh] = None
    output_texture: Optional[np.ndarray] = None    # Shape (H, W, 3) or (H, W, 4) uint8

    texture_width: int = 0
    texture_height: int = 0

    total_texture_pixels: int = 0                  # Pixels covered by target UV triangles
    valid_texture_pixels: int = 0                  # Pixels within max correspondence distance
    uncovered_texture_pixels: int = 0              # Pixels exceeding max correspondence distance
    coverage_ratio: float = 0.0                    # valid / total

    median_distance: float = 0.0
    p95_distance: float = 0.0
    max_distance: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


class STETextureBakingService:
    """
    Production service for surface-wide texture baking onto LiDAR target UV parameterization.
    """

    @staticmethod
    def load_texture_image(texture_input: Union[str, np.ndarray, Image.Image, o3d.geometry.Image]) -> np.ndarray:
        """
        Normalize texture input to a uint8 RGB/RGBA NumPy array of shape (H, W, C).
        """
        if isinstance(texture_input, str):
            if not os.path.exists(texture_input):
                raise FileNotFoundError(f"Texture file not found: {texture_input}")
            img = Image.open(texture_input).convert('RGB')
            return np.array(img, dtype=np.uint8)
        elif isinstance(texture_input, np.ndarray):
            if texture_input.dtype != np.uint8:
                if np.max(texture_input) <= 1.0:
                    return np.clip(texture_input * 255.0, 0, 255).astype(np.uint8)
                return np.clip(texture_input, 0, 255).astype(np.uint8)
            return texture_input
        elif isinstance(texture_input, Image.Image):
            return np.array(texture_input.convert('RGB'), dtype=np.uint8)
        elif isinstance(texture_input, o3d.geometry.Image):
            arr = np.asarray(texture_input)
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).astype(np.uint8)
            return arr
        else:
            raise ValueError(f"Unsupported texture input type: {type(texture_input)}")

    @staticmethod
    def sample_texture_bilinear(texture: np.ndarray, uvs: np.ndarray) -> np.ndarray:
        """
        Vectorized bilinear interpolation sampling of source texture at normalized (u, v) coordinates.

        Args:
            texture: (H, W, C) uint8 or float NumPy array.
            uvs: (N, 2) normalized float64 coordinates in [0, 1].

        Returns:
            (N, C) sampled colors (uint8).
        """
        H, W = texture.shape[:2]
        C = texture.shape[2] if texture.ndim == 3 else 1

        u = np.clip(uvs[:, 0], 0.0, 1.0)
        v = np.clip(uvs[:, 1], 0.0, 1.0)

        # Convert UV to continuous pixel coordinates
        # V=1 is top (y=0), V=0 is bottom (y=H-1)
        px_x = u * (W - 1)
        px_y = (1.0 - v) * (H - 1)

        x0 = np.floor(px_x).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y0 = np.floor(px_y).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, H - 1)

        fx = (px_x - x0)[:, np.newaxis]
        fy = (px_y - y0)[:, np.newaxis]

        tex_flat = texture if texture.ndim == 3 else texture[:, :, np.newaxis]

        c00 = tex_flat[y0, x0].astype(np.float64)
        c10 = tex_flat[y0, x1].astype(np.float64)
        c01 = tex_flat[y1, x0].astype(np.float64)
        c11 = tex_flat[y1, x1].astype(np.float64)

        top = c00 * (1.0 - fx) + c10 * fx
        bottom = c01 * (1.0 - fx) + c11 * fx
        sampled = top * (1.0 - fy) + bottom * fy

        return np.clip(sampled, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def apply_texture_padding(texture: np.ndarray, mask: np.ndarray, padding: int = 4) -> np.ndarray:
        """
        Apply deterministic Euclidean-distance-based seam dilation / texture padding.

        Args:
            texture: (H, W, C) uint8 baked texture image.
            mask: (H, W) boolean mask where True indicates valid baked pixels.
            padding: Radius in pixels to dilate valid boundary colors outward.

        Returns:
            (H, W, C) padded texture.
        """
        if padding <= 0 or not np.any(mask) or np.all(mask):
            return texture.copy()

        padded = texture.copy()
        # Compute exact Euclidean distance transform to nearest True pixel
        dist, (idx_y, idx_x) = ndi.distance_transform_edt(~mask, return_indices=True)

        pad_zone = (~mask) & (dist <= float(padding))
        padded[pad_zone] = texture[idx_y[pad_zone], idx_x[pad_zone]]

        return padded

    @classmethod
    def rasterize_target_uv_triangles(
        cls,
        uv_triangles: np.ndarray,
        texture_width: int,
        texture_height: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized rasterization of target UV triangles into pixel space.

        Args:
            uv_triangles: (F, 3, 2) float64 array of triangle UV coordinates.
            texture_width: Width in pixels.
            texture_height: Height in pixels.

        Returns:
            pixel_coords: (N, 2) int32 [px, py] covered pixel coordinates.
            triangle_indices: (N,) int32 index of target triangle covering each pixel.
            barycentric_coords: (N, 3) float64 barycentric weights [w0, w1, w2].
        """
        F = uv_triangles.shape[0]
        W = texture_width
        H = texture_height

        px_coords = np.zeros_like(uv_triangles)
        px_coords[:, :, 0] = uv_triangles[:, :, 0] * (W - 1)
        px_coords[:, :, 1] = (1.0 - uv_triangles[:, :, 1]) * (H - 1)

        all_px = []
        all_tri_idx = []
        all_bary = []

        eps = -1e-4

        for i in range(F):
            p0 = px_coords[i, 0]
            p1 = px_coords[i, 1]
            p2 = px_coords[i, 2]

            xmin = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
            xmax = min(W - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
            ymin = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
            ymax = min(H - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

            if xmin > xmax or ymin > ymax:
                continue

            xs = np.arange(xmin, xmax + 1)
            ys = np.arange(ymin, ymax + 1)
            gx, gy = np.meshgrid(xs, ys)
            pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

            # 2D barycentric coords
            v0 = p1 - p0
            v1 = p2 - p0
            v2 = pts - p0

            d00 = v0[0] * v0[0] + v0[1] * v0[1]
            d01 = v0[0] * v1[0] + v0[1] * v1[1]
            d11 = v1[0] * v1[0] + v1[1] * v1[1]
            denom = d00 * d11 - d01 * d01

            if abs(denom) < 1e-12:
                continue

            d20 = v2[:, 0] * v0[0] + v2[:, 1] * v0[1]
            d21 = v2[:, 0] * v1[0] + v2[:, 1] * v1[1]

            w1 = (d11 * d20 - d01 * d21) / denom
            w2 = (d00 * d21 - d01 * d20) / denom
            w0 = 1.0 - w1 - w2

            inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
            if np.any(inside):
                w = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
                w = np.clip(w, 0.0, 1.0)
                row_sums = np.sum(w, axis=1, keepdims=True)
                row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
                w /= row_sums

                all_px.append(pts[inside].astype(np.int32))
                all_tri_idx.append(np.full(int(np.sum(inside)), i, dtype=np.int32))
                all_bary.append(w)

        if all_px:
            return np.vstack(all_px), np.concatenate(all_tri_idx), np.vstack(all_bary)
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float64)

    @classmethod
    def bake(
        cls,
        photogrammetry_mesh: Union[o3d.geometry.TriangleMesh, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        photogrammetry_texture: Union[str, np.ndarray, Image.Image, o3d.geometry.Image],
        lidar_surface_mesh: Union[o3d.geometry.TriangleMesh, Tuple[np.ndarray, np.ndarray]],
        lidar_target_uvs: np.ndarray,
        alignment_transform: STEAlignmentResult,
        texture_resolution: int = 4096,
        max_correspondence_distance: float = 0.05,
        texture_padding: int = 4,
        batch_size: int = 100000,
        progress_fn: Optional[Callable[[float, str], None]] = None,
        log_fn: Optional[Callable[[str], None]] = None
    ) -> TextureBakeResult:
        """
        Execute production surface-wide texture baking.
        """
        def _log(msg: str):
            if log_fn:
                log_fn(msg)

        def _progress(pct: float, stage: str):
            if progress_fn:
                progress_fn(pct, stage)

        _progress(0.02, "Validating Inputs")
        _log("Initiating STE Texture Baking pipeline...")

        # 1. Validate alignment result
        if alignment_transform is None or not alignment_transform.success:
            return TextureBakeResult(
                success=False,
                status="failed_input",
                status_message="Valid alignment transform is required for texture baking."
            )

        # 2. Extract Photogrammetry Mesh Data
        if isinstance(photogrammetry_mesh, o3d.geometry.TriangleMesh):
            verts_photo = np.asarray(photogrammetry_mesh.vertices, dtype=np.float64)
            tris_photo = np.asarray(photogrammetry_mesh.triangles, dtype=np.int32)
            uvs_photo = np.asarray(photogrammetry_mesh.triangle_uvs, dtype=np.float64)
            if uvs_photo.shape[0] == 0 and hasattr(photogrammetry_mesh, 'vertex_uvs'):
                uvs_photo = np.asarray(photogrammetry_mesh.vertex_uvs, dtype=np.float64)
        elif isinstance(photogrammetry_mesh, (tuple, list)) and len(photogrammetry_mesh) >= 3:
            verts_photo = np.ascontiguousarray(photogrammetry_mesh[0], dtype=np.float64)
            tris_photo = np.ascontiguousarray(photogrammetry_mesh[1], dtype=np.int32)
            uvs_photo = np.ascontiguousarray(photogrammetry_mesh[2], dtype=np.float64)
        else:
            return TextureBakeResult(
                success=False,
                status="failed_input",
                status_message="Invalid photogrammetry mesh format."
            )

        # 3. Extract LiDAR Mesh Data (Native LiDAR coordinates preserved)
        if isinstance(lidar_surface_mesh, o3d.geometry.TriangleMesh):
            verts_lidar = np.asarray(lidar_surface_mesh.vertices, dtype=np.float64)
            tris_lidar = np.asarray(lidar_surface_mesh.triangles, dtype=np.int32)
        elif isinstance(lidar_surface_mesh, (tuple, list)) and len(lidar_surface_mesh) >= 2:
            verts_lidar = np.ascontiguousarray(lidar_surface_mesh[0], dtype=np.float64)
            tris_lidar = np.ascontiguousarray(lidar_surface_mesh[1], dtype=np.int32)
        else:
            return TextureBakeResult(
                success=False,
                status="failed_input",
                status_message="Invalid LiDAR surface mesh format."
            )

        V_lidar = verts_lidar.shape[0]
        F_lidar = tris_lidar.shape[0]
        if V_lidar < 3 or F_lidar < 1 or not np.all(np.isfinite(verts_lidar)):
            return TextureBakeResult(
                success=False,
                status="failed_input",
                status_message=f"LiDAR mesh has invalid or non-finite geometry (V={V_lidar}, F={F_lidar})."
            )

        # 4. Load Photogrammetry Source Texture
        try:
            source_tex = cls.load_texture_image(photogrammetry_texture)
        except Exception as e:
            return TextureBakeResult(
                success=False,
                status="failed_input",
                status_message=f"Failed to load photogrammetry source texture: {str(e)}"
            )

        # 5. Format Target UVs
        target_uvs = np.ascontiguousarray(lidar_target_uvs, dtype=np.float64)
        if target_uvs.ndim != 2 or not np.all(np.isfinite(target_uvs)):
            return TextureBakeResult(
                success=False,
                status="failed_uv",
                status_message="Target UVs must be a finite 2D array."
            )

        if target_uvs.shape[0] == F_lidar * 3:
            uv_triangles = target_uvs.reshape(F_lidar, 3, 2)
        elif target_uvs.shape[0] == V_lidar:
            uv_triangles = target_uvs[tris_lidar]  # (F, 3, 2)
        else:
            return TextureBakeResult(
                success=False,
                status="failed_uv",
                status_message=f"Target UVs count {target_uvs.shape[0]} matches neither V ({V_lidar}) nor 3*F ({F_lidar*3})."
            )

        # Check for overlaps in target UVs
        has_overlaps, overlap_count = STELiDARUVService.detect_overlapping_triangles(target_uvs, tris_lidar)

        W_target = int(texture_resolution)
        H_target = int(texture_resolution)

        _progress(0.10, "Rasterizing Target UV Space")
        _log(f"Rasterizing {F_lidar} target triangles into {W_target}x{H_target} texture space...")

        # 6. Rasterize target UV triangles into pixel space
        px_coords, tri_indices, bary_weights = cls.rasterize_target_uv_triangles(
            uv_triangles=uv_triangles,
            texture_width=W_target,
            texture_height=H_target
        )

        total_pixels = px_coords.shape[0]
        if total_pixels == 0:
            return TextureBakeResult(
                success=False,
                status="failed_baking",
                status_message="Target UV rasterization produced 0 covered pixels."
            )

        _log(f"Rasterization produced {total_pixels:,} target surface pixels.")
        _progress(0.25, "Building Acceleration Structure")

        # 7. Build Open3D RaycastingScene for photogrammetry mesh
        mesh_legacy = o3d.geometry.TriangleMesh()
        mesh_legacy.vertices = o3d.utility.Vector3dVector(verts_photo)
        mesh_legacy.triangles = o3d.utility.Vector3iVector(tris_photo)
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(t_mesh)

        # 8. Allocate target texture image and valid mask
        channels = source_tex.shape[2] if source_tex.ndim == 3 else 3
        baked_texture = np.zeros((H_target, W_target, channels), dtype=np.uint8)
        valid_mask = np.zeros((H_target, W_target), dtype=bool)
        uncovered_mask = np.zeros((H_target, W_target), dtype=bool)

        num_source_triangles = tris_photo.shape[0]

        # 9. Interpolate 3D LiDAR positions and transform into Photogrammetry space
        # P_lidar = w0*V0 + w1*V1 + w2*V2
        matched_lidar_tris = tris_lidar[tri_indices]  # (N, 3)
        v0_lidar = verts_lidar[matched_lidar_tris[:, 0]]
        v1_lidar = verts_lidar[matched_lidar_tris[:, 1]]
        v2_lidar = verts_lidar[matched_lidar_tris[:, 2]]

        w0 = bary_weights[:, 0:1]
        w1 = bary_weights[:, 1:2]
        w2 = bary_weights[:, 2:3]
        pts_lidar_interp = w0 * v0_lidar + w1 * v1_lidar + w2 * v2_lidar  # (N, 3)

        # Transform to photogrammetry space: P_photo = s * R * P_lidar + t
        pts_photo_query = alignment_transform.apply(pts_lidar_interp)

        _progress(0.35, "Projecting Surface & Sampling Texture")
        _log("Executing batched geometric correspondence queries...")

        all_distances = []
        valid_pixel_count = 0
        uncovered_pixel_count = 0

        # Process in batches
        num_batches = int(np.ceil(total_pixels / batch_size))
        for b in range(num_batches):
            b_start = b * batch_size
            b_end = min(total_pixels, (b + 1) * batch_size)

            batch_queries = pts_photo_query[b_start:b_end]
            batch_px = px_coords[b_start:b_end]

            # Query closest point on photogrammetry surface
            q_tensor = o3d.core.Tensor(batch_queries, dtype=o3d.core.Dtype.Float32)
            closest_dict = scene.compute_closest_points(q_tensor)

            closest_pts = closest_dict['points'].numpy().astype(np.float64)
            primitive_ids = closest_dict['primitive_ids'].numpy().astype(np.int32)
            primitive_ids = np.clip(primitive_ids, 0, num_source_triangles - 1)

            # Compute Euclidean correspondence distance
            batch_dists = np.linalg.norm(batch_queries - closest_pts, axis=1)
            all_distances.append(batch_dists)

            # Distance threshold validation
            batch_valid = batch_dists <= max_correspondence_distance

            # For valid points, calculate source barycentrics and sample texture
            if np.any(batch_valid):
                valid_idx = np.where(batch_valid)[0]
                val_prim_ids = primitive_ids[valid_idx]
                val_closest = closest_pts[valid_idx]
                val_px = batch_px[valid_idx]

                # Source triangle vertices
                val_source_tris = tris_photo[val_prim_ids]
                v0_src = verts_photo[val_source_tris[:, 0]]
                v1_src = verts_photo[val_source_tris[:, 1]]
                v2_src = verts_photo[val_source_tris[:, 2]]

                # Barycentric coordinates on source triangle
                src_bary = STELiDARUVService.compute_face_normals  # placeholder ref
                # Compute source barycentric coords
                v0_vec = v1_src - v0_src
                v1_vec = v2_src - v0_src
                v2_vec = val_closest - v0_src

                d00 = np.sum(v0_vec * v0_vec, axis=1)
                d01 = np.sum(v0_vec * v1_vec, axis=1)
                d11 = np.sum(v1_vec * v1_vec, axis=1)
                d20 = np.sum(v2_vec * v0_vec, axis=1)
                d21 = np.sum(v2_vec * v1_vec, axis=1)

                denom = (d00 * d11) - (d01 * d01)
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

                sw1 = (d11 * d20 - d01 * d21) / denom
                sw2 = (d00 * d21 - d01 * d20) / denom
                sw0 = 1.0 - sw1 - sw2

                sw = np.stack([sw0, sw1, sw2], axis=1)
                sw = np.clip(sw, 0.0, 1.0)
                sw_sum = np.sum(sw, axis=1, keepdims=True)
                sw_sum = np.where(sw_sum < 1e-12, 1.0, sw_sum)
                sw /= sw_sum

                # Retrieve source UVs for matched source triangles
                if uvs_photo.ndim == 2 and uvs_photo.shape[0] == (num_source_triangles * 3):
                    uv0_s = uvs_photo[val_prim_ids * 3 + 0]
                    uv1_s = uvs_photo[val_prim_ids * 3 + 1]
                    uv2_s = uvs_photo[val_prim_ids * 3 + 2]
                elif uvs_photo.ndim == 3 and uvs_photo.shape[0] == num_source_triangles:
                    uv0_s = uvs_photo[val_prim_ids, 0]
                    uv1_s = uvs_photo[val_prim_ids, 1]
                    uv2_s = uvs_photo[val_prim_ids, 2]
                elif uvs_photo.ndim == 2 and uvs_photo.shape[0] == verts_photo.shape[0]:
                    uv0_s = uvs_photo[val_source_tris[:, 0]]
                    uv1_s = uvs_photo[val_source_tris[:, 1]]
                    uv2_s = uvs_photo[val_source_tris[:, 2]]
                else:
                    uv0_s = v0_src[:, :2]
                    uv1_s = v1_src[:, :2]
                    uv2_s = v2_src[:, :2]

                src_uvs_interp = sw[:, 0:1] * uv0_s + sw[:, 1:2] * uv1_s + sw[:, 2:3] * uv2_s

                # Bilinear sample source texture
                sampled_colors = cls.sample_texture_bilinear(source_tex, src_uvs_interp)

                # Write to target texture
                baked_texture[val_px[:, 1], val_px[:, 0]] = sampled_colors
                valid_mask[val_px[:, 1], val_px[:, 0]] = True
                valid_pixel_count += len(valid_idx)

            if np.any(~batch_valid):
                inval_idx = np.where(~batch_valid)[0]
                inval_px = batch_px[inval_idx]
                uncovered_mask[inval_px[:, 1], inval_px[:, 0]] = True
                uncovered_pixel_count += len(inval_idx)

            pct = 0.35 + 0.45 * ((b + 1) / num_batches)
            _progress(pct, f"Baking Surface ({int(pct*100)}%)")

        # 10. Compute Distance Statistics
        all_dists_arr = np.concatenate(all_distances)
        min_dist = float(np.min(all_dists_arr))
        median_dist = float(np.median(all_dists_arr))
        p95_dist = float(np.percentile(all_dists_arr, 95))
        max_dist = float(np.max(all_dists_arr))

        # Total unique surface pixels
        total_surf_pixels = int(np.sum(valid_mask | uncovered_mask))
        valid_surf_pixels = int(np.sum(valid_mask))
        uncovered_surf_pixels = total_surf_pixels - valid_surf_pixels
        coverage_ratio = float(valid_surf_pixels / total_surf_pixels) if total_surf_pixels > 0 else 0.0

        _progress(0.85, "Applying Texture Padding")
        _log(f"Applying Euclidean seam dilation (padding = {texture_padding}px)...")

        # 11. Apply Texture Padding / Seam Dilation
        padded_texture = cls.apply_texture_padding(
            texture=baked_texture,
            mask=valid_mask,
            padding=texture_padding
        )

        _progress(0.95, "Assembling Derived Output Mesh")

        # 12. Create Derived Output Mesh (Native LiDAR coordinates preserved)
        output_mesh = o3d.geometry.TriangleMesh()
        output_mesh.vertices = o3d.utility.Vector3dVector(verts_lidar.copy())
        output_mesh.triangles = o3d.utility.Vector3iVector(tris_lidar.copy())

        # Assign per-wedge triangle UVs (shape (F*3, 2))
        if target_uvs.shape[0] == F_lidar * 3:
            output_mesh.triangle_uvs = o3d.utility.Vector2dVector(target_uvs.copy())
        else:
            output_mesh.triangle_uvs = o3d.utility.Vector2dVector(target_uvs[tris_lidar].reshape(-1, 2))

        # Attach baked texture
        output_mesh.textures = [o3d.geometry.Image(padded_texture)]

        # 13. Sanity Validations
        if (len(output_mesh.vertices) == 0 or
            len(output_mesh.triangles) == 0 or
            len(output_mesh.triangle_uvs) != F_lidar * 3 or
            padded_texture.shape[0] != H_target or
            padded_texture.shape[1] != W_target):
            return TextureBakeResult(
                success=False,
                status="failed_sanity_check",
                status_message="Derived mesh sanity check failed."
            )

        _progress(1.0, "Texture Baking Complete")
        _log(f"Baking succeeded: {coverage_ratio*100.0:.2f}% coverage, {median_dist*100.0:.2f} cm median dist.")

        status = "ready" if coverage_ratio >= 0.80 else "VALID_WITH_PARTIAL_COVERAGE"
        status_message = f"Texture baking complete: {valid_surf_pixels:,} valid pixels ({coverage_ratio*100.0:.1f}% coverage)."

        return TextureBakeResult(
            success=True,
            status=status,
            status_message=status_message,
            output_mesh=output_mesh,
            output_texture=padded_texture,
            texture_width=W_target,
            texture_height=H_target,
            total_texture_pixels=total_surf_pixels,
            valid_texture_pixels=valid_surf_pixels,
            uncovered_texture_pixels=uncovered_surf_pixels,
            coverage_ratio=coverage_ratio,
            median_distance=median_dist,
            p95_distance=p95_dist,
            max_distance=max_dist,
            metadata={
                "has_overlapping_uvs": has_overlaps,
                "overlapping_triangle_count": overlap_count,
                "texture_padding": texture_padding,
                "max_correspondence_distance": max_correspondence_distance,
                "resolution": texture_resolution
            }
        )


class STETextureBakingWorker(QThread):
    """
    Asynchronous Qt background worker for non-blocking texture baking.
    """
    if PYSIDE_AVAILABLE:
        progress = Signal(float, str)
        finished = Signal(object)
        error = Signal(str)

    def __init__(
        self,
        photogrammetry_mesh,
        photogrammetry_texture,
        lidar_surface_mesh,
        lidar_target_uvs,
        alignment_transform,
        texture_resolution: int = 4096,
        max_correspondence_distance: float = 0.05,
        texture_padding: int = 4,
        batch_size: int = 100000,
        parent=None
    ):
        super().__init__(parent)
        self.photogrammetry_mesh = photogrammetry_mesh
        self.photogrammetry_texture = photogrammetry_texture
        self.lidar_surface_mesh = lidar_surface_mesh
        self.lidar_target_uvs = lidar_target_uvs
        self.alignment_transform = alignment_transform
        self.texture_resolution = texture_resolution
        self.max_correspondence_distance = max_correspondence_distance
        self.texture_padding = texture_padding
        self.batch_size = batch_size
        self.result: Optional[TextureBakeResult] = None

    def run(self):
        try:
            def _on_progress(pct: float, stage: str):
                if PYSIDE_AVAILABLE:
                    self.progress.emit(pct, stage)

            result = STETextureBakingService.bake(
                photogrammetry_mesh=self.photogrammetry_mesh,
                photogrammetry_texture=self.photogrammetry_texture,
                lidar_surface_mesh=self.lidar_surface_mesh,
                lidar_target_uvs=self.lidar_target_uvs,
                alignment_transform=self.alignment_transform,
                texture_resolution=self.texture_resolution,
                max_correspondence_distance=self.max_correspondence_distance,
                texture_padding=self.texture_padding,
                batch_size=self.batch_size,
                progress_fn=_on_progress
            )
            self.result = result
            if PYSIDE_AVAILABLE:
                self.finished.emit(result)
        except Exception as e:
            if PYSIDE_AVAILABLE:
                self.error.emit(str(e))
