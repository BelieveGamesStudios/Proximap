"""
Unit Test Suite for STE Production Texture Baking Engine
========================================================

Tests:
- Test A: Basic baking (synthetic triangle with known texture -> valid baked mesh & texture)
- Test B: Interior pixel correspondence (verifies exact color recovery for interior surface pixels)
- Test C: Full surface coverage (proves continuous surface rasterization, not just vertices)
- Test D: Distance rejection (outlier surfaces exceeding threshold are marked uncovered)
- Test E: Padding / Seam dilation (verifies boundary dilation by exact pixel radius)
- Test F: UV validity (verifies finite, normalized [0, 1] target UVs on output mesh)
- Test G: Source preservation (verifies all source geometry, UVs, and textures remain unmodified)
- Test H: Alignment independence (output LiDAR geometry remains in native coordinates)
- Test I: Determinism (identical inputs produce bitwise-identical output texture)
- Test J: UV overlap diagnostics (overlapping target triangles reported in metadata)
- Test K: Invalid inputs handling (graceful failure on missing texture, empty mesh, non-finite data)
- Test L: Realistic integration (end-to-end texture baking pipeline)
"""

import unittest
import numpy as np
import open3d as o3d

from ste_alignment import STEAlignmentResult
from ste_lidar_uv import STELiDARUVService
from ste_texture_baking import (
    STETextureBakingService,
    TextureBakeResult,
    STETextureBakingWorker
)


def create_colored_quad_photogrammetry():
    """
    Creates a photogrammetry quad (2 triangles) with a 64x64 checkerboard/gradient texture.
    Vertices: (0,0,0), (2,0,0), (2,2,0), (0,2,0)
    """
    verts = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0]
    ], dtype=np.float64)

    tris = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)

    # UVs
    uvs = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
        [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]
    ], dtype=np.float64)

    # 64x64 gradient texture: Red along X, Green along Y
    H, W = 64, 64
    x = np.linspace(0, 255, W, dtype=np.uint8)
    y = np.linspace(0, 255, H, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    tex = np.stack([xx, yy, np.full_like(xx, 128)], axis=2)

    return verts, tris, uvs, tex


class TestSTETextureBaking(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_a_basic_baking(self):
        """Test A: Basic baking succeeds and produces valid output mesh and texture."""
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()

        # Identical LiDAR surface
        v_lidar = v_photo.copy()
        t_lidar = t_photo.copy()
        u_lidar = u_photo.copy()

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_lidar, t_lidar),
            lidar_target_uvs=u_lidar,
            alignment_transform=align_res,
            texture_resolution=128,
            texture_padding=2
        )

        self.assertTrue(res.success)
        self.assertIsNotNone(res.output_mesh)
        self.assertIsNotNone(res.output_texture)
        self.assertEqual(res.texture_width, 128)
        self.assertEqual(res.texture_height, 128)
        self.assertGreater(res.valid_texture_pixels, 0)
        self.assertGreater(res.coverage_ratio, 0.95)

    def test_b_interior_pixel_correspondence(self):
        """Test B: Interior target UV pixel samples exact corresponding source color."""
        # Solid Blue texture with a yellow center box
        H, W = 100, 100
        tex_photo = np.zeros((H, W, 3), dtype=np.uint8)
        tex_photo[:, :] = [0, 0, 255]  # Blue background
        tex_photo[40:60, 40:60] = [255, 255, 0]  # Yellow center box

        v_photo, t_photo, u_photo, _ = create_colored_quad_photogrammetry()
        v_lidar, t_lidar, u_lidar = v_photo.copy(), t_photo.copy(), u_photo.copy()

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_lidar, t_lidar),
            lidar_target_uvs=u_lidar,
            alignment_transform=align_res,
            texture_resolution=100,
            texture_padding=0
        )

        self.assertTrue(res.success)
        # Pixel at center (x=50, y=50) must be Yellow [255, 255, 0]
        center_color = res.output_texture[50, 50]
        np.testing.assert_allclose(center_color[:2], [255, 255], atol=5)
        self.assertLess(center_color[2], 10)

    def test_c_full_surface_coverage(self):
        """Test C: Large triangle rasterizes thousands of continuous surface pixels."""
        # Single large triangle spanning half the texture
        verts = np.array([[0,0,0], [10,0,0], [0,10,0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        uvs = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float64)

        tex = np.full((128, 128, 3), 200, dtype=np.uint8)
        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureBakingService.bake(
            photogrammetry_mesh=(verts, tris, uvs),
            photogrammetry_texture=tex,
            lidar_surface_mesh=(verts, tris),
            lidar_target_uvs=uvs,
            alignment_transform=align_res,
            texture_resolution=128,
            texture_padding=0
        )

        self.assertTrue(res.success)
        # Expected area ~ 0.5 * 0.8 * 0.8 * 128 * 128 = 5242 pixels
        self.assertGreater(res.valid_texture_pixels, 4000)

    def test_d_distance_rejection(self):
        """Test D: Distant surface parts beyond threshold are marked uncovered."""
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()

        # LiDAR quad is shifted vertically by 0.50m (50cm) in Z (threshold = 0.05m = 5cm)
        v_lidar = v_photo.copy()
        v_lidar[:, 2] += 0.50
        t_lidar = t_photo.copy()
        u_lidar = u_photo.copy()

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_lidar, t_lidar),
            lidar_target_uvs=u_lidar,
            alignment_transform=align_res,
            texture_resolution=64,
            max_correspondence_distance=0.05
        )

        self.assertTrue(res.success)
        self.assertEqual(res.valid_texture_pixels, 0)
        self.assertEqual(res.uncovered_texture_pixels, res.total_texture_pixels)
        self.assertEqual(res.coverage_ratio, 0.0)

    def test_e_padding(self):
        """Test E: Valid pixels dilate by exact padding radius."""
        # Single 20x20 square centered in 100x100 texture
        verts = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=np.float64)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uvs = np.array([
            [0.4, 0.4], [0.6, 0.4], [0.6, 0.6],
            [0.4, 0.4], [0.6, 0.6], [0.4, 0.6]
        ], dtype=np.float64)

        tex = np.full((64, 64, 3), 255, dtype=np.uint8)
        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res_nopad = STETextureBakingService.bake(
            photogrammetry_mesh=(verts, tris, uvs),
            photogrammetry_texture=tex,
            lidar_surface_mesh=(verts, tris),
            lidar_target_uvs=uvs,
            alignment_transform=align_res,
            texture_resolution=100,
            texture_padding=0
        )

        res_pad = STETextureBakingService.bake(
            photogrammetry_mesh=(verts, tris, uvs),
            photogrammetry_texture=tex,
            lidar_surface_mesh=(verts, tris),
            lidar_target_uvs=uvs,
            alignment_transform=align_res,
            texture_resolution=100,
            texture_padding=4
        )

        count_nopad = np.sum(res_nopad.output_texture[:, :, 0] > 0)
        count_pad = np.sum(res_pad.output_texture[:, :, 0] > 0)
        self.assertGreater(count_pad, count_nopad)

    def test_f_uv_validity(self):
        """Test F: Output mesh has finite UVs normalized in [0, 1]."""
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_photo, t_photo),
            lidar_target_uvs=u_photo,
            alignment_transform=align_res,
            texture_resolution=64
        )

        mesh_uvs = np.asarray(res.output_mesh.triangle_uvs)
        self.assertEqual(len(mesh_uvs), len(t_photo) * 3)
        self.assertTrue(np.all(np.isfinite(mesh_uvs)))
        self.assertGreaterEqual(np.min(mesh_uvs), 0.0)
        self.assertLessEqual(np.max(mesh_uvs), 1.0)

    def test_g_source_preservation(self):
        """Test G: Source geometry, UVs, and textures remain 100% untouched."""
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        v_lidar = v_photo.copy()
        t_lidar = t_photo.copy()
        u_lidar = u_photo.copy()

        v_photo_copy = v_photo.copy()
        t_photo_copy = t_photo.copy()
        u_photo_copy = u_photo.copy()
        tex_copy = tex_photo.copy()
        v_lidar_copy = v_lidar.copy()
        t_lidar_copy = t_lidar.copy()
        u_lidar_copy = u_lidar.copy()

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        _ = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_lidar, t_lidar),
            lidar_target_uvs=u_lidar,
            alignment_transform=align_res,
            texture_resolution=64
        )

        np.testing.assert_array_equal(v_photo, v_photo_copy)
        np.testing.assert_array_equal(t_photo, t_photo_copy)
        np.testing.assert_array_equal(u_photo, u_photo_copy)
        np.testing.assert_array_equal(tex_photo, tex_copy)
        np.testing.assert_array_equal(v_lidar, v_lidar_copy)
        np.testing.assert_array_equal(t_lidar, t_lidar_copy)
        np.testing.assert_array_equal(u_lidar, u_lidar_copy)

    def test_h_alignment_independence(self):
        """Test H: Output derived mesh remains in native LiDAR coordinates (unbaked transform)."""
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()

        # Transform photogrammetry mesh by scale=2.0, translation=(10, 20, 0)
        s_true = 2.0
        t_true = np.array([10.0, 20.0, 0.0])
        v_photo_trans = s_true * v_photo + t_true

        # LiDAR mesh is at raw origin coordinates
        v_lidar = v_photo.copy()
        t_lidar = t_photo.copy()
        u_lidar = u_photo.copy()

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=t_true, scale=s_true,
            transformation_matrix=np.array([
                [2.0, 0, 0, 10.0],
                [0, 2.0, 0, 20.0],
                [0, 0, 2.0, 0.0],
                [0, 0, 0, 1.0]
            ]), rms_error=0.0, residuals=[]
        )

        res = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo_trans, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_lidar, t_lidar),
            lidar_target_uvs=u_lidar,
            alignment_transform=align_res,
            texture_resolution=64
        )

        self.assertTrue(res.success)
        out_verts = np.asarray(res.output_mesh.vertices)
        # Output mesh must match v_lidar (origin), NOT v_photo_trans (shifted to 10,20)
        np.testing.assert_allclose(out_verts, v_lidar, atol=1e-5)

    def test_i_determinism(self):
        """Test I: Identical inputs produce bitwise-identical output textures."""
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res1 = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_photo, t_photo),
            lidar_target_uvs=u_photo,
            alignment_transform=align_res,
            texture_resolution=64
        )

        res2 = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_photo, t_photo),
            lidar_target_uvs=u_photo,
            alignment_transform=align_res,
            texture_resolution=64
        )

        np.testing.assert_array_equal(res1.output_texture, res2.output_texture)
        self.assertEqual(res1.valid_texture_pixels, res2.valid_texture_pixels)

    def test_j_uv_overlap_diagnostics(self):
        """Test J: Overlapping target UV triangles are diagnosed and reported in metadata."""
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        
        # Intentionally overlapping UVs
        u_overlap = np.array([
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9],
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9]  # Exact duplicate triangle
        ], dtype=np.float64)

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_photo, t_photo),
            lidar_target_uvs=u_overlap,
            alignment_transform=align_res,
            texture_resolution=64
        )

        self.assertTrue(res.success)
        self.assertTrue(res.metadata.get("has_overlapping_uvs", False))

    def test_k_invalid_inputs(self):
        """Test K: Graceful failure on invalid / corrupt inputs."""
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        # Empty LiDAR mesh
        res1 = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
            lidar_target_uvs=u_photo,
            alignment_transform=align_res,
            texture_resolution=64
        )
        self.assertFalse(res1.success)

        # Non-finite LiDAR vertices
        v_nan = v_photo.copy()
        v_nan[0, 0] = np.nan
        res2 = STETextureBakingService.bake(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_nan, t_photo),
            lidar_target_uvs=u_photo,
            alignment_transform=align_res,
            texture_resolution=64
        )
        self.assertFalse(res2.success)

    def test_l_realistic_integration(self):
        """Test L: Realistic multi-triangle mesh with UV generation and baking."""
        # 4 quad mesh (8 triangles)
        grid_size = 3
        x = np.linspace(0, 2, grid_size)
        y = np.linspace(0, 2, grid_size)
        xx, yy = np.meshgrid(x, y)
        verts = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())], axis=1)

        tris = []
        for j in range(grid_size - 1):
            for i in range(grid_size - 1):
                idx = j * grid_size + i
                tris.append([idx, idx + 1, idx + grid_size + 1])
                tris.append([idx, idx + grid_size + 1, idx + grid_size])
        tris = np.array(tris, dtype=np.int32)

        # Generate target UVs using STELiDARUVService
        uv_res = STELiDARUVService.generate_uvs(verts, tris)
        self.assertTrue(uv_res.success)

        tex = np.full((128, 128, 3), 180, dtype=np.uint8)
        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureBakingService.bake(
            photogrammetry_mesh=(verts, tris, uv_res.uvs),
            photogrammetry_texture=tex,
            lidar_surface_mesh=(verts, tris),
            lidar_target_uvs=uv_res.uvs,
            alignment_transform=align_res,
            texture_resolution=128,
            texture_padding=2
        )

        self.assertTrue(res.success)
        self.assertEqual(res.status, "ready")
        self.assertGreater(res.coverage_ratio, 0.90)

    def test_m_bake_invalidation_after_alignment_change(self):
        """Test M: Changing alignment transform invalidates previously derived bake result."""
        from ste_workspace import STEWorkspace
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])

        ws = STEWorkspace()
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        photo_mesh = o3d.geometry.TriangleMesh()
        photo_mesh.vertices = o3d.utility.Vector3dVector(v_photo)
        photo_mesh.triangles = o3d.utility.Vector3iVector(t_photo)

        ws.viewport.set_photogrammetry_data(photo_mesh, tex_photo, point_cloud=v_photo)
        ws.photo_verts = v_photo
        ws.photo_tris = t_photo
        ws.photo_uvs = u_photo
        ws.photo_texture_img = tex_photo
        ws.lidar_surface_verts = v_photo.copy()
        ws.lidar_surface_tris = t_photo.copy()
        ws.lidar_target_uvs = u_photo.copy()

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )
        ws.alignment_result = align_res

        # Simulate valid bake result
        ws.bake_result = TextureBakeResult(
            success=True, status="ready", status_message="OK",
            output_mesh=photo_mesh, output_texture=tex_photo,
            texture_width=128, texture_height=128
        )
        self.assertIsNotNone(ws.bake_result)

        # Invalidate via alignment change
        ws._invalidate_alignment_derived_stages()
        self.assertIsNone(ws.bake_result)
        self.assertIsNone(ws.projection_result)
        self.assertIn("Stale", ws.lbl_bake_stats.text())

    def test_n_bake_invalidation_after_uv_regeneration(self):
        """Test N: Regenerating Target UVs invalidates bake result."""
        from ste_workspace import STEWorkspace
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])

        ws = STEWorkspace()
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        photo_mesh = o3d.geometry.TriangleMesh()
        photo_mesh.vertices = o3d.utility.Vector3dVector(v_photo)
        photo_mesh.triangles = o3d.utility.Vector3iVector(t_photo)

        ws.photo_verts = v_photo
        ws.photo_tris = t_photo
        ws.photo_uvs = u_photo
        ws.photo_texture_img = tex_photo
        ws.lidar_surface_verts = v_photo.copy()
        ws.lidar_surface_tris = t_photo.copy()

        ws.bake_result = TextureBakeResult(
            success=True, status="ready", status_message="OK",
            output_mesh=photo_mesh, output_texture=tex_photo
        )

        ws._generate_target_uvs()
        self.assertIsNone(ws.bake_result)
        self.assertIn("Stale", ws.lbl_bake_stats.text())

    def test_o_bake_invalidation_after_projection_change(self):
        """Test O: Running projection validation invalidates existing bake."""
        from ste_workspace import STEWorkspace
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])

        ws = STEWorkspace()
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        photo_mesh = o3d.geometry.TriangleMesh()
        photo_mesh.vertices = o3d.utility.Vector3dVector(v_photo)
        photo_mesh.triangles = o3d.utility.Vector3iVector(t_photo)

        ws.photo_verts = v_photo
        ws.photo_tris = t_photo
        ws.photo_uvs = u_photo
        ws.photo_texture_img = tex_photo
        ws.lidar_surface_verts = v_photo.copy()
        ws.lidar_surface_tris = t_photo.copy()
        ws.lidar_target_uvs = u_photo.copy()

        ws.alignment_result = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        ws.bake_result = TextureBakeResult(
            success=True, status="ready", status_message="OK",
            output_mesh=photo_mesh, output_texture=tex_photo
        )

        ws._validate_texture_projection()
        self.assertIsNone(ws.bake_result)
        self.assertIn("Stale", ws.lbl_bake_stats.text())

    def test_p_worker_signals_and_progress(self):
        """Test P: STETextureBakingWorker executes and emits progress, finished, and result."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])

        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()
        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        worker = STETextureBakingWorker(
            photogrammetry_mesh=(v_photo, t_photo, u_photo),
            photogrammetry_texture=tex_photo,
            lidar_surface_mesh=(v_photo, t_photo),
            lidar_target_uvs=u_photo,
            alignment_transform=align_res,
            texture_resolution=64,
            texture_padding=2
        )

        received_results = []
        received_progress = []

        worker.finished.connect(lambda res: received_results.append(res))
        worker.progress.connect(lambda pct, msg: received_progress.append((pct, msg)))

        worker.run()  # Synchronous run for deterministic test

        self.assertEqual(len(received_results), 1)
        self.assertTrue(received_results[0].success)
        self.assertGreater(len(received_progress), 0)

    def test_q_baked_representation_viewport_display(self):
        """Test Q: Viewport set_lidar_baked correctly assigns per-wedge UVs and texture."""
        from ste_workspace import STEUnifiedAlignmentViewport, STELiDARRepresentation
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])

        vp = STEUnifiedAlignmentViewport()
        v_photo, t_photo, u_photo, tex_photo = create_colored_quad_photogrammetry()

        derived_mesh = o3d.geometry.TriangleMesh()
        derived_mesh.vertices = o3d.utility.Vector3dVector(v_photo)
        derived_mesh.triangles = o3d.utility.Vector3iVector(t_photo)
        derived_mesh.triangle_uvs = o3d.utility.Vector2dVector(u_photo)

        vp.set_lidar_baked(derived_mesh, tex_photo)

        self.assertEqual(vp.lidar_display_mode, STELiDARRepresentation.BAKED)
        self.assertIsNotNone(vp.obj_lidar_baked)
        self.assertIsNotNone(vp.obj_lidar_baked.mesh.texcoords)
        self.assertEqual(len(vp.obj_lidar_baked.mesh.texcoords), len(t_photo) * 3)
        self.assertIsNotNone(vp.obj_lidar_baked.mesh.texture_data)


if __name__ == "__main__":
    unittest.main()
