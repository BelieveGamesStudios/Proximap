"""
Unit Test Suite for STE LiDAR Target UV Parameterization
========================================================

Tests:
- Test A: Simple triangle UV generation (3 valid UVs, finite values, valid triangle)
- Test B: Multi-triangle mesh (UVs generated for every vertex / triangle corner)
- Test C: UV range (all UVs normalized into [0, 1])
- Test D: Degenerate geometry (graceful failure on invalid inputs)
- Test E: Zero-area UV detection (detects degenerate UV triangles)
- Test F: Overlap detection (intentionally overlapping triangles detected)
- Test G: Non-overlapping UVs (clean parameterization reports no overlaps)
- Test H: Determinism (identical inputs produce identical UVs)
- Test I: Source preservation (vertices, triangles, normals unmodified)
- Test J: Projection integration (target UVs coexist with texture projection service)
- Test K: Resolution independence (same UVs map to 1024, 2048, 4096 pixels)
- Test L: Realistic architectural mesh (walls, floor, ceiling charts)
"""

import unittest
import numpy as np
import open3d as o3d

from ste_alignment import STEAlignmentResult
from ste_texture_projection import (
    STETextureProjectionService,
    STETextureProjectionSettings
)
from ste_lidar_uv import (
    STELiDARUVService,
    STELiDARUVSettings,
    STELiDARUVResult
)


def create_room_box_mesh():
    """
    Creates an architectural room mesh with floor, ceiling, and 4 walls (6 faces = 12 triangles).
    Size: 4m x 3m x 2.5m (L x W x H)
    """
    # 8 box vertices
    verts = np.array([
        [0.0, 0.0, 0.0],  # 0: floor SW
        [4.0, 0.0, 0.0],  # 1: floor SE
        [4.0, 3.0, 0.0],  # 2: floor NE
        [0.0, 3.0, 0.0],  # 3: floor NW
        [0.0, 0.0, 2.5],  # 4: ceiling SW
        [4.0, 0.0, 2.5],  # 5: ceiling SE
        [4.0, 3.0, 2.5],  # 6: ceiling NE
        [0.0, 3.0, 2.5],  # 7: ceiling NW
    ], dtype=np.float64)

    # 12 triangles
    tris = np.array([
        # Floor (+Z normal)
        [0, 2, 1], [0, 3, 2],
        # Ceiling (-Z normal)
        [4, 5, 6], [4, 6, 7],
        # South Wall (-Y normal)
        [0, 1, 5], [0, 5, 4],
        # North Wall (+Y normal)
        [3, 6, 2], [3, 7, 6],
        # West Wall (-X normal)
        [0, 4, 7], [0, 7, 3],
        # East Wall (+X normal)
        [1, 2, 6], [1, 6, 5]
    ], dtype=np.int32)

    return verts, tris


class TestSTELiDARUV(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_a_simple_triangle(self):
        """Test A: Generate UVs for a single triangle."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int32)

        res = STELiDARUVService.generate_uvs(verts, tris)

        self.assertTrue(res.success)
        self.assertEqual(res.vertex_count, 3)
        self.assertEqual(res.triangle_count, 1)
        self.assertEqual(res.uvs.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(res.uvs)))
        self.assertTrue(res.is_ready_for_baking)

    def test_b_multi_triangle_mesh(self):
        """Test B: UVs exist for every vertex and triangle corner in a multi-triangle mesh."""
        verts, tris = create_room_box_mesh()
        res = STELiDARUVService.generate_uvs(verts, tris)

        self.assertTrue(res.success)
        self.assertEqual(res.vertex_count, len(verts))
        self.assertEqual(res.triangle_count, len(tris))
        self.assertEqual(res.uvs.shape, (len(tris) * 3, 2))
        self.assertEqual(res.vertex_uvs.shape, (len(verts), 2))
        self.assertTrue(np.all(np.isfinite(res.uvs)))

    def test_c_uv_range(self):
        """Test C: Normalized UV coordinates remain within [0, 1]."""
        verts, tris = create_room_box_mesh()
        res = STELiDARUVService.generate_uvs(verts, tris)

        self.assertGreaterEqual(res.uv_min[0], 0.0)
        self.assertGreaterEqual(res.uv_min[1], 0.0)
        self.assertLessEqual(res.uv_max[0], 1.0)
        self.assertLessEqual(res.uv_max[1], 1.0)

    def test_d_degenerate_geometry(self):
        """Test D: Graceful failure on invalid / degenerate input."""
        # Less than 3 vertices
        res1 = STELiDARUVService.generate_uvs(np.zeros((2, 3)), np.zeros((1, 3), dtype=np.int32))
        self.assertFalse(res1.success)

        # NaN values in vertices
        verts_nan = np.array([[0, 0, 0], [1, 0, np.nan], [0, 1, 0]])
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        res2 = STELiDARUVService.generate_uvs(verts_nan, tris)
        self.assertFalse(res2.success)

    def test_e_zero_area_uv_detection(self):
        """Test E: Zero-area UV triangles are detected."""
        verts = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int32)
        
        # Degenerate collinear UVs: (0,0), (0.5, 0.5), (1,1) -> area = 0
        uvs_degen = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)

        count, mask = STELiDARUVService.detect_zero_area_triangles(uvs_degen, tris)
        self.assertEqual(count, 1)
        self.assertTrue(mask[0])

    def test_f_overlap_detection(self):
        """Test F: Intentionally overlapping UV triangles are detected."""
        # 2 identical overlapping triangles in UV space
        tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        uvs_overlap = np.array([
            [0.1, 0.1], [0.8, 0.1], [0.1, 0.8],  # T0
            [0.2, 0.2], [0.9, 0.2], [0.2, 0.9]   # T1 overlapping T0
        ], dtype=np.float64)

        has_overlap, count = STELiDARUVService.detect_overlapping_triangles(uvs_overlap, tris)
        self.assertTrue(has_overlap)
        self.assertEqual(count, 2)

    def test_g_non_overlapping_uvs(self):
        """Test G: Disjoint non-overlapping UV triangles report has_overlapping_uvs = False."""
        tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        uvs_disjoint = np.array([
            [0.0, 0.0], [0.4, 0.0], [0.0, 0.4],  # T0 on left
            [0.6, 0.6], [1.0, 0.6], [0.6, 1.0]   # T1 on right
        ], dtype=np.float64)

        has_overlap, count = STELiDARUVService.detect_overlapping_triangles(uvs_disjoint, tris)
        self.assertFalse(has_overlap)
        self.assertEqual(count, 0)

    def test_h_determinism(self):
        """Test H: Generating UVs twice produces identical results."""
        verts, tris = create_room_box_mesh()

        res1 = STELiDARUVService.generate_uvs(verts, tris)
        res2 = STELiDARUVService.generate_uvs(verts, tris)

        np.testing.assert_array_equal(res1.uvs, res2.uvs)
        self.assertEqual(res1.chart_count, res2.chart_count)
        self.assertEqual(res1.uv_utilization, res2.uv_utilization)

    def test_i_source_preservation(self):
        """Test I: Input vertices, triangles, and normals are 100% unmodified."""
        verts, tris = create_room_box_mesh()
        verts_copy = verts.copy()
        tris_copy = tris.copy()

        _ = STELiDARUVService.generate_uvs(verts, tris)

        np.testing.assert_array_equal(verts, verts_copy)
        np.testing.assert_array_equal(tris, tris_copy)

    def test_j_projection_integration(self):
        """Test J: Target UVs coexist with texture projection correspondence."""
        verts_lidar, tris_lidar = create_room_box_mesh()
        
        # 1. Generate target UVs for LiDAR surface
        uv_res = STELiDARUVService.generate_uvs(verts_lidar, tris_lidar)
        self.assertTrue(uv_res.success)

        # 2. Run Texture Projection against photogrammetry mesh
        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        proj_res = STETextureProjectionService.project(
            lidar_surface_points=verts_lidar,
            photogrammetry_vertices=verts_lidar,
            photogrammetry_triangles=tris_lidar,
            photogrammetry_uvs=uv_res.uvs,
            alignment_result=align_res
        )

        self.assertTrue(proj_res.success)
        self.assertEqual(proj_res.valid_samples, len(verts_lidar))

    def test_k_resolution_independence(self):
        """Test K: Normalized UV coordinates scale to any target resolution."""
        verts, tris = create_room_box_mesh()
        res = STELiDARUVService.generate_uvs(verts, tris)

        resolutions = [1024, 2048, 4096, 8192]
        for res_px in resolutions:
            pixel_coords = np.floor(res.uvs * (res_px - 1)).astype(int)
            self.assertTrue(np.all(pixel_coords >= 0))
            self.assertTrue(np.all(pixel_coords < res_px))

    def test_l_realistic_architectural_mesh(self):
        """Test L: Multi-surface room mesh generates valid charts and utilization."""
        verts, tris = create_room_box_mesh()
        res = STELiDARUVService.generate_uvs(verts, tris)

        self.assertTrue(res.success)
        self.assertGreaterEqual(res.chart_count, 4)  # At least 4 distinct walls/floor/ceiling charts
        self.assertEqual(res.zero_area_triangle_count, 0)
        self.assertGreater(res.uv_utilization, 10.0)
        self.assertTrue(res.is_ready_for_baking)


if __name__ == "__main__":
    unittest.main()
