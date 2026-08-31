"""
Unit Test Suite for STE Texture Projection Foundation
=====================================================

Tests:
- Test A: Perfect correspondence (identical surfaces -> 0 distance, 100% coverage)
- Test B: Known transformation (verifies alignment is correctly applied)
- Test C: Barycentric coordinates correctness (w0+w1+w2=1, interior reconstruction)
- Test D: UV interpolation accuracy (w0*UV0 + w1*UV1 + w2*UV2)
- Test E: Distance rejection (points exceeding threshold are marked invalid)
- Test F: Coverage calculation (valid + invalid = total, 0 <= coverage <= 1)
- Test G: Non-destructive behavior (input geometry & UV arrays unmodified)
- Test H: Determinism (identical inputs -> identical outputs)
- Test I: Multiple triangles correspondence (correct primitive IDs assigned)
- Test J: Realistic transformed surface (multi-triangle mesh with s=7.0, rotation, translation)
"""

import unittest
import numpy as np
import open3d as o3d

from ste_alignment import STEAlignmentResult
from ste_texture_projection import (
    STETextureProjectionService,
    STETextureProjectionSettings,
    STETextureProjectionResult
)


def create_simple_quad_mesh():
    """
    Creates a simple 2-triangle quad in the XY plane:
    Vertices:
      0: (0, 0, 0) -> UV: (0.0, 0.0)
      1: (1, 0, 0) -> UV: (1.0, 0.0)
      2: (1, 1, 0) -> UV: (1.0, 1.0)
      3: (0, 1, 0) -> UV: (0.0, 1.0)
    Triangles:
      T0: [0, 1, 2]
      T1: [0, 2, 3]
    """
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64)

    tris = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)

    # Per-corner UVs (6 UVs for 2 triangles)
    uvs = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],  # T0: (0, 1, 2)
        [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]   # T1: (0, 2, 3)
    ], dtype=np.float64)

    return verts, tris, uvs


def make_rotation_matrix(angle_rad: float, axis: str = 'z'):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


class TestSTETextureProjection(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_a_perfect_correspondence(self):
        """Test A: Identical surfaces produce near-zero distance and 100% coverage."""
        verts, tris, uvs = create_simple_quad_mesh()
        
        # Test points directly on the surface
        pts_lidar = np.array([
            [0.2, 0.2, 0.0],
            [0.8, 0.4, 0.0],
            [0.3, 0.7, 0.0]
        ], dtype=np.float64)

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureProjectionService.project(
            lidar_surface_points=pts_lidar,
            photogrammetry_vertices=verts,
            photogrammetry_triangles=tris,
            photogrammetry_uvs=uvs,
            alignment_result=align_res
        )

        self.assertTrue(res.success)
        self.assertEqual(res.coverage_ratio, 1.0)
        self.assertEqual(res.valid_samples, 3)
        self.assertLess(res.max_distance, 1e-4)
        self.assertTrue(res.is_ready_for_baking)

    def test_b_known_transformation(self):
        """Test B: Alignment transform is correctly applied before projection."""
        verts, tris, uvs = create_simple_quad_mesh()

        # True transformation: s=2.0, translation=(5, 10, 0)
        s = 2.0
        R = np.eye(3)
        t = np.array([5.0, 10.0, 0.0])

        # A point in LiDAR coordinates: (0.25, 0.25, 0.0)
        # When transformed by s*R*P + t: 2*(0.25, 0.25, 0) + (5, 10, 0) = (5.5, 10.5, 0.0)
        # Now if photogrammetry mesh is located at (5, 10, 0) with size 2x2:
        verts_photo = s * verts + t

        p_lidar = np.array([[0.25, 0.25, 0.0]], dtype=np.float64)

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=R, translation=t, scale=s,
            transformation_matrix=np.array([
                [2.0, 0.0, 0.0, 5.0],
                [0.0, 2.0, 0.0, 10.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]),
            rms_error=0.0, residuals=[]
        )

        res = STETextureProjectionService.project(
            lidar_surface_points=p_lidar,
            photogrammetry_vertices=verts_photo,
            photogrammetry_triangles=tris,
            photogrammetry_uvs=uvs,
            alignment_result=align_res
        )

        self.assertTrue(res.success)
        self.assertLess(res.max_distance, 1e-4)
        self.assertEqual(res.valid_samples, 1)

    def test_c_barycentric_correctness(self):
        """Test C: Barycentric coordinates sum to 1 and accurately reconstruct point."""
        v0 = np.array([[0.0, 0.0, 0.0]])
        v1 = np.array([[1.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 1.0, 0.0]])

        # Known interior point: P = 0.2*V0 + 0.5*V1 + 0.3*V2 = (0.5, 0.3, 0.0)
        p = np.array([[0.5, 0.3, 0.0]])

        bary = STETextureProjectionService.compute_barycentric_coordinates(p, v0, v1, v2)
        w0, w1, w2 = bary[0]

        self.assertAlmostEqual(w0 + w1 + w2, 1.0, places=5)
        self.assertAlmostEqual(w0, 0.2, places=4)
        self.assertAlmostEqual(w1, 0.5, places=4)
        self.assertAlmostEqual(w2, 0.3, places=4)

        # Reconstructed point
        p_reconstructed = w0 * v0 + w1 * v1 + w2 * v2
        np.testing.assert_allclose(p_reconstructed, p, atol=1e-5)

    def test_d_uv_interpolation(self):
        """Test D: Interpolated UV matches w0*UV0 + w1*UV1 + w2*UV2."""
        verts, tris, uvs = create_simple_quad_mesh()
        
        # Test point at center of T0 (triangle 0: [0,0,0], [1,0,0], [1,1,0])
        # Barycentric coordinates for centroid = (1/3, 1/3, 1/3)
        # Expected UV = (1/3)*(0,0) + (1/3)*(1,0) + (1/3)*(1,1) = (2/3, 1/3) = (0.6667, 0.3333)
        p_centroid_t0 = np.array([[2.0/3.0, 1.0/3.0, 0.0]], dtype=np.float64)

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureProjectionService.project(
            lidar_surface_points=p_centroid_t0,
            photogrammetry_vertices=verts,
            photogrammetry_triangles=tris,
            photogrammetry_uvs=uvs,
            alignment_result=align_res
        )

        self.assertTrue(res.success)
        self.assertEqual(res.triangle_ids[0], 0)
        expected_uv = np.array([2.0/3.0, 1.0/3.0])
        np.testing.assert_allclose(res.source_uvs[0], expected_uv, atol=1e-4)

    def test_e_distance_rejection(self):
        """Test E: Points exceeding max_correspondence_distance are marked invalid."""
        verts, tris, uvs = create_simple_quad_mesh()
        
        # 1 point on surface (z=0.0) and 1 point far above (z=0.50m)
        pts_lidar = np.array([
            [0.5, 0.5, 0.01],   # 1cm above -> valid (< 5cm)
            [0.5, 0.5, 0.50]    # 50cm above -> invalid (> 5cm)
        ], dtype=np.float64)

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        settings = STETextureProjectionSettings(max_correspondence_distance=0.05)
        res = STETextureProjectionService.project(
            lidar_surface_points=pts_lidar,
            photogrammetry_vertices=verts,
            photogrammetry_triangles=tris,
            photogrammetry_uvs=uvs,
            alignment_result=align_res,
            settings=settings
        )

        self.assertTrue(res.valid_mask[0])
        self.assertFalse(res.valid_mask[1])
        self.assertEqual(res.valid_samples, 1)
        self.assertEqual(res.invalid_samples, 1)
        self.assertEqual(res.coverage_ratio, 0.5)

    def test_f_coverage_calculation(self):
        """Test F: valid + invalid = total and 0 <= coverage <= 1."""
        verts, tris, uvs = create_simple_quad_mesh()
        
        pts = np.random.uniform(0.0, 1.0, size=(20, 3))
        # Set arbitrary z offsets
        pts[:, 2] = np.linspace(0.0, 0.20, 20)

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        settings = STETextureProjectionSettings(max_correspondence_distance=0.08)
        res = STETextureProjectionService.project(
            lidar_surface_points=pts,
            photogrammetry_vertices=verts,
            photogrammetry_triangles=tris,
            photogrammetry_uvs=uvs,
            alignment_result=align_res,
            settings=settings
        )

        self.assertEqual(res.valid_samples + res.invalid_samples, res.total_samples)
        self.assertGreaterEqual(res.coverage_ratio, 0.0)
        self.assertLessEqual(res.coverage_ratio, 1.0)

    def test_g_non_destructive_behavior(self):
        """Test G: Input geometry and UV arrays remain 100% unmodified."""
        verts, tris, uvs = create_simple_quad_mesh()
        pts_lidar = np.array([[0.3, 0.4, 0.0], [0.6, 0.7, 0.0]])

        verts_copy = verts.copy()
        tris_copy = tris.copy()
        uvs_copy = uvs.copy()
        pts_copy = pts_lidar.copy()

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        _ = STETextureProjectionService.project(
            lidar_surface_points=pts_lidar,
            photogrammetry_vertices=verts,
            photogrammetry_triangles=tris,
            photogrammetry_uvs=uvs,
            alignment_result=align_res
        )

        np.testing.assert_array_equal(verts, verts_copy)
        np.testing.assert_array_equal(tris, tris_copy)
        np.testing.assert_array_equal(uvs, uvs_copy)
        np.testing.assert_array_equal(pts_lidar, pts_copy)

    def test_h_determinism(self):
        """Test H: Running identical projection twice produces identical output."""
        verts, tris, uvs = create_simple_quad_mesh()
        pts_lidar = np.array([[0.2, 0.3, 0.01], [0.7, 0.8, 0.02]])

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res1 = STETextureProjectionService.project(pts_lidar, verts, tris, uvs, align_res)
        res2 = STETextureProjectionService.project(pts_lidar, verts, tris, uvs, align_res)

        np.testing.assert_array_equal(res1.triangle_ids, res2.triangle_ids)
        np.testing.assert_array_equal(res1.barycentric_coordinates, res2.barycentric_coordinates)
        np.testing.assert_array_equal(res1.source_uvs, res2.source_uvs)
        np.testing.assert_array_equal(res1.distances, res2.distances)

    def test_i_multiple_triangles(self):
        """Test I: Correct triangle IDs are returned for points on different triangles."""
        verts, tris, uvs = create_simple_quad_mesh()
        
        # Point on T0 (bottom-right: [0.8, 0.2, 0])
        # Point on T1 (top-left: [0.2, 0.8, 0])
        pts_lidar = np.array([
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0]
        ], dtype=np.float64)

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureProjectionService.project(pts_lidar, verts, tris, uvs, align_res)

        self.assertEqual(res.triangle_ids[0], 0)
        self.assertEqual(res.triangle_ids[1], 1)

    def test_j_realistic_transformed_surface(self):
        """Test J: Multi-triangle mesh with scale=7.0, rotation, translation."""
        # Create a grid mesh (4 quads = 8 triangles)
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

        # UVs matching normalized coords
        uvs = verts[:, :2] / 2.0

        # Transform Photogrammetry mesh by scale=7.0, rotation=30deg, translation=(10, -5, 8)
        s_true = 7.0
        R_true = make_rotation_matrix(np.pi / 6, 'z')
        t_true = np.array([10.0, -5.0, 8.0])

        verts_photo = s_true * (verts @ R_true.T) + t_true

        # LiDAR samples are defined in original raw LiDAR space
        pts_lidar = np.array([
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.5, 1.5, 0.0]
        ], dtype=np.float64)

        align_res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=R_true, translation=t_true, scale=s_true,
            transformation_matrix=np.eye(4), rms_error=0.0, residuals=[]
        )

        res = STETextureProjectionService.project(
            lidar_surface_points=pts_lidar,
            photogrammetry_vertices=verts_photo,
            photogrammetry_triangles=tris,
            photogrammetry_uvs=uvs,
            alignment_result=align_res
        )

        self.assertTrue(res.success)
        self.assertEqual(res.valid_samples, 3)
        self.assertLess(res.max_distance, 1e-4)
        self.assertEqual(res.coverage_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
