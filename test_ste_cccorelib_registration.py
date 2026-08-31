"""
Unit Test Suite for Isolated CCCoreLib Registration Backend

Tests:
- Test A: CCCoreLib availability
- Test B: Rigid registration (known R, T, scale=1.0)
- Test C: Scale-aware registration (known s=7.0, R, T)
- Test D: Transformation correctness (transformed source points match reference target points)
- Test E: Determinism (identical input produces identical result)
- Test F: Failure handling (degenerate input, <3 points, NaN/Inf, mismatched dimensions)
- Test G: ICP registration (point cloud alignment)
"""

import unittest
import numpy as np
import ste_cccorelib_registration as ccr


def make_random_rotation_matrix():
    """Generates a random valid 3D orthonormal rotation matrix."""
    q, r = np.linalg.qr(np.random.randn(3, 3))
    # Ensure determinant is +1 (proper rotation, no reflection)
    d = np.linalg.det(q)
    if d < 0:
        q[:, 0] *= -1
    return q


class TestCCCoreLibRegistration(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_a_availability(self):
        """Test A: CCCoreLib backend availability."""
        available = ccr.is_cccorelib_available()
        self.assertTrue(available, "CCCoreLib backend should be compiled and available.")

    def test_b_rigid_registration(self):
        """Test B: Rigid registration with known rotation and translation (scale = 1.0)."""
        # Generate synthetic 3D points
        N = 25
        p_aligned = np.random.uniform(-10.0, 10.0, size=(N, 3))

        # Known transformation
        R_true = make_random_rotation_matrix()
        T_true = np.array([3.5, -2.1, 7.8], dtype=np.float64)
        s_true = 1.0

        # P_ref = s * (P_aligned @ R.T) + T
        p_ref = s_true * (p_aligned @ R_true.T) + T_true

        # Solve using CCCoreLib with adjust_scale=False
        result = ccr.register_point_pairs(p_aligned, p_ref, adjust_scale=False)

        self.assertTrue(result.success, f"Rigid registration failed: {result.error_message}")
        self.assertAlmostEqual(result.scale, 1.0, places=5)
        np.testing.assert_allclose(result.rotation, R_true, atol=1e-4)
        np.testing.assert_allclose(result.translation, T_true, atol=1e-4)
        self.assertLess(result.rms, 1e-4)

    def test_c_scale_aware_registration(self):
        """Test C: Scale-aware registration with known scale s = 7.0."""
        N = 30
        p_aligned = np.random.uniform(-5.0, 5.0, size=(N, 3))

        # Known transformation with large scale factor
        R_true = make_random_rotation_matrix()
        T_true = np.array([15.0, -8.4, 4.2], dtype=np.float64)
        s_true = 7.0

        p_ref = s_true * (p_aligned @ R_true.T) + T_true

        # Solve using CCCoreLib with adjust_scale=True
        result = ccr.register_point_pairs(p_aligned, p_ref, adjust_scale=True)

        self.assertTrue(result.success, f"Scale registration failed: {result.error_message}")
        self.assertAlmostEqual(result.scale, s_true, places=4)
        np.testing.assert_allclose(result.rotation, R_true, atol=1e-4)
        np.testing.assert_allclose(result.translation, T_true, atol=1e-3)
        self.assertLess(result.rms, 1e-4)

    def test_d_transformation_correctness(self):
        """Test D: Transformed source points accurately match target reference points."""
        N = 15
        p_aligned = np.random.uniform(-20.0, 20.0, size=(N, 3))

        R_true = make_random_rotation_matrix()
        T_true = np.array([-12.3, 45.6, -7.89], dtype=np.float64)
        s_true = 3.14159

        p_ref = s_true * (p_aligned @ R_true.T) + T_true

        result = ccr.register_point_pairs(p_aligned, p_ref, adjust_scale=True)
        self.assertTrue(result.success)

        # Apply the recovered transformation
        p_transformed = result.apply(p_aligned)

        # Verify point-by-point closeness
        np.testing.assert_allclose(p_transformed, p_ref, atol=1e-4)
        # Verify RMS calculation consistency
        computed_rms = np.sqrt(np.mean(np.sum((p_ref - p_transformed) ** 2, axis=1)))
        self.assertAlmostEqual(result.rms, computed_rms, places=5)

    def test_e_determinism(self):
        """Test E: Identical inputs produce identical registration results."""
        N = 20
        p_aligned = np.random.uniform(-10.0, 10.0, size=(N, 3))
        R_true = make_random_rotation_matrix()
        T_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        p_ref = 2.5 * (p_aligned @ R_true.T) + T_true

        res1 = ccr.register_point_pairs(p_aligned, p_ref, adjust_scale=True)
        res2 = ccr.register_point_pairs(p_aligned, p_ref, adjust_scale=True)

        self.assertTrue(res1.success)
        self.assertTrue(res2.success)
        np.testing.assert_array_equal(res1.transform, res2.transform)
        self.assertEqual(res1.scale, res2.scale)
        self.assertEqual(res1.rms, res2.rms)

    def test_f_failure_handling(self):
        """Test F: Failure handling for invalid, degenerate, and edge-case inputs."""
        # 1. Fewer than 3 points
        p_2pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        res = ccr.register_point_pairs(p_2pts, p_2pts)
        self.assertFalse(res.success)
        self.assertIn("3 point pairs", res.error_message)

        # 2. Empty arrays
        p_empty = np.zeros((0, 3))
        res = ccr.register_point_pairs(p_empty, p_empty)
        self.assertFalse(res.success)

        # 3. Point count mismatch
        p_3pts = np.random.randn(3, 3)
        p_4pts = np.random.randn(4, 3)
        res = ccr.register_point_pairs(p_3pts, p_4pts)
        self.assertFalse(res.success)
        self.assertIn("mismatch", res.error_message.lower())

        # 4. NaN / Inf inputs
        p_nan = np.random.randn(4, 3)
        p_nan[1, 1] = np.nan
        res = ccr.register_point_pairs(p_nan, p_3pts[:3])
        self.assertFalse(res.success)

        # 5. Degenerate collinear points handling
        p_line = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float64)
        res_line = ccr.register_point_pairs(p_line, p_line)
        # Should gracefully return without crashing the host process
        self.assertIsInstance(res_line, ccr.CCCoreLibRegistrationResult)

    def test_g_icp_fine_registration(self):
        """Test G: Fine ICP registration between synthetic point clouds."""
        # Generate a synthetic point cloud (sphere or surface patch)
        u = np.linspace(0, np.pi, 20)
        v = np.linspace(0, 2 * np.pi, 20)
        u, v = np.meshgrid(u, v)
        x = np.sin(u) * np.cos(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(u)
        model_pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # Create slight rigid offset
        R_small = np.array([
            [0.9998, -0.0175,  0.0099],
            [0.0176,  0.9998, -0.0051],
            [-0.0098,  0.0053,  0.9999]
        ])
        T_small = np.array([0.05, -0.03, 0.02])

        # data_pts is the inverse transformed model
        data_pts = (model_pts - T_small) @ R_small

        # Run ICP registration to align data_pts onto model_pts
        result = ccr.refine_icp(
            model_pts=model_pts,
            data_pts=data_pts,
            adjust_scale=False,
            min_rms_decrease=1e-6,
            max_iterations=50
        )

        self.assertTrue(result.success, f"ICP failed: {result.error_message}")
        self.assertLess(result.rms, 0.05, f"ICP RMS error too high: {result.rms}")


if __name__ == "__main__":
    unittest.main()
