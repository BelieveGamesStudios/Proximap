"""
Comprehensive Unit Test Suite for Production STE Alignment & ICP Refinement Engine
===================================================================================

Tests:
- Test A: Three-point rigid alignment (non-collinear, scale=1.0)
- Test B: Four+ point rigid alignment (all valid control points used)
- Test C: Scale-aware alignment (known s, R, t recovered)
- Test D: 7.0x and 7.17x scale (representative of real dataset)
- Test E: Residual calculation & independent verification
- Test F: Three non-collinear points accepted
- Test G: Collinear points rejected gracefully
- Test H: Coincident points rejected gracefully
- Test I: Incomplete control points ignored (only complete pairs used)
- Test J: Non-destructive preview (source geometry never mutated)
- Test K: Reset alignment (reverts to Identity, geometry intact)
- Test L: Invalid/non-finite data handling
- Test ICP A: ICP unavailable / missing initial alignment handling
- Test ICP B: ICP improves synthetic alignment
- Test ICP C: Initial transform is respected
- Test ICP D: Transformation composition (T_final = T_icp @ T_init)
- Test ICP E: Scale preservation (s = 7.0 preserved under adjust_scale=False)
- Test ICP F: Non-destructive behavior (source LiDAR geometry intact)
- Test ICP G: Bad ICP result rejection (preserves initial alignment)
- Test ICP H: Complete reset clears ICP preview
- Test ICP I: Realistic multi-point + ICP end-to-end pipeline
"""

import unittest
import numpy as np

from ste_alignment import (
    STEControlPoint,
    STEControlPointManager,
    STEAlignmentResult,
    STEAlignmentService,
    STEAlignmentState,
    STEICPRefinementSettings,
    STEICPRefinementResult,
    STEICPRefinementService
)


def make_random_rotation():
    """Create a random orthonormal 3D rotation matrix with det = +1."""
    q, r = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def generate_synthetic_surface(num_points: int = 500) -> np.ndarray:
    """Generate a curved 3D surface patch (e.g. dome/curved wall)."""
    u = np.linspace(-2.0, 2.0, int(np.sqrt(num_points)))
    v = np.linspace(-2.0, 2.0, int(np.sqrt(num_points)))
    uu, vv = np.meshgrid(u, v)
    zz = 0.5 * np.sin(uu) + 0.3 * np.cos(vv)
    pts = np.stack([uu.ravel(), vv.ravel(), zz.ravel()], axis=1)
    return pts


class TestSTEAlignment(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345)

    def test_a_three_point_rigid_alignment(self):
        """Test A: Three non-collinear control-point pairs in rigid mode (scale = 1.0)."""
        p_lidar = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ], dtype=np.float64)

        R_true = make_random_rotation()
        T_true = np.array([5.0, -3.0, 2.5], dtype=np.float64)
        p_photo = (p_lidar @ R_true.T) + T_true

        res = STEAlignmentService.solve(p_lidar, p_photo, adjust_scale=False)

        self.assertTrue(res.success, f"Failed: {res.status_message}")
        self.assertEqual(res.control_point_count, 3)
        self.assertAlmostEqual(res.scale, 1.0, places=5)
        np.testing.assert_allclose(res.rotation, R_true, atol=1e-4)
        np.testing.assert_allclose(res.translation, T_true, atol=1e-4)
        self.assertLess(res.rms_error, 1e-4)

    def test_b_four_plus_point_rigid_alignment(self):
        """Test B: Four or more control points, verifying all valid points are used."""
        N = 6
        p_lidar = np.random.uniform(-10.0, 10.0, size=(N, 3))
        R_true = make_random_rotation()
        T_true = np.array([-1.5, 4.2, 8.0], dtype=np.float64)
        p_photo = (p_lidar @ R_true.T) + T_true

        res = STEAlignmentService.solve(p_lidar, p_photo, adjust_scale=False)

        self.assertTrue(res.success)
        self.assertEqual(res.control_point_count, N)
        self.assertEqual(len(res.residuals), N)
        self.assertAlmostEqual(res.scale, 1.0, places=5)
        np.testing.assert_allclose(res.rotation, R_true, atol=1e-4)
        np.testing.assert_allclose(res.translation, T_true, atol=1e-4)
        self.assertLess(res.rms_error, 1e-4)

    def test_c_scale_aware_alignment(self):
        """Test C: Scale-aware alignment recovering arbitrary scale factor."""
        N = 5
        p_lidar = np.random.uniform(-5.0, 5.0, size=(N, 3))
        R_true = make_random_rotation()
        T_true = np.array([12.0, -14.0, 6.5], dtype=np.float64)
        s_true = 3.456

        p_photo = s_true * (p_lidar @ R_true.T) + T_true

        res = STEAlignmentService.solve(p_lidar, p_photo, adjust_scale=True)

        self.assertTrue(res.success, f"Failed: {res.status_message}")
        self.assertAlmostEqual(res.scale, s_true, places=4)
        np.testing.assert_allclose(res.rotation, R_true, atol=1e-4)
        np.testing.assert_allclose(res.translation, T_true, atol=1e-3)
        self.assertLess(res.rms_error, 1e-4)

    def test_d_seven_point_one_seven_scale(self):
        """Test D: Scale = 7.0 and scale = 7.17 (representative of real dataset)."""
        for s_target in [7.0, 7.17]:
            p_lidar = np.array([
                [0.5, 1.2, -0.8],
                [-1.4, 2.0, 0.3],
                [1.8, -0.5, 1.1],
                [-0.2, -1.5, -0.9]
            ], dtype=np.float64)

            R_true = make_random_rotation()
            T_true = np.array([20.0, -35.0, 15.0], dtype=np.float64)
            p_photo = s_target * (p_lidar @ R_true.T) + T_true

            res = STEAlignmentService.solve(p_lidar, p_photo, adjust_scale=True)

            self.assertTrue(res.success)
            self.assertAlmostEqual(res.scale, s_target, places=4)
            np.testing.assert_allclose(res.rotation, R_true, atol=1e-4)
            np.testing.assert_allclose(res.translation, T_true, atol=1e-3)
            self.assertLess(res.rms_error, 1e-4)

    def test_e_residual_calculation(self):
        """Test E: Independent residual and RMS error verification."""
        N = 4
        p_lidar = np.array([
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [1.0, 2.0, 3.0]
        ], dtype=np.float64)

        R_true = make_random_rotation()
        T_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        s_true = 2.0

        p_photo = s_true * (p_lidar @ R_true.T) + T_true
        p_photo[0] += np.array([0.03, 0.04, 0.0], dtype=np.float64)  # 0.05m offset

        res = STEAlignmentService.solve(p_lidar, p_photo, adjust_scale=True)
        self.assertTrue(res.success)

        p_pred = res.apply(p_lidar)
        diff = p_photo - p_pred
        indep_residuals = np.linalg.norm(diff, axis=1)
        indep_rms = float(np.sqrt(np.mean(indep_residuals ** 2)))

        self.assertAlmostEqual(res.rms_error, indep_rms, places=5)
        np.testing.assert_allclose(res.residuals, indep_residuals, atol=1e-5)

    def test_f_three_non_collinear_points_accepted(self):
        """Test F: Exactly three non-collinear points must pass."""
        p_lidar = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0]
        ], dtype=np.float64)
        p_photo = p_lidar * 2.0 + np.array([1.0, 2.0, 3.0])

        res = STEAlignmentService.solve(p_lidar, p_photo, adjust_scale=True)
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.scale, 2.0, places=5)

    def test_g_collinear_points_rejected(self):
        """Test G: Collinear points along a 1D line must be rejected gracefully."""
        p_collinear = np.array([
            [1.0, 2.0, 0.0],
            [2.0, 4.0, 0.0],
            [3.0, 6.0, 0.0]
        ], dtype=np.float64)
        p_target = p_collinear * 3.0

        res = STEAlignmentService.solve(p_collinear, p_target, adjust_scale=True)
        self.assertFalse(res.success)
        self.assertIn("collinear", res.status_message.lower())

    def test_h_coincident_points_rejected(self):
        """Test H: Coincident points must be rejected gracefully."""
        p_coincident = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=np.float64)
        p_target = p_coincident * 2.0

        res = STEAlignmentService.solve(p_coincident, p_target, adjust_scale=True)
        self.assertFalse(res.success)
        self.assertIn("coincident", res.status_message.lower())

    def test_i_incomplete_control_points_ignored(self):
        """Test I: Manager ignores incomplete control points and passes only complete pairs."""
        mgr = STEControlPointManager()
        mgr.set_photo_marker("CP1", np.array([1.0, 0.0, 0.0]))
        mgr.set_lidar_marker("CP1", np.array([2.0, 0.0, 0.0]))
        mgr.set_photo_marker("CP2", np.array([0.0, 1.0, 0.0]))
        mgr.set_photo_marker("CP3", np.array([0.0, 1.0, 0.0]))
        mgr.set_lidar_marker("CP3", np.array([0.0, 2.0, 0.0]))
        mgr.set_lidar_marker("CP4", np.array([0.0, 0.0, 2.0]))
        mgr.set_photo_marker("CP5", np.array([0.0, 0.0, 1.0]))
        mgr.set_lidar_marker("CP5", np.array([0.0, 0.0, 2.0]))

        self.assertEqual(mgr.count, 5)
        self.assertEqual(mgr.complete_count, 3)

        lidar_pts, photo_pts, ids = mgr.get_complete_pairs()
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids, ["CP1", "CP3", "CP5"])

        res = STEAlignmentService.solve_from_manager(mgr, adjust_scale=True)
        self.assertTrue(res.success)
        self.assertEqual(res.control_point_count, 3)
        self.assertEqual(res.control_point_ids, ["CP1", "CP3", "CP5"])

    def test_j_non_destructive_preview(self):
        """Test J: Preview alignment does NOT mutate original LiDAR coordinates."""
        p_lidar_orig = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 1.0, 2.0]
        ], dtype=np.float64)
        p_lidar_copy = p_lidar_orig.copy()

        p_photo = 5.0 * p_lidar_orig + np.array([10.0, -10.0, 5.0])

        state = STEAlignmentState()
        res = STEAlignmentService.solve(p_lidar_orig, p_photo, adjust_scale=True)
        state.set_initial_result(res)

        preview_mat = state.get_preview_transform()
        self.assertFalse(np.array_equal(preview_mat, np.eye(4)))

        p_visual = res.apply(p_lidar_orig)

        np.testing.assert_array_equal(p_lidar_orig, p_lidar_copy)
        self.assertFalse(np.array_equal(p_visual, p_lidar_orig))

    def test_k_reset_alignment(self):
        """Test K: Reset alignment reverts transform to Identity and clears preview."""
        p_lidar = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        p_photo = p_lidar * 2.0

        state = STEAlignmentState()
        res = STEAlignmentService.solve(p_lidar, p_photo, adjust_scale=True)
        state.set_initial_result(res)
        state.accept()

        self.assertTrue(state.is_accepted)
        self.assertTrue(state.preview_active)

        state.reset()
        self.assertFalse(state.is_accepted)
        self.assertFalse(state.preview_active)
        self.assertIsNone(state.result)
        np.testing.assert_array_equal(state.get_preview_transform(), np.eye(4))
        np.testing.assert_array_equal(state.get_committed_transform(), np.eye(4))

    def test_l_invalid_non_finite_data(self):
        """Test L: Graceful failure on NaN/Inf, mismatched dimensions, fewer than 3 points."""
        p_nan = np.array([[np.nan, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        p_valid = np.random.randn(3, 3)
        res = STEAlignmentService.solve(p_nan, p_valid)
        self.assertFalse(res.success)

        res2 = STEAlignmentService.solve(p_valid[:2], p_valid[:2])
        self.assertFalse(res2.success)
        self.assertEqual(res2.status, "insufficient_points")

        res3 = STEAlignmentService.solve(p_valid, p_valid[:2])
        self.assertFalse(res3.success)
        self.assertEqual(res3.status, "count_mismatch")

    # -------------------------------------------------------------------------
    # ICP REFINEMENT TESTS (Prompt 3)
    # -------------------------------------------------------------------------

    def test_icp_a_unavailable_or_no_initial_alignment(self):
        """Test ICP A: Verify graceful failure when initial alignment is missing or invalid."""
        pts_lidar = generate_synthetic_surface(100)
        pts_photo = generate_synthetic_surface(100)

        # 1. No initial alignment passed
        res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            initial_alignment=None
        )
        self.assertFalse(res.success)
        self.assertEqual(res.status, "no_initial_alignment")
        self.assertIn("Perform control-point alignment", res.status_message)

        # 2. Failed initial alignment passed
        bad_init = STEAlignmentResult(
            success=False, status="failed", status_message="error",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=-1.0, residuals=[]
        )
        res2 = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            initial_alignment=bad_init
        )
        self.assertFalse(res2.success)
        self.assertEqual(res2.status, "no_initial_alignment")

    def test_icp_b_improves_synthetic_alignment(self):
        """Test ICP B: Verify ICP improves an intentionally imperfect initial alignment."""
        pts_lidar = generate_synthetic_surface(400)
        
        # True transformation: s=3.0, rotation, translation
        R_true = np.array([
            [0.9998, -0.0175,  0.0099],
            [0.0176,  0.9998, -0.0051],
            [-0.0098,  0.0053,  0.9999]
        ], dtype=np.float64)
        t_true = np.array([2.5, -1.8, 4.2], dtype=np.float64)
        s_true = 3.0

        pts_photo = s_true * (pts_lidar @ R_true.T) + t_true

        # Construct an initial control-point alignment with slight residual error (e.g. 5cm translation error)
        t_init = t_true + np.array([0.05, -0.04, 0.03], dtype=np.float64)
        init_transform = np.eye(4, dtype=np.float64)
        init_transform[:3, :3] = s_true * R_true
        init_transform[:3, 3] = t_init

        # Pre-alignment RMS
        init_pred = (pts_lidar @ (s_true * R_true).T) + t_init
        initial_rms = float(np.sqrt(np.mean(np.sum((pts_photo - init_pred)**2, axis=1))))

        init_res = STEAlignmentResult(
            success=True,
            status="complete",
            status_message="Initial CP solve",
            rotation=R_true,
            translation=t_init,
            scale=s_true,
            transformation_matrix=init_transform,
            rms_error=initial_rms,
            residuals=[initial_rms] * 4,
            control_point_count=4,
            control_point_ids=["CP1", "CP2", "CP3", "CP4"]
        )

        # Run ICP Refinement
        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            initial_alignment=init_res,
            settings=STEICPRefinementSettings(adjust_scale=False, max_iterations=50)
        )

        self.assertTrue(icp_res.success, f"ICP failed: {icp_res.status_message}")
        self.assertLess(icp_res.final_rms, initial_rms, "ICP should improve alignment RMS error.")
        self.assertLess(icp_res.final_rms, 0.01, f"Final RMS should be small (<1cm), got {icp_res.final_rms:.4f}")

    def test_icp_c_initial_transform_is_respected(self):
        """Test ICP C: Verify ICP starts from the initial control-point alignment rather than raw coordinates."""
        pts_lidar = generate_synthetic_surface(200)
        
        # Raw coordinates are separated by large scale (7.0x) and large offset (50m)
        s_true = 7.0
        R_true = np.eye(3, dtype=np.float64)
        t_true = np.array([50.0, 50.0, 50.0], dtype=np.float64)

        pts_photo = s_true * (pts_lidar @ R_true.T) + t_true

        # Initial alignment correctly matches this large offset
        init_transform = np.eye(4, dtype=np.float64)
        init_transform[:3, :3] = s_true * R_true
        init_transform[:3, 3] = t_true

        init_res = STEAlignmentResult(
            success=True, status="complete", status_message="Initial OK",
            rotation=R_true, translation=t_true, scale=s_true,
            transformation_matrix=init_transform, rms_error=0.0,
            residuals=[0.0] * 3, control_point_count=3, control_point_ids=["CP1", "CP2", "CP3"]
        )

        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            initial_alignment=init_res
        )

        self.assertTrue(icp_res.success)
        # Verify final transform maintains the large translation and scale
        np.testing.assert_allclose(icp_res.translation, t_true, atol=0.05)
        self.assertAlmostEqual(icp_res.scale, s_true, places=3)

    def test_icp_d_transformation_composition(self):
        """Test ICP D: Verify T_final = T_icp @ T_init correctly maps points."""
        pts_lidar = generate_synthetic_surface(150)
        
        s_cp = 4.0
        R_cp = make_random_rotation()
        t_cp = np.array([10.0, -5.0, 8.0], dtype=np.float64)

        # Initial transform
        T_init = np.eye(4, dtype=np.float64)
        T_init[:3, :3] = s_cp * R_cp
        T_init[:3, 3] = t_cp

        init_res = STEAlignmentResult(
            success=True, status="complete", status_message="Init",
            rotation=R_cp, translation=t_cp, scale=s_cp,
            transformation_matrix=T_init, rms_error=0.02,
            residuals=[0.02] * 3, control_point_count=3
        )

        # Target photo is slightly rotated/translated relative to initial LiDAR
        R_small = np.array([
            [ 0.9998, -0.0150,  0.0100],
            [ 0.0151,  0.9998, -0.0050],
            [-0.0099,  0.0052,  0.9999]
        ])
        t_small = np.array([0.02, -0.03, 0.01])
        pts_init = (pts_lidar @ (s_cp * R_cp).T) + t_cp
        pts_photo = (pts_init @ R_small.T) + t_small

        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            initial_alignment=init_res,
            settings=STEICPRefinementSettings(adjust_scale=False, max_iterations=30)
        )

        self.assertTrue(icp_res.success)
        # Verify matrix equality: T_final == T_delta @ T_init
        expected_T_final = icp_res.icp_delta_transform @ T_init
        np.testing.assert_allclose(icp_res.transformation_matrix, expected_T_final, atol=1e-6)

        # Verify point mapping accuracy
        pts_final = (pts_lidar @ (icp_res.scale * icp_res.rotation).T) + icp_res.translation
        diff = np.linalg.norm(pts_photo - pts_final, axis=1)
        self.assertLess(np.mean(diff), 0.02)

    def test_icp_e_scale_preservation(self):
        """Test ICP E: Verify scale s = 7.0 established by control points is locked when adjust_scale=False."""
        pts_lidar = generate_synthetic_surface(200)
        s_established = 7.0
        R_established = make_random_rotation()
        t_established = np.array([1.0, 2.0, 3.0])

        T_init = np.eye(4)
        T_init[:3, :3] = s_established * R_established
        T_init[:3, 3] = t_established

        pts_photo = s_established * (pts_lidar @ R_established.T) + t_established

        init_res = STEAlignmentResult(
            success=True, status="complete", status_message="Init",
            rotation=R_established, translation=t_established, scale=s_established,
            transformation_matrix=T_init, rms_error=0.0, residuals=[]
        )

        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            initial_alignment=init_res,
            settings=STEICPRefinementSettings(adjust_scale=False)
        )

        self.assertTrue(icp_res.success)
        self.assertEqual(icp_res.scale, s_established)
        self.assertEqual(icp_res.icp_scale_delta, 1.0)

    def test_icp_f_non_destructive_behavior(self):
        """Test ICP F: Verify source LiDAR coordinates are byte-for-byte unmodified by ICP."""
        pts_lidar_orig = generate_synthetic_surface(100)
        pts_lidar_copy = pts_lidar_orig.copy()

        pts_photo = pts_lidar_orig * 2.0 + np.array([5.0, 5.0, 5.0])

        init_res = STEAlignmentService.solve(pts_lidar_orig[:4], pts_photo[:4], adjust_scale=True)
        
        state = STEAlignmentState()
        state.set_initial_result(init_res)

        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar_orig,
            target_photogrammetry_points=pts_photo,
            initial_alignment=init_res
        )
        state.set_icp_result(icp_res)

        # Verify underlying source array was never mutated
        np.testing.assert_array_equal(pts_lidar_orig, pts_lidar_copy)

    def test_icp_g_bad_result_rejection(self):
        """Test ICP G: When ICP produces runaway drift or invalid transform, initial alignment is preserved."""
        pts_lidar = generate_synthetic_surface(100)
        # Target points are located 100 meters away (unrelated geometry)
        pts_photo_far = generate_synthetic_surface(100) + 100.0

        init_res = STEAlignmentResult(
            success=True, status="complete", status_message="Init",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.01, residuals=[0.01] * 3
        )

        # Enforce max allowed drift = 5.0m
        settings = STEICPRefinementSettings(max_allowed_translation_drift=5.0)
        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo_far,
            initial_alignment=init_res,
            settings=settings
        )

        # Should reject runaway ICP and preserve initial alignment values
        self.assertFalse(icp_res.success)
        self.assertEqual(icp_res.status, "excessive_drift")
        self.assertIn("Preserving initial", icp_res.status_message)
        np.testing.assert_array_equal(icp_res.transformation_matrix, init_res.transformation_matrix)

    def test_icp_h_complete_reset_clears_icp(self):
        """Test ICP H: Reset clears all ICP and initial preview transforms."""
        pts_lidar = generate_synthetic_surface(100)
        pts_photo = pts_lidar * 2.0

        state = STEAlignmentState()
        init_res = STEAlignmentService.solve(pts_lidar[:4], pts_photo[:4], adjust_scale=True)
        state.set_initial_result(init_res)

        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            initial_alignment=init_res
        )
        state.set_icp_result(icp_res)
        state.accept()

        self.assertTrue(state.is_accepted)
        self.assertFalse(np.array_equal(state.get_preview_transform(), np.eye(4)))

        # Reset
        state.reset()
        self.assertFalse(state.is_accepted)
        self.assertFalse(state.preview_active)
        self.assertIsNone(state.result)
        self.assertIsNone(state.initial_result)
        self.assertIsNone(state.icp_result)
        np.testing.assert_array_equal(state.get_preview_transform(), np.eye(4))

    def test_icp_i_realistic_multipoint_pipeline(self):
        """Test ICP I: End-to-end multi-point control point solve + ICP refinement."""
        mgr = STEControlPointManager()

        # True world mapping
        R_true = make_random_rotation()
        t_true = np.array([4.5, -6.8, 12.3], dtype=np.float64)
        s_true = 7.17

        # 5 Control points in LiDAR space
        cp_lidar = [
            np.array([ 0.5,  1.2, -0.4]),
            np.array([-1.2,  0.8,  0.6]),
            np.array([ 0.9, -0.7,  1.1]),
            np.array([-0.4, -1.5, -0.8]),
            np.array([ 1.4,  0.3,  0.2]),
        ]

        for i, pt in enumerate(cp_lidar):
            cp_id = f"CP{i+1}"
            photo_pt = s_true * (pt @ R_true.T) + t_true
            mgr.set_lidar_marker(cp_id, pt)
            mgr.set_photo_marker(cp_id, photo_pt)

        # 1. Solve Control Points
        init_res = STEAlignmentService.solve_from_manager(mgr, adjust_scale=True)
        self.assertTrue(init_res.success)
        self.assertAlmostEqual(init_res.scale, s_true, places=3)
        self.assertLess(init_res.rms_error, 1e-4)

        # 2. Dense surface registration
        surf_lidar = generate_synthetic_surface(300)
        surf_photo = s_true * (surf_lidar @ R_true.T) + t_true

        # 3. Refine with ICP
        icp_res = STEICPRefinementService.refine(
            source_lidar_points=surf_lidar,
            target_photogrammetry_points=surf_photo,
            initial_alignment=init_res,
            settings=STEICPRefinementSettings(adjust_scale=False, max_iterations=30),
            cp_manager=mgr
        )

        self.assertTrue(icp_res.success)
        self.assertAlmostEqual(icp_res.scale, s_true, places=3)
        self.assertLess(icp_res.final_rms, 0.01)
        self.assertEqual(len(icp_res.final_cp_residuals), 5)

    def test_icp_roi_cropping_and_cp_degradation_protection(self):
        """Test: ICP automatically crops reference Photogrammetry points to LiDAR ROI and protects against semantic CP drift."""
        # 1. Target object at origin (box: -1 to +1)
        pts_obj = np.random.uniform(-1.0, 1.0, size=(200, 3))
        # Distant background clutter (e.g. at 50 to 100 meters)
        pts_clutter = np.random.uniform(50.0, 100.0, size=(500, 3))
        pts_photo_all = np.vstack([pts_obj, pts_clutter])

        # LiDAR object
        pts_lidar = pts_obj.copy()

        mgr = STEControlPointManager()
        for i in range(4):
            cp_id = f"CP{i+1}"
            pt = pts_obj[i * 10]
            mgr.set_lidar_marker(cp_id, pt)
            mgr.set_photo_marker(cp_id, pt)

        init_res = STEAlignmentService.solve_from_manager(mgr, adjust_scale=False)
        self.assertTrue(init_res.success)

        # ICP should filter out the 500 clutter points and align cleanly to the object
        settings = STEICPRefinementSettings(overlap_ratio=0.80, roi_margin_ratio=0.5)
        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo_all,
            initial_alignment=init_res,
            settings=settings,
            cp_manager=mgr
        )
        self.assertTrue(icp_res.success)
        self.assertLess(icp_res.final_rms, 0.05)


if __name__ == "__main__":
    unittest.main()
