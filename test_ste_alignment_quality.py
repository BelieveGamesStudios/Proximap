"""
Unit Test Suite for STE Alignment Quality Gate & Diagnostics
============================================================

Tests:
- Test A: Perfectly aligned surfaces (EXCELLENT rating, small distances, high overlap, READY)
- Test B: Known translation error (detected in surface distances & spatial bins)
- Test C: Known scale mismatch (detected by quality gate)
- Test D: Correct control points but poor surface alignment (not reporting READY just because CP RMS is low)
- Test E: High surface overlap computation
- Test F: Low surface overlap triggers NOT_READY
- Test G: ICP improves alignment (diagnostics confirm improvement)
- Test H: ICP makes alignment worse (diagnostics detect ICP_DEGRADATION)
- Test I: Non-destructive behavior (source geometry unmodified)
- Test J: Determinism (identical inputs produce identical diagnostics)
"""

import unittest
import numpy as np

from ste_alignment import (
    STEAlignmentResult,
    STEAlignmentService,
    STEICPRefinementService,
    STEICPRefinementSettings
)
from ste_alignment_quality import (
    STEAlignmentQualityGate,
    STEAlignmentQualityReport,
    AlignmentQualityRating,
    TextureTransferReadiness,
    QualityGateThresholds
)


def generate_curved_room_geometry(num_points: int = 1000) -> np.ndarray:
    """Generate a realistic curved room surface patch."""
    u = np.linspace(-3.0, 3.0, int(np.sqrt(num_points)))
    v = np.linspace(-3.0, 3.0, int(np.sqrt(num_points)))
    uu, vv = np.meshgrid(u, v)
    zz = 0.4 * np.sin(uu) + 0.2 * np.cos(vv)
    return np.stack([uu.ravel(), vv.ravel(), zz.ravel()], axis=1)


class TestSTEAlignmentQuality(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_a_perfectly_aligned_surfaces(self):
        """Test A: Perfectly aligned surfaces produce EXCELLENT rating and READY status."""
        pts = generate_curved_room_geometry(600)
        
        # Perfect Identity transformation
        result = STEAlignmentResult(
            success=True,
            status="complete",
            status_message="OK",
            rotation=np.eye(3),
            translation=np.zeros(3),
            scale=1.0,
            transformation_matrix=np.eye(4),
            rms_error=0.001,
            residuals=[0.001] * 4,
            control_point_count=4
        )

        report = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts,
            target_photogrammetry_points=pts,
            alignment_result=result
        )

        self.assertEqual(report.rating, AlignmentQualityRating.EXCELLENT)
        self.assertEqual(report.readiness, TextureTransferReadiness.READY_FOR_TEXTURE_TRANSFER)
        self.assertTrue(report.is_ready)
        self.assertLess(report.surface_median_dist, 0.01)
        self.assertGreater(report.overlap_metrics.overlap_ratio, 0.95)

    def test_b_known_translation_error(self):
        """Test B: Known translation offset is accurately detected in surface distances."""
        pts_lidar = generate_curved_room_geometry(500)
        # Shift target normal to surface by 0.25m (25cm in Z)
        shift = np.array([0.0, 0.0, 0.25])
        pts_photo = pts_lidar + shift

        # Alignment claiming no translation (erroneous alignment)
        result = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.25, residuals=[0.25] * 4
        )

        report = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            alignment_result=result
        )

        # Surface distance should clearly show the 25cm offset
        self.assertGreater(report.surface_median_dist, 0.20)
        self.assertGreater(report.surface_p95_dist, 0.20)
        self.assertEqual(report.readiness, TextureTransferReadiness.NOT_READY_FOR_TEXTURE_TRANSFER)

    def test_c_known_scale_mismatch(self):
        """Test C: Known scale mismatch triggers POSSIBLE_SCALE_MISMATCH or NOT_READY."""
        pts_lidar = generate_curved_room_geometry(500)
        pts_photo = pts_lidar * 1.5  # 50% larger

        # Alignment claiming scale = 1.0 (unscaled alignment on scaled target)
        result = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.5, residuals=[0.5] * 4
        )

        report = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            alignment_result=result
        )

        self.assertEqual(report.readiness, TextureTransferReadiness.NOT_READY_FOR_TEXTURE_TRANSFER)
        self.assertIn(report.rating, (AlignmentQualityRating.POOR, AlignmentQualityRating.VERY_POOR))

    def test_d_low_cp_rms_but_poor_surface_alignment(self):
        """Test D: Low CP RMS on 3 localized markers does NOT fool quality gate when surfaces mismatch."""
        pts_lidar = generate_curved_room_geometry(800)
        # Warp the photo surface globally while keeping 3 localized control points at the center identical
        pts_photo = pts_lidar.copy()
        radius = np.linalg.norm(pts_photo[:, :2], axis=1)
        pts_photo[:, 2] += 0.35 * (radius > 1.5)  # 35cm bulge on outer perimeter

        cp_rms = 0.005  # 5mm on control points

        result = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=cp_rms, residuals=[cp_rms] * 4
        )

        report = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            alignment_result=result
        )

        # Control point RMS is low, but surface P95 reflects the real outer distortion
        self.assertLess(report.control_point_rms, 0.01)
        self.assertGreater(report.surface_p95_dist, 0.20)
        self.assertIn("LOCAL_ALIGNMENT_FAILURE", report.failure_modes)
        self.assertEqual(report.readiness, TextureTransferReadiness.NOT_READY_FOR_TEXTURE_TRANSFER)

    def test_e_high_surface_overlap(self):
        """Test E: High surface overlap correctly computes overlap ratio."""
        pts = generate_curved_room_geometry(400)
        result = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.01, residuals=[0.01] * 4
        )

        report = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts,
            target_photogrammetry_points=pts,
            alignment_result=result
        )

        self.assertGreater(report.overlap_metrics.overlap_ratio, 0.90)

    def test_f_low_surface_overlap(self):
        """Test F: Low bounding box overlap triggers LOW_SURFACE_OVERLAP and NOT_READY."""
        pts_lidar = generate_curved_room_geometry(400)
        # Shift photogrammetry target 10m away
        pts_photo = pts_lidar + np.array([10.0, 10.0, 0.0])

        result = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.10, residuals=[0.10] * 4
        )

        report = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            alignment_result=result
        )

        self.assertIn("LOW_SURFACE_OVERLAP", report.failure_modes)
        self.assertEqual(report.readiness, TextureTransferReadiness.NOT_READY_FOR_TEXTURE_TRANSFER)

    def test_g_icp_improves_alignment(self):
        """Test G: Quality diagnostics confirm improvement when ICP refines an initial alignment."""
        pts_lidar = generate_curved_room_geometry(600)
        R_true = np.eye(3)
        t_true = np.array([1.0, 2.0, 3.0])
        s_true = 2.0
        pts_photo = s_true * (pts_lidar @ R_true.T) + t_true

        # Initial alignment with 8cm offset
        t_init = t_true + np.array([0.08, 0.0, 0.0])
        init_res = STEAlignmentResult(
            success=True, status="complete", status_message="Init",
            rotation=R_true, translation=t_init, scale=s_true,
            transformation_matrix=np.array([
                [2.0, 0, 0, t_init[0]],
                [0, 2.0, 0, t_init[1]],
                [0, 0, 2.0, t_init[2]],
                [0, 0, 0, 1.0]
            ]), rms_error=0.08, residuals=[0.08] * 4
        )

        # Refined alignment with ICP
        icp_res = STEICPRefinementService.refine(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            initial_alignment=init_res,
            settings=STEICPRefinementSettings(adjust_scale=False, max_iterations=30)
        )
        final_res = icp_res.to_alignment_result()

        # Evaluate quality comparing initial vs final
        report = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            alignment_result=final_res,
            initial_alignment_result=init_res
        )

        self.assertTrue(report.icp_improved_surface)
        self.assertLess(report.surface_dist_change_pct, 0.0)  # Distance decreased
        self.assertEqual(report.rating, AlignmentQualityRating.EXCELLENT)
        self.assertEqual(report.readiness, TextureTransferReadiness.READY_FOR_TEXTURE_TRANSFER)

    def test_h_icp_degradation_detection(self):
        """Test H: Diagnostics detect when a bad alignment has degraded surface distances."""
        pts_lidar = generate_curved_room_geometry(400)
        pts_photo = pts_lidar.copy()

        # Good initial alignment
        init_res = STEAlignmentResult(
            success=True, status="complete", status_message="Good Init",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.005, residuals=[0.005] * 4
        )

        # Degraded refined alignment (e.g. 15cm offset)
        degraded_transform = np.eye(4)
        degraded_transform[0, 3] = 0.15
        degraded_res = STEAlignmentResult(
            success=True, status="complete", status_message="Degraded",
            rotation=np.eye(3), translation=np.array([0.15, 0.0, 0.0]), scale=1.0,
            transformation_matrix=degraded_transform, rms_error=0.15, residuals=[0.15] * 4
        )

        report = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            alignment_result=degraded_res,
            initial_alignment_result=init_res
        )

        self.assertFalse(report.icp_improved_surface)
        self.assertIn("ICP_DEGRADATION", report.failure_modes)

    def test_i_non_destructive_behavior(self):
        """Test I: Quality evaluation does NOT modify source or target geometry arrays."""
        pts_lidar = generate_curved_room_geometry(300)
        pts_photo = generate_curved_room_geometry(300)
        
        lidar_copy = pts_lidar.copy()
        photo_copy = pts_photo.copy()

        res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.02, residuals=[0.02] * 4
        )

        _ = STEAlignmentQualityGate.evaluate(
            source_lidar_points=pts_lidar,
            target_photogrammetry_points=pts_photo,
            alignment_result=res
        )

        np.testing.assert_array_equal(pts_lidar, lidar_copy)
        np.testing.assert_array_equal(pts_photo, photo_copy)

    def test_j_determinism(self):
        """Test J: Identical inputs produce identical diagnostics."""
        pts_lidar = generate_curved_room_geometry(300)
        pts_photo = pts_lidar + np.array([0.02, -0.01, 0.03])

        res = STEAlignmentResult(
            success=True, status="complete", status_message="OK",
            rotation=np.eye(3), translation=np.zeros(3), scale=1.0,
            transformation_matrix=np.eye(4), rms_error=0.03, residuals=[0.03] * 4
        )

        rep1 = STEAlignmentQualityGate.evaluate(pts_lidar, pts_photo, res)
        rep2 = STEAlignmentQualityGate.evaluate(pts_lidar, pts_photo, res)

        self.assertEqual(rep1.rating, rep2.rating)
        self.assertEqual(rep1.readiness, rep2.readiness)
        self.assertAlmostEqual(rep1.surface_median_dist, rep2.surface_median_dist, places=5)
        self.assertAlmostEqual(rep1.overlap_metrics.overlap_ratio, rep2.overlap_metrics.overlap_ratio, places=5)


if __name__ == "__main__":
    unittest.main()
