import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import open3d as o3d

from deep_mesh_fusion import DeepMeshFusionConfig, DeepMeshFusionReconstructionService, EvidenceBasedGapRepairService
from deep_mesh_fusion.fusion import ConsensusFusionOutput
from deep_mesh_fusion.reconstruction import ArchitectureMeshOutput
from tests.test_architecture_reconstruction import synthetic_apartment


def planar_fused():
    values = np.arange(0.05, 1.0, 0.10)
    points = np.asarray([[x, y, 0.0] for x in values for y in values])
    count = len(points)
    return ConsensusFusionOutput(
        points=points,
        normals=np.tile([[0.0, 0.0, 1.0]], (count, 1)),
        colors=np.tile([[0.7, 0.7, 0.7]], (count, 1)),
        confidence=np.full(count, 0.96),
        observation_counts=np.full(count, 3, dtype=np.uint16),
        method_codes=np.zeros(count, dtype=np.uint8),
        provenance=[],
        suppressed_observation_count=0,
    )


def subset_fused(fused, keep):
    return ConsensusFusionOutput(
        points=fused.points[keep], normals=fused.normals[keep], colors=fused.colors[keep],
        confidence=fused.confidence[keep], observation_counts=fused.observation_counts[keep],
        method_codes=fused.method_codes[keep], provenance=[], suppressed_observation_count=0,
    )


class GapRecoveryTests(unittest.TestCase):
    def _config(self, **overrides):
        values = dict(
            voxel_size=0.10,
            fusion_cell_size=0.10,
            architecture_up_axis="z",
            architecture_plane_distance=0.02,
            architecture_plane_min_points=20,
            architecture_grid_size=0.10,
            architecture_grid_closing_iterations=0,
            opening_min_width=0.30,
            opening_min_height=0.30,
            opening_min_area=0.10,
            gap_max_planar_area=0.50,
        )
        values.update(overrides)
        return DeepMeshFusionConfig(**values)

    @staticmethod
    def _remove_faces(architecture, predicate):
        centers = architecture.vertices[architecture.faces].mean(axis=1)
        keep = ~predicate(centers)
        summary = replace(architecture.summary, face_count=int(np.sum(keep)))
        return ArchitectureMeshOutput(
            vertices=architecture.vertices,
            faces=architecture.faces[keep],
            normals=architecture.normals,
            colors=architecture.colors,
            confidence=architecture.confidence,
            class_codes=architecture.class_codes,
            surface_ids=architecture.surface_ids,
            planes=architecture.planes,
            edges=architecture.edges,
            corners=architecture.corners,
            summary=summary,
        )

    def test_uses_observed_fused_geometry_before_inference(self):
        config = self._config()
        fused = planar_fused()
        architecture = DeepMeshFusionReconstructionService(config).reconstruct(fused)
        damaged = self._remove_faces(
            architecture,
            lambda centers: (centers[:, 0] >= 0.40) & (centers[:, 0] < 0.60) & (centers[:, 1] >= 0.40) & (centers[:, 1] < 0.60),
        )
        output = EvidenceBasedGapRepairService(config).recover(damaged, fused)
        observed = [gap for gap in output.gaps if gap.classification == "observed-geometry"]
        self.assertTrue(observed)
        self.assertTrue(all(gap.decision == "repair" for gap in observed))
        self.assertGreater(output.summary.observed_geometry_repair_count, 0)
        self.assertGreater(output.summary.final_face_count, len(damaged.faces))

    def test_confident_planar_continuation_repairs_unobserved_small_gap(self):
        config = self._config()
        fused = planar_fused()
        architecture = DeepMeshFusionReconstructionService(config).reconstruct(fused)
        in_gap = (
            (fused.points[:, 0] >= 0.40) & (fused.points[:, 0] < 0.60)
            & (fused.points[:, 1] >= 0.40) & (fused.points[:, 1] < 0.60)
        )
        damaged = self._remove_faces(
            architecture,
            lambda centers: (centers[:, 0] >= 0.40) & (centers[:, 0] < 0.60) & (centers[:, 1] >= 0.40) & (centers[:, 1] < 0.60),
        )
        output = EvidenceBasedGapRepairService(config).recover(damaged, subset_fused(fused, ~in_gap))
        inferred = [gap for gap in output.gaps if gap.classification == "planar-continuation"]
        self.assertTrue(inferred)
        self.assertGreater(output.summary.planar_continuation_count, 0)
        self.assertTrue(all(gap.repair_confidence >= 0.78 for gap in inferred))

    def test_large_unsupported_gap_remains_open_for_manual_review(self):
        config = self._config(gap_max_planar_area=0.02)
        fused = planar_fused()
        architecture = DeepMeshFusionReconstructionService(config).reconstruct(fused)
        in_gap = (
            (fused.points[:, 0] >= 0.30) & (fused.points[:, 0] < 0.70)
            & (fused.points[:, 1] >= 0.30) & (fused.points[:, 1] < 0.70)
        )
        damaged = self._remove_faces(
            architecture,
            lambda centers: (centers[:, 0] >= 0.30) & (centers[:, 0] < 0.70) & (centers[:, 1] >= 0.30) & (centers[:, 1] < 0.70),
        )
        output = EvidenceBasedGapRepairService(config).recover(damaged, subset_fused(fused, ~in_gap))
        unresolved = [gap for gap in output.gaps if gap.review_required]
        self.assertTrue(unresolved)
        self.assertEqual(output.summary.unresolved_gap_count, len(unresolved))
        self.assertEqual(output.summary.final_face_count, len(damaged.faces))

    def test_intentional_openings_are_preserved_and_reports_export(self):
        config = self._config(
            architecture_plane_min_points=80,
            opening_min_area=0.10,
            doorway_floor_tolerance=0.20,
            complex_poisson_depth=6,
        )
        fused = synthetic_apartment()
        reconstruction = DeepMeshFusionReconstructionService(config)
        architecture = reconstruction.reconstruct(fused)
        service = EvidenceBasedGapRepairService(config)
        output = service.recover(architecture, fused)
        self.assertGreaterEqual(output.summary.intentional_opening_count, 2)
        # A watertight Poisson residual needs no secondary complex-hole patch;
        # otherwise the conservative interpolation path remains available.
        self.assertTrue(
            output.summary.surface_interpolation_count >= 1
            or reconstruction.last_complex_report["watertight"]
        )
        intentional = [gap for gap in output.gaps if gap.classification == "intentional-opening"]
        self.assertTrue(all(gap.decision == "preserve" and gap.repaired_face_count == 0 for gap in intentional))

        with tempfile.TemporaryDirectory() as temporary:
            result = service.export(
                output,
                str(Path(temporary) / "repaired.ply"),
                str(Path(temporary) / "gap_report.json"),
                str(Path(temporary) / "gap_review.ply"),
            )
            report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
            self.assertEqual(report["hierarchy"], ["observed-geometry", "confident-inference", "manual-review"])
            mesh = o3d.io.read_triangle_mesh(result.repaired_mesh_path)
            review = o3d.io.read_point_cloud(result.review_path)
            self.assertGreater(len(mesh.triangles), 0)
            self.assertEqual(len(review.points), output.summary.detected_gap_count)


if __name__ == "__main__":
    unittest.main()
