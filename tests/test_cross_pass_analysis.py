import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from deep_mesh_fusion import DeepMeshFusionConfig, DeepMeshFusionWorkspace, RegistrationMetrics


def make_surface(*, hole=None, raised=None):
    values = np.arange(0.01, 1.0, 0.025)
    points = np.asarray([[x, y, 0.0] for x in values for y in values], dtype=float)
    if hole is not None:
        x0, x1, y0, y1 = hole
        inside = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
        points = points[~inside]
    if raised is not None:
        x0, x1, y0, y1, height = raised
        inside = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
        points[inside, 2] = height
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(np.tile([[0.6, 0.6, 0.6]], (len(points), 1)))
    return cloud


class CrossPassAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _workspace(self):
        config = DeepMeshFusionConfig(
            voxel_size=0.05,
            analysis_cell_size=0.20,
            missing_neighbor_count=2,
            conflict_distance_multiplier=0.50,
        )
        workspace = DeepMeshFusionWorkspace(str(self.root / "fusion"), config)
        clouds = [
            make_surface(),
            make_surface(hole=(0.20, 0.40, 0.20, 0.40)),
            make_surface(raised=(0.60, 0.80, 0.60, 0.80, 0.16)),
        ]
        for index, cloud in enumerate(clouds, 1):
            path = self.root / f"Pass_{index:02d}.ply"
            self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))
            scan_pass = workspace.add_pass(str(path))
            scan_pass.registration = RegistrationMetrics(
                reference_pass_id="pass-01",
                fitness=1.0,
                inlier_rmse=0.0,
                overlap_ratio=1.0,
                method="test-identity",
                accepted=True,
                message="Synthetic common-space pass",
            )
        return workspace

    def test_evidence_map_detects_missing_and_conflicting_observations(self):
        workspace = self._workspace()
        result = workspace.analyze_cross_passes()

        evidence_path = Path(result.evidence_map_path)
        visualization_path = Path(result.confidence_cloud_path)
        self.assertTrue(evidence_path.is_file())
        self.assertTrue(visualization_path.is_file())
        self.assertTrue(all(workspace.verify_sources_unchanged().values()))
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
        summary = data["summary"]
        self.assertGreater(summary["total_regions"], 20)
        self.assertGreater(summary["conflict_regions"], 0)
        self.assertGreater(summary["missing_observation_regions"], 0)
        self.assertGreater(summary["overlap_ratio"], 0.70)
        self.assertEqual(set(summary["per_pass_coverage"]), {"pass-01", "pass-02", "pass-03"})

        missing = [region for region in data["regions"] if "pass-02" in region["missing_pass_ids"]]
        conflicts = [region for region in data["regions"] if "cross-pass-distance" in region["conflict_reasons"]]
        shared = [region for region in data["regions"] if region["observation_count"] == 3 and not region["conflict"]]
        self.assertTrue(missing)
        self.assertTrue(conflicts)
        self.assertTrue(shared)
        self.assertGreater(np.mean([item["confidence"] for item in shared]), np.mean([item["confidence"] for item in missing]))
        self.assertTrue(all(0.0 <= item["confidence"] <= 1.0 for item in data["regions"]))

    def test_confidence_map_contains_explainable_scores_and_provenance(self):
        workspace = self._workspace()
        result = workspace.analyze_cross_passes("evidence.json", "confidence.ply")
        data = json.loads(Path(result.evidence_map_path).read_text(encoding="utf-8"))

        self.assertAlmostEqual(sum(data["scoring_weights"].values()), 1.0)
        self.assertEqual(set(data["source_provenance"]), {"pass-01", "pass-02", "pass-03"})
        for source in data["source_provenance"].values():
            self.assertEqual(len(source["source_sha256"]), 64)
            self.assertEqual(len(source["transform_sha256"]), 64)
        region = next(item for item in data["regions"] if item["observation_count"] == 3)
        self.assertEqual(sum(region["provenance"].values()), region["total_point_count"])
        self.assertEqual(
            set(region["pass_evidence"][0]["score_components"]),
            {"observation", "density", "distance", "normal", "surface", "registration"},
        )

        header = Path(result.confidence_cloud_path).read_text(encoding="ascii").split("end_header", 1)[0]
        self.assertIn("property float confidence", header)
        self.assertIn("property ushort observation_count", header)
        self.assertIn("property uchar conflict", header)
        visual_cloud = o3d.io.read_point_cloud(result.confidence_cloud_path)
        self.assertEqual(len(visual_cloud.points), data["summary"]["total_regions"])
        self.assertTrue(visual_cloud.has_colors())
        manifest = json.loads((workspace.root / "workspace.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["cross_pass_analysis"]["derived"])

    def test_analysis_requires_two_accepted_passes(self):
        workspace = self._workspace()
        workspace.passes[1].registration.accepted = False
        workspace.passes[2].registration.accepted = False
        with self.assertRaisesRegex(ValueError, "at least two accepted"):
            workspace.analyze_cross_passes()


if __name__ == "__main__":
    unittest.main()
