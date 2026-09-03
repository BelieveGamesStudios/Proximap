import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from deep_mesh_fusion import (
    DeepMeshFusionConfig,
    DeepMeshFusionService,
    DeepMeshFusionWorkspace,
    RegistrationMetrics,
    ScanPass,
)


def noisy_plane(seed, *, transient=False, floater=False):
    rng = np.random.default_rng(seed)
    values = np.arange(0.01, 1.0, 0.025)
    points = np.asarray([[x, y, 0.01 + rng.normal(0.0, 0.004)] for x in values for y in values])
    if transient:
        patch = (points[:, 0] >= 0.40) & (points[:, 0] < 0.60) & (points[:, 1] >= 0.40) & (points[:, 1] < 0.60)
        points[patch, 2] = 0.09 + rng.normal(0.0, 0.001, int(np.sum(patch)))
    if floater:
        points = np.vstack((points, [[2.0, 2.0, 2.0]]))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(np.tile([[0.4, 0.6, 0.8]], (len(points), 1)))
    return cloud


class ConsensusFusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _workspace(self):
        config = DeepMeshFusionConfig(
            voxel_size=0.10,
            fusion_cell_size=0.10,
            analysis_cell_size=0.20,
            correspondence_distance_multiplier=0.50,
            single_pass_min_neighbors=2,
        )
        workspace = DeepMeshFusionWorkspace(str(self.root / "fusion"), config)
        sources = []
        for index, cloud in enumerate((noisy_plane(1), noisy_plane(2), noisy_plane(3, transient=True, floater=True)), 1):
            path = self.root / f"Pass_{index:02d}.ply"
            self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))
            sources.append(path)
            scan_pass = workspace.add_pass(str(path))
            scan_pass.registration = RegistrationMetrics(
                reference_pass_id="pass-01",
                fitness=0.99,
                inlier_rmse=0.002,
                overlap_ratio=1.0,
                method="test-identity",
                accepted=True,
            )
        return workspace, sources

    def test_consensus_reduces_noise_and_suppresses_transient_geometry(self):
        workspace, sources = self._workspace()
        source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        result = workspace.fuse_registered()

        cloud = o3d.io.read_point_cloud(result.fused_cloud_path)
        points = np.asarray(cloud.points)
        self.assertTrue(cloud.has_normals())
        self.assertTrue(cloud.has_colors())
        patch = (points[:, 0] >= 0.40) & (points[:, 0] < 0.60) & (points[:, 1] >= 0.40) & (points[:, 1] < 0.60)
        self.assertTrue(np.any(patch))
        self.assertLess(float(np.max(points[patch, 2])), 0.04)
        self.assertFalse(np.any(np.linalg.norm(points - [2.0, 2.0, 2.0], axis=1) < 0.2))
        self.assertGreater(result.consensus_point_count, 0)
        self.assertGreater(result.artifact_suppressed_point_count, 0)
        self.assertGreater(result.mean_confidence, 0.70)
        self.assertLess(float(np.mean(np.abs(points[:, 2] - 0.01))), 0.0025)
        self.assertTrue(all(workspace.verify_sources_unchanged().values()))
        for path, digest in source_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_point_level_provenance_and_confidence_are_exported(self):
        workspace, _sources = self._workspace()
        result = workspace.fuse_registered("consensus.ply", "consensus.provenance.json")
        provenance = json.loads(Path(result.provenance_path).read_text(encoding="utf-8"))

        self.assertEqual(provenance["summary"]["output_point_count"], result.fused_point_count)
        self.assertEqual(len(provenance["points"]), result.fused_point_count)
        self.assertEqual(set(provenance["sources"]), {"pass-01", "pass-02", "pass-03"})
        consensus = next(point for point in provenance["points"] if point["fusion_method"] == "consensus")
        self.assertGreaterEqual(len(consensus["contributions"]), 2)
        self.assertAlmostEqual(sum(item["weight"] for item in consensus["contributions"]), 1.0)
        self.assertTrue(all(len(source["source_sha256"]) == 64 for source in provenance["sources"].values()))
        header = Path(result.fused_cloud_path).read_text(encoding="ascii").split("end_header", 1)[0]
        self.assertIn("property float confidence", header)
        self.assertIn("property ushort observation_count", header)
        self.assertIn("property uchar fusion_method", header)
        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(manifest["fused_artifact"]["provenance_path"], result.provenance_path)
        self.assertTrue(manifest["artifact_suppression"]["derived"])
        self.assertTrue(Path(result.artifact_report_path).is_file())
        self.assertTrue(Path(result.rejected_geometry_path).is_file())
        self.assertTrue(Path(result.reconstructed_mesh_path).is_file())
        self.assertTrue(Path(result.reconstruction_report_path).is_file())
        self.assertGreater(result.reconstructed_vertex_count, 0)
        self.assertGreater(result.reconstructed_face_count, 0)
        self.assertTrue(manifest["architecture_reconstruction"]["derived"])
        self.assertTrue(Path(result.repaired_mesh_path).is_file())
        self.assertTrue(Path(result.gap_report_path).is_file())
        self.assertTrue(Path(result.gap_review_path).is_file())
        self.assertTrue(manifest["gap_recovery"]["derived"])

    def test_normal_disagreement_selects_best_observation(self):
        config = DeepMeshFusionConfig(voxel_size=0.20, fusion_cell_size=0.20)
        service = DeepMeshFusionService(config)
        rng = np.random.default_rng(9)
        horizontal = np.column_stack((rng.uniform(0.02, 0.08, 80), rng.uniform(0.02, 0.08, 80), np.full(80, 0.05)))
        vertical = np.column_stack((np.full(80, 0.05), rng.uniform(0.02, 0.08, 80), rng.uniform(0.02, 0.08, 80)))
        inputs = []
        for index, points in enumerate((horizontal, horizontal.copy(), vertical), 1):
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            scan_pass = ScanPass(
                pass_id=f"pass-{index:02d}",
                name=f"Pass {index}",
                source_path="synthetic",
                source_sha256=str(index) * 64,
                source_size=0,
                registration=RegistrationMetrics("pass-01", fitness=1.0, inlier_rmse=0.0, accepted=True),
            )
            inputs.append((scan_pass, cloud))
        output = service.fuse(inputs)
        self.assertEqual(len(output.points), 1)
        self.assertEqual(int(output.method_codes[0]), service.METHOD_CODES["best-observation"])
        weights = [item.weight for item in output.provenance[0].contributions]
        self.assertEqual(sum(weight > 0 for weight in weights), 1)


if __name__ == "__main__":
    unittest.main()
