import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from deep_mesh_fusion import DeepMeshFusionConfig, DeepMeshFusionWorkspace
from deep_mesh_fusion.registration import axis_aligned_overlap


def make_room_cloud(offset=(0.0, 0.0, 0.0)):
    """Three asymmetric perpendicular patches with enough 3-D feature structure."""
    rng = np.random.default_rng(11)
    points = []
    colors = []
    for _ in range(750):
        x, y = rng.uniform([0.0, 0.0], [3.0, 2.1])
        points.append([x, y, rng.normal(0.0, 0.002)])
        colors.append([0.8, 0.2, 0.2])
    for _ in range(600):
        y, z = rng.uniform([0.0, 0.0], [2.1, 1.6])
        points.append([rng.normal(0.0, 0.002), y, z])
        colors.append([0.2, 0.8, 0.2])
    for _ in range(450):
        x, z = rng.uniform([0.7, 0.0], [3.0, 1.6])
        points.append([x, rng.normal(2.1, 0.002), z])
        colors.append([0.2, 0.2, 0.8])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points) + np.asarray(offset)))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return cloud


class DeepMeshFusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_cloud(self, name, cloud):
        path = self.root / name
        self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))
        return path

    def test_ingest_analyze_register_fuse_preserves_sources(self):
        source_a = self.write_cloud("Pass_01.ply", make_room_cloud())
        source_b = self.write_cloud("Pass_02.ply", make_room_cloud(offset=(0.008, -0.004, 0.003)))
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source_a, source_b)}
        workspace = DeepMeshFusionWorkspace(
            str(self.root / "fusion"),
            DeepMeshFusionConfig(voxel_size=0.08, min_registration_fitness=0.5),
        )

        workspace.add_pass(str(source_a))
        workspace.add_pass(str(source_b))
        workspace.analyze_passes()
        workspace.register_passes()
        analysis = workspace.analyze_cross_passes()
        registered_lossless = workspace.export_registered_cloud(str(self.root / "registered_lossless.ply"))
        registered_voxel = workspace.export_registered_cloud(str(self.root / "registered_voxel.ply"), voxel_downsampled=True)
        point_result = workspace.fuse_points_registered()
        self.assertTrue(Path(point_result.fused_cloud_path).is_file())
        self.assertGreater(point_result.fused_point_count, 100)
        self.assertGreaterEqual(len(o3d.io.read_point_cloud(str(registered_lossless)).points), len(o3d.io.read_point_cloud(str(registered_voxel)).points))
        result = workspace.reconstruct_fused_surface()
        cleaned_validation = workspace.validate_cleaned_mesh(result.validated_mesh_path)

        self.assertEqual(result.source_pass_count, 2)
        self.assertEqual(result.registered_pass_count, 2)
        self.assertGreater(result.fused_point_count, 100)
        self.assertTrue(Path(result.fused_cloud_path).is_file())
        self.assertTrue(Path(result.validated_mesh_path).is_file())
        self.assertTrue(Path(result.validation_report_path).is_file())
        self.assertTrue(Path(result.quality_map_path).is_file())
        self.assertGreater(cleaned_validation.summary.face_count, 0)
        self.assertTrue(Path(analysis.evidence_map_path).is_file())
        self.assertGreater(analysis.region_count, 0)
        self.assertTrue(all(workspace.verify_sources_unchanged().values()))
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

        manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_policy"], "immutable-reference")
        self.assertTrue(manifest["fused_artifact"]["derived"])
        self.assertEqual(manifest["schema_version"], 11)
        self.assertTrue(manifest["geometry_validation"]["derived"])
        self.assertIn("ready_for_appearance_processing", manifest["geometry_validation"]["summary"])
        self.assertTrue(all(item["registration"]["accepted"] for item in manifest["passes"]))

    def test_duplicate_source_is_rejected(self):
        source = self.write_cloud("Pass.ply", make_room_cloud())
        workspace = DeepMeshFusionWorkspace(str(self.root / "fusion"), DeepMeshFusionConfig(voxel_size=0.08))
        workspace.add_pass(str(source))
        with self.assertRaisesRegex(ValueError, "already in the workspace"):
            workspace.add_pass(str(source))

    def test_overlap_is_symmetric_and_bounded(self):
        a = ([0, 0, 0], [2, 2, 2])
        b = ([1, 1, 1], [3, 3, 3])
        self.assertAlmostEqual(axis_aligned_overlap(a, b), 1 / 8)
        self.assertAlmostEqual(axis_aligned_overlap(a, b), axis_aligned_overlap(b, a))
        self.assertEqual(axis_aligned_overlap(a, ([5, 5, 5], [6, 6, 6])), 0.0)

    def test_manual_transform_is_quality_checked(self):
        source_a = self.write_cloud("Reference.ply", make_room_cloud())
        source_b = self.write_cloud("Moved.ply", make_room_cloud(offset=(0.4, -0.2, 0.1)))
        workspace = DeepMeshFusionWorkspace(
            str(self.root / "fusion"),
            DeepMeshFusionConfig(voxel_size=0.1, min_registration_fitness=0.5),
        )
        reference = workspace.add_pass(str(source_a))
        moved = workspace.add_pass(str(source_b))
        transform = np.eye(4)
        transform[:3, 3] = [-0.4, 0.2, -0.1]
        metrics = workspace.set_manual_transform(moved.pass_id, transform, reference.pass_id)
        self.assertTrue(metrics.accepted)
        self.assertEqual(metrics.method, "manual")
        self.assertGreater(metrics.fitness, 0.99)

    def test_coarse_registration_recovers_displaced_pass(self):
        source_a = self.write_cloud("Reference.ply", make_room_cloud())
        source_b = self.write_cloud("Displaced.ply", make_room_cloud(offset=(0.45, -0.2, 0.12)))
        workspace = DeepMeshFusionWorkspace(
            str(self.root / "fusion"),
            DeepMeshFusionConfig(voxel_size=0.1, min_registration_fitness=0.5),
        )
        reference = workspace.add_pass(str(source_a))
        displaced = workspace.add_pass(str(source_b))
        workspace.register_passes(reference.pass_id)
        metrics = displaced.registration
        self.assertTrue(metrics.accepted)
        self.assertEqual(metrics.method, "fpfh-ransac+point-to-plane-icp")
        self.assertGreater(metrics.fitness, 0.95)
        self.assertTrue(np.allclose(np.asarray(metrics.transform)[:3, 3], [-0.45, 0.2, -0.12], atol=0.03))


if __name__ == "__main__":
    unittest.main()
