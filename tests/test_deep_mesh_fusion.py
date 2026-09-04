import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from deep_mesh_fusion import DeepMeshFusionConfig, DeepMeshFusionWorkspace, RegistrationMetrics
from deep_mesh_fusion.registration import axis_aligned_overlap
from mesh_cleanup import _find_python310, _find_worker_script


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

    def test_manual_alignment_batch_requires_every_secondary_and_commits_together(self):
        paths = [self.write_cloud(f"Batch_{index}.ply", make_room_cloud(offset=(index * .01, 0, 0))) for index in range(3)]
        workspace = DeepMeshFusionWorkspace(str(self.root / "batch"), DeepMeshFusionConfig(voxel_size=.08))
        passes = [workspace.add_pass(str(path)) for path in paths]
        workspace.analyze_passes()
        accepted = lambda reference: RegistrationMetrics(
            reference_pass_id=reference, transform=np.eye(4).tolist(), fitness=1.0,
            inlier_rmse=0.0, overlap_ratio=1.0, accepted=True,
            method="manual+point-to-plane-icp", message="ICP accepted",
        )
        with self.assertRaisesRegex(ValueError, "every secondary"):
            workspace.commit_manual_alignment_batch(passes[0].pass_id, {passes[1].pass_id: accepted(passes[0].pass_id)})
        self.assertTrue(all(item.registration is None for item in passes))

        workspace.commit_manual_alignment_batch(
            passes[0].pass_id,
            {passes[1].pass_id: accepted(passes[0].pass_id), passes[2].pass_id: accepted(passes[0].pass_id)},
        )
        self.assertEqual(passes[0].registration.method, "reference")
        self.assertTrue(all(item.registration.accepted for item in passes))
        manifest = json.loads((workspace.root / "workspace.json").read_text())
        self.assertTrue(all(item["registration"]["accepted"] for item in manifest["passes"]))

    def test_point_removal_outputs_replace_aligned_inputs_without_touching_sources(self):
        paths = [self.write_cloud(f"Removal_{index}.ply", make_room_cloud(offset=(index * .01, 0, 0))) for index in range(2)]
        workspace = DeepMeshFusionWorkspace(str(self.root / "removal"), DeepMeshFusionConfig(voxel_size=.08))
        passes = [workspace.add_pass(str(path)) for path in paths]
        workspace.analyze_passes()
        accepted = RegistrationMetrics(
            reference_pass_id=passes[0].pass_id, transform=np.eye(4).tolist(), fitness=1.0,
            inlier_rmse=0.0, overlap_ratio=1.0, accepted=True,
            method="manual+point-to-plane-icp", message="ICP accepted",
        )
        workspace.commit_manual_alignment_batch(passes[0].pass_id, {passes[1].pass_id: accepted})
        source_bytes = [path.read_bytes() for path in paths]
        edited = {
            scan_pass.pass_id: (
                np.asarray([[index, 0, 0], [index, 1, 0]], dtype=float),
                np.asarray([[1, 0, 0], [0, 1, 0]], dtype=float),
            )
            for index, scan_pass in enumerate(passes)
        }

        outputs = workspace.commit_point_removal(edited)

        self.assertEqual(set(outputs), {item.pass_id for item in passes})
        self.assertEqual(workspace._aligned_pymeshlab_inputs(passes), [outputs[item.pass_id] for item in passes])
        self.assertTrue(all(Path(path).is_file() for path in outputs.values()))
        self.assertEqual([path.read_bytes() for path in paths], source_bytes)
        for scan_pass in passes:
            cloud = o3d.io.read_point_cloud(outputs[scan_pass.pass_id])
            self.assertEqual(len(cloud.points), 2)
            self.assertTrue(cloud.has_colors())

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

    @unittest.skipUnless(_find_python310() and _find_worker_script(), "PyMeshLab sidecar is unavailable")
    def test_strict_pymeshlab_pipeline_aligns_fuses_and_reconstructs_without_fallback(self):
        source_a = self.write_cloud("PyMeshLab_A.ply", make_room_cloud())
        source_b = self.write_cloud("PyMeshLab_B.ply", make_room_cloud(offset=(0.08, -0.04, 0.03)))
        workspace = DeepMeshFusionWorkspace(
            str(self.root / "pymeshlab_only"),
            DeepMeshFusionConfig(
                voxel_size=0.05,
                pymeshlab_only_pipeline=True,
                complex_poisson_depth=6,
            ),
        )
        workspace.add_pass(str(source_a)); workspace.add_pass(str(source_b)); workspace.analyze_passes()
        workspace.register_passes()
        self.assertTrue(all(item.registration.method == "pymeshlab-pairwise-icp" for item in workspace.passes))
        fused = workspace.fuse_points_registered()
        fused_cloud = o3d.io.read_point_cloud(fused.fused_cloud_path)
        self.assertTrue(fused_cloud.has_normals())
        result = workspace.reconstruct_fused_surface()
        report = json.loads(Path(result.reconstruction_report_path).read_text(encoding="utf-8"))
        self.assertGreater(fused.fused_point_count, 100)
        self.assertEqual(report["strategy"], "pymeshlab-only")
        self.assertEqual(report["worker"]["backend"], "pymeshlab-screened-poisson")
        self.assertIn("unsupported_vertex_count", report["worker"])
        self.assertGreater(report["worker"]["support_distance"], 0)
        self.assertGreater(result.reconstructed_face_count, 0)


if __name__ == "__main__":
    unittest.main()
