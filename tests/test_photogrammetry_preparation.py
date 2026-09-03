import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from deep_mesh_fusion import ColmapTextModelLoader, DeepMeshFusionConfig, DeepMeshFusionWorkspace, PhotogrammetryPreparationService


class PhotogrammetryPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "sparse" / "0"
        self.images = self.root / "images"
        self.model.mkdir(parents=True)
        self.images.mkdir()
        (self.model / "cameras.txt").write_text(
            "# Camera list\n1 PINHOLE 128 128 100 100 64 64\n", encoding="utf-8"
        )
        image_lines = ["# Image list"]
        for image_id, x in ((1, 0.35), (2, 0.65)):
            name = f"view_{image_id}.png"
            translation = f"{-x} -0.5 2"
            image_lines.extend((f"{image_id} 1 0 0 0 {translation} 1 {name}", "1 1 1 2 2 2"))
            yy, xx = np.indices((128, 128))
            pixels = (((xx // 8 + yy // 8) % 2) * 150 + 50).astype(np.uint8)
            Image.fromarray(pixels).save(self.images / name)
        (self.model / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        points = [
            "1 0 0 0 200 200 200 0.1 1 0 2 0",
            "2 1 0 0 200 200 200 0.1 1 1 2 1",
            "3 1 1 0 200 200 200 0.1 1 2 2 2",
            "4 0 1 0 200 200 200 0.1 1 3 2 3",
        ]
        (self.model / "points3D.txt").write_text("# Point list\n" + "\n".join(points) + "\n", encoding="utf-8")
        self.vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        self.faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.normals = np.tile([[0., 0., 1.]], (4, 1))

    def tearDown(self):
        self.temp.cleanup()

    def config(self):
        return DeepMeshFusionConfig(
            voxel_size=0.1, photogrammetry_voxel_size=0.15,
            photogrammetry_min_registration_fitness=0.2,
            photogrammetry_min_camera_mesh_coverage=0.5,
            photogrammetry_min_image_quality=0.35,
            texture_atlas_size=64,
        )

    def test_colmap_text_model_loads_cameras_points_and_immutable_hashes(self):
        dataset = ColmapTextModelLoader().load(str(self.model), str(self.images))
        self.assertEqual(len(dataset.cameras), 2)
        self.assertEqual(len(dataset.points), 4)
        self.assertEqual(dataset.cameras[0].registered_observation_count, 2)
        self.assertAlmostEqual(dataset.cameras[0].center[2], -2.0)
        self.assertIn(str((self.images / "view_1.png").resolve()), dataset.source_hashes)

    def test_colmap_binary_model_is_supported_without_conversion(self):
        binary = self.root / "binary"
        binary.mkdir()
        with (binary / "cameras.bin").open("wb") as handle:
            handle.write(struct.pack("<QiiQQ4d", 1, 1, 1, 128, 128, 100., 100., 64., 64.))
        with (binary / "images.bin").open("wb") as handle:
            handle.write(struct.pack("<Q", 1))
            handle.write(struct.pack("<i7di", 1, 1., 0., 0., 0., -0.5, -0.5, 2., 1))
            handle.write(b"view_1.png\x00")
            handle.write(struct.pack("<Qddq", 1, 64., 64., 1))
        with (binary / "points3D.bin").open("wb") as handle:
            handle.write(struct.pack("<Q", 3))
            for point_id, point in enumerate(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), 1):
                handle.write(struct.pack("<Q3d3BdQii", point_id, *point, 200, 200, 200, 0.1, 1, 1, point_id - 1))
        dataset = ColmapTextModelLoader().load(str(binary), str(self.images))
        self.assertEqual(len(dataset.points), 3)
        self.assertEqual(dataset.cameras[0].registered_observation_count, 1)
        self.assertIn(str((binary / "cameras.bin").resolve()), dataset.source_hashes)

    def test_registration_camera_validation_coverage_and_quality(self):
        service = PhotogrammetryPreparationService(self.config())
        output = service.prepare(
            str(self.model), str(self.images), self.vertices, self.faces, self.normals,
            manual_transform=np.eye(4),
        )
        self.assertTrue(output.registration.accepted)
        self.assertEqual(output.registration.correspondence_count, 4)
        self.assertEqual(sum(item.relationship_valid for item in output.camera_validation), 2)
        self.assertAlmostEqual(output.summary.coverage, 1.0)
        self.assertAlmostEqual(output.summary.multi_view_coverage, 1.0)
        self.assertTrue(output.summary.texture_ready)
        self.assertTrue(all(item.score > 0.35 for item in output.texture_quality))

    def test_automatic_similarity_registration_recovers_scale_rotation_and_translation(self):
        rng = np.random.default_rng(3)
        target = np.vstack((
            np.c_[rng.uniform(0, 3, 240), rng.uniform(0, 2, 240), np.zeros(240)],
            np.c_[np.zeros(180), rng.uniform(0, 2, 180), rng.uniform(0, 1.5, 180)],
            np.c_[rng.uniform(0, 3, 140), np.full(140, 2.), rng.uniform(0, 1.5, 140)],
        ))
        angle = 0.6
        rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        expected_scale, translation = 1.7, np.asarray([4., -2., 0.8])
        source = (target - translation) @ rotation / expected_scale
        metrics, aligned = PhotogrammetryPreparationService(self.config())._register(source, target, None, None)
        self.assertTrue(metrics.accepted)
        self.assertAlmostEqual(metrics.scale, expected_scale, delta=0.05)
        self.assertGreater(metrics.mutual_correspondence_ratio, 0.75)
        self.assertLess(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))), 0.04)

    def test_exports_registered_source_and_texture_preparation_artifacts(self):
        service = PhotogrammetryPreparationService(self.config())
        output = service.prepare(
            str(self.model), str(self.images), self.vertices, self.faces, self.normals,
            manual_transform=np.eye(4),
        )
        result = service.export(output, str(self.root / "prepared"))
        for path in (
            result.registration_path, result.aligned_source_path, result.camera_report_path,
            result.coverage_map_path, result.report_path,
        ):
            self.assertTrue(Path(path).is_file())
        registration = json.loads(Path(result.registration_path).read_text())
        report = json.loads(Path(result.report_path).read_text())
        self.assertEqual(registration["registration"]["method"], "manual-similarity")
        self.assertTrue(report["summary"]["texture_ready"])
        self.assertIn("view_count", Path(result.coverage_map_path).read_text().split("end_header")[0])

    def test_workspace_enforces_validation_gate_and_records_preparation(self):
        workspace_root = self.root / "fusion"
        workspace = DeepMeshFusionWorkspace(str(workspace_root), self.config())
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices), o3d.utility.Vector3iVector(self.faces.astype(np.int32))
        )
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(workspace_root / "derived" / "validated_lidar_surface.ply"), mesh)
        validation_path = workspace_root / "analysis" / "geometry_validation.json"
        validation_path.write_text(json.dumps({"summary": {"ready_for_appearance_processing": True}}), encoding="utf-8")
        result = workspace.prepare_photogrammetry(
            str(self.model), str(self.images), manual_transform=np.eye(4)
        )
        manifest = json.loads((workspace_root / "workspace.json").read_text())
        self.assertTrue(result.summary.texture_ready)
        self.assertTrue(manifest["photogrammetry_preparation"]["derived"])
        baked = workspace.bake_textures()
        manifest = json.loads((workspace_root / "workspace.json").read_text())
        self.assertTrue(Path(baked.texture_atlas_path).is_file())
        self.assertTrue(manifest["texture_baking"]["derived"])
        final = workspace.finalize_asset()
        manifest = json.loads((workspace_root / "workspace.json").read_text())
        self.assertTrue(Path(final.final_obj_path).is_file())
        self.assertTrue(manifest["final_asset"]["derived"])
        quality = workspace.evaluate_tour_readiness()
        manifest = json.loads((workspace_root / "workspace.json").read_text())
        self.assertTrue(Path(quality.html_report_path).is_file())
        self.assertFalse(quality.summary.tour_ready)
        self.assertTrue(manifest["tour_readiness"]["derived"])
        self.assertEqual(manifest["schema_version"], 11)


if __name__ == "__main__":
    unittest.main()
