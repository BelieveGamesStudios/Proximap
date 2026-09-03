import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from deep_mesh_fusion import DeepMeshFusionConfig, IntelligentTextureBakingService, PhotogrammetryPreparationService


class TextureBakingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model"; self.images = self.root / "images"
        self.model.mkdir(); self.images.mkdir()
        (self.model / "cameras.txt").write_text("1 PINHOLE 128 128 100 100 64 64\n", encoding="utf-8")
        image_lines = []
        yy, xx = np.indices((128, 128))
        pattern = (((xx // 8 + yy // 8) % 2) * 140 + 55).astype(np.uint8)
        rgb = np.dstack((pattern, np.roll(pattern, 2, axis=1), np.roll(pattern, 2, axis=0)))
        for image_id, camera_x, brightness in ((1, 0.35, 1.0), (2, 0.65, 0.9)):
            name = f"view_{image_id}.png"
            Image.fromarray(np.clip(rgb * brightness, 0, 255).astype(np.uint8)).save(self.images / name)
            image_lines.extend((f"{image_id} 1 0 0 0 {-camera_x} -0.5 2 1 {name}", "1 1 1"))
        (self.model / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        (self.model / "points3D.txt").write_text(
            "1 0 0 0 200 200 200 0.1 1 0\n2 1 0 0 200 200 200 0.1 1 0\n"
            "3 1 1 0 200 200 200 0.1 1 0\n4 0 1 0 200 200 200 0.1 1 0\n",
            encoding="utf-8",
        )
        self.vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        self.faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.config = DeepMeshFusionConfig(
            voxel_size=0.1, photogrammetry_voxel_size=0.15,
            photogrammetry_min_registration_fitness=0.2,
            photogrammetry_min_camera_mesh_coverage=0.5,
            photogrammetry_min_image_quality=0.3,
            texture_atlas_size=64, texture_atlas_padding=2,
            texture_min_observation_score=0.2,
        )
        self.preparation = PhotogrammetryPreparationService(self.config).prepare(
            str(self.model), str(self.images), self.vertices, self.faces,
            np.tile([[0., 0., 1.]], (4, 1)), manual_transform=np.eye(4),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_projects_selects_blends_and_propagates_geometry_confidence(self):
        baker = IntelligentTextureBakingService(self.config)
        high = baker.bake(self.preparation, np.full(4, 0.95))
        low = baker.bake(self.preparation, np.full(4, 0.25))
        self.assertGreater(high.summary.textured_texel_count, 0)
        self.assertGreater(high.summary.blended_texel_count, 0)
        self.assertGreater(high.summary.texture_coverage, 0.95)
        self.assertGreater(high.summary.mean_texture_confidence, low.summary.mean_texture_confidence)
        self.assertTrue(high.summary.texture_ready)
        self.assertEqual(high.face_uvs.shape, (2, 3, 2))
        self.assertTrue(np.all((high.face_uvs >= 0) & (high.face_uvs <= 1)))

    def test_exports_textured_asset_atlas_confidence_and_selection_report(self):
        service = IntelligentTextureBakingService(self.config)
        output = service.bake(self.preparation, np.full(4, 0.9))
        result = service.export(output, str(self.root / "texture"))
        for path in (
            result.textured_obj_path, result.material_path, result.texture_atlas_path,
            result.confidence_map_path, result.selection_report_path,
        ):
            self.assertTrue(Path(path).is_file())
        atlas = np.asarray(Image.open(result.texture_atlas_path))
        confidence = np.asarray(Image.open(result.confidence_map_path))
        report = json.loads(Path(result.selection_report_path).read_text())
        self.assertEqual(atlas.shape, (64, 64, 4))
        self.assertEqual(confidence.shape, (64, 64, 4))
        self.assertEqual(report["hole_treatment"], "unobserved texels remain transparent and are not synthesized")
        self.assertEqual(len(report["camera_usage"]), 2)
        self.assertIn("map_Kd environment_albedo.png", Path(result.material_path).read_text())

    def test_low_quality_camera_is_rejected_when_a_better_source_exists(self):
        qualities = list(self.preparation.texture_quality)
        qualities[0] = replace(qualities[0], score=0.05)
        preparation = replace(self.preparation, texture_quality=qualities)
        output = IntelligentTextureBakingService(self.config).bake(preparation, np.full(4, 0.95))
        usage = {item.image_id: item for item in output.camera_usage}
        self.assertEqual(usage[1].selected_texel_count, 0)
        self.assertGreater(usage[2].selected_texel_count, 0)

    def test_occluded_or_out_of_frustum_surface_remains_transparent(self):
        vertices = np.vstack((self.vertices, [[0., 0., -3.], [1., 0., -3.], [0., 1., -3.]]))
        faces = np.vstack((self.faces, [[4, 5, 6]])).astype(np.int64)
        preparation = PhotogrammetryPreparationService(self.config).prepare(
            str(self.model), str(self.images), vertices, faces,
            np.tile([[0., 0., 1.]], (len(vertices), 1)), manual_transform=np.eye(4),
        )
        output = IntelligentTextureBakingService(self.config).bake(preparation, np.full(len(vertices), 0.95))
        self.assertGreater(output.summary.uncovered_texel_count, 0)
        self.assertFalse(output.summary.texture_ready)
        self.assertTrue(np.any(output.atlas[:, :, 3] == 0))


if __name__ == "__main__":
    unittest.main()
