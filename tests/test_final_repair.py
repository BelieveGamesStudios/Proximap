import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from deep_mesh_fusion import FinalSurfaceTextureRepairService, IntelligentTextureBakingService
from tests import test_texture_baking as texture_test_module


class FinalSurfaceRepairTests(unittest.TestCase):
    def setUp(self):
        self.source = texture_test_module.TextureBakingTests()
        self.source.setUp()
        self.config = self.source.config
        self.baked = IntelligentTextureBakingService(self.config).bake(
            self.source.preparation, np.full(4, 0.95)
        )

    def tearDown(self):
        self.source.tearDown()

    def test_clean_asset_passes_final_surface_and_texture_validation(self):
        output = FinalSurfaceTextureRepairService(self.config).repair(self.baked, np.full(4, 0.95))
        self.assertTrue(output.summary.polished_asset_ready)
        self.assertEqual(output.summary.remaining_issue_count, 0)
        self.assertGreater(output.summary.quality.overall, 0.9)

    def test_high_geometry_low_texture_confidence_is_flagged_for_review(self):
        weak = replace(
            self.baked,
            texture_confidence=np.where(self.baked.valid_texels, 0.10, 0.0).astype(np.float32),
        )
        output = FinalSurfaceTextureRepairService(self.config).repair(weak, np.full(4, 0.95))
        categories = {issue.category for issue in output.remaining_issues}
        self.assertIn("wrong-projection", categories)
        self.assertFalse(output.summary.texture_valid)
        self.assertGreater(output.summary.review_region_count, 0)

    def test_small_bounded_missing_region_is_repaired_but_large_gap_is_preserved(self):
        valid = self.baked.valid_texels.copy()
        atlas = self.baked.atlas.copy()
        # Remove a compact interior patch from the first UV chart.
        valid[12:15, 12:15] = False; atlas[12:15, 12:15] = 0
        damaged = replace(self.baked, atlas=atlas, valid_texels=valid)
        output = FinalSurfaceTextureRepairService(self.config).repair(damaged, np.full(4, 0.95))
        self.assertTrue(any(issue.category == "missing-texture" and issue.repaired for issue in output.initial_issues))

        valid_large = self.baked.valid_texels.copy(); atlas_large = self.baked.atlas.copy()
        valid_large[:32, :32] = False; atlas_large[:32, :32] = 0
        large = replace(self.baked, atlas=atlas_large, valid_texels=valid_large)
        unresolved = FinalSurfaceTextureRepairService(self.config).repair(large, np.full(4, 0.95))
        self.assertTrue(any(issue.category == "missing-texture" for issue in unresolved.remaining_issues))
        self.assertFalse(unresolved.summary.polished_asset_ready)

    def test_moderate_seam_is_repaired_and_abnormal_uv_stretch_is_reported(self):
        atlas = self.baked.atlas.copy()
        service = FinalSurfaceTextureRepairService(self.config)
        ys, xs = service._face_pixels(self.baked.face_uvs[1], atlas.shape[0])
        atlas[ys, xs, :3] = np.clip(atlas[ys, xs, :3].astype(int) + 55, 0, 255).astype(np.uint8)
        seam_output = service.repair(replace(self.baked, atlas=atlas), np.full(4, 0.95))
        self.assertTrue(any(issue.category == "texture-seam" and issue.repaired for issue in seam_output.initial_issues))

        stretch_config = replace(self.config, final_texture_stretch_ratio=1.2)
        stretched_vertices = self.baked.mesh_vertices.copy(); stretched_vertices[1, 0] = 4.0
        stretched = replace(self.baked, mesh_vertices=stretched_vertices)
        stretch_output = FinalSurfaceTextureRepairService(stretch_config).repair(stretched, np.full(4, 0.95))
        self.assertTrue(any(issue.category == "stretched-texture" for issue in stretch_output.remaining_issues))

    def test_exports_polished_asset_and_before_after_review_report(self):
        service = FinalSurfaceTextureRepairService(self.config)
        output = service.repair(self.baked, np.full(4, 0.95))
        with tempfile.TemporaryDirectory() as directory:
            result = service.export(output, directory)
            for path in (
                result.final_obj_path, result.final_material_path, result.final_texture_path,
                result.final_confidence_path, result.review_map_path, result.report_path,
            ):
                self.assertTrue(Path(path).is_file())
            report = json.loads(Path(result.report_path).read_text())
            self.assertIn("initial_issues", report)
            self.assertIn("remaining_issues", report)
            self.assertEqual(np.asarray(Image.open(result.review_map_path)).shape, (64, 64, 4))


if __name__ == "__main__":
    unittest.main()
