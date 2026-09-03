import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from deep_mesh_fusion import DeepMeshFusionConfig, TourReadinessQualityGate


class TourReadinessQualityGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "analysis").mkdir()
        (self.root / "photogrammetry").mkdir()
        (self.root / "final").mkdir()
        self.lidar = self.root / "Pass_01.ply"; self.lidar.write_text("immutable lidar")
        self.photo = self.root / "view.jpg"; self.photo.write_text("immutable photo")
        self.geometry = {
            "summary": {"scores": {"overall": 0.94, "completeness": 0.97, "consistency": 0.95}, "review_region_count": 0},
            "issues": [],
        }
        self.gaps = {"summary": {"unresolved_gap_count": 0}}
        self.registration = {
            "registration": {"accepted": True},
            "source_hashes": {str(self.photo): self.sha(self.photo)},
        }
        self.final_report = {
            "summary": {
                "polished_asset_ready": True, "review_region_count": 0,
                "quality": {"geometry": 0.94, "texture": 0.91, "coverage": 0.93, "consistency": 0.95, "overall": 0.93},
            },
            "remaining_issues": [],
        }
        self.write_reports()
        final_paths = {}
        for key, name in {
            "final_obj_path": "final_environment.obj", "final_material_path": "final_environment.mtl",
            "final_texture_path": "final_albedo.png", "final_confidence_path": "final_texture_confidence.png",
            "review_map_path": "final_review_map.png", "report_path": "final_asset_validation.json",
        }.items():
            path = self.root / "final" / name
            if name != "final_asset_validation.json": path.write_bytes(f"asset:{name}".encode())
            final_paths[key] = str(path)
        self.manifest = {
            "schema_version": 10,
            "passes": [{
                "pass_id": "pass-01", "enabled": True, "source_path": str(self.lidar),
                "source_sha256": self.sha(self.lidar), "registration": {"accepted": True},
            }],
            "final_asset": final_paths,
        }
        self.write_json(self.root / "workspace.json", self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def write_reports(self):
        self.write_json(self.root / "analysis" / "geometry_validation.json", self.geometry)
        self.write_json(self.root / "analysis" / "gap_recovery.json", self.gaps)
        self.write_json(self.root / "photogrammetry" / "photogrammetry_registration.json", self.registration)
        self.write_json(self.root / "final" / "final_asset_validation.json", self.final_report)

    @staticmethod
    def write_json(path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_ready_asset_passes_every_production_gate_and_exports_handoff_manifest(self):
        service = TourReadinessQualityGate(DeepMeshFusionConfig())
        output = service.evaluate(str(self.root))
        self.assertTrue(output.summary.tour_ready)
        self.assertEqual(output.summary.blocking_issue_count, 0)
        self.assertTrue(all(check.passed for check in output.checks))
        result = service.export(output, str(self.root / "quality"))
        self.assertTrue(Path(result.report_path).is_file())
        self.assertIn("TOUR READY", Path(result.html_report_path).read_text())
        handoff = json.loads(Path(result.asset_manifest_path).read_text())
        self.assertTrue(handoff["tour_ready"])
        self.assertTrue(all(asset["sha256"] for asset in handoff["assets"]))

    def test_gaps_texture_regions_and_registration_conflict_block_release(self):
        self.gaps["summary"]["unresolved_gap_count"] = 2
        self.registration["registration"]["accepted"] = False
        self.final_report["summary"]["polished_asset_ready"] = False
        self.final_report["summary"]["review_region_count"] = 7
        self.final_report["remaining_issues"] = [
            {"severity": "error", "category": "wrong-projection"} for _ in range(7)
        ]
        self.write_reports()
        output = TourReadinessQualityGate(DeepMeshFusionConfig()).evaluate(str(self.root))
        categories = {issue.category for issue in output.issues}
        self.assertFalse(output.summary.tour_ready)
        self.assertIn("geometry-gaps", categories)
        self.assertIn("texture-confidence-regions", categories)
        self.assertIn("registration-conflict", categories)
        self.assertGreater(output.summary.blocking_issue_count, 0)

    def test_changed_source_and_missing_final_artifact_fail_integrity_checks(self):
        self.photo.write_text("changed photo")
        Path(self.manifest["final_asset"]["final_texture_path"]).unlink()
        output = TourReadinessQualityGate(DeepMeshFusionConfig()).evaluate(str(self.root))
        self.assertFalse(output.summary.source_integrity_verified)
        self.assertFalse(output.summary.artifact_integrity_verified)
        self.assertFalse(output.summary.tour_ready)
        self.assertTrue(any(issue.category == "source-integrity" for issue in output.issues))
        self.assertTrue(any(issue.category == "artifact-integrity" for issue in output.issues))


if __name__ == "__main__":
    unittest.main()
