import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from deep_mesh_fusion import DeepMeshFusionConfig, RegistrationMetrics, ScanPass, TransientArtifactSuppressionService


def scan_pass(index):
    return ScanPass(
        pass_id=f"pass-{index:02d}",
        name=f"Pass {index}",
        source_path=f"Pass_{index:02d}.ply",
        source_sha256=str(index) * 64,
        source_size=0,
        registration=RegistrationMetrics("pass-01", fitness=1.0, inlier_rmse=0.0, accepted=True),
    )


def cloud(points, normals=None):
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=float)))
    if normals is not None:
        result.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    return result


def room_shell():
    values = np.arange(0.025, 1.0, 0.05)
    points = []
    for a in values:
        for b in values:
            points.extend(([a, b, 0.0], [a, b, 1.0], [0.0, a, b], [1.0, a, b], [a, 0.0, b], [a, 1.0, b]))
    return np.asarray(points)


def box_points(x_range, y_range, z_range, spacing=0.035):
    xs = np.arange(x_range[0], x_range[1], spacing)
    ys = np.arange(y_range[0], y_range[1], spacing)
    zs = np.arange(z_range[0], z_range[1], spacing)
    return np.asarray([[x, y, z] for x in xs for y in ys for z in zs])


class ArtifactSuppressionTests(unittest.TestCase):
    def test_pass_specific_people_furniture_and_curtain_patterns_are_suppressed(self):
        base = room_shell()
        person = box_points((0.44, 0.56), (0.44, 0.56), (0.10, 0.82))
        furniture = box_points((0.18, 0.42), (0.60, 0.84), (0.08, 0.38), spacing=0.05)
        curtain = box_points((0.70, 0.76), (0.18, 0.82), (0.18, 0.88), spacing=0.05)
        inputs = [
            (scan_pass(1), cloud(np.vstack((base, person)))),
            (scan_pass(2), cloud(np.vstack((base, furniture)))),
            (scan_pass(3), cloud(np.vstack((base, curtain)))),
        ]
        service = TransientArtifactSuppressionService(DeepMeshFusionConfig(
            voxel_size=0.10,
            artifact_cell_size=0.10,
            artifact_suppression_threshold=0.65,
        ))
        output = service.suppress(inputs)

        self.assertGreaterEqual(output.summary.suppressed_component_count, 3)
        transient_points = len(person) + len(furniture) + len(curtain)
        self.assertGreaterEqual(output.summary.suppressed_point_count, int(transient_points * 0.95))
        for scan, filtered in output.filtered_clouds:
            self.assertLessEqual(len(filtered.points), len(base) + 30, scan.pass_id)
        suppressed_classes = {item.classification for item in output.reports if item.suppressed}
        self.assertIn("non-persistent-object", suppressed_classes)

    def test_coplanar_single_pass_wall_patch_is_retained(self):
        values = np.arange(0.01, 1.0, 0.025)
        full = np.asarray([[x, 0.0, z] for x in values for z in values])
        patch = (full[:, 0] >= 0.30) & (full[:, 0] < 0.60) & (full[:, 2] >= 0.30) & (full[:, 2] < 0.60)
        with_hole = full[~patch]
        full_normals = np.tile([[0.0, 1.0, 0.0]], (len(full), 1))
        hole_normals = np.tile([[0.0, 1.0, 0.0]], (len(with_hole), 1))
        inputs = [
            (scan_pass(1), cloud(full, full_normals)),
            (scan_pass(2), cloud(with_hole, hole_normals)),
            (scan_pass(3), cloud(with_hole, hole_normals)),
        ]
        service = TransientArtifactSuppressionService(DeepMeshFusionConfig(
            voxel_size=0.10,
            artifact_cell_size=0.10,
            artifact_structural_continuity_threshold=0.55,
        ))
        output = service.suppress(inputs)
        structural = [item for item in output.reports if item.pass_id == "pass-01" and item.classification == "retained-structural-single-pass"]
        self.assertTrue(structural)
        self.assertTrue(all(not item.suppressed for item in structural))
        self.assertEqual(len(output.filtered_clouds[0][1].points), len(full))

    def test_report_and_rejected_geometry_are_auditable(self):
        base = room_shell()
        floater = np.asarray([[0.5, 0.5, 0.5]])
        inputs = [
            (scan_pass(1), cloud(np.vstack((base, floater)))),
            (scan_pass(2), cloud(base)),
            (scan_pass(3), cloud(base)),
        ]
        service = TransientArtifactSuppressionService(DeepMeshFusionConfig(voxel_size=0.10, artifact_cell_size=0.10))
        output = service.suppress(inputs)
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            rejected_path = Path(temporary) / "rejected.ply"
            result = service.export(output, str(report_path), str(rejected_path))
            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(data["principle"], "persistence-across-independent-passes")
            self.assertEqual(set(data["sources"]), {"pass-01", "pass-02", "pass-03"})
            self.assertGreater(data["summary"]["suppressed_point_count"], 0)
            self.assertTrue(any(item["classification"] == "floating-or-isolated-fragment" for item in data["components"]))
            rejected = o3d.io.read_point_cloud(result.rejected_geometry_path)
            self.assertGreater(len(rejected.points), 0)
            self.assertTrue(rejected.has_colors())


if __name__ == "__main__":
    unittest.main()
