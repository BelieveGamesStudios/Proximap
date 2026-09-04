import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from deep_mesh_fusion import DeepMeshFusionConfig, DeepMeshFusionReconstructionService
from deep_mesh_fusion.fusion import ConsensusFusionOutput


def synthetic_apartment():
    spacing = 0.10
    xs = np.arange(0.05, 4.0, spacing)
    ys = np.arange(0.05, 3.0, spacing)
    zs = np.arange(0.05, 2.5, spacing)
    points = set()
    for x in xs:
        for y in ys:
            points.add((round(x, 4), round(y, 4), 0.0))
            points.add((round(x, 4), round(y, 4), 2.5))
    for x in xs:
        for z in zs:
            if not (1.20 <= x < 2.10 and z < 2.05):
                points.add((round(x, 4), 0.0, round(z, 4)))
            if not (2.40 <= x < 3.20 and 0.90 <= z < 1.80):
                points.add((round(x, 4), 3.0, round(z, 4)))
    for y in ys:
        for z in zs:
            points.add((0.0, round(y, 4), round(z, 4)))
            points.add((4.0, round(y, 4), round(z, 4)))

    # A rounded non-planar furnishing exercises the general reconstruction path.
    rng = np.random.default_rng(21)
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    furnishing = directions * [0.35, 0.35, 0.45] + [2.0, 1.5, 0.65]
    geometry = np.vstack((np.asarray(sorted(points), dtype=float), furnishing))
    count = len(geometry)
    return ConsensusFusionOutput(
        points=geometry,
        normals=np.zeros((count, 3), dtype=float),
        colors=np.tile([[0.65, 0.68, 0.72]], (count, 1)),
        confidence=np.full(count, 0.95, dtype=float),
        observation_counts=np.full(count, 3, dtype=np.uint16),
        method_codes=np.zeros(count, dtype=np.uint8),
        provenance=[],
        suppressed_observation_count=0,
    )


class ArchitectureReconstructionTests(unittest.TestCase):
    def _service(self):
        return DeepMeshFusionReconstructionService(DeepMeshFusionConfig(
            voxel_size=0.10,
            fusion_cell_size=0.10,
            architecture_up_axis="z",
            architecture_plane_distance=0.025,
            architecture_plane_min_points=80,
            architecture_grid_size=0.10,
            architecture_grid_closing_iterations=0,
            opening_min_width=0.30,
            opening_min_height=0.30,
            opening_min_area=0.10,
            doorway_floor_tolerance=0.20,
            complex_poisson_depth=6,
            complex_alpha_multiplier=2.5,
        ))

    def test_detects_room_planes_openings_edges_and_corners(self):
        output = self._service().reconstruct(synthetic_apartment())
        summary = output.summary
        self.assertGreaterEqual(summary.wall_count, 4)
        self.assertEqual(summary.floor_count, 1)
        self.assertEqual(summary.ceiling_count, 1)
        self.assertGreaterEqual(summary.doorway_count, 1)
        self.assertGreaterEqual(summary.window_count, 1)
        self.assertGreaterEqual(summary.edge_count, 8)
        self.assertGreaterEqual(summary.corner_count, 4)
        self.assertGreater(summary.complex_point_count, 100)
        self.assertGreater(summary.vertex_count, 1000)
        self.assertGreater(summary.face_count, 1000)

        centers = output.vertices[output.faces].mean(axis=1)
        doorway_faces = (
            (np.abs(centers[:, 1]) < 0.03)
            & (centers[:, 0] >= 1.20)
            & (centers[:, 0] < 2.10)
            & (centers[:, 2] < 2.0)
        )
        self.assertFalse(np.any(doorway_faces), "Door opening was incorrectly bridged by wall triangles")

    def test_exports_classified_mesh_and_reconstruction_report(self):
        service = self._service()
        output = service.reconstruct(synthetic_apartment())
        with tempfile.TemporaryDirectory() as temporary:
            mesh_path = Path(temporary) / "architecture_mesh.ply"
            report_path = Path(temporary) / "architecture_reconstruction.json"
            result = service.export(output, str(mesh_path), str(report_path))
            mesh = o3d.io.read_triangle_mesh(result.mesh_path)
            self.assertEqual(len(mesh.vertices), output.summary.vertex_count)
            self.assertEqual(len(mesh.triangles), output.summary.face_count)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["strategy"], "architecture-aware-planar+screened-poisson")
            self.assertEqual(report["complex_reconstruction"]["method"], "screened-poisson")
            self.assertEqual(report["complex_reconstruction"]["requested_backend"], "pymeshlab")
            self.assertIn(report["complex_reconstruction"]["backend"], {
                "pymeshlab-screened-poisson", "open3d-screened-poisson",
            })
            self.assertGreater(report["complex_reconstruction"]["cleaned_point_count"], 0)
            self.assertGreater(report["complex_reconstruction"]["generated_face_count"], 0)
            self.assertGreaterEqual(len(report["planes"]), 6)
            self.assertTrue(any(plane["openings"] for plane in report["planes"] if plane["classification"] == "wall"))
            header = mesh_path.read_text(encoding="ascii").split("end_header", 1)[0]
            self.assertIn("property uchar architecture_class", header)
            self.assertIn("property int surface_id", header)
            self.assertIn("property float confidence", header)

    def test_screened_poisson_cleans_duplicates_and_low_confidence_residuals(self):
        service = self._service()
        rng = np.random.default_rng(42)
        directions = rng.normal(size=(700, 3)); directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = directions * [0.55, 0.45, 0.65] + [1.0, 1.0, 1.0]
        duplicates = points[:100].copy()
        outliers = np.asarray([[8.0, 8.0, 8.0], [-7.0, 6.0, 9.0]])
        points = np.vstack((points, duplicates, outliers))
        colors = np.vstack((
            np.tile([[0.2, 0.7, 0.4]], (700, 1)),
            np.tile([[1.0, 0.0, 0.0]], (100, 1)),
            np.tile([[1.0, 0.0, 1.0]], (2, 1)),
        ))
        confidence = np.r_[np.full(700, .95), np.full(100, .40), np.full(2, .05)]
        normals = np.vstack((directions, directions[:100], np.zeros((2, 3))))

        mesh = service._mesh_complex(points, colors, confidence, normals)
        self.assertIsNotNone(mesh)
        self.assertGreater(len(mesh["faces"]), 100)
        self.assertGreater(service.last_complex_report["rejected_point_count"], 100)
        self.assertGreaterEqual(service.last_complex_report["point_component_count"], 1)
        self.assertTrue(service.last_complex_report["watertight"])
        self.assertLess(np.max(np.linalg.norm(mesh["vertices"] - [1.0, 1.0, 1.0], axis=1)), 1.5)
        self.assertLessEqual(float(np.max(mesh["confidence"])), .95)


if __name__ == "__main__":
    unittest.main()
