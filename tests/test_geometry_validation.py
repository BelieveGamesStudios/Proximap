import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from deep_mesh_fusion import DeepMeshFusionConfig, GeometryValidationService
from deep_mesh_fusion.gaps import GapRepairOutput
from deep_mesh_fusion.models import ArchitectureReconstructionSummary, GapRegion, GapRepairSummary
from deep_mesh_fusion.reconstruction import ArchitectureMeshOutput, DeepMeshFusionReconstructionService


def clean_gap_output(unresolved=False):
    vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    summary = ArchitectureReconstructionSummary(
        input_point_count=4, plane_count=0, wall_count=0, floor_count=0, ceiling_count=0,
        doorway_count=0, window_count=0, edge_count=0, corner_count=0, complex_point_count=4,
        vertex_count=4, face_count=2, boundary_edge_count=4, nonmanifold_edge_count=0,
        connected_component_count=1, surface_area=1.0,
    )
    mesh = ArchitectureMeshOutput(
        vertices=vertices, faces=faces,
        normals=np.tile([[0., 0., 1.]], (4, 1)), colors=np.tile([[0.7, 0.7, 0.7]], (4, 1)),
        confidence=np.full(4, 0.96), class_codes=np.full(4, 5, dtype=np.uint8),
        surface_ids=np.full(4, -1, dtype=np.int32), planes=[], edges=[], corners=[], summary=summary,
    )
    gaps = []
    if unresolved:
        gaps.append(GapRegion(
            gap_id="gap-review", surface_id=-1, plane_id=None, classification="uncertain",
            decision="manual-review", area=0.35, perimeter=1.2, boundary_confidence=0.4,
            repair_confidence=0.2, observed_point_count=0, evidence_observation_count=0,
            bounds_min=[0.2, 0.2, 0.0], bounds_max=[0.8, 0.8, 0.0], repaired_face_count=0,
            review_required=True, reason="Insufficient evidence",
        ))
    gap_summary = GapRepairSummary(
        detected_gap_count=len(gaps), repaired_gap_count=0, observed_geometry_repair_count=0,
        planar_continuation_count=0, surface_interpolation_count=0, intentional_opening_count=0,
        exterior_boundary_count=1, unresolved_gap_count=int(unresolved), added_vertex_count=0,
        added_face_count=0, final_vertex_count=4, final_face_count=2,
    )
    return GapRepairOutput(mesh, gaps, gap_summary, np.empty((0, 3)), np.empty(0, dtype=np.uint8))


class GeometryValidationTests(unittest.TestCase):
    def config(self):
        return DeepMeshFusionConfig(
            voxel_size=0.1, architecture_grid_size=0.1, architecture_up_axis="z",
            validation_min_completeness=0.8, validation_min_surface_quality=0.8,
            validation_min_consistency=0.8, validation_min_confidence=0.7,
            validation_min_overall_quality=0.8,
        )

    def test_clean_mesh_is_ready_and_intentional_exterior_is_not_a_hole(self):
        output = GeometryValidationService(self.config()).validate(clean_gap_output())
        self.assertTrue(output.summary.ready_for_appearance_processing)
        self.assertEqual(output.summary.unclassified_hole_count, 0)
        self.assertAlmostEqual(output.summary.scores.completeness, 1.0)
        self.assertGreater(output.summary.scores.surface_quality, 0.95)

    def test_detects_corrupt_geometry_and_blocks_readiness(self):
        gap_output = clean_gap_output()
        mesh = gap_output.mesh
        extra = np.asarray([
            [0.25, 0.25, -0.5], [0.25, 0.25, 0.5], [0.75, 0.25, 0.0],
            [4.0, 0.0, 0.0], [4.001, 0.0, 0.0], [4.0, 1.0, 0.0],
        ])
        vertices = np.vstack((mesh.vertices, extra))
        faces = np.vstack((mesh.faces, [[4, 5, 6], [7, 8, 9], [0, 0, 1]])).astype(np.int64)
        normals = DeepMeshFusionReconstructionService(self.config())._vertex_normals(vertices, faces)
        corrupt = replace(
            mesh, vertices=vertices, faces=faces, normals=normals,
            colors=np.vstack((mesh.colors, np.tile([[0.7, 0.7, 0.7]], (6, 1)))),
            confidence=np.r_[mesh.confidence, np.full(6, 0.9)],
            class_codes=np.r_[mesh.class_codes, np.full(6, 5, dtype=np.uint8)],
            surface_ids=np.r_[mesh.surface_ids, np.full(6, -1, dtype=np.int32)],
            summary=replace(mesh.summary, vertex_count=len(vertices), face_count=len(faces)),
        )
        gap_output = replace(gap_output, mesh=corrupt)
        output = GeometryValidationService(self.config()).validate(gap_output)
        categories = {issue.category for issue in output.issues}
        self.assertGreater(output.summary.degenerate_triangle_count, 0)
        self.assertGreater(output.summary.stretched_triangle_count, 0)
        self.assertGreater(output.summary.self_intersection_count, 0)
        self.assertGreater(output.summary.disconnected_component_count, 1)
        self.assertIn("self-intersections", categories)
        self.assertFalse(output.summary.ready_for_appearance_processing)

    def test_unresolved_gap_reduces_completeness_and_exports_quality_artifacts(self):
        service = GeometryValidationService(self.config())
        output = service.validate(clean_gap_output(unresolved=True))
        self.assertLess(output.summary.scores.completeness, 1.0)
        self.assertGreater(output.summary.review_region_count, 0)
        self.assertFalse(output.summary.ready_for_appearance_processing)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = service.export(output, root / "validated.ply", root / "validation.json", root / "quality.ply")
            report = json.loads(Path(result.report_path).read_text())
            self.assertIn("issues", report)
            self.assertIn("ready_for_appearance_processing", report["summary"])
            self.assertIn("quality_score", (root / "quality.ply").read_text().split("end_header")[0])
            self.assertTrue(Path(result.validated_mesh_path).is_file())


if __name__ == "__main__":
    unittest.main()
