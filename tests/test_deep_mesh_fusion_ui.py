import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import open3d as o3d
from PySide6.QtCore import QMimeData, Qt, QUrl
from PySide6.QtWidgets import QApplication

from deep_mesh_fusion.ui import DeepMeshFusionPanel, PIPELINE_STAGES
from deep_mesh_fusion.viewport import DeepMeshFusionViewport
from deep_mesh_fusion.workspace import DeepMeshFusionWorkspace


def write_cloud(path: Path, offset=0.0):
    points = [(offset + x * .01, y * .01, (x + y) * .002) for x in range(12) for y in range(10)]
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z", "end_header",
    ]
    lines.extend(f"{x} {y} {z}" for x, y, z in points)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class DeepMeshFusionPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        model = self.root / "colmap" / "sparse" / "0"; model.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"): (model / name).write_bytes(b"model")
        (self.root / "input_images").mkdir(); (self.root / "input_images" / "frame_001.jpg").write_bytes(b"image")
        mvs = self.root / "mvs"; mvs.mkdir(); write_cloud(mvs / "scene_dense.ply")
        self.panel = DeepMeshFusionPanel(reconstruction_root=str(self.root))

    def tearDown(self):
        self.panel.deleteLater(); self.temp.cleanup()

    def test_exposes_seven_clickable_dependency_gated_stages(self):
        self.assertEqual(len(PIPELINE_STAGES), 7)
        self.assertEqual(PIPELINE_STAGES[0], "Scan preparation")
        self.assertIn("Cleanup", PIPELINE_STAGES)
        self.assertIn("Texture", PIPELINE_STAGES)
        self.assertEqual(PIPELINE_STAGES[-1], "Final quality")
        self.assertEqual(len(self.panel.stage_labels), 7)
        self.assertTrue(self.panel.stage_labels[0].isEnabled())
        self.assertFalse(self.panel.stage_labels[1].isEnabled())
        self.assertFalse(hasattr(self.panel, "stage_table"))
        self.assertFalse(hasattr(self.panel, "status_badge"))
        self.assertEqual(self.panel.action_button.text(), "Continue")
        self.assertFalse(self.panel.action_button.isEnabled())

    def test_autodetects_reconstruction_without_exposing_raw_path_fields(self):
        self.assertEqual(Path(self.panel.photogrammetry_paths[0]), self.root / "colmap" / "sparse" / "0")
        self.assertEqual(Path(self.panel.photogrammetry_paths[1]), self.root / "input_images")
        self.assertEqual(Path(self.panel.photogrammetry_paths[2]), self.root / "mvs" / "scene_dense.ply")
        self.assertEqual(self.panel.photo_button.text(), "Load Reconstructed Model")
        self.assertFalse(hasattr(self.panel, "model_edit"))
        self.assertFalse(hasattr(self.panel, "image_edit"))
        self.assertFalse(hasattr(self.panel, "dense_edit"))

    def test_autodetects_recovered_colmap_images_layout(self):
        recovered = self.root / "recovered"
        model = recovered / "colmap" / "sparse" / "0"; model.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"): (model / name).write_bytes(b"model")
        images = recovered / "colmap" / "images"; images.mkdir(parents=True); (images / "frame.jpg").write_bytes(b"image")
        dense = recovered / "mvs" / "scene_dense.ply"; dense.parent.mkdir(parents=True); write_cloud(dense)
        panel = DeepMeshFusionPanel(reconstruction_root=str(recovered))
        try:
            self.assertEqual(Path(panel.photogrammetry_paths[1]), images)
            self.assertEqual(Path(panel.photogrammetry_paths[2]), dense)
            self.assertEqual(panel.photo_button.text(), "Load Reconstructed Model")
        finally:
            panel.deleteLater()

    def test_partial_reconstruction_without_dense_cloud_is_not_offered(self):
        partial = self.root / "partial"
        model = partial / "colmap" / "sparse" / "0"; model.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"): (model / name).write_bytes(b"model")
        images = partial / "colmap" / "images"; images.mkdir(parents=True); (images / "frame.jpg").write_bytes(b"image")
        panel = DeepMeshFusionPanel(reconstruction_root=str(partial))
        try:
            self.assertIsNone(panel.photogrammetry_paths)
            self.assertEqual(panel.photo_button.text(), "Run Reconstruction")
        finally:
            panel.deleteLater()

    def test_import_loads_distinct_viewport_layers_and_fits_combined_scans(self):
        first = self.root / "Pass_01.ply"; second = self.root / "Pass_02.ply"
        write_cloud(first); write_cloud(second, .05)
        self.panel._add_scan_paths((str(first), str(second)))
        self.assertEqual(len(self.panel.scan_previews), 2)
        self.assertEqual(len(self.panel.scan_rows), 2)
        self.assertIn("scan-1", self.panel.viewport.layers)
        self.assertIn("scan-2", self.panel.viewport.layers)
        self.assertNotEqual(self.panel.scan_previews["scan-1"].color, self.panel.scan_previews["scan-2"].color)
        self.assertEqual(self.panel.action_button.text(), "Continue")
        self.assertEqual(self.panel.scan_count_label.text(), "2 scans")
        self.assertTrue(self.panel.scan_count_label.alignment() & Qt.AlignRight)
        self.assertTrue(self.panel.action_button.isEnabled())

    def test_visibility_and_remove_controls_update_viewport_and_state(self):
        first = self.root / "Pass_01.ply"; second = self.root / "Pass_02.ply"
        write_cloud(first); write_cloud(second, .05); self.panel._add_scan_paths((str(first), str(second)))
        self.panel._set_scan_visible("scan-1", False)
        self.assertFalse(self.panel.viewport.layers["scan-1"]["visible"])
        self.panel._remove_scan("scan-2")
        self.assertEqual(len(self.panel.scan_previews), 1)
        self.assertEqual(self.panel.action_button.text(), "Continue")
        self.assertFalse(self.panel.action_button.isEnabled())

    def test_photogrammetry_controls_are_scoped_to_texture_stage(self):
        self.assertTrue(self.panel.texture_panel.isHidden())
        self.assertFalse(self.panel.photo_panel.isHidden())
        self.assertFalse(self.panel.advanced_toggle.isHidden())
        self.assertFalse(self.panel.advanced_panel.isVisible())
        self.assertFalse(self.panel.console.isVisible())
        self.panel._toggle_diagnostics(True)
        self.assertFalse(self.panel.console.isHidden())

    def test_full_height_sidebar_and_resizable_viewport_console_layout(self):
        self.panel.resize(1000, 700); self.panel.show(); self.app.processEvents()
        self.assertEqual(self.panel.main_splitter.orientation(), Qt.Horizontal)
        self.assertEqual(self.panel.detail_splitter.orientation(), Qt.Vertical)
        self.assertIs(self.panel.sidebar.parentWidget(), self.panel.main_splitter)
        self.assertIs(self.panel.detail_splitter.parentWidget(), self.panel.main_splitter)
        self.assertIs(self.panel.viewport.parentWidget(), self.panel.detail_splitter)
        self.assertIs(self.panel.diagnostics_pane.parentWidget(), self.panel.detail_splitter)
        self.assertTrue(self.panel.sidebar.isAncestorOf(self.panel.action_button))
        self.assertFalse(self.panel.diagnostics_pane.isAncestorOf(self.panel.action_button))
        self.assertEqual(self.panel.diagnostics_pane.minimumHeight(), 44)
        self.assertEqual(self.panel.sidebar.height(), self.panel.detail_splitter.height())
        action_center = self.panel.action_button.mapTo(self.panel.sidebar, self.panel.action_button.rect().center()).x()
        self.assertLessEqual(abs(action_center - self.panel.sidebar.width() // 2), 2)

    def test_reuses_locked_mesh_editor_viewport_without_picking_or_transforms(self):
        from mesh_editor.viewport import MeshEditorViewport

        self.assertIsInstance(self.panel.viewport, DeepMeshFusionViewport)
        self.assertIsInstance(self.panel.viewport, MeshEditorViewport)
        self.assertFalse(self.panel.viewport.transform_enabled)
        self.assertFalse(self.panel.viewport.enable_gizmo)
        self.assertFalse(self.panel.viewport.picking_enabled)
        self.panel.viewport.set_transform_enabled(True)
        self.assertFalse(self.panel.viewport.transform_enabled)
        self.assertIsNone(self.panel.viewport._pick_object_at(10, 10))

    def test_dragging_diagnostics_splitter_expands_and_collapses_console(self):
        self.panel.resize(1000, 700); self.panel.show(); self.app.processEvents()
        self.panel.detail_splitter.setSizes((400, 220)); self.panel._diagnostics_splitter_moved(400, 1)
        self.assertFalse(self.panel.console.isHidden())
        self.assertTrue(self.panel.diagnostics_toggle.isChecked())
        self.panel.detail_splitter.setSizes((570, 44)); self.panel._diagnostics_splitter_moved(570, 1)
        self.assertTrue(self.panel.console.isHidden())
        self.assertFalse(self.panel.diagnostics_toggle.isChecked())

    def test_uses_proximap_charcoal_and_green_application_palette(self):
        self.assertIn("#121212", self.panel.styleSheet())
        self.assertIn("#00E676", self.panel.styleSheet())
        self.panel._set_action_style("ready")
        self.assertIn("#00E676", self.panel.action_button.styleSheet())
        self.assertEqual(self.panel.COLORS[0], "#00E676")

    def test_stage_navigation_is_locked_until_prerequisite_completes(self):
        self.panel._select_stage(2)
        self.assertEqual(self.panel.current_stage, 0)
        self.panel.unlocked_stage = 2; self.panel.completed_stages.update(("diagnostics", "alignment")); self.panel._update_state()
        self.panel._select_stage(2)
        self.assertEqual(self.panel.current_stage, 2)
        self.assertEqual(self.panel.action_button.text(), "Fuse Point Clouds")
        self.assertFalse(self.panel.preparation_panel.isVisible())
        self.assertFalse(self.panel.alignment_panel.isVisible())
        self.assertFalse(self.panel.fusion_panel.isHidden())

    def test_stage_specific_controls_and_registered_export_modes_exist(self):
        self.assertEqual(self.poisson_depth_range(), (5, 12))
        self.assertEqual(self.panel.poisson_depth.value(), 8)
        self.assertEqual([action.text() for action in self.panel.export_registered.menu().actions()],
                         ["Lossless registered PLY", "Voxel-downsampled PLY"])
        self.assertEqual(self.panel.cleanup_reduce_button.text(), "Cleanup & Reduce Mesh")
        self.assertEqual(self.panel.repair_nonmanifold_button.text(), "Repair Non-Manifold Edges")
        self.assertEqual(self.panel.close_holes_button.text(), "Close Holes")
        self.assertEqual(self.panel.merge_vertices_button.text(), "Merge Close Vertices")
        self.assertEqual(self.panel.smooth_button.text(), "Smooth Mesh")

    def poisson_depth_range(self):
        return self.panel.poisson_depth.minimum(), self.panel.poisson_depth.maximum()

    def test_primary_action_becomes_disabled_processing_indicator(self):
        self.panel.worker = SimpleNamespace(task="point_fusion")
        self.panel.processing = True; self.panel._update_state()
        self.assertEqual(self.panel.action_button.text(), "Fusing Point Clouds…")
        self.assertFalse(self.panel.action_button.isEnabled())
        self.assertIn("#FFB74D", self.panel.action_button.styleSheet())
        self.assertFalse(hasattr(self.panel, "status_badge"))

        self.panel.processing = False; self.panel.workspace = SimpleNamespace(); self.panel.current_stage = 2; self.panel.unlocked_stage = 2; self.panel.completed_stages.add("alignment"); self.panel._update_state()
        self.assertEqual(self.panel.action_button.text(), "Fuse Point Clouds")
        self.assertTrue(self.panel.action_button.isEnabled())
        self.assertIn("#00E676", self.panel.action_button.styleSheet())

    def test_current_atlas_setting_is_propagated_to_active_workspace(self):
        workspace = DeepMeshFusionWorkspace(str(self.root / "fusion_workspace"))
        self.assertEqual(workspace.config.texture_atlas_size, 2048)
        self.panel.workspace = workspace; self.panel.atlas.setValue(4096)
        self.panel._sync_appearance_config()
        manifest = json.loads((workspace.root / "workspace.json").read_text())
        self.assertEqual(workspace.config.texture_atlas_size, 4096)
        self.assertEqual(manifest["config"]["texture_atlas_size"], 4096)

    def test_restores_persisted_diagnostic_stage_and_parameters(self):
        first = self.root / "Recovered_01.ply"; second = self.root / "Recovered_02.ply"
        write_cloud(first); write_cloud(second, .03)
        workspace = DeepMeshFusionWorkspace(str(self.root / "deep_mesh_fusion_workspace"))
        workspace.add_pass(str(first), "Kitchen A"); workspace.add_pass(str(second), "Kitchen B"); workspace.analyze_passes()
        workspace.save_workflow_state({"current_stage": 1, "completed": ["diagnostics"], "cleanup_history": [],
                                       "parameters": {"voxel_size": .045, "poisson_depth": 9}})
        panel = DeepMeshFusionPanel(reconstruction_root=str(self.root))
        try:
            self.assertEqual(len(panel.scan_previews), 2)
            self.assertIn("diagnostics", panel.completed_stages)
            self.assertEqual(panel.current_stage, 1)
            self.assertEqual(panel.voxel.value(), .045)
            self.assertEqual(panel.poisson_depth.value(), 9)
        finally:
            panel.deleteLater()

    def test_dragging_ply_files_into_viewport_imports_and_filters_files(self):
        cloud = self.root / "Dragged_Pass.ply"; write_cloud(cloud)
        unsupported = self.root / "notes.txt"; unsupported.write_text("not a point cloud")
        mime = QMimeData(); mime.setUrls((QUrl.fromLocalFile(str(cloud)), QUrl.fromLocalFile(str(unsupported))))

        class DropEvent:
            accepted = False
            def mimeData(self): return mime
            def acceptProposedAction(self): self.accepted = True
            def ignore(self): self.accepted = False

        event = DropEvent(); self.panel.viewport.dropEvent(event)
        self.assertTrue(event.accepted)
        self.assertEqual(len(self.panel.scan_previews), 1)
        self.assertEqual(next(iter(self.panel.scan_previews.values())).source_path, str(cloud.resolve()))
        self.assertTrue(self.panel.viewport.acceptDrops())

    def test_analysis_applies_registration_transform_to_viewport_preview(self):
        cloud = self.root / "Rotated_Pass.ply"; write_cloud(cloud)
        self.panel._add_scan_paths((str(cloud),)); raw = self.panel.scan_previews["scan-1"].points.copy()
        transform = [[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]]
        registration = SimpleNamespace(accepted=True, transform=transform, method="fpfh-ransac+point-to-plane-icp", fitness=.82, message="accepted")
        self.panel.workspace = SimpleNamespace(passes=[SimpleNamespace(source_path=str(cloud), registration=registration)])
        self.panel._show_registered_scans()
        expected = raw @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]
        self.assertTrue(np.allclose(self.panel.viewport.layers["scan-1"]["points"], expected))
        self.assertIn("Registered · 82%", self.panel.scan_rows["scan-1"].file_label.text())

    def test_alignment_summary_uses_fitness_instead_of_acceptance_count(self):
        evidence = self.root / "evidence.json"; evidence.write_text('{"summary":{"overlap_ratio":0.65}}')
        reference = SimpleNamespace(method="reference", fitness=1.0, accepted=True)
        first = SimpleNamespace(method="fpfh-ransac+point-to-plane-icp", fitness=.70, accepted=True)
        second = SimpleNamespace(method="fpfh-ransac+point-to-plane-icp", fitness=.50, accepted=True)
        self.panel.workspace = SimpleNamespace(passes=[SimpleNamespace(registration=item) for item in (reference, first, second)])
        result = SimpleNamespace(evidence_map_path=str(evidence), conflict_region_count=30, region_count=100, mean_confidence=.67)
        self.panel._show_analysis(result)
        self.assertEqual(self.panel.metric_labels["Alignment"].text(), "Review · 60%")
        self.assertTrue(self.panel.analysis_requires_review)

    def test_fusion_result_displays_validated_solid_mesh_instead_of_points(self):
        mesh_path = self.root / "validated_lidar_surface.ply"
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)),
            o3d.utility.Vector3iVector(np.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=np.int32)),
        )
        self.assertTrue(o3d.io.write_triangle_mesh(str(mesh_path), mesh))
        result = SimpleNamespace(validated_mesh_path=str(mesh_path), fused_cloud_path=str(self.root / "unused.ply"), mean_confidence=.88)
        self.panel._show_fused_geometry(result)
        self.assertIsNotNone(self.panel.viewport.mesh_layer)
        self.assertEqual(len(self.panel.viewport.mesh_layer["faces"]), 4)
        self.assertNotIn("fused-geometry", self.panel.viewport.layers)
        self.assertEqual(self.panel.metric_labels["Confidence"].text(), "88%")


if __name__ == "__main__": unittest.main()
