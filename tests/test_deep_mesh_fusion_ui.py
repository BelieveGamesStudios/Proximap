import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import open3d as o3d
from PySide6.QtCore import QMimeData, Qt, QUrl
from PySide6.QtWidgets import QApplication, QDialog

from deep_mesh_fusion.ui import (
    DeepMeshFusionPanel, PairAlignmentDialog, PIPELINE_STAGES,
    _cloudcompare_style_icp,
)
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
        self.assertEqual(PIPELINE_STAGES[2], "Point removal")
        self.assertEqual(PIPELINE_STAGES[3], "Geometry reconstruction")
        self.assertNotIn("Surface fusion", PIPELINE_STAGES)
        self.assertIn("Cleanup", PIPELINE_STAGES)
        self.assertIn("Texture", PIPELINE_STAGES)
        self.assertEqual(PIPELINE_STAGES[-1], "Final quality")
        self.assertEqual(len(self.panel.stage_labels), 7)
        self.assertEqual(len(self.panel.stage_panels), 7)
        self.assertTrue(self.panel.stage_labels[0].isEnabled())
        self.assertFalse(self.panel.stage_labels[1].isEnabled())
        self.assertFalse(hasattr(self.panel, "stage_table"))
        self.assertFalse(hasattr(self.panel, "status_badge"))
        self.assertEqual(self.panel.action_button.text(), "Continue")
        self.assertFalse(self.panel.action_button.isEnabled())
        self.assertEqual(self.panel.new_project_button.text(), "New Project")

    def test_new_project_request_is_exposed_by_header_button(self):
        requests = []
        self.panel.new_project_requested.connect(lambda: requests.append(True))
        self.panel.new_project_button.click()
        self.assertEqual(requests, [True])

    def test_clear_project_state_removes_recovery_workspace_but_preserves_sources(self):
        first = self.root / "Pass_01.ply"; second = self.root / "Pass_02.ply"
        write_cloud(first); write_cloud(second, .05)
        self.panel._add_scan_paths((str(first), str(second)))
        workspace_root = Path(self.panel.workspace_edit.text())
        workspace = DeepMeshFusionWorkspace(str(workspace_root))
        workspace.add_pass(str(first), "Pass 1"); workspace.add_pass(str(second), "Pass 2")
        (workspace_root / "derived").mkdir(exist_ok=True)
        (workspace_root / "derived" / "recovered.ply").write_text("derived")
        self.panel.workspace = workspace

        self.panel.clear_project_state(remove_workspace=True)

        self.assertFalse(workspace_root.exists())
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        self.assertEqual(self.panel.scan_previews, {})
        self.assertEqual(self.panel.scan_rows, {})
        self.assertEqual(self.panel.current_stage, 0)
        self.assertEqual(self.panel.completed_stages, set())
        self.assertEqual(self.panel.scan_count_label.text(), "0 scans")
        self.assertTrue(self.panel.retry_alignment.isHidden())
        self.assertTrue(self.panel.translate_alignment.isHidden())
        self.assertTrue(self.panel.rotate_alignment.isHidden())
        self.assertTrue(self.panel.run_pair_icp.isHidden())

    def test_diagnostics_completion_opens_pair_chooser_after_stage_transition(self):
        first = self.root / "Pass_01.ply"; second = self.root / "Pass_02.ply"
        write_cloud(first); write_cloud(second, .05)
        self.panel._add_scan_paths((str(first), str(second)))
        workspace = DeepMeshFusionWorkspace(str(self.root / "diagnostics_workspace"))
        workspace.add_pass(str(first), "Pass 1"); workspace.add_pass(str(second), "Pass 2")
        diagnostics = workspace.analyze_passes()

        with patch("deep_mesh_fusion.ui.QTimer.singleShot") as single_shot:
            self.panel._on_task_complete("diagnostics", {"workspace": workspace, "passes": diagnostics})

        self.assertEqual(self.panel.current_stage, 1)
        self.assertFalse(self.panel.retry_alignment.isHidden())
        self.assertTrue(self.panel.retry_alignment.isEnabled())
        self.assertIn("choose the fixed and moving", self.panel.status_text.text())
        single_shot.assert_called_once_with(0, self.panel._run_alignment)

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

    def test_import_preserves_vertex_colors_for_alignment_viewport(self):
        path = self.root / "Colored_Pass.ply"
        points = np.asarray([(x * .01, y * .01, (x + y) * .002) for x in range(12) for y in range(10)], dtype=float)
        colors = np.asarray([(x / 11, y / 9, .25) for x in range(12) for y in range(10)], dtype=float)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))

        self.panel._add_scan_paths((str(path),))

        preview = self.panel.scan_previews["scan-1"]
        self.assertTrue(np.allclose(preview.vertex_colors, colors, atol=1 / 255))
        self.assertTrue(np.allclose(self.panel.viewport.layers["scan-1"]["vertex_colors"], colors, atol=1 / 255))

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

    def test_viewport_stays_locked_outside_an_explicit_pair_session(self):
        from mesh_editor.viewport import MeshEditorViewport

        self.assertIsInstance(self.panel.viewport, DeepMeshFusionViewport)
        self.assertIsInstance(self.panel.viewport, MeshEditorViewport)
        self.assertFalse(self.panel.viewport.transform_enabled)
        self.assertFalse(self.panel.viewport.enable_gizmo)
        self.assertFalse(self.panel.viewport.picking_enabled)
        self.panel.viewport.set_transform_enabled(True)
        self.assertFalse(self.panel.viewport.transform_enabled)
        self.assertIsNone(self.panel.viewport._pick_object_at(10, 10))

    def test_pair_session_isolates_scans_and_reuses_translate_rotate_gizmo(self):
        first = self.root / "Pair_01.ply"; second = self.root / "Pair_02.ply"; third = self.root / "Pair_03.ply"
        write_cloud(first); write_cloud(second, .03); write_cloud(third, .06)
        self.panel._add_scan_paths((str(first), str(second), str(third)))
        workspace = DeepMeshFusionWorkspace(str(self.root / "pair_workspace"))
        for path in (first, second, third): workspace.add_pass(str(path))
        workspace.analyze_passes(); self.panel.workspace = workspace; self.panel._initialize_alignment_state()

        self.panel._begin_pair_alignment("scan-1", "scan-3", .2, .05, .05)

        self.assertTrue(self.panel.viewport.transform_enabled)
        self.assertEqual(self.panel.viewport._alignment_proxy.layer_id, "scan-3")
        self.assertEqual(self.panel.viewport.gizmo.space, "global")
        self.assertTrue(self.panel.viewport.layers["scan-1"]["visible"])
        self.assertFalse(self.panel.viewport.layers["scan-2"]["visible"])
        self.assertTrue(self.panel.viewport.layers["scan-3"]["visible"])
        self.panel.rotate_alignment.click()
        self.assertEqual(self.panel.viewport.gizmo.operation, "rotate")
        self.panel.translate_alignment.click()
        self.assertEqual(self.panel.viewport.gizmo.operation, "translate")

        moved = np.eye(4); moved[:3, 3] = [1, 2, 3]
        self.panel.viewport.set_layer_transform("scan-3", moved)
        self.panel._alignment_transforms["scan-3"] = moved
        self.panel._cancel_pair_alignment()
        self.assertTrue(np.allclose(self.panel.viewport.layer_transform("scan-3"), np.eye(4)))
        self.assertFalse(self.panel.viewport.transform_enabled)
        self.assertTrue(all(self.panel.viewport.layers[key]["visible"] for key in ("scan-1", "scan-2", "scan-3")))

    def test_icp_pair_dialog_lists_primary_scan_in_both_slots(self):
        dialog = PairAlignmentDialog(
            [("scan-1", "Scan 01"), ("scan-2", "Scan 02")],
            "scan-1", (.2, .045, .05), self.panel,
        )
        try:
            self.assertEqual([dialog.fixed_combo.itemData(i) for i in range(dialog.fixed_combo.count())],
                             ["scan-1", "scan-2"])
            self.assertEqual([dialog.moving_combo.itemData(i) for i in range(dialog.moving_combo.count())],
                             ["scan-1", "scan-2"])
            self.assertEqual(dialog.fixed_combo.currentData(), "scan-1")
            self.assertEqual(dialog.moving_combo.currentData(), "scan-2")
            self.assertEqual(dialog.rms_decrease.value(), 1e-5)
            self.assertEqual(dialog.final_overlap.value(), 1.0)
            self.assertEqual(dialog.max_iterations.value(), 20)
            self.assertEqual(dialog.sampling_limit.value(), 50_000)
        finally:
            dialog.deleteLater()

    def test_first_fixed_choice_is_respected_and_becomes_batch_reference(self):
        first = self.root / "Pair_01.ply"; second = self.root / "Pair_02.ply"
        write_cloud(first); write_cloud(second, .03)
        self.panel._add_scan_paths((str(first), str(second)))
        workspace = DeepMeshFusionWorkspace(str(self.root / "fixed_choice_workspace"))
        workspace.add_pass(str(first), "Scan 01"); workspace.add_pass(str(second), "Scan 02")
        workspace.analyze_passes(); self.panel.workspace = workspace
        self.panel._initialize_alignment_state()

        selection = ("scan-2", "scan-1", .2, .05, .05, 1e-5, .15, 50, 50_000)
        with patch("deep_mesh_fusion.ui.PairAlignmentDialog") as dialog_class:
            dialog = dialog_class.return_value
            dialog.exec.return_value = QDialog.Accepted
            dialog.selection.return_value = selection
            self.panel._run_alignment()

        self.assertEqual(self.panel._alignment_reference_scan_id, "scan-2")
        self.assertEqual(self.panel._active_alignment_pair["fixed"], "scan-2")
        self.assertEqual(self.panel._active_alignment_pair["moving"], "scan-1")
        self.assertEqual(self.panel.viewport._alignment_proxy.layer_id, "scan-1")
        self.assertIn("Batch reference: Scan 02", self.panel.auto_reference.text())
        self.assertIn("Fixed: Scan 02", self.panel.alignment_summary.text())
        self.assertIn("Moving: Scan 01", self.panel.alignment_summary.text())

    def test_cloudcompare_style_icp_recovers_rigid_translation_without_scaling(self):
        rng = np.random.default_rng(11)
        target = rng.uniform(-1, 1, size=(4000, 3))
        offset = np.asarray([.025, -.018, .012])
        source = target + offset

        result = _cloudcompare_style_icp(
            source, target, final_overlap=1.0, min_rms_decrease=1e-7,
            max_iterations=40, sampling_limit=4000,
        )

        self.assertLess(result["final_rms"], result["baseline_rms"])
        self.assertTrue(np.allclose(result["transform"][:3, 3], -offset, atol=1e-4))
        self.assertTrue(np.allclose(result["transform"][:3, :3].T @ result["transform"][:3, :3], np.eye(3), atol=1e-8))
        self.assertAlmostEqual(np.linalg.det(result["transform"][:3, :3]), 1.0, places=8)

    def test_cloudcompare_style_icp_keeps_an_already_exact_manual_pose(self):
        rng = np.random.default_rng(13)
        points = rng.normal(size=(1000, 3))

        result = _cloudcompare_style_icp(points, points, min_rms_decrease=1e-5)

        self.assertTrue(result["manual_pose_retained"])
        self.assertTrue(np.allclose(result["transform"], np.eye(4)))
        self.assertAlmostEqual(result["baseline_rms"], result["final_rms"])

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
        self.panel.unlocked_stage = 3; self.panel.completed_stages.update(("diagnostics", "alignment", "point_removal")); self.panel._update_state()
        self.panel._select_stage(3)
        self.assertEqual(self.panel.current_stage, 3)
        self.assertEqual(self.panel.action_button.text(), "Reconstruct Geometry")
        self.assertFalse(self.panel.preparation_panel.isVisible())
        self.assertFalse(self.panel.alignment_panel.isVisible())
        self.assertTrue(self.panel.fusion_panel.isHidden())
        self.assertFalse(self.panel.validation_panel.isHidden())

    def test_alignment_completion_advances_to_point_removal(self):
        reference = SimpleNamespace(enabled=True, source_path="reference.ply", registration=SimpleNamespace(accepted=True, method="reference", fitness=1.0))
        aligned = SimpleNamespace(enabled=True, source_path="aligned.ply", registration=SimpleNamespace(accepted=True, method="pymeshlab-pairwise-icp", fitness=.8))
        self.panel.workspace = SimpleNamespace(passes=[reference, aligned])
        self.panel.current_stage = 1; self.panel.unlocked_stage = 1

        with patch.object(self.panel, "_prepare_point_removal"):
            self.panel._on_task_complete("alignment", [])

        self.assertIn("alignment", self.panel.completed_stages)
        self.assertEqual(self.panel.current_stage, 2)
        self.assertEqual(self.panel.stage_counter.text(), "Stage 3 of 7")
        self.assertEqual(self.panel.action_button.text(), "Continue to Reconstruction")
        self.assertFalse(self.panel.point_removal_panel.isHidden())
        self.assertTrue(self.panel.validation_panel.isHidden())

    def test_point_removal_stage_exposes_auto_rectangle_lasso_and_delete_controls(self):
        self.assertEqual(self.panel.auto_remove_overlap.text(), "Auto Remove Existing Areas")
        self.assertEqual(self.panel.rectangle_select.text(), "Rectangle")
        self.assertEqual(self.panel.lasso_select.text(), "Lasso")
        self.assertEqual(self.panel.stop_point_selection.text(), "Stop Selection / View")
        self.assertEqual(self.panel.remove_selected_points.text(), "Remove Selected Points")
        self.assertGreater(self.panel.removal_distance.maximum(), self.panel.removal_distance.minimum())

    def test_point_selection_can_be_disabled_by_toggle_button_and_escape(self):
        self.panel.rectangle_select.click()
        self.assertEqual(self.panel.removal_selection_overlay.get_mode(), "box")
        self.assertTrue(self.panel.rectangle_select.isChecked())
        self.assertFalse(self.panel.removal_selection_overlay.isHidden())

        self.panel.rectangle_select.click()
        self.assertEqual(self.panel.removal_selection_overlay.get_mode(), "none")
        self.assertFalse(self.panel.rectangle_select.isChecked())
        self.assertTrue(self.panel.removal_selection_overlay.isHidden())

        self.panel.lasso_select.click()
        self.assertEqual(self.panel.removal_selection_overlay.get_mode(), "lasso")
        self.panel.stop_removal_selection_shortcut.activated.emit()
        self.assertEqual(self.panel.removal_selection_overlay.get_mode(), "none")
        self.assertFalse(self.panel.lasso_select.isChecked())
        self.assertIn("camera controls restored", self.panel.status_text.text())

    def test_point_projection_uses_the_renderers_row_vector_matrix_convention(self):
        view = np.eye(4, dtype=float); view[3, 0] = .5
        with patch.object(self.panel.viewport.camera, "get_view_matrix", return_value=view), \
             patch.object(self.panel.viewport.camera, "get_projection_matrix", return_value=np.eye(4)):
            screen, visible = self.panel.viewport.project_points(np.asarray([[0, 0, 0]], dtype=float))
        self.assertTrue(visible[0])
        self.assertAlmostEqual(screen[0, 0], self.panel.viewport.width() * .75)
        self.assertAlmostEqual(screen[0, 1], self.panel.viewport.height() * .5)

    def test_auto_removal_discards_only_secondary_points_near_primary(self):
        first = self.root / "Reference.ply"; second = self.root / "Secondary.ply"
        write_cloud(first); write_cloud(second, .05); self.panel._add_scan_paths((str(first), str(second)))
        reference = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        secondary = np.asarray([[.01, 0, 0], [1.02, 0, 0], [3, 0, 0]], dtype=np.float32)
        colors = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.panel._alignment_reference_scan_id = "scan-1"
        self.panel._removal_clouds = {
            "scan-1": {"points": reference, "colors": None},
            "scan-2": {"points": secondary.copy(), "colors": colors.copy()},
        }
        self.panel._removal_originals = {
            key: {"points": value["points"].copy(), "colors": None if value["colors"] is None else value["colors"].copy()}
            for key, value in self.panel._removal_clouds.items()
        }
        self.panel.removal_cloud_combo.clear()
        self.panel.removal_cloud_combo.addItem("Reference", "scan-1")
        self.panel.removal_cloud_combo.addItem("Secondary", "scan-2")
        self.panel.removal_cloud_combo.setCurrentIndex(1)
        self.panel.removal_distance.setValue(.03)

        self.panel._auto_remove_existing_areas()

        self.assertTrue(np.allclose(self.panel._removal_clouds["scan-1"]["points"], reference))
        self.assertTrue(np.allclose(self.panel._removal_clouds["scan-2"]["points"], [[3, 0, 0]]))
        self.assertTrue(np.allclose(self.panel._removal_clouds["scan-2"]["colors"], [[0, 0, 1]]))

    def test_rectangle_selection_and_delete_edit_only_active_cloud(self):
        first = self.root / "Reference.ply"; second = self.root / "Secondary.ply"
        write_cloud(first); write_cloud(second, .05); self.panel._add_scan_paths((str(first), str(second)))
        points = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        self.panel._removal_clouds = {"scan-2": {"points": points.copy(), "colors": None}}
        self.panel._removal_originals = {"scan-2": {"points": points.copy(), "colors": None}}
        self.panel.removal_cloud_combo.clear(); self.panel.removal_cloud_combo.addItem("Secondary", "scan-2")
        projected = np.asarray([[10, 10], [20, 20], [50, 50]], dtype=float)
        with patch.object(self.panel.viewport, "project_points", return_value=(projected, np.ones(3, dtype=bool))):
            self.panel._on_removal_shape(("box", (5, 5, 25, 25)))
        self.assertEqual(self.panel._removal_selected_indices.tolist(), [0, 1])

        self.panel._remove_selected_points()

        self.assertTrue(np.allclose(self.panel._removal_clouds["scan-2"]["points"], [[2, 0, 0]]))
        self.assertEqual(len(self.panel._removal_selected_indices), 0)

    def test_alignment_with_only_reference_accepted_blocks_geometry(self):
        reference = SimpleNamespace(enabled=True, source_path="reference.ply", registration=SimpleNamespace(accepted=True, method="reference", fitness=1.0))
        rejected = SimpleNamespace(enabled=True, source_path="rejected.ply", registration=SimpleNamespace(accepted=False, method="pymeshlab-pairwise-icp", fitness=0.0))
        self.panel.workspace = SimpleNamespace(passes=[reference, rejected])
        self.panel.current_stage = 1; self.panel.unlocked_stage = 1

        with patch("deep_mesh_fusion.ui.QMessageBox.warning") as warning:
            self.panel._on_task_complete("alignment", [])

        self.assertNotIn("alignment", self.panel.completed_stages)
        self.assertEqual(self.panel.current_stage, 1)
        self.assertEqual(self.panel.unlocked_stage, 1)
        self.assertEqual(self.panel.action_button.text(), "Confirm All Alignments")
        self.assertIn("fewer than two", self.panel.status_text.text())
        warning.assert_called_once()

    def test_stage_specific_controls_and_registered_export_modes_exist(self):
        self.assertEqual(self.poisson_depth_range(), (5, 12))
        self.assertEqual(self.panel.poisson_depth.value(), 8)
        self.assertEqual(self.panel.normal_neighbors.value(), 30)
        self.assertEqual(self.panel.poisson_samples.value(), 3.0)
        self.assertEqual(self.panel.surface_support.value(), 2.0)
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
        self.panel.worker = SimpleNamespace(task="surface")
        self.panel.processing = True; self.panel._update_state()
        self.assertEqual(self.panel.action_button.text(), "Reconstructing Surface…")
        self.assertFalse(self.panel.action_button.isEnabled())
        self.assertIn("#FFB74D", self.panel.action_button.styleSheet())
        self.assertFalse(hasattr(self.panel, "status_badge"))

        self.panel.processing = False; self.panel.workspace = SimpleNamespace(); self.panel.current_stage = 3; self.panel.unlocked_stage = 3; self.panel.completed_stages.update(("alignment", "point_removal")); self.panel._update_state()
        self.assertEqual(self.panel.action_button.text(), "Reconstruct Geometry")
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
                                       "parameters": {"voxel_size": .045, "poisson_depth": 9,
                                                      "reconstruction_backend": "pymeshlab"}})
        panel = DeepMeshFusionPanel(reconstruction_root=str(self.root))
        try:
            self.assertEqual(len(panel.scan_previews), 2)
            self.assertIn("diagnostics", panel.completed_stages)
            self.assertEqual(panel.current_stage, 1)
            self.assertEqual(panel.voxel.value(), .045)
            self.assertEqual(panel.poisson_depth.value(), 9)
            self.assertEqual(panel.reconstruction_backend.currentData(), "pymeshlab")
            self.assertFalse(panel.retry_alignment.isHidden())
            self.assertTrue(panel.retry_alignment.isEnabled())
            self.assertEqual(panel.retry_alignment.text(), "Choose ICP Pair…")
            self.assertIn("choose an ICP pair", panel.status_text.text())
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
