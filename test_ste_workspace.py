"""
test_ste_workspace.py
=====================

Comprehensive test suite for the Spatial Texture Engine (STE) Unified Alignment Workspace.
Verifies the unified CloudCompare-style shared viewport, in-viewport control point picking,
non-destructive alignment preview, Points/Surface/Baked display modes, workflow state machine,
derived data invalidation cascades, and source asset preservation.
"""

import os
import unittest
import numpy as np
import open3d as o3d
import pyrr
from PIL import Image

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton

from ste_alignment import STEAlignmentResult, STEControlPointManager
from ste_texture_baking import TextureBakeResult
from ste_workspace import (
    STEWorkspace,
    STEWorkflowState,
    STELiDARRepresentation,
    STEUnifiedAlignmentViewport
)


# Ensure QApplication instance exists for Qt widget tests
app = QApplication.instance()
if app is None:
    app = QApplication([])


def create_mock_scene_data():
    """Generates synthetic photogrammetry and transformed LiDAR datasets."""
    # Photogrammetry planar surface mesh
    verts_photo = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.5]
    ], dtype=np.float64)

    tris_photo = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ], dtype=np.int32)

    uvs_photo = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, 0.5],
        [1.0, 0.0], [1.0, 1.0], [0.5, 0.5],
        [1.0, 1.0], [0.0, 1.0], [0.5, 0.5],
        [0.0, 1.0], [0.0, 0.0], [0.5, 0.5]
    ], dtype=np.float64)

    tex_photo = np.full((128, 128, 3), 180, dtype=np.uint8)

    # Scale = 7.0, Rotation around Z = 45 deg, Translation = [10, 20, 5]
    scale_true = 7.0
    theta = np.radians(45.0)
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ], dtype=np.float64)
    t_true = np.array([10.0, 20.0, 5.0], dtype=np.float64)

    p_photo = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.5],
        [0.0, 2.0, 0.0]
    ], dtype=np.float64)

    p_lidar = ((p_photo - t_true) @ R_true) / scale_true

    return verts_photo, tris_photo, uvs_photo, tex_photo, p_photo, p_lidar, scale_true, R_true, t_true


class TestSTEUnifiedWorkspace(unittest.TestCase):

    def setUp(self):
        self.ws = STEWorkspace()

    def test_a_unified_viewport_initialization(self):
        """TEST A: STE creates a single unified shared 3D alignment viewport."""
        self.assertIsInstance(self.ws.viewport, STEUnifiedAlignmentViewport)
        self.assertEqual(self.ws.state, STEWorkflowState.NO_DATA)
        self.assertTrue(self.ws.viewport.photo_visible)
        self.assertTrue(self.ws.viewport.lidar_visible)
        self.assertEqual(self.ws.viewport.lidar_display_mode, STELiDARRepresentation.POINTS)

    def test_b_attach_photogrammetry(self):
        """TEST B: Photogrammetry dataset can be attached to unified viewport."""
        verts, tris, uvs, tex, _, _, _, _, _ = create_mock_scene_data()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)

        self.ws.photo_verts = verts
        self.ws.photo_tris = tris
        self.ws.photo_uvs = uvs
        self.ws.photo_texture_img = tex
        self.ws.viewport.set_photogrammetry_data(mesh, tex)
        self.ws._update_workflow_state()

        self.assertEqual(self.ws.state, STEWorkflowState.PHOTOGRAMMETRY_READY)
        self.assertIsNotNone(self.ws.photo_verts)
        self.assertTrue(self.ws.viewport.visual_photo_cloud.visible)
        self.assertFalse(self.ws.viewport.visual_photo_mesh.visible)

    def test_c_attach_lidar(self):
        """TEST C: LiDAR point cloud can be attached to unified viewport."""
        verts, _, _, _, _, _, _, _, _ = create_mock_scene_data()
        self.ws.lidar_source_points = verts
        self.ws.viewport.set_lidar_points(verts)
        self.ws._update_workflow_state()

        self.assertEqual(self.ws.state, STEWorkflowState.LIDAR_READY)
        self.assertIsNotNone(self.ws.lidar_source_points)
        self.assertTrue(self.ws.viewport.visual_lidar_points.visible)

    def test_d_datasets_coexist_in_unified_viewport(self):
        """TEST D: Photogrammetry point cloud and LiDAR point cloud coexist in the same shared coordinate space."""
        verts, tris, uvs, tex, p_photo, p_lidar, _, _, _ = create_mock_scene_data()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)

        self.ws.photo_verts = verts
        self.ws.photo_tris = tris
        self.ws.lidar_source_points = p_lidar

        self.ws.viewport.set_photogrammetry_data(mesh)
        self.ws.viewport.set_lidar_points(p_lidar)
        self.ws._update_workflow_state()

        self.assertEqual(self.ws.state, STEWorkflowState.CONTROL_POINTS_READY)
        self.assertTrue(self.ws.viewport.visual_photo_cloud.visible)
        self.assertFalse(self.ws.viewport.visual_photo_mesh.visible)
        self.assertTrue(self.ws.viewport.visual_lidar_points.visible)

    def test_e_independent_dataset_visibility(self):
        """TEST E: Viewport allows independent visibility toggling for each dataset."""
        self.test_d_datasets_coexist_in_unified_viewport()

        # Hide Photogrammetry
        self.ws.chk_photo_vis.setChecked(False)
        self.assertFalse(self.ws.viewport.visual_photo_cloud.visible)
        self.assertTrue(self.ws.viewport.visual_lidar_points.visible)

        # Hide LiDAR
        self.ws.chk_lidar_vis.setChecked(False)
        self.assertFalse(self.ws.viewport.visual_photo_cloud.visible)
        self.assertFalse(self.ws.viewport.visual_lidar_points.visible)

        # Restore both
        self.ws.chk_photo_vis.setChecked(True)
        self.ws.chk_lidar_vis.setChecked(True)
        self.assertTrue(self.ws.viewport.visual_photo_cloud.visible)
        self.assertTrue(self.ws.viewport.visual_lidar_points.visible)

    def test_f_in_viewport_control_point_placement(self):
        """TEST F: Control points can be picked directly in the unified viewport."""
        verts, tris, uvs, tex, p_photo, p_lidar, _, _, _ = create_mock_scene_data()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)

        self.ws.photo_verts = verts
        self.ws.photo_tris = tris
        self.ws.lidar_source_points = p_lidar
        self.ws.viewport.set_photogrammetry_data(mesh)
        self.ws.viewport.set_lidar_points(p_lidar)

        # Add CP1 and pick coordinates via signal
        self.ws._add_control_point()
        self.assertEqual(self.ws.active_cp_id, "CP1")

        self.ws.viewport.point_picked.emit("photo", p_photo[0])
        self.ws.viewport.point_picked.emit("lidar", p_lidar[0])

        cp1 = self.ws.cp_manager._points["CP1"]
        self.assertTrue(cp1.is_complete)
        np.testing.assert_allclose(cp1.photo_pos, p_photo[0])
        np.testing.assert_allclose(cp1.lidar_pos, p_lidar[0])

    def test_g_valid_control_points_enable_alignment(self):
        """TEST G: 3 complete control points solve CCCoreLib scale-aware alignment."""
        verts, tris, uvs, tex, p_photo, p_lidar, scale_true, _, _ = create_mock_scene_data()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)

        self.ws.photo_verts = verts
        self.ws.photo_tris = tris
        self.ws.lidar_source_points = p_lidar
        self.ws.viewport.set_photogrammetry_data(mesh)
        self.ws.viewport.set_lidar_points(p_lidar)

        for i in range(3):
            self.ws._add_control_point()
            self.ws.cp_manager.set_photo_marker(f"CP{i+1}", p_photo[i])
            self.ws.cp_manager.set_lidar_marker(f"CP{i+1}", p_lidar[i])

        self.ws._auto_solve_alignment_if_ready()
        self.ws._update_workflow_state()

        self.assertIsNotNone(self.ws.alignment_result)
        self.assertTrue(self.ws.alignment_result.success)
        self.assertAlmostEqual(self.ws.alignment_result.scale, scale_true, places=2)
        self.assertAlmostEqual(self.ws.alignment_result.rms_error, 0.0, places=4)

    def test_h_alignment_preview_non_destructive(self):
        """TEST H: Alignment preview transforms rendering matrix without modifying source geometry."""
        self.test_g_valid_control_points_enable_alignment()

        p_lidar_original = self.ws.lidar_source_points.copy()

        # Activate Preview
        self.ws._toggle_alignment_preview()
        self.assertTrue(self.ws.preview_active)
        self.assertFalse(np.allclose(self.ws.viewport._preview_matrix, np.eye(4)))

        # Verify source data unmodified
        np.testing.assert_allclose(self.ws.lidar_source_points, p_lidar_original)

    def test_i_reset_alignment_restores_native_state(self):
        """TEST I: Reset alignment immediately returns LiDAR to native coordinate system."""
        self.test_h_alignment_preview_non_destructive()

        self.ws._reset_alignment()
        self.assertFalse(self.ws.preview_active)
        self.assertIsNone(self.ws.alignment_result)
        np.testing.assert_allclose(self.ws.viewport._preview_matrix, np.eye(4))

    def test_j_points_surface_baked_display_modes(self):
        """TEST J: Points / Surface / Baked modes switch representations within the same viewport."""
        verts, tris, _, _, _, _, _, _, _ = create_mock_scene_data()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)

        self.ws.lidar_source_points = verts
        self.ws.viewport.set_lidar_points(verts)

        # 1. Points Mode
        self.ws._set_lidar_display_mode(STELiDARRepresentation.POINTS)
        self.assertTrue(self.ws.viewport.visual_lidar_points.visible)
        self.assertFalse(self.ws.viewport.visual_lidar_surface.visible)
        self.assertFalse(self.ws.viewport.visual_lidar_baked.visible)

        # 2. Surface Mode
        self.ws.viewport.set_lidar_surface(mesh)
        self.ws._set_lidar_display_mode(STELiDARRepresentation.SURFACE)
        self.assertFalse(self.ws.viewport.visual_lidar_points.visible)
        self.assertTrue(self.ws.viewport.visual_lidar_surface.visible)
        self.assertFalse(self.ws.viewport.visual_lidar_baked.visible)

        # 3. Baked Mode
        self.ws.viewport.set_lidar_baked(mesh)
        self.ws._set_lidar_display_mode(STELiDARRepresentation.BAKED)
        self.assertFalse(self.ws.viewport.visual_lidar_points.visible)
        self.assertFalse(self.ws.viewport.visual_lidar_surface.visible)
        self.assertTrue(self.ws.viewport.visual_lidar_baked.visible)

    def test_k_replacing_lidar_invalidates_derived_stages(self):
        """TEST K: Replacing LiDAR point cloud invalidates all downstream derived assets."""
        verts, tris, _, _, _, _, _, _, _ = create_mock_scene_data()
        self.ws.lidar_surface_verts = verts
        self.ws.lidar_surface_tris = tris
        self.ws.lidar_target_uvs = np.zeros((len(tris)*3, 2))
        self.ws._invalidate_lidar_derived_stages()

        self.assertIsNone(self.ws.lidar_surface_verts)
        self.assertIsNone(self.ws.lidar_target_uvs)
        self.assertIsNone(self.ws.projection_result)
        self.assertIsNone(self.ws.bake_result)

    def test_l_source_assets_preserved_throughout(self):
        """TEST L: Original Photogrammetry & LiDAR source geometries remain 100% untouched."""
        verts_photo, tris_photo, uvs_photo, tex_photo, p_photo, p_lidar, _, _, _ = create_mock_scene_data()
        self.ws.photo_verts = verts_photo.copy()
        self.ws.photo_tris = tris_photo.copy()
        self.ws.photo_uvs = uvs_photo.copy()
        self.ws.photo_texture_img = tex_photo.copy()
        self.ws.lidar_source_points = p_lidar.copy()

        # Add CPs and solve
        for i in range(3):
            self.ws._add_control_point()
            self.ws.cp_manager.set_photo_marker(f"CP{i+1}", p_photo[i])
            self.ws.cp_manager.set_lidar_marker(f"CP{i+1}", p_lidar[i])

        self.ws._solve_alignment()
        self.ws._toggle_alignment_preview()

        # Verify bitwise preservation
        np.testing.assert_allclose(self.ws.photo_verts, verts_photo)
        np.testing.assert_allclose(self.ws.photo_tris, tris_photo)
        np.testing.assert_allclose(self.ws.lidar_source_points, p_lidar)

    def test_m_ste_mode_disables_transformation_tools(self):
        """TEST M: STE mode disables transformation gizmos while Mesh Editor mode retains them."""
        from mesh_editor.viewport import MeshEditorViewport

        # 1. Standard Mesh Editor viewport retains transform tools
        me_viewport = MeshEditorViewport()
        self.assertTrue(me_viewport.transform_enabled)
        self.assertTrue(me_viewport.enable_gizmo)

        # 2. STE viewport explicitly disables transform gizmos
        ste_viewport = self.ws.viewport
        self.assertFalse(ste_viewport.transform_enabled)
        self.assertFalse(ste_viewport.enable_gizmo)

        # 3. Setting transform mode dynamically works
        me_viewport.set_transform_enabled(False)
        self.assertFalse(me_viewport.transform_enabled)
        self.assertFalse(me_viewport.enable_gizmo)

        me_viewport.set_transform_enabled(True)
        self.assertTrue(me_viewport.transform_enabled)
        self.assertTrue(me_viewport.enable_gizmo)

    def test_n_in_row_control_point_buttons(self):
        """TEST N: Control points table provides in-cell pick buttons and inline deletion."""
        self.ws._add_control_point()
        self.assertEqual(self.ws.table_cps.rowCount(), 1)
        self.assertEqual(self.ws.table_cps.columnCount(), 5)

        # Verify cell widgets exist
        btn_photo = self.ws.table_cps.cellWidget(0, 1)
        btn_lidar = self.ws.table_cps.cellWidget(0, 2)
        btn_del = self.ws.table_cps.cellWidget(0, 4)

        self.assertIsInstance(btn_photo, QPushButton)
        self.assertIsInstance(btn_lidar, QPushButton)
        self.assertIsInstance(btn_del, QPushButton)
        self.assertEqual(btn_photo.text(), "Pick Photogrammetry")
        self.assertEqual(btn_lidar.text(), "Pick LiDAR")
        self.assertEqual(btn_del.text(), "✕")

        # Clicking in-row Photo button activates picking
        btn_photo.click()
        self.assertEqual(self.ws.viewport.picking_target, "photo")
        self.assertEqual(self.ws.viewport.active_cp_id, "CP1")

        # Placing point updates button text to coordinates
        self.ws.viewport.point_picked.emit("photo", np.array([1.23, 4.56, 7.89]))
        btn_photo_updated = self.ws.table_cps.cellWidget(0, 1)
        self.assertIn("1.23", btn_photo_updated.text())

        # Inline delete removes the point
        btn_del_updated = self.ws.table_cps.cellWidget(0, 4)
        btn_del_updated.click()
        self.assertEqual(self.ws.table_cps.rowCount(), 0)
        self.assertEqual(self.ws.cp_manager.count, 0)

    def test_o_repick_and_replace_control_point_coordinates(self):
        """TEST O: Clicking placed value buttons activates repicking/replacement mode and toggles."""
        self.ws._add_control_point()
        # Place initial photo coordinate
        self.ws.viewport.point_picked.emit("photo", np.array([1.0, 2.0, 3.0]))
        btn_photo = self.ws.table_cps.cellWidget(0, 1)
        self.assertIn("1.00", btn_photo.text())

        # Clicking the placed value button activates repicking
        btn_photo.click()
        self.assertEqual(self.ws.viewport.picking_target, "photo")
        self.assertEqual(self.ws.viewport.active_cp_id, "CP1")
        btn_photo_repick = self.ws.table_cps.cellWidget(0, 1)
        self.assertIn("Repick Photo", btn_photo_repick.text())

        # Clicking the repick button again cancels/toggles picking mode
        btn_photo_repick.click()
        self.assertIsNone(self.ws.viewport.picking_target)
        btn_photo_restored = self.ws.table_cps.cellWidget(0, 1)
        self.assertIn("1.00", btn_photo_restored.text())

        # Re-activate and pick replacement point
        btn_photo_restored.click()
        self.assertEqual(self.ws.viewport.picking_target, "photo")
        self.ws.viewport.point_picked.emit("photo", np.array([9.5, 8.5, 7.5]))
        self.assertIsNone(self.ws.viewport.picking_target)
        btn_photo_new = self.ws.table_cps.cellWidget(0, 1)
        self.assertIn("9.50", btn_photo_new.text())
        np.testing.assert_allclose(self.ws.cp_manager.get_point("CP1").photo_pos, np.array([9.5, 8.5, 7.5]))

        # Test LiDAR repicking via cell click
        self.ws.viewport.point_picked.emit("lidar", np.array([0.1, 0.2, 0.3]))
        self.ws._on_table_cell_clicked(0, 2)
        self.assertEqual(self.ws.viewport.picking_target, "lidar")
        btn_lidar_repick = self.ws.table_cps.cellWidget(0, 2)
        self.assertIn("Repick LiDAR", btn_lidar_repick.text())

        # Replace LiDAR point
        self.ws.viewport.point_picked.emit("lidar", np.array([0.4, 0.5, 0.6]))
        self.assertIsNone(self.ws.viewport.picking_target)
        btn_lidar_new = self.ws.table_cps.cellWidget(0, 2)
        self.assertIn("0.40", btn_lidar_new.text())
        np.testing.assert_allclose(self.ws.cp_manager.get_point("CP1").lidar_pos, np.array([0.4, 0.5, 0.6]))

    def test_p_photogrammetry_alignment_cloud_and_texture_source_distinction(self):
        """TEST P: Photogrammetry alignment geometry is exposed as point cloud, textured mesh remains texture source."""
        # Create test textured mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32))
        mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64))
        tex_img = np.full((64, 64, 3), 255, dtype=np.uint8)

        # Separate dense point cloud (50 points)
        dense_pts = np.random.randn(50, 3).astype(np.float64)

        self.ws.viewport.set_photogrammetry_data(mesh, tex_img, point_cloud=dense_pts)
        self.ws.photo_mesh = mesh
        self.ws.photo_verts = np.asarray(mesh.vertices)
        self.ws.photo_uvs = np.asarray(mesh.triangle_uvs)
        self.ws.photo_texture_img = tex_img
        self.ws.photo_alignment_cloud = dense_pts

        # Verify alignment point cloud is used for alignment candidates
        self.assertEqual(len(self.ws.photo_alignment_cloud), 50)
        self.assertEqual(len(self.ws.viewport._photo_alignment_cloud), 50)

        # Verify textured mesh is preserved as texture source
        self.assertIsNotNone(self.ws.photo_mesh)
        self.assertEqual(len(self.ws.photo_verts), 3)
        self.assertEqual(len(self.ws.photo_uvs), 3)
        self.assertIsNotNone(self.ws.photo_texture_img)

    def test_q_control_point_markers_visualization_lifecycle(self):
        """TEST Q: CP1 and CP2 create distinct P1/L1 and P2/L2 visual markers with complete lifecycle."""
        # 1. Create CP1 and pick coordinates
        self.ws._add_control_point()
        p1 = np.array([1.0, 2.0, 3.0])
        l1 = np.array([10.0, 20.0, 30.0])
        self.ws.viewport.point_picked.emit("photo", p1)
        self.ws.viewport.point_picked.emit("lidar", l1)

        # Verify markers dictionary in viewport
        self.assertIn("CP1", self.ws.viewport._photo_markers)
        self.assertIn("CP1", self.ws.viewport._lidar_markers)
        np.testing.assert_allclose(self.ws.viewport._photo_markers["CP1"], p1)
        np.testing.assert_allclose(self.ws.viewport._lidar_markers["CP1"], l1)

        # 2. Create CP2 and pick coordinates
        self.ws._add_control_point()
        p2 = np.array([4.0, 5.0, 6.0])
        l2 = np.array([40.0, 50.0, 60.0])
        self.ws.viewport.point_picked.emit("photo", p2)
        self.ws.viewport.point_picked.emit("lidar", l2)

        self.assertIn("CP2", self.ws.viewport._photo_markers)
        self.assertIn("CP2", self.ws.viewport._lidar_markers)
        np.testing.assert_allclose(self.ws.viewport._photo_markers["CP2"], p2)
        np.testing.assert_allclose(self.ws.viewport._lidar_markers["CP2"], l2)

        # 3. Re-pick P1 replaces old P1
        p1_new = np.array([1.5, 2.5, 3.5])
        self.ws._set_picking_target("photo", "CP1")
        self.ws.viewport.point_picked.emit("photo", p1_new)
        np.testing.assert_allclose(self.ws.viewport._photo_markers["CP1"], p1_new)

        # 4. Delete CP1 removes both P1 and L1 markers
        self.ws._remove_control_point_by_id("CP1")
        self.assertNotIn("CP1", self.ws.viewport._photo_markers)
        self.assertNotIn("CP1", self.ws.viewport._lidar_markers)
        self.assertIn("CP2", self.ws.viewport._photo_markers)
        self.assertIn("CP2", self.ws.viewport._lidar_markers)

        # 5. Clear all removes all markers
        self.ws.cp_manager.clear()
        self.ws._refresh_control_points_table()
        self.assertEqual(len(self.ws.viewport._photo_markers), 0)
        self.assertEqual(len(self.ws.viewport._lidar_markers), 0)

    def test_r_non_destructive_datasets_during_alignment(self):
        """TEST R: Source photogrammetry mesh, UVs, texture, and LiDAR points are unmodified by alignment and preview."""
        photo_verts_orig = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        photo_uvs_orig = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        lidar_pts_orig = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 7.0, 0.0]], dtype=np.float64)

        self.ws.photo_verts = photo_verts_orig.copy()
        self.ws.photo_uvs = photo_uvs_orig.copy()
        self.ws.lidar_source_points = lidar_pts_orig.copy()
        self.ws.viewport.set_lidar_points(lidar_pts_orig)

        # Add 3 control points (1:7 scale)
        cp1 = self.ws.cp_manager.create_control_point("CP1")
        self.ws.cp_manager.set_photo_marker("CP1", np.array([0.0, 0.0, 0.0]))
        self.ws.cp_manager.set_lidar_marker("CP1", np.array([0.0, 0.0, 0.0]))

        cp2 = self.ws.cp_manager.create_control_point("CP2")
        self.ws.cp_manager.set_photo_marker("CP2", np.array([1.0, 0.0, 0.0]))
        self.ws.cp_manager.set_lidar_marker("CP2", np.array([7.0, 0.0, 0.0]))

        cp3 = self.ws.cp_manager.create_control_point("CP3")
        self.ws.cp_manager.set_photo_marker("CP3", np.array([0.0, 1.0, 0.0]))
        self.ws.cp_manager.set_lidar_marker("CP3", np.array([0.0, 7.0, 0.0]))

        # Solve alignment
        self.ws._solve_alignment()
        self.assertTrue(self.ws.alignment_result.success)

        # Toggle preview on and off
        self.ws._toggle_alignment_preview()
        self.assertTrue(self.ws.preview_active)
        self.assertFalse(np.allclose(self.ws.viewport._preview_matrix, np.eye(4)))

        self.ws._reset_alignment()
        self.assertFalse(self.ws.preview_active)
        self.assertTrue(np.allclose(self.ws.viewport._preview_matrix, np.eye(4)))

        # Verify source arrays are identical to original
        np.testing.assert_allclose(self.ws.photo_verts, photo_verts_orig)
        np.testing.assert_allclose(self.ws.photo_uvs, photo_uvs_orig)
        np.testing.assert_allclose(self.ws.lidar_source_points, lidar_pts_orig)

    def test_s_marker_screen_projection_and_render(self):
        """TEST S: Markers are properly stored and 3D camera projection produces valid 2D coordinates."""
        self.ws.viewport.resize(800, 600)
        self.ws.viewport.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.ws.viewport.camera.distance = 10.0

        p1 = np.array([0.0, 0.0, 0.0])
        l1 = np.array([1.0, 0.0, 0.0])
        self.ws.viewport.set_control_markers({"CP1": p1}, {"CP1": l1}, active_id="CP1")

        self.assertIn("CP1", self.ws.viewport._photo_markers)
        self.assertIn("CP1", self.ws.viewport._lidar_markers)

        # Verify screen projection math with camera
        aspect = 800.0 / 600.0
        v_mat = self.ws.viewport.camera.get_view_matrix()
        p_mat = self.ws.viewport.camera.get_projection_matrix(aspect)
        vp_mat = pyrr.matrix44.multiply(v_mat, p_mat)

        p4 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        clip = pyrr.matrix44.apply_to_vector(vp_mat, p4)
        self.assertGreater(clip[3], 0.0)  # in front of camera
        ndc = clip[:3] / clip[3]
        sx = (ndc[0] + 1.0) * 0.5 * 800.0
        sy = (1.0 - ndc[1]) * 0.5 * 600.0
        self.assertAlmostEqual(sx, 400.0, delta=15.0)
        self.assertAlmostEqual(sy, 300.0, delta=15.0)

    def test_t_projected_point_cloud_picking(self):
        """TEST T: High-precision point cloud picking on point cloud surface vs rejection on empty space."""
        self.ws.viewport.resize(800, 600)
        self.ws.viewport.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.ws.viewport.camera.distance = 10.0

        aspect = 800.0 / 600.0
        v_mat = self.ws.viewport.camera.get_view_matrix()
        p_mat = self.ws.viewport.camera.get_projection_matrix(aspect)
        vp_mat = pyrr.matrix44.multiply(v_mat, p_mat)

        # Points at center (0,0,0) and offset (1,0,0)
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)

        # 1. Clicking directly in the center screen (400, 300) should pick (0, 0, 0)
        picked = self.ws.viewport._pick_closest_point(400.0, 300.0, vp_mat, 800.0, 600.0, pts)
        self.assertIsNotNone(picked)
        np.testing.assert_allclose(picked, [0.0, 0.0, 0.0])

        # 2. Clicking far away in empty space (e.g. corner at 50, 50) should return None
        picked_empty = self.ws.viewport._pick_closest_point(50.0, 50.0, vp_mat, 800.0, 600.0, pts)
        self.assertIsNone(picked_empty)

    def test_u_staging_transform_gizmo_modes(self):
        """TEST U: Interactive LiDAR staging transforms (translate, rotate, scale, reset) and tool states."""
        lidar_pts = np.array([
            [10.0, 20.0, 30.0],
            [12.0, 20.0, 30.0],
            [10.0, 22.0, 30.0]
        ], dtype=np.float64)
        self.ws.viewport.set_lidar_points(lidar_pts)

        # Initial staging should be identity
        self.assertTrue(np.allclose(self.ws.viewport.get_staging_matrix(), np.eye(4)))
        self.assertFalse(self.ws.viewport.staging_enabled)

        # Set tool to translate
        self.ws.viewport.set_staging_tool("translate")
        self.assertTrue(self.ws.viewport.staging_enabled)
        self.assertEqual(self.ws.viewport.gizmo.operation, "translate")

        # Simulate translation staging
        self.ws.viewport._lidar_staging_delta_pos = np.array([5.0, -3.0, 2.0], dtype=np.float32)
        M = self.ws.viewport.get_staging_matrix("lidar")
        self.assertFalse(np.allclose(M, np.eye(4)))

        # Center should move by delta_pos
        centroid = np.mean(lidar_pts, axis=0)
        c_transformed = (M @ np.append(centroid, 1.0))[:3]
        np.testing.assert_allclose(c_transformed, centroid + [5.0, -3.0, 2.0], atol=1e-5)

        # Simulate scale staging
        self.ws.viewport.set_staging_tool("scale")
        self.assertEqual(self.ws.viewport.gizmo.operation, "scale")
        self.ws.viewport._lidar_staging_scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        M_scaled = self.ws.viewport.get_staging_matrix("lidar")
        
        # Point offset from centroid should scale by 2.0
        p1 = lidar_pts[1]
        p1_trans = (M_scaled @ np.append(p1, 1.0))[:3]
        expected_p1 = (centroid + [5.0, -3.0, 2.0]) + 2.0 * (p1 - centroid)
        np.testing.assert_allclose(p1_trans, expected_p1, atol=1e-5)

        # Reset staging
        self.ws.viewport.reset_staging_transform("lidar")
        self.assertTrue(np.allclose(self.ws.viewport.get_staging_matrix("lidar"), np.eye(4)))
        np.testing.assert_allclose(self.ws.viewport._lidar_staging_delta_pos, [0, 0, 0])
        np.testing.assert_allclose(self.ws.viewport._lidar_staging_scale, [1, 1, 1])

    def test_v_point_picking_under_staging_transform(self):
        """TEST V: Point picking through staging transform returns exact native coordinate."""
        self.ws.viewport.resize(800, 600)

        # Native LiDAR point at (0, 5, 0)
        raw_native_point = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        lidar_pts = np.array([raw_native_point, [1.0, 5.0, 0.0]], dtype=np.float64)
        self.ws.viewport.set_lidar_points(lidar_pts)

        self.ws.viewport.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.ws.viewport.camera.distance = 10.0

        aspect = 800.0 / 600.0
        v_mat = self.ws.viewport.camera.get_view_matrix()
        p_mat = self.ws.viewport.camera.get_projection_matrix(aspect)
        vp_mat = pyrr.matrix44.multiply(v_mat, p_mat)

        # Apply staging translation: move (0, 5, 0) down to (0, 0, 0) where the camera looks
        self.ws.viewport._lidar_staging_delta_pos = np.array([0.0, -5.0, 0.0], dtype=np.float32)
        staging_mat = self.ws.viewport.get_staging_matrix("lidar")

        # In screen space, clicking at screen center (400, 300) hits the visually staged point
        picked = self.ws.viewport._pick_closest_point(
            400.0, 300.0, vp_mat, 800.0, 600.0, lidar_pts, transform_matrix=staging_mat
        )
        self.assertIsNotNone(picked)
        # MUST return the raw native coordinate [0.0, 5.0, 0.0], NOT the staged [0.0, 0.0, 0.0]
        np.testing.assert_allclose(picked, raw_native_point)

    def test_w_alignment_solving_with_staged_control_points(self):
        """TEST W: Registration solver receives exact native coordinates and calculates accurate Horn transform."""
        verts_photo, tris_photo, uvs_photo, tex_photo, p_photo, p_lidar, scale_true, R_true, t_true = create_mock_scene_data()

        photo_mesh = o3d.geometry.TriangleMesh()
        photo_mesh.vertices = o3d.utility.Vector3dVector(verts_photo)
        photo_mesh.triangles = o3d.utility.Vector3iVector(tris_photo)

        self.ws.photo_verts = verts_photo
        self.ws.photo_tris = tris_photo
        self.ws.photo_uvs = uvs_photo
        self.ws.photo_texture_img = tex_photo
        self.ws.viewport.set_photogrammetry_data(photo_mesh, tex_photo, point_cloud=verts_photo)

        self.ws.lidar_source_points = p_lidar
        self.ws.viewport.set_lidar_points(p_lidar)
        self.ws._update_workflow_state()

        # Apply heavy staging transform to viewport
        self.ws.viewport._lidar_staging_delta_pos = np.array([100.0, -50.0, 25.0], dtype=np.float32)
        self.ws.viewport._lidar_staging_scale = np.array([3.5, 3.5, 3.5], dtype=np.float32)
        self.ws.viewport._lidar_staging_rotation = np.array([45.0, 30.0, -15.0], dtype=np.float32)

        # Add control points (simulating user picking on staged geometry, which yields native p_lidar coords)
        for i in range(len(p_photo)):
            self.ws._add_control_point()
            cp_id = f"CP{i+1}"
            self.ws.cp_manager.set_photo_marker(cp_id, p_photo[i])
            self.ws.cp_manager.set_lidar_marker(cp_id, p_lidar[i])

        # Solve alignment
        self.ws._solve_alignment()

        self.assertIsNotNone(self.ws.alignment_result)
        self.assertTrue(self.ws.alignment_result.success)
        self.assertAlmostEqual(self.ws.alignment_result.scale, scale_true, delta=0.01)
        self.assertLess(self.ws.alignment_result.rms_error, 0.01)

    def test_x_photo_staging_transform_and_target_switching(self):
        """TEST X: Photogrammetry staging transform and target dataset switching."""
        photo_pts = np.array([
            [0.0, 10.0, 20.0],
            [1.0, 10.0, 20.0],
            [0.0, 11.0, 20.0]
        ], dtype=np.float64)
        lidar_pts = np.array([
            [50.0, 60.0, 70.0],
            [51.0, 60.0, 70.0],
            [50.0, 61.0, 70.0]
        ], dtype=np.float64)

        self.ws.viewport.set_photogrammetry_point_cloud(photo_pts)
        self.ws.viewport.set_lidar_points(lidar_pts)

        # Switch staging target to 'photo'
        self.ws.viewport.set_staging_target("photo")
        self.assertEqual(self.ws.viewport.staging_target, "photo")

        # Set tool to translate
        self.ws.viewport.set_staging_tool("translate")
        self.assertTrue(self.ws.viewport.staging_enabled)

        # Check gizmo pivot points to photo center
        pivot = self.ws.viewport._get_gizmo_pivot()
        self.assertIsNotNone(pivot)
        np.testing.assert_allclose(pivot.position, np.mean(photo_pts, axis=0), atol=1e-5)

        # Apply photo translation
        self.ws.viewport._photo_staging_delta_pos = np.array([10.0, 0.0, -5.0], dtype=np.float32)
        M_photo = self.ws.viewport.get_staging_matrix("photo")
        M_lidar = self.ws.viewport.get_staging_matrix("lidar")

        # Photo matrix should have transform; LiDAR matrix should remain identity
        self.assertFalse(np.allclose(M_photo, np.eye(4)))
        self.assertTrue(np.allclose(M_lidar, np.eye(4)))

        # Switch back to 'lidar'
        self.ws.viewport.set_staging_target("lidar")
        pivot_lidar = self.ws.viewport._get_gizmo_pivot()
        self.assertIsNotNone(pivot_lidar)
        np.testing.assert_allclose(pivot_lidar.position, np.mean(lidar_pts, axis=0), atol=1e-5)

        # Reset photo staging
        self.ws.viewport.reset_staging_transform("photo")
        self.assertTrue(np.allclose(self.ws.viewport.get_staging_matrix("photo"), np.eye(4)))

    def test_y_photo_point_picking_under_staging_transform(self):
        """TEST Y: Screen picking on staged Photogrammetry point cloud returns exact native raw coordinate."""
        self.ws.viewport.resize(800, 600)

        # Native Photogrammetry point at (0, 8, 0)
        raw_native_photo_pt = np.array([0.0, 8.0, 0.0], dtype=np.float64)
        photo_pts = np.array([raw_native_photo_pt, [1.0, 8.0, 0.0]], dtype=np.float64)
        self.ws.viewport.set_photogrammetry_point_cloud(photo_pts)

        self.ws.viewport.camera.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.ws.viewport.camera.distance = 10.0

        # Apply staging transform: translate (0, 8, 0) down to (0, 0, 0) where the camera looks
        self.ws.viewport._photo_staging_delta_pos = np.array([0.0, -8.0, 0.0], dtype=np.float32)
        self.ws.viewport.set_picking_mode("photo", cp_id="CP1")

        # In screen space, clicking at screen center (400, 300) hits the visually staged point
        picked = self.ws.viewport._pick_3d_point(400.0, 300.0)
        self.assertIsNotNone(picked)
        # MUST return the raw native photogrammetry coordinate [0.0, 8.0, 0.0]
        np.testing.assert_allclose(picked, raw_native_photo_pt)

    def test_z_dual_staging_simultaneous_alignment(self):
        """TEST Z: Simultaneous heavy staging of both Photogrammetry and LiDAR maintains 100% exact Horn alignment."""
        verts_photo, tris_photo, uvs_photo, tex_photo, p_photo, p_lidar, scale_true, R_true, t_true = create_mock_scene_data()

        photo_mesh = o3d.geometry.TriangleMesh()
        photo_mesh.vertices = o3d.utility.Vector3dVector(verts_photo)
        photo_mesh.triangles = o3d.utility.Vector3iVector(tris_photo)

        self.ws.photo_verts = verts_photo
        self.ws.photo_tris = tris_photo
        self.ws.photo_uvs = uvs_photo
        self.ws.photo_texture_img = tex_photo
        self.ws.viewport.set_photogrammetry_data(photo_mesh, tex_photo, point_cloud=verts_photo)

        self.ws.lidar_source_points = p_lidar
        self.ws.viewport.set_lidar_points(p_lidar)
        self.ws._update_workflow_state()

        # Heavily stage Photogrammetry
        self.ws.viewport._photo_staging_delta_pos = np.array([-40.0, 20.0, 15.0], dtype=np.float32)
        self.ws.viewport._photo_staging_scale = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.ws.viewport._photo_staging_rotation = np.array([30.0, -45.0, 60.0], dtype=np.float32)

        # Heavily stage LiDAR
        self.ws.viewport._lidar_staging_delta_pos = np.array([80.0, -100.0, 50.0], dtype=np.float32)
        self.ws.viewport._lidar_staging_scale = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        self.ws.viewport._lidar_staging_rotation = np.array([-60.0, 90.0, 15.0], dtype=np.float32)

        # Place CPs (picking on both staged geometries yields exact native p_photo and p_lidar coords)
        for i in range(len(p_photo)):
            self.ws._add_control_point()
            cp_id = f"CP{i+1}"
            self.ws.cp_manager.set_photo_marker(cp_id, p_photo[i])
            self.ws.cp_manager.set_lidar_marker(cp_id, p_lidar[i])

        # Solve alignment
        self.ws._solve_alignment()

        self.assertIsNotNone(self.ws.alignment_result)
        self.assertTrue(self.ws.alignment_result.success)
        self.assertAlmostEqual(self.ws.alignment_result.scale, scale_true, delta=0.01)
        self.assertLess(self.ws.alignment_result.rms_error, 0.01)

        # Trigger Alignment Preview
        self.ws._toggle_alignment_preview()
        self.assertTrue(self.ws.preview_active)
        # During preview: Photogrammetry MUST be in true native space (identity), LiDAR at preview matrix
        self.assertTrue(np.allclose(self.ws.viewport.get_active_photo_transform(), np.eye(4)))
        self.assertTrue(np.allclose(self.ws.viewport.get_active_lidar_transform(), self.ws.alignment_result.transformation_matrix))

        # Exit Alignment Preview
        self.ws._toggle_alignment_preview()
        self.assertFalse(self.ws.preview_active)
        # After preview: Staging transforms are restored for both
        self.assertFalse(np.allclose(self.ws.viewport.get_active_photo_transform(), np.eye(4)))
        self.assertFalse(np.allclose(self.ws.viewport.get_active_lidar_transform(), np.eye(4)))

    def test_za_end_to_end_texture_baking_pipeline(self):
        """TEST ZA: Complete end-to-end STE workflow through production texture baking."""
        # 1. Setup Photogrammetry Quad with Texture
        verts_photo = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0]
        ], dtype=np.float64)
        tris_photo = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        uvs_photo = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
            [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ], dtype=np.float64)
        tex_photo = np.full((128, 128, 3), 220, dtype=np.uint8)

        photo_mesh = o3d.geometry.TriangleMesh()
        photo_mesh.vertices = o3d.utility.Vector3dVector(verts_photo)
        photo_mesh.triangles = o3d.utility.Vector3iVector(tris_photo)

        self.ws.photo_verts = verts_photo.copy()
        self.ws.photo_tris = tris_photo.copy()
        self.ws.photo_uvs = uvs_photo.copy()
        self.ws.photo_texture_img = tex_photo.copy()
        self.ws.viewport.set_photogrammetry_data(photo_mesh, tex_photo, point_cloud=verts_photo)

        # 2. Setup LiDAR Surface Mesh (identical geometry)
        v_lidar = verts_photo.copy()
        t_lidar = tris_photo.copy()
        self.ws.lidar_source_points = v_lidar.copy()
        self.ws.lidar_surface_verts = v_lidar.copy()
        self.ws.lidar_surface_tris = t_lidar.copy()

        lidar_mesh = o3d.geometry.TriangleMesh()
        lidar_mesh.vertices = o3d.utility.Vector3dVector(v_lidar)
        lidar_mesh.triangles = o3d.utility.Vector3iVector(t_lidar)
        self.ws.viewport.set_lidar_surface(lidar_mesh)

        # 3. Control Points and Alignment
        for i in range(3):
            self.ws._add_control_point()
            cp_id = f"CP{i+1}"
            self.ws.cp_manager.set_photo_marker(cp_id, verts_photo[i])
            self.ws.cp_manager.set_lidar_marker(cp_id, v_lidar[i])

        self.ws._solve_alignment()
        self.assertTrue(self.ws.alignment_result.success)

        # 4. Target UV Generation
        self.ws._generate_target_uvs()
        self.assertIsNotNone(self.ws.lidar_target_uvs)
        self.assertTrue(self.ws.lidar_uv_result.success)

        # 5. Projection Validation
        self.ws._validate_texture_projection()
        self.assertIsNotNone(self.ws.projection_result)
        self.assertTrue(self.ws.projection_result.is_ready_for_baking)
        self.assertTrue(self.ws.btn_bake_texture.isEnabled())

        # 6. Execute Texture Bake
        self.ws.combo_resolution.setCurrentText("1024")
        self.ws.spin_padding.setValue(2)
        self.ws._bake_texture()

        # Run worker synchronously for testing
        if hasattr(self.ws, '_bake_worker') and self.ws._bake_worker is not None:
            self.ws._bake_worker.run()

        self.assertIsNotNone(self.ws.bake_result)
        self.assertTrue(self.ws.bake_result.success)
        self.assertEqual(self.ws.state, STEWorkflowState.BAKED)
        self.assertEqual(self.ws.viewport.lidar_display_mode, STELiDARRepresentation.BAKED)
        self.assertEqual(self.ws.bake_result.texture_width, 1024)
        self.assertEqual(self.ws.bake_result.texture_height, 1024)
        self.assertGreater(self.ws.bake_result.valid_texture_pixels, 0)

        # 7. Invalidation check on settings change
        self.ws.combo_resolution.setCurrentText("2048")
        self.assertIsNone(self.ws.bake_result)
        self.assertIn("Stale", self.ws.lbl_bake_stats.text())

        # Source data preservation
        np.testing.assert_array_equal(self.ws.photo_verts, verts_photo)
        np.testing.assert_array_equal(self.ws.photo_tris, tris_photo)
        np.testing.assert_array_equal(self.ws.photo_texture_img, tex_photo)


if __name__ == "__main__":
    unittest.main()


