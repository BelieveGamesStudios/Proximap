import os
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

# Ensure QApplication exists for QWidget instantiation in headless tests
app = QApplication.instance() or QApplication(sys.argv)

from viewport_tool_system import (
    EditorWindowHostModal, FloatingToolboxWidget,
    TransformToolWindow, MeshCleanupToolWindow, MeshCutToolWindow,
    CameraRegistrationToolWindow
)
from camera_frustum_manager import CameraFrustumManager


class SidebarToolsVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.parent_widget = QWidget()
        self.parent_widget.resize(800, 600)

        self.host_modal = EditorWindowHostModal(self.parent_widget)
        self.toolbox = FloatingToolboxWidget(self.parent_widget, host_modal=self.host_modal)

        self.cam_tool = CameraRegistrationToolWindow()
        self.transform_tool = TransformToolWindow()
        self.cleanup_tool = MeshCleanupToolWindow()
        self.cut_tool = MeshCutToolWindow()

        self.toolbox.register_tool(self.cam_tool)
        self.toolbox.register_tool(self.transform_tool)
        self.toolbox.register_tool(self.cleanup_tool)
        self.toolbox.register_tool(self.cut_tool)

    def tearDown(self):
        self.parent_widget.deleteLater()

    def test_registered_tool_ids(self):
        """Verify registered tool IDs match exact expected strings."""
        self.assertEqual(self.cam_tool.tool_id, "camera_registration")
        self.assertEqual(self.transform_tool.tool_id, "transform")
        self.assertEqual(self.cleanup_tool.tool_id, "cleanup")
        self.assertEqual(self.cut_tool.tool_id, "cut")

    def test_visibility_matrix_modes(self):
        """Verify visibility across all 3 view modes."""
        # Helper applying the visibility rule
        def apply_mode_visibility(mode_index: int):
            is_sparse = (mode_index == 0)
            is_textured_mesh = (mode_index == 2)
            self.toolbox.set_tool_visible("camera_registration", is_sparse)
            self.toolbox.set_tool_visible("transform", True)
            self.toolbox.set_tool_visible("cleanup", is_textured_mesh)
            self.toolbox.set_tool_visible("cut", is_textured_mesh)

        # Mode 0: Sparse Point Cloud & Cameras
        apply_mode_visibility(0)
        self.assertFalse(self.toolbox.tool_buttons["camera_registration"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["transform"].isHidden())
        self.assertTrue(self.toolbox.tool_buttons["cleanup"].isHidden())
        self.assertTrue(self.toolbox.tool_buttons["cut"].isHidden())

        # Mode 1: Dense Point Cloud
        apply_mode_visibility(1)
        self.assertTrue(self.toolbox.tool_buttons["camera_registration"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["transform"].isHidden())
        self.assertTrue(self.toolbox.tool_buttons["cleanup"].isHidden())
        self.assertTrue(self.toolbox.tool_buttons["cut"].isHidden())

        # Mode 2: Textured Mesh
        apply_mode_visibility(2)
        self.assertTrue(self.toolbox.tool_buttons["camera_registration"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["transform"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["cleanup"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["cut"].isHidden())

    def test_hiding_open_tool_closes_host_modal(self):
        """Verify that hiding a tool currently open in host modal automatically closes it."""
        # Open camera registration tool
        self.host_modal.open_tool(self.cam_tool)
        self.assertFalse(self.host_modal.isHidden())
        self.assertEqual(self.host_modal.current_tool, self.cam_tool)

        # Hide camera registration tool
        self.toolbox.set_tool_visible("camera_registration", False)
        self.assertTrue(self.host_modal.isHidden())
        self.assertIsNone(self.host_modal.current_tool)

    def test_set_tool_visible_idempotency(self):
        """Verify that repeating set_tool_visible calls does not crash or corrupt layout."""
        self.toolbox.set_tool_visible("camera_registration", True)
        self.toolbox.set_tool_visible("camera_registration", True)
        self.assertFalse(self.toolbox.tool_buttons["camera_registration"].isHidden())

        self.toolbox.set_tool_visible("camera_registration", False)
        self.toolbox.set_tool_visible("camera_registration", False)
        self.assertTrue(self.toolbox.tool_buttons["camera_registration"].isHidden())

    def test_camera_settings_applied_after_manager_creation(self):
        """Verify that changes made to camera settings before camera data loads are preserved and applied."""
        mock_scene = Mock()
        mock_view = Mock()
        mock_view.scene = mock_scene

        # Instantiate manager
        manager = CameraFrustumManager(mock_view)

        # Custom values set before camera load
        custom_scale = 1.8
        custom_opacity = 0.45
        custom_frustums_vis = False
        custom_photos_vis = True

        manager.set_scale(custom_scale)
        manager.set_opacity(custom_opacity)
        manager.set_frustums_visible(custom_frustums_vis)
        manager.set_planes_visible(custom_photos_vis)

        self.assertAlmostEqual(manager._scale, custom_scale)
        self.assertAlmostEqual(manager._opacity, custom_opacity)
        self.assertEqual(manager._frustums_visible, custom_frustums_vis)
        self.assertEqual(manager._planes_visible, custom_photos_vis)

    def test_camera_tool_set_busy(self):
        """Verify that set_busy enables/disables controls on the CameraRegistrationToolWindow."""
        # Open tool in host modal so UI widgets are constructed
        self.host_modal.open_tool(self.cam_tool)
        self.assertIsNotNone(self.cam_tool.btn_show_cameras)
        self.assertIsNotNone(self.cam_tool.btn_show_photos)
        self.assertIsNotNone(self.cam_tool.scale_slider)
        self.assertIsNotNone(self.cam_tool.opacity_slider)

        self.cam_tool.set_busy(True)
        self.assertFalse(self.cam_tool.btn_show_cameras.isEnabled())
        self.assertFalse(self.cam_tool.btn_show_photos.isEnabled())
        self.assertFalse(self.cam_tool.scale_slider.isEnabled())
        self.assertFalse(self.cam_tool.opacity_slider.isEnabled())

        self.cam_tool.set_busy(False)
        self.assertTrue(self.cam_tool.btn_show_cameras.isEnabled())
        self.assertTrue(self.cam_tool.btn_show_photos.isEnabled())
        self.assertTrue(self.cam_tool.scale_slider.isEnabled())
        self.assertTrue(self.cam_tool.opacity_slider.isEnabled())


    def test_main_window_set_view_mode_helper(self):
        """Verify MainWindow._set_view_mode sets index and updates sidebar visibility."""
        import main_window
        win = Mock()
        win.floating_toolbox = self.toolbox
        win.viewer_widget = Mock()
        win.viewer_widget.mode_select = Mock()
        win.viewer_widget.update_crop_box_state = Mock()

        # Bind methods from MainWindow
        win._update_sidebar_tools_visibility = main_window.MainWindow._update_sidebar_tools_visibility.__get__(win)
        win._set_view_mode = main_window.MainWindow._set_view_mode.__get__(win)

        # Set to mode 0 (Sparse)
        win._set_view_mode(0)
        self.assertFalse(self.toolbox.tool_buttons["camera_registration"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["transform"].isHidden())
        self.assertTrue(self.toolbox.tool_buttons["cleanup"].isHidden())
        self.assertTrue(self.toolbox.tool_buttons["cut"].isHidden())

        # Set to mode 2 (Textured Mesh)
        win._set_view_mode(2)
        self.assertTrue(self.toolbox.tool_buttons["camera_registration"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["transform"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["cleanup"].isHidden())
        self.assertFalse(self.toolbox.tool_buttons["cut"].isHidden())


class CameraFrustumPickingTests(unittest.TestCase):
    def setUp(self):
        self.manager = CameraFrustumManager(Mock())
        self.manager._view.scene.transform.map.side_effect = lambda point: point
        self.manager._cameras_data = [{
            "center": np.array([100.0, 100.0, 0.0]),
            "R": np.eye(3),
        }]
        self.manager._world_corners_for = Mock(return_value=[
            np.array([200.0, 80.0, 0.0]),
            np.array([200.0, 120.0, 0.0]),
            np.array([160.0, 120.0, 0.0]),
            np.array([160.0, 80.0, 0.0]),
        ])

    def test_picks_using_canvas_pixel_coordinates(self):
        """Projected coordinates are already pixels after homogeneous division."""
        self.assertEqual(self.manager.pick_camera(100, 100, 800, 600), 0)

    def test_picks_visible_frustum_edge_not_just_camera_origin(self):
        """Clicking a rendered wireframe segment should select its camera."""
        self.assertEqual(self.manager.pick_camera(180, 80, 800, 600), 0)

    def test_does_not_pick_when_camera_geometry_is_hidden(self):
        self.manager.set_frustums_visible(False)
        self.manager.set_planes_visible(False)
        self.assertEqual(self.manager.pick_camera(100, 100, 800, 600), -1)

    def test_camera_look_through_orientation_math(self):
        """Verify look-through azimuth and elevation match VisPy forward view vector."""
        import main_window
        win = Mock()
        win.frustum_manager = Mock()
        win.view = Mock()
        win.view.camera = Mock()
        win.view.camera.azimuth = 45.0
        win.view.camera.elevation = 30.0
        win.view.camera.center = [0.0, 0.0, 0.0]
        win.view.camera.distance = 5.0
        win._show_look_through_badge = Mock()
        win._look_through_timer = Mock()
        win._look_through_timer.start = Mock()

        # Camera pointing toward -Z in world space (R = eye(3))
        cam_data = {
            "center": np.array([2.0, 3.0, 4.0]),
            "R": np.eye(3),
            "image_name": "test_cam.jpg"
        }
        win.frustum_manager.get_camera.return_value = cam_data

        # Bind and invoke _on_look_through_camera
        win._on_look_through_camera = main_window.MainWindow._on_look_through_camera.__get__(win)
        win._on_look_through_camera(0)

        target = win._look_through_target
        # forward is [0, 0, -1] -> az = 90 deg, el = 0 deg
        self.assertAlmostEqual(target["azimuth"] % 360.0, 90.0, places=3)
        self.assertAlmostEqual(target["elevation"], 0.0, places=3)
        # target_center = C + forward * 2.0 = [2.0, 3.0, 4.0] + [0, 0, -2.0] = [2.0, 3.0, 2.0]
        self.assertAlmostEqual(target["center"][0], 2.0, places=3)
        self.assertAlmostEqual(target["center"][1], 3.0, places=3)
        self.assertAlmostEqual(target["center"][2], 2.0, places=3)
        self.assertEqual(target["distance"], 2.0)


if __name__ == "__main__":
    unittest.main()

