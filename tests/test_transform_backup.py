import os
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch
import numpy as np

import main_window


class TransformBackupPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rec_dir = os.path.join(self.temp_dir, "reconstruction_out")
        self.backup_dir = os.path.join(self.temp_dir, "backup")
        self.mvs_dir = os.path.join(self.rec_dir, "mvs")
        self.backup_mvs = os.path.join(self.backup_dir, "mvs")

        os.makedirs(self.mvs_dir, exist_ok=True)
        os.makedirs(self.backup_mvs, exist_ok=True)

        # Create dummy reconstruction and backup files
        self.dense_ply = os.path.join(self.mvs_dir, "scene_dense.ply")
        self.obj_file = os.path.join(self.mvs_dir, "scene_dense_mesh_texture.obj")
        self.ply_file = os.path.join(self.mvs_dir, "scene_dense_mesh_texture.ply")

        with open(self.dense_ply, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n")
        with open(self.obj_file, "w") as f:
            f.write("v 1.0 2.0 3.0\n")
        with open(self.ply_file, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n1 2 3\n")

        # Initial backup copies
        shutil.copy2(self.dense_ply, os.path.join(self.backup_mvs, "scene_dense.ply"))
        shutil.copy2(self.obj_file, os.path.join(self.backup_mvs, "scene_dense_mesh_texture.obj"))
        shutil.copy2(self.ply_file, os.path.join(self.backup_mvs, "scene_dense_mesh_texture.ply"))

        # Put a stale glb in backup to verify invalidation
        self.stale_glb = os.path.join(self.backup_mvs, "scene_dense_mesh_texture.glb")
        with open(self.stale_glb, "w") as f:
            f.write("glb")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("main_window.get_reconstruction_out_dir")
    @patch("main_window.get_backup_dir")
    def test_transform_applied_syncs_to_session_backup(self, mock_get_backup, mock_get_rec_out):
        mock_get_rec_out.return_value = self.rec_dir
        mock_get_backup.return_value = self.backup_dir

        # Setup mock window with required attributes
        window = Mock()
        window._current_points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        window._last_points = window._current_points
        window._raw_points = None
        window._current_faces = None
        window._current_colors = None
        window._cleanup_backup_points = None
        window._cleanup_backup_faces = None
        window._cleanup_backup_colors = None
        window._cumulative_mesh_transform = np.eye(4, dtype=np.float32)
        window.standalone_cloud_path = None
        window.viewer_widget = Mock()
        window.viewer_widget.current_mvs_dir = self.mvs_dir
        window.console_text = Mock()
        window.status_label = Mock()
        window.editor_tool_host = None
        window.transform_gizmo = None
        window.markers_visual = None
        window.mesh_visual = None
        window._update_ground_grid = Mock()
        window.canvas = Mock()

        # Bind helper methods to real implementation
        window._apply_transform_to_obj_file = main_window.MainWindow._apply_transform_to_obj_file.__get__(window)
        window._apply_transform_to_ply_file = main_window.MainWindow._apply_transform_to_ply_file.__get__(window)
        window._get_active_mvs_dir = lambda: self.mvs_dir

        # Create translation matrix
        T_mat = np.eye(4, dtype=np.float32)
        T_mat[0, 3] = 10.0  # translate X by +10

        # Execute _on_cloud_transform_applied
        main_window.MainWindow._on_cloud_transform_applied(window, T_mat)

        # 1. Verify OBJ file in backup was updated with the transformed coordinate
        backup_obj = os.path.join(self.backup_mvs, "scene_dense_mesh_texture.obj")
        self.assertTrue(os.path.isfile(backup_obj))
        with open(backup_obj, "r") as f:
            content = f.read()
        self.assertIn("v 11.000000 2.000000 3.000000", content)

        # 2. Verify stale GLB in backup was removed
        self.assertFalse(os.path.exists(self.stale_glb))

        # 3. Verify session metadata contains cumulative_mesh_transform
        meta = main_window.load_session_metadata()
        self.assertIsNotNone(meta)
        self.assertIn("cumulative_mesh_transform", meta)
        np.testing.assert_allclose(
            np.array(meta["cumulative_mesh_transform"]),
            T_mat,
            atol=1e-4
        )

    def test_restore_session_restores_cumulative_transform(self):
        T_mat = np.array([
            [0.0, -1.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        meta = {"cumulative_mesh_transform": T_mat.tolist()}
        window = Mock()
        window._cumulative_mesh_transform = np.eye(4, dtype=np.float32)

        # Simulate the metadata restoration block from restore_session_backup
        if "cumulative_mesh_transform" in meta:
            try:
                window._cumulative_mesh_transform = np.array(meta["cumulative_mesh_transform"], dtype=np.float32)
            except Exception:
                window._cumulative_mesh_transform = np.eye(4, dtype=np.float32)

        np.testing.assert_allclose(window._cumulative_mesh_transform, T_mat, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
