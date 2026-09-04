"""
viewport_gizmo.py - Combined 3D Transform Gizmo (Translation & Rotation) for VisPy Viewport
Provides a Unity/Blender-style interactive 3D gizmo combining translation arrows and rotation rings.
"""

import numpy as np
from vispy import scene


class CombinedTransformGizmo:
    """
    Interactive 3D Transform Gizmo combining:
    - Translation Axis Arrows (X: Red, Y: Green, Z: Blue)
    - Rotation Rings (X: Red, Y: Green, Z: Blue)
    - Center Pivot Marker
    """

    # Color definitions
    COL_X = np.array([1.0, 0.32, 0.32, 0.95], dtype=np.float32)  # #FF5252
    COL_Y = np.array([0.0, 0.90, 0.46, 0.95], dtype=np.float32)  # #00E676
    COL_Z = np.array([0.27, 0.55, 1.00, 0.95], dtype=np.float32)  # #448AFF
    COL_HIGHLIGHT = np.array([1.0, 0.90, 0.20, 1.0], dtype=np.float32)  # Gold hover/active

    def __init__(self, parent_scene=None):
        self.parent_scene = parent_scene
        self.visible = False

        # Transform state
        self.pivot_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # in degrees (rx, ry, rz)

        # Scale parameters
        self.arrow_length = 2.0
        self.ring_radius = 1.5

        # Visuals
        self.trans_lines = None
        self.trans_heads = None
        self.rot_lines = None
        self.pivot_marker = None

        # Drag state
        self.is_dragging = False
        self.active_handle = None  # 'TRANS_X', 'TRANS_Y', 'TRANS_Z', 'ROT_X', 'ROT_Y', 'ROT_Z'
        self.drag_start_mouse = None
        self.drag_last_mouse = None
        self.drag_start_pos = None
        self.drag_start_rot = None

        if self.parent_scene is not None:
            self._create_visuals()

    def set_parent(self, parent_scene):
        self.parent_scene = parent_scene
        self._create_visuals()

    def _create_visuals(self):
        if self.parent_scene is None:
            return

        self.remove_visuals()

        # 1. Translation Axis Shafts
        self.trans_lines = scene.visuals.Line(
            pos=np.zeros((6, 3), dtype=np.float32),
            color=np.zeros((6, 4), dtype=np.float32),
            width=2.5,
            connect='segments',
            method='gl',
            parent=self.parent_scene
        )

        # 2. Translation Arrowhead Markers
        self.trans_heads = scene.visuals.Markers(parent=self.parent_scene)

        # 3. Rotation Rings
        self.rot_lines = scene.visuals.Line(
            pos=np.zeros((96, 3), dtype=np.float32),
            color=np.zeros((96, 4), dtype=np.float32),
            width=2.5,
            connect='segments',
            method='gl',
            parent=self.parent_scene
        )

        # 4. Center Pivot Marker
        self.pivot_marker = scene.visuals.Markers(parent=self.parent_scene)

        self._update_geometry()
        self.set_visible(self.visible)

    def remove_visuals(self):
        for v in (self.trans_lines, self.trans_heads, self.rot_lines, self.pivot_marker):
            if v is not None:
                v.parent = None
        self.trans_lines = None
        self.trans_heads = None
        self.rot_lines = None
        self.pivot_marker = None

    def set_visible(self, visible: bool):
        self.visible = visible
        for v in (self.trans_lines, self.trans_heads, self.rot_lines, self.pivot_marker):
            if v is not None:
                v.visible = visible

    def set_pivot(self, pivot_xyz):
        """Sets the baseline reference pivot (e.g. centroid of active point cloud/mesh)."""
        self.pivot_center = np.array(pivot_xyz, dtype=np.float32)
        self._update_geometry()

    def set_transform(self, position=None, rotation=None):
        """Updates the live position and rotation (deg) of the gizmo."""
        if position is not None:
            self.position = np.array(position, dtype=np.float32)
        if rotation is not None:
            self.rotation = np.array(rotation, dtype=np.float32)
        self._update_geometry()

    def get_origin(self) -> np.ndarray:
        """Returns 3D world position of gizmo center."""
        return self.pivot_center + self.position

    def _get_rotation_matrix(self) -> np.ndarray:
        rx, ry, rz = np.radians(self.rotation)
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
        return Rz @ Ry @ Rx

    def _update_geometry(self):
        if self.trans_lines is None or self.rot_lines is None:
            return

        origin = self.get_origin()
        R = self._get_rotation_matrix()

        # Axis directions in world space
        x_dir = R @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_dir = R @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_dir = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        L = self.arrow_length
        R_ring = self.ring_radius

        # --- 1. Translation Shafts ---
        t_verts = [
            origin, origin + L * x_dir,
            origin, origin + L * y_dir,
            origin, origin + L * z_dir
        ]
        t_colors = [
            self.COL_X, self.COL_X,
            self.COL_Y, self.COL_Y,
            self.COL_Z, self.COL_Z
        ]
        self.trans_lines.set_data(
            pos=np.array(t_verts, dtype=np.float32),
            color=np.array(t_colors, dtype=np.float32)
        )

        # --- 2. Translation Arrow Heads ---
        head_pos = np.array([
            origin + L * x_dir,
            origin + L * y_dir,
            origin + L * z_dir
        ], dtype=np.float32)
        head_colors = np.array([self.COL_X, self.COL_Y, self.COL_Z], dtype=np.float32)
        self.trans_heads.set_data(
            pos=head_pos,
            face_color=head_colors,
            edge_color='white',
            edge_width=1.0,
            size=11
        )

        # --- 3. Rotation Rings (32 segments each) ---
        num_segs = 32
        angles = np.linspace(0, 2 * np.pi, num_segs, endpoint=False)
        r_verts = []
        r_colors = []

        # X-Ring (around X-axis: in Y-Z plane)
        for i in range(num_segs):
            a1 = angles[i]
            a2 = angles[(i + 1) % num_segs]
            p1 = origin + R_ring * (np.cos(a1) * y_dir + np.sin(a1) * z_dir)
            p2 = origin + R_ring * (np.cos(a2) * y_dir + np.sin(a2) * z_dir)
            r_verts.extend([p1, p2])
            r_colors.extend([self.COL_X, self.COL_X])

        # Y-Ring (around Y-axis: in X-Z plane)
        for i in range(num_segs):
            a1 = angles[i]
            a2 = angles[(i + 1) % num_segs]
            p1 = origin + R_ring * (np.cos(a1) * x_dir + np.sin(a1) * z_dir)
            p2 = origin + R_ring * (np.cos(a2) * x_dir + np.sin(a2) * z_dir)
            r_verts.extend([p1, p2])
            r_colors.extend([self.COL_Y, self.COL_Y])

        # Z-Ring (around Z-axis: in X-Y plane)
        for i in range(num_segs):
            a1 = angles[i]
            a2 = angles[(i + 1) % num_segs]
            p1 = origin + R_ring * (np.cos(a1) * x_dir + np.sin(a1) * y_dir)
            p2 = origin + R_ring * (np.cos(a2) * x_dir + np.sin(a2) * y_dir)
            r_verts.extend([p1, p2])
            r_colors.extend([self.COL_Z, self.COL_Z])

        self.rot_lines.set_data(
            pos=np.array(r_verts, dtype=np.float32),
            color=np.array(r_colors, dtype=np.float32)
        )

        # --- 4. Center Pivot Marker ---
        self.pivot_marker.set_data(
            pos=np.array([origin], dtype=np.float32),
            face_color=np.array([[1.0, 1.0, 1.0, 0.9]], dtype=np.float32),
            edge_color='#00E676',
            edge_width=1.5,
            size=8
        )

    # -----------------------------------------------------------------------
    # Hit Testing & Drag Interaction
    # -----------------------------------------------------------------------
    def hit_test(self, mouse_xy, visual_transform) -> str:
        """
        Raycasts screen-space distance against gizmo translation arrowheads and rotation rings.
        Returns: 'TRANS_X', 'TRANS_Y', 'TRANS_Z', 'ROT_X', 'ROT_Y', 'ROT_Z', or None.
        """
        if not self.visible or visual_transform is None:
            return None

        origin = self.get_origin()
        R = self._get_rotation_matrix()
        x_dir = R @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_dir = R @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_dir = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        L = self.arrow_length
        R_ring = self.ring_radius
        mouse_pt = np.array(mouse_xy, dtype=np.float32)

        # 1. Test Translation Arrow Heads (Priority 1)
        head_pts_3d = np.array([
            origin + L * x_dir,
            origin + L * y_dir,
            origin + L * z_dir
        ], dtype=np.float32)

        m_heads = visual_transform.map(head_pts_3d)
        heads_2d = m_heads[:, :2] / m_heads[:, 3:4]
        head_dists = np.linalg.norm(heads_2d - mouse_pt, axis=1)

        min_h_idx = int(np.argmin(head_dists))
        if head_dists[min_h_idx] <= 20.0:  # 20px tolerance
            return ['TRANS_X', 'TRANS_Y', 'TRANS_Z'][min_h_idx]

        # 2. Test Rotation Rings (Sample 24 points per ring)
        num_samples = 24
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

        # X Ring
        rx_pts = np.array([origin + R_ring * (np.cos(a) * y_dir + np.sin(a) * z_dir) for a in angles], dtype=np.float32)
        m_rx = visual_transform.map(rx_pts)
        rx_2d = m_rx[:, :2] / m_rx[:, 3:4]
        if np.min(np.linalg.norm(rx_2d - mouse_pt, axis=1)) <= 14.0:
            return 'ROT_X'

        # Y Ring
        ry_pts = np.array([origin + R_ring * (np.cos(a) * x_dir + np.sin(a) * z_dir) for a in angles], dtype=np.float32)
        m_ry = visual_transform.map(ry_pts)
        ry_2d = m_ry[:, :2] / m_ry[:, 3:4]
        if np.min(np.linalg.norm(ry_2d - mouse_pt, axis=1)) <= 14.0:
            return 'ROT_Y'

        # Z Ring
        rz_pts = np.array([origin + R_ring * (np.cos(a) * x_dir + np.sin(a) * y_dir) for a in angles], dtype=np.float32)
        m_rz = visual_transform.map(rz_pts)
        rz_2d = m_rz[:, :2] / m_rz[:, 3:4]
        if np.min(np.linalg.norm(rz_2d - mouse_pt, axis=1)) <= 14.0:
            return 'ROT_Z'

        return None

    def start_drag(self, handle: str, mouse_xy: tuple):
        self.is_dragging = True
        self.active_handle = handle
        self.drag_start_mouse = np.array(mouse_xy, dtype=np.float32)
        self.drag_last_mouse = np.array(mouse_xy, dtype=np.float32)
        self.drag_start_pos = np.copy(self.position)
        self.drag_start_rot = np.copy(self.rotation)

    def update_drag(self, mouse_xy: tuple, visual_transform, camera=None):
        """
        Computes delta and returns updated (pos_x, pos_y, pos_z), (rot_x, rot_y, rot_z).
        """
        if not self.is_dragging or self.active_handle is None:
            return None, None

        curr_mouse = np.array(mouse_xy, dtype=np.float32)
        dx_scr = curr_mouse[0] - self.drag_last_mouse[0]
        dy_scr = curr_mouse[1] - self.drag_last_mouse[1]

        if abs(dx_scr) < 1e-4 and abs(dy_scr) < 1e-4:
            return self.position, self.rotation

        origin = self.get_origin()
        R = self._get_rotation_matrix()
        x_dir = R @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_dir = R @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_dir = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # --- Translation Drag ---
        if self.active_handle.startswith('TRANS_'):
            axis_map = {'TRANS_X': (0, x_dir), 'TRANS_Y': (1, y_dir), 'TRANS_Z': (2, z_dir)}
            axis_idx, axis_vec = axis_map[self.active_handle]

            p0_3d = origin
            p1_3d = origin + axis_vec

            m = visual_transform.map(np.array([p0_3d, p1_3d], dtype=np.float32))
            s0 = (m[0, :2] / m[0, 3:4])
            s1 = (m[1, :2] / m[1, 3:4])

            scr_dir = s1 - s0
            len_sq = np.dot(scr_dir, scr_dir)
            if len_sq > 1e-5:
                # 3D translation displacement
                delta_world = (dx_scr * scr_dir[0] + dy_scr * scr_dir[1]) / len_sq
                self.position[axis_idx] += delta_world
                self.drag_last_mouse = curr_mouse
                self._update_geometry()
                return np.copy(self.position), np.copy(self.rotation)

        # --- Rotation Drag ---
        elif self.active_handle.startswith('ROT_'):
            axis_map = {'ROT_X': 0, 'ROT_Y': 1, 'ROT_Z': 2}
            axis_idx = axis_map[self.active_handle]

            m_orig = visual_transform.map(np.array([origin], dtype=np.float32))
            s_orig = (m_orig[0, :2] / m_orig[0, 3:4])

            v_prev = self.drag_last_mouse - s_orig
            v_curr = curr_mouse - s_orig

            ang_prev = np.arctan2(v_prev[1], v_prev[0])
            ang_curr = np.arctan2(v_curr[1], v_curr[0])
            d_theta = np.degrees(ang_curr - ang_prev)

            # Invert sign if needed for intuitive rotation
            if axis_idx == 1:
                d_theta = -d_theta

            self.rotation[axis_idx] += d_theta
            self.drag_last_mouse = curr_mouse
            self._update_geometry()
            return np.copy(self.position), np.copy(self.rotation)

        return self.position, self.rotation

    def end_drag(self):
        self.is_dragging = False
        self.active_handle = None
        self.drag_start_mouse = None
        self.drag_last_mouse = None
