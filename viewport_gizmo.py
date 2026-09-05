"""
viewport_gizmo.py - High-Precision Combined 3D Transform Gizmo (Global Space) for VisPy Viewport
Provides a Unity/Blender-grade interactive 3D gizmo combining translation shafts/heads and rotation rings.
Features:
- Segment-based hit-testing (click anywhere along shaft, head, or ring)
- 3D ray-line & ray-plane intersection for zero-latency, natural mouse dragging
- Active hover and drag highlighting in gold
"""

import numpy as np
from vispy import scene


def dist_to_segment_2d(M: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """Computes Euclidean distance from point M to line segment AB in 2D screen space."""
    ab = B - A
    denom = np.dot(ab, ab)
    if denom < 1e-10:
        return float(np.linalg.norm(M - A))
    t = np.clip(np.dot(M - A, ab) / denom, 0.0, 1.0)
    closest = A + t * ab
    return float(np.linalg.norm(M - closest))


class CombinedTransformGizmo:
    """
    Interactive 3D Transform Gizmo locked to Global (World) Space:
    - Translation Axis Shafts & Heads (X: Red, Y: Green, Z: Blue)
    - Rotation Rings (X: Red, Y: Green, Z: Blue)
    - Center Pivot Marker
    """

    # Colors
    COL_X = np.array([1.00, 0.32, 0.32, 0.95], dtype=np.float32)  # Red
    COL_Y = np.array([0.00, 0.90, 0.46, 0.95], dtype=np.float32)  # Green
    COL_Z = np.array([0.27, 0.55, 1.00, 0.95], dtype=np.float32)  # Blue
    COL_HIGHLIGHT = np.array([1.00, 0.85, 0.20, 1.00], dtype=np.float32)  # Gold

    def __init__(self, parent_scene=None):
        self.parent_scene = parent_scene
        self.visible = False

        # Transform state
        self.pivot_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # (rx, ry, rz) in degrees

        # Sizing
        self.arrow_length = 2.4
        self.ring_radius = 1.8

        # Visuals
        self.trans_lines = None
        self.trans_heads = None
        self.rot_lines = None
        self.pivot_marker = None

        # Interaction & Hover state
        self.hovered_handle = None
        self.active_handle = None
        self.is_dragging = False

        # Drag tracking bookmarks
        self.drag_start_pos = None
        self.drag_start_rot = None
        self.drag_start_mouse = None
        self._drag_t_start = 0.0
        self._drag_angle_start = 0.0
        self._drag_plane_normal = None
        self._drag_origin_start = None

        if self.parent_scene is not None:
            self._create_visuals()

    def set_parent(self, parent_scene):
        self.parent_scene = parent_scene
        self._create_visuals()

    def _create_visuals(self):
        if self.parent_scene is None:
            return

        self.remove_visuals()

        # 1. Translation Shafts
        self.trans_lines = scene.visuals.Line(
            pos=np.zeros((6, 3), dtype=np.float32),
            color=np.zeros((6, 4), dtype=np.float32),
            width=3.5,
            connect='segments',
            method='gl',
            parent=self.parent_scene
        )

        # 2. Translation Arrowhead Markers
        self.trans_heads = scene.visuals.Markers(parent=self.parent_scene)

        # 3. Rotation Rings (32 segments each = 96 vertices)
        self.rot_lines = scene.visuals.Line(
            pos=np.zeros((96 * 2, 3), dtype=np.float32),
            color=np.zeros((96 * 2, 4), dtype=np.float32),
            width=3.0,
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
        self.pivot_center = np.array(pivot_xyz, dtype=np.float32)
        self._update_geometry()

    def set_transform(self, position=None, rotation=None):
        if position is not None:
            self.position = np.array(position, dtype=np.float32)
        if rotation is not None:
            self.rotation = np.array(rotation, dtype=np.float32)
        self._update_geometry()

    def get_origin(self) -> np.ndarray:
        return self.pivot_center + self.position

    def _update_geometry(self):
        if self.trans_lines is None or self.rot_lines is None:
            return

        origin = self.get_origin()
        x_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        L = self.arrow_length
        R = self.ring_radius

        active = self.active_handle or self.hovered_handle

        col_x = self.COL_HIGHLIGHT if active == 'TRANS_X' else self.COL_X
        col_y = self.COL_HIGHLIGHT if active == 'TRANS_Y' else self.COL_Y
        col_z = self.COL_HIGHLIGHT if active == 'TRANS_Z' else self.COL_Z

        col_rot_x = self.COL_HIGHLIGHT if active == 'ROT_X' else self.COL_X
        col_rot_y = self.COL_HIGHLIGHT if active == 'ROT_Y' else self.COL_Y
        col_rot_z = self.COL_HIGHLIGHT if active == 'ROT_Z' else self.COL_Z

        # --- 1. Translation Shafts ---
        t_verts = [
            origin, origin + L * x_dir,
            origin, origin + L * y_dir,
            origin, origin + L * z_dir
        ]
        t_colors = [
            col_x, col_x,
            col_y, col_y,
            col_z, col_z
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
        head_colors = np.array([col_x, col_y, col_z], dtype=np.float32)
        self.trans_heads.set_data(
            pos=head_pos,
            face_color=head_colors,
            edge_color='white',
            edge_width=1.5,
            size=14
        )

        # --- 3. Rotation Rings (32 segments) ---
        num_segs = 32
        angles = np.linspace(0, 2 * np.pi, num_segs, endpoint=False)
        r_verts = []
        r_colors = []

        # X-Ring (YZ plane)
        for i in range(num_segs):
            a1, a2 = angles[i], angles[(i + 1) % num_segs]
            r_verts.extend([
                origin + R * (np.cos(a1) * y_dir + np.sin(a1) * z_dir),
                origin + R * (np.cos(a2) * y_dir + np.sin(a2) * z_dir)
            ])
            r_colors.extend([col_rot_x, col_rot_x])

        # Y-Ring (XZ plane)
        for i in range(num_segs):
            a1, a2 = angles[i], angles[(i + 1) % num_segs]
            r_verts.extend([
                origin + R * (np.cos(a1) * x_dir + np.sin(a1) * z_dir),
                origin + R * (np.cos(a2) * x_dir + np.sin(a2) * z_dir)
            ])
            r_colors.extend([col_rot_y, col_rot_y])

        # Z-Ring (XY plane)
        for i in range(num_segs):
            a1, a2 = angles[i], angles[(i + 1) % num_segs]
            r_verts.extend([
                origin + R * (np.cos(a1) * x_dir + np.sin(a1) * y_dir),
                origin + R * (np.cos(a2) * x_dir + np.sin(a2) * y_dir)
            ])
            r_colors.extend([col_rot_z, col_rot_z])

        self.rot_lines.set_data(
            pos=np.array(r_verts, dtype=np.float32),
            color=np.array(r_colors, dtype=np.float32)
        )

        # --- 4. Pivot Marker ---
        self.pivot_marker.set_data(
            pos=np.array([origin], dtype=np.float32),
            face_color=np.array([[1.0, 1.0, 1.0, 0.9]], dtype=np.float32),
            edge_color='#00E676',
            edge_width=1.5,
            size=9
        )

    # -----------------------------------------------------------------------
    # 3D Raycasting & Math Helpers
    # -----------------------------------------------------------------------
    def get_ray_from_mouse(self, mouse_xy: tuple, visual_transform):
        """Calculates a 3D ray (origin, unit_direction) from 2D screen coordinates."""
        mx, my = mouse_xy
        try:
            # Map near plane (z=0) and far plane (z=1) into 3D scene visual coordinates
            inv_tr = visual_transform.inverse
            p_near = inv_tr.map(np.array([[mx, my, 0.0, 1.0]], dtype=np.float32))
            p_far = inv_tr.map(np.array([[mx, my, 1.0, 1.0]], dtype=np.float32))

            o = (p_near[0, :3] / p_near[0, 3]).astype(np.float32)
            f = (p_far[0, :3] / p_far[0, 3]).astype(np.float32)

            d = f - o
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                d = d / norm
            return o, d
        except Exception:
            return None, None

    def closest_point_on_line(self, ray_o, ray_d, line_pt, line_dir) -> float:
        """Returns distance parameter t along line_pt + t * line_dir closest to 3D ray."""
        w0 = ray_o - line_pt
        a = float(np.dot(line_dir, line_dir))
        b = float(np.dot(line_dir, ray_d))
        c = float(np.dot(ray_d, ray_d))
        d = float(np.dot(line_dir, w0))
        e = float(np.dot(ray_d, w0))

        denom = a * c - b * b
        if abs(denom) < 1e-6:
            return 0.0
        return (b * e - c * d) / denom

    def ray_plane_intersection(self, ray_o, ray_d, plane_pt, plane_normal):
        """Finds intersection distance parameter t along ray_o + t * ray_d with plane."""
        denom = float(np.dot(ray_d, plane_normal))
        if abs(denom) < 1e-5:
            return None
        t = float(np.dot(plane_pt - ray_o, plane_normal)) / denom
        return t if t > 0 else None

    # -----------------------------------------------------------------------
    # Hit Testing (Full shaft & ring segment testing)
    # -----------------------------------------------------------------------
    def hit_test(self, mouse_xy: tuple, visual_transform) -> str:
        """
        Tests screen distance against full translation shafts and rotation rings.
        Tolerance is generous (22px) for effortless clicking and dragging.
        """
        if not self.visible or visual_transform is None:
            return None

        origin = self.get_origin()
        x_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        z_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        L = self.arrow_length
        R = self.ring_radius
        mouse_pt = np.array(mouse_xy, dtype=np.float32)

        # 1. Project origin and axis tips to screen
        pts_3d = np.array([
            origin,
            origin + L * x_dir,
            origin + L * y_dir,
            origin + L * z_dir
        ], dtype=np.float32)

        m_pts = visual_transform.map(pts_3d)
        scr_pts = m_pts[:, :2] / m_pts[:, 3:4]
        s_orig = scr_pts[0]
        s_x = scr_pts[1]
        s_y = scr_pts[2]
        s_z = scr_pts[3]

        best_handle = None
        min_dist = 22.0  # 22-pixel pick tolerance

        # Test Translation Shafts & Heads (Entire segment from center to tip)
        d_x = dist_to_segment_2d(mouse_pt, s_orig, s_x)
        if d_x < min_dist:
            min_dist = d_x
            best_handle = 'TRANS_X'

        d_y = dist_to_segment_2d(mouse_pt, s_orig, s_y)
        if d_y < min_dist:
            min_dist = d_y
            best_handle = 'TRANS_Y'

        d_z = dist_to_segment_2d(mouse_pt, s_orig, s_z)
        if d_z < min_dist:
            min_dist = d_z
            best_handle = 'TRANS_Z'

        # Test Rotation Rings (32 screen segments per ring)
        num_segs = 32
        angles = np.linspace(0, 2 * np.pi, num_segs, endpoint=False)

        # X-Ring (YZ plane)
        rx_3d = np.array([origin + R * (np.cos(a) * y_dir + np.sin(a) * z_dir) for a in angles], dtype=np.float32)
        m_rx = visual_transform.map(rx_3d)
        rx_scr = m_rx[:, :2] / m_rx[:, 3:4]
        for i in range(num_segs):
            d = dist_to_segment_2d(mouse_pt, rx_scr[i], rx_scr[(i + 1) % num_segs])
            if d < min_dist:
                min_dist = d
                best_handle = 'ROT_X'

        # Y-Ring (XZ plane)
        ry_3d = np.array([origin + R * (np.cos(a) * x_dir + np.sin(a) * z_dir) for a in angles], dtype=np.float32)
        m_ry = visual_transform.map(ry_3d)
        ry_scr = m_ry[:, :2] / m_ry[:, 3:4]
        for i in range(num_segs):
            d = dist_to_segment_2d(mouse_pt, ry_scr[i], ry_scr[(i + 1) % num_segs])
            if d < min_dist:
                min_dist = d
                best_handle = 'ROT_Y'

        # Z-Ring (XY plane)
        rz_3d = np.array([origin + R * (np.cos(a) * x_dir + np.sin(a) * y_dir) for a in angles], dtype=np.float32)
        m_rz = visual_transform.map(rz_3d)
        rz_scr = m_rz[:, :2] / m_rz[:, 3:4]
        for i in range(num_segs):
            d = dist_to_segment_2d(mouse_pt, rz_scr[i], rz_scr[(i + 1) % num_segs])
            if d < min_dist:
                min_dist = d
                best_handle = 'ROT_Z'

        return best_handle

    def set_hover(self, handle: str):
        if self.hovered_handle != handle:
            self.hovered_handle = handle
            if not self.is_dragging:
                self._update_geometry()

    # -----------------------------------------------------------------------
    # Smooth 3D Dragging Pipeline
    # -----------------------------------------------------------------------
    def start_drag(self, handle: str, mouse_xy: tuple, visual_transform):
        self.is_dragging = True
        self.active_handle = handle
        self.drag_start_pos = np.copy(self.position)
        self.drag_start_rot = np.copy(self.rotation)
        self.drag_start_mouse = np.array(mouse_xy, dtype=np.float32)
        self._drag_origin_start = self.get_origin()

        ray_o, ray_d = self.get_ray_from_mouse(mouse_xy, visual_transform)
        if ray_o is None:
            return

        axes = {
            'TRANS_X': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'TRANS_Y': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'TRANS_Z': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'ROT_X': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'ROT_Y': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'ROT_Z': np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

        if handle.startswith('TRANS_'):
            axis_vec = axes[handle]
            self._drag_t_start = self.closest_point_on_line(ray_o, ray_d, self._drag_origin_start, axis_vec)

        elif handle.startswith('ROT_'):
            pn = axes[handle]
            self._drag_plane_normal = pn
            t = self.ray_plane_intersection(ray_o, ray_d, self._drag_origin_start, pn)
            if t is not None:
                pt = ray_o + t * ray_d
                v = pt - self._drag_origin_start
                # Orthogonal basis on rotation plane
                if handle == 'ROT_X':
                    px, py = np.array([0, 1, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32)
                elif handle == 'ROT_Y':
                    px, py = np.array([1, 0, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32)
                else:  # ROT_Z
                    px, py = np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)

                self._drag_angle_start = np.arctan2(float(np.dot(v, py)), float(np.dot(v, px)))
            else:
                self._drag_angle_start = 0.0

        self._update_geometry()

    def update_drag(self, mouse_xy: tuple, visual_transform, camera=None):
        """Calculates exact 3D displacement and updates position and rotation."""
        if not self.is_dragging or self.active_handle is None:
            return None, None

        ray_o, ray_d = self.get_ray_from_mouse(mouse_xy, visual_transform)
        if ray_o is None:
            return self.position, self.rotation

        axes = {
            'TRANS_X': (0, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            'TRANS_Y': (1, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            'TRANS_Z': (2, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            'ROT_X': (0, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            'ROT_Y': (1, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            'ROT_Z': (2, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        }

        # --- Translation Drag ---
        if self.active_handle.startswith('TRANS_'):
            axis_idx, axis_vec = axes[self.active_handle]
            t_curr = self.closest_point_on_line(ray_o, ray_d, self._drag_origin_start, axis_vec)
            delta = t_curr - self._drag_t_start

            new_pos = np.copy(self.drag_start_pos)
            new_pos[axis_idx] -= delta
            self.position = new_pos
            self._update_geometry()
            return np.copy(self.position), np.copy(self.rotation)

        # --- Rotation Drag ---
        elif self.active_handle.startswith('ROT_'):
            axis_idx, pn = axes[self.active_handle]
            t = self.ray_plane_intersection(ray_o, ray_d, self._drag_origin_start, pn)

            if t is not None:
                pt = ray_o + t * ray_d
                v = pt - self._drag_origin_start

                if self.active_handle == 'ROT_X':
                    px, py = np.array([0, 1, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32)
                elif self.active_handle == 'ROT_Y':
                    px, py = np.array([1, 0, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32)
                else:  # ROT_Z
                    px, py = np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)

                cur_angle = np.arctan2(float(np.dot(v, py)), float(np.dot(v, px)))
                delta_rad = cur_angle - self._drag_angle_start
                delta_deg = np.degrees(delta_rad)

                # Invert Y-axis for standard turntable yaw
                if axis_idx == 1:
                    delta_deg = -delta_deg

                new_rot = np.copy(self.drag_start_rot)
                new_rot[axis_idx] -= delta_deg
                self.rotation = new_rot
                self._update_geometry()
                return np.copy(self.position), np.copy(self.rotation)

        return self.position, self.rotation

    def end_drag(self):
        self.is_dragging = False
        self.active_handle = None
        self.drag_start_mouse = None
        self._update_geometry()
