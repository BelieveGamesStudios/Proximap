"""
crop_box.py - 3D Bounding Box Crop Visual Overlay & Cropping Engine for VisPy
Provides a RealityScan / RealityCapture style 3D bounding box overlay in VisPy
with draggable handles and fast numpy-based mesh/point-cloud trimming (Remove Inside / Remove Outside).
"""

import numpy as np
from vispy import scene

class CropBoxOverlay:
    """
    VisPy visual overlay representing an axis-aligned 3D cropping volume with 6 face handles.
    """

    HANDLE_NAMES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']

    def __init__(self, parent_scene=None):
        self.parent_scene = parent_scene
        self.visible = False

        # Volume bounds: [min_x, min_y, min_z], [max_x, max_y, max_z]
        self.bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self.bounds_max = np.array([ 1.0,  1.0,  1.0], dtype=np.float32)

        # VisPy Visuals
        self.box_lines = None
        self.handle_markers = None

        # Dragging state
        self.active_handle_idx = None
        self.is_dragging = False
        self.drag_start_pos = None

        if self.parent_scene is not None:
            self._create_visuals()

    def set_parent(self, parent_scene):
        self.parent_scene = parent_scene
        self._create_visuals()

    def _create_visuals(self):
        if self.parent_scene is None:
            return
        
        # Clean existing visuals if any
        self.remove_visuals()

        # Wireframe box using VisPy Line visual (12 edges = 24 vertices)
        self.box_lines = scene.visuals.Line(
            pos=np.zeros((24, 3), dtype=np.float32),
            color='#00E676',
            width=2.5,
            connect='segments',
            parent=self.parent_scene
        )

        # 6 Handle markers for +X, -X, +Y, -Y, +Z, -Z
        self.handle_markers = scene.visuals.Markers(parent=self.parent_scene)

        self._update_geometry()
        self.set_visible(self.visible)

    def remove_visuals(self):
        if self.box_lines is not None:
            self.box_lines.parent = None
            self.box_lines = None
        if self.handle_markers is not None:
            self.handle_markers.parent = None
            self.handle_markers = None

    def fit_to_points(self, points, padding_ratio=0.02):
        """Initializes the box bounds to wrap tight around the given 3D points with padding."""
        if points is None or len(points) == 0:
            self.bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            self.bounds_max = np.array([ 1.0,  1.0,  1.0], dtype=np.float32)
        else:
            p_min = np.min(points, axis=0)
            p_max = np.max(points, axis=0)
            size = p_max - p_min
            # Prevent zero thickness
            size = np.maximum(size, 0.01)
            padding = size * padding_ratio

            self.bounds_min = (p_min - padding).astype(np.float32)
            self.bounds_max = (p_max + padding).astype(np.float32)

        self._update_geometry()

    def set_bounds(self, bounds_min, bounds_max):
        self.bounds_min = np.array(bounds_min, dtype=np.float32)
        self.bounds_max = np.array(bounds_max, dtype=np.float32)
        # Ensure min <= max
        for i in range(3):
            if self.bounds_min[i] > self.bounds_max[i]:
                self.bounds_min[i], self.bounds_max[i] = self.bounds_max[i], self.bounds_min[i]
        self._update_geometry()

    def get_bounds(self):
        return np.copy(self.bounds_min), np.copy(self.bounds_max)

    def get_handle_positions(self):
        x0, y0, z0 = self.bounds_min
        x1, y1, z1 = self.bounds_max
        c_x = (x0 + x1) * 0.5
        c_y = (y0 + y1) * 0.5
        c_z = (z0 + z1) * 0.5

        return np.array([
            [x1, c_y, c_z],  # 0: +X
            [x0, c_y, c_z],  # 1: -X
            [c_x, y1, c_z],  # 2: +Y
            [c_x, y0, c_z],  # 3: -Y
            [c_x, c_y, z1],  # 4: +Z
            [c_x, c_y, z0],  # 5: -Z
        ], dtype=np.float32)

    def set_visible(self, visible: bool):
        self.visible = visible
        if self.box_lines is not None:
            self.box_lines.visible = visible
        if self.handle_markers is not None:
            self.handle_markers.visible = visible

    def _update_geometry(self):
        if self.box_lines is None:
            return

        x0, y0, z0 = self.bounds_min
        x1, y1, z1 = self.bounds_max

        # 8 box corners
        c = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ], dtype=np.float32)

        # 12 line segments (24 points)
        edges = [
            # Bottom face
            c[0], c[1],  c[1], c[2],  c[2], c[3],  c[3], c[0],
            # Top face
            c[4], c[5],  c[5], c[6],  c[6], c[7],  c[7], c[4],
            # Vertical edges
            c[0], c[4],  c[1], c[5],  c[2], c[6],  c[3], c[7]
        ]
        line_pos = np.array(edges, dtype=np.float32)
        self.box_lines.set_data(pos=line_pos)

        # 6 Handle positions (centers of the 6 faces)
        c_x = (x0 + x1) * 0.5
        c_y = (y0 + y1) * 0.5
        c_z = (z0 + z1) * 0.5

        handles = np.array([
            [x1, c_y, c_z],  # +X (Redish/Green)
            [x0, c_y, c_z],  # -X
            [c_x, y1, c_z],  # +Y
            [c_x, y0, c_z],  # -Y
            [c_x, c_y, z1],  # +Z
            [c_x, c_y, z0],  # -Z
        ], dtype=np.float32)

        colors = np.array([
            [1.0, 0.3, 0.3, 1.0],  # +X (Red)
            [1.0, 0.5, 0.5, 1.0],  # -X
            [0.3, 1.0, 0.3, 1.0],  # +Y (Green)
            [0.5, 1.0, 0.5, 1.0],  # -Y
            [0.3, 0.5, 1.0, 1.0],  # +Z (Blue)
            [0.5, 0.7, 1.0, 1.0],  # -Z
        ], dtype=np.float32)

        self.handle_markers.set_data(
            pos=handles,
            face_color=colors,
            edge_color='white',
            edge_width=1.5,
            size=12
        )

    # -----------------------------------------------------------------------
    # Cropping Logic (Trimming numpy points/faces)
    # -----------------------------------------------------------------------
    def crop_points_and_mesh(self, points, colors=None, faces=None, texcoords=None, keep_inside=True):
        """
        Trims 3D points and faces based on current box volume bounds.
        Returns: (new_points, new_colors, new_faces, new_texcoords)
        """
        if points is None or len(points) == 0:
            return points, colors, faces, texcoords

        b_min, b_max = self.get_bounds()

        # Vertex containment mask: True if point inside [b_min, b_max]
        inside_mask = np.all((points >= b_min) & (points <= b_max), axis=1)

        if faces is not None and len(faces) > 0:
            # Mesh cropping by face centroids or vertex containment
            # Face is inside if all 3 vertices (or centroid) are inside
            f_v0 = inside_mask[faces[:, 0]]
            f_v1 = inside_mask[faces[:, 1]]
            f_v2 = inside_mask[faces[:, 2]]

            if keep_inside:
                # Keep faces where all 3 vertices (or majority) are inside
                keep_face_mask = f_v0 & f_v1 & f_v2
            else:
                # Keep faces where at least one vertex is outside (Remove Inside)
                keep_face_mask = ~(f_v0 & f_v1 & f_v2)

            sub_faces = faces[keep_face_mask]
            if len(sub_faces) == 0:
                return np.zeros((0, 3), dtype=np.float32), None, np.zeros((0, 3), dtype=np.int32), None

            # Re-index remaining vertices to remove unreferenced orphaned vertices
            unique_v_idx, new_faces = np.unique(sub_faces, return_inverse=True)
            new_faces = new_faces.reshape(sub_faces.shape).astype(np.int32)

            new_points = points[unique_v_idx]
            new_colors = colors[unique_v_idx] if colors is not None and len(colors) == len(points) else colors
            new_texcoords = texcoords[unique_v_idx] if texcoords is not None and len(texcoords) == len(points) else texcoords

            return new_points, new_colors, new_faces, new_texcoords

        else:
            # Point-cloud cropping
            keep_mask = inside_mask if keep_inside else ~inside_mask

            new_points = points[keep_mask]
            new_colors = colors[keep_mask] if colors is not None and len(colors) == len(points) else colors

            return new_points, new_colors, None, None
