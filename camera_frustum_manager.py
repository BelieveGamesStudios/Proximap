"""
camera_frustum_manager.py
=========================
Manages the Reality Capture-style in-viewport camera display:

  - CameraThumbnailAtlasWorker  – background thread: downscale all source photos
    to 256 px thumbnails and pack them into a single 2-D RGBA numpy array
    (texture atlas).  Emits (atlas_rgba, uv_rects, cameras_data) when done.

  - CameraFrustumManager        – owns the VisPy Line (wireframes) and Mesh
    (textured image planes) visuals; handles scale / opacity changes; does
    screen-space hit-testing when the user clicks in the viewport.
"""

import os
import math
import numpy as np

from PySide6.QtCore import QThread, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_thumbnail(path: str, size: int = 256) -> np.ndarray:
    """Open *path* and return an (H, W, 4) uint8 RGBA array scaled to *size* px.
    Returns a plain grey tile on any error."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        # Keep aspect ratio, fit into size x size box
        img.thumbnail((size, size), Image.LANCZOS)
        # Paste onto a solid background to guarantee exact size
        canvas = Image.new("RGBA", (size, size), (30, 30, 30, 255))
        offset = ((size - img.width) // 2, (size - img.height) // 2)
        canvas.paste(img, offset)
        return np.array(canvas, dtype=np.uint8)
    except Exception:
        arr = np.full((size, size, 4), 50, dtype=np.uint8)
        arr[:, :, 3] = 255
        return arr


def _pack_atlas(thumbnails: list, tile_size: int = 256) -> tuple:
    """Pack *thumbnails* into a square RGBA texture atlas.

    Returns:
        atlas   – (atlas_h, atlas_w, 4) uint8 numpy array
        uv_list – list of (u0, v0, u1, v1) normalised UV coords per thumbnail
    """
    n = len(thumbnails)
    if n == 0:
        dummy = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        return dummy, []

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    atlas_w = cols * tile_size
    atlas_h = rows * tile_size
    atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    uv_list = []

    for idx, thumb in enumerate(thumbnails):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_size
        x0 = col * tile_size
        atlas[y0:y0 + tile_size, x0:x0 + tile_size] = thumb

        u0 = x0 / atlas_w
        v0 = y0 / atlas_h
        u1 = (x0 + tile_size) / atlas_w
        v1 = (y0 + tile_size) / atlas_h
        uv_list.append((u0, v0, u1, v1))

    return atlas, uv_list


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class CameraThumbnailAtlasWorker(QThread):
    """Downscale each camera's source image to a 256 px thumbnail and pack
    them into a single RGBA numpy texture atlas.

    Emits:
        finished(atlas_rgba, uv_list, cameras_data)
            atlas_rgba  – (H, W, 4) uint8 texture atlas
            uv_list     – list of (u0, v0, u1, v1) per camera (same order)
            cameras_data – the original list passed in (unchanged)
        error(str)
    """
    finished = Signal(object, object, object)
    error = Signal(str)

    def __init__(self, cameras_data: list, tile_size: int = 256, parent=None):
        super().__init__(parent)
        self.cameras_data = cameras_data
        self.tile_size = tile_size

    def run(self):
        try:
            thumbnails = []
            for cam in self.cameras_data:
                path = cam.get("image_path", "")
                thumbnails.append(_load_thumbnail(path, self.tile_size))

            atlas, uv_list = _pack_atlas(thumbnails, self.tile_size)
            self.finished.emit(atlas, uv_list, self.cameras_data)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

# Default frustum dimensions (world units)
_DEFAULT_DEPTH  = 0.15
_DEFAULT_HALF_W = 0.12
_DEFAULT_HALF_H = 0.08
_DEFAULT_OPACITY = 0.85


class CameraFrustumManager:
    """Owns and manages all camera-related VisPy visuals.

    Usage (in ProximapApp)::

        self.frustum_manager = CameraFrustumManager(self.view)
        cameras_data = self._read_images_binary(images_bin)
        self.frustum_manager.load(cameras_data)

    Then connect toolbar buttons::

        viewer_widget.btn_show_cameras.toggled.connect(
            self.frustum_manager.set_frustums_visible)
        viewer_widget.btn_show_photos.toggled.connect(
            self.frustum_manager.set_planes_visible)
    """

    def __init__(self, view):
        self._view = view
        self._cameras_data = []
        self._scale = 1.0
        self._opacity = _DEFAULT_OPACITY

        # VisPy visuals
        self._frustums_visual = None
        self._planes_visual = None
        self._highlight_visual = None

        self._atlas = None
        self._uv_list = []
        self._selected_index = -1
        self._planes_visible = True
        self._frustums_visible = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, cameras_data: list):
        """Set a new camera dataset and rebuild frustum wireframes immediately."""
        self._cameras_data = cameras_data
        self._selected_index = -1
        self._clear_all_visuals()
        if cameras_data:
            self._build_frustums()

    def apply_atlas(self, atlas: np.ndarray, uv_list: list, cameras_data: list):
        """Called from the UI thread once CameraThumbnailAtlasWorker finishes."""
        self._atlas = atlas
        self._uv_list = uv_list
        self._cameras_data = cameras_data
        self._build_planes()

    def set_frustums_visible(self, visible: bool):
        self._frustums_visible = visible
        if self._frustums_visual is not None:
            self._frustums_visual.visible = visible

    def set_planes_visible(self, visible: bool):
        self._planes_visible = visible
        if self._planes_visual is not None:
            self._planes_visual.visible = visible

    def set_scale(self, scale: float):
        self._scale = max(0.05, scale)
        if self._cameras_data:
            self._clear_frustum_visual()
            self._build_frustums()
            if self._atlas is not None and len(self._uv_list) > 0:
                self._clear_planes_visual()
                self._build_planes()
            if self._selected_index >= 0:
                self._clear_highlight_visual()
                self._build_highlight(self._selected_index)

    def set_opacity(self, opacity: float):
        self._opacity = max(0.0, min(1.0, opacity))
        if self._planes_visual is not None:
            self._clear_planes_visual()
            self._build_planes()

    def clear(self):
        self._cameras_data = []
        self._atlas = None
        self._uv_list = []
        self._selected_index = -1
        self._clear_all_visuals()

    def pick_camera(self, canvas_x: float, canvas_y: float,
                    canvas_width: float, canvas_height: float,
                    radius_px: float = 18.0) -> int:
        """Return the camera whose projected frustum is closest to the click.

        VisPy's scene transform maps world positions directly to homogeneous
        canvas coordinates.  After dividing by ``w``, ``x`` and ``y`` are
        already pixels; they must not be converted from NDC a second time.
        """
        if not self._cameras_data or not (self._frustums_visible or self._planes_visible):
            return -1

        try:
            tr = self._view.scene.transform
            best_idx = -1
            best_dist = radius_px

            def project(world_point):
                mapped = tr.map(np.array([
                    world_point[0], world_point[1], world_point[2], 1.0
                ], dtype=np.float64))
                if not np.all(np.isfinite(mapped)) or mapped[3] <= 1e-6:
                    return None
                return np.array([mapped[0] / mapped[3], mapped[1] / mapped[3]])

            def point_segment_distance(point, start, end):
                segment = end - start
                length_sq = float(np.dot(segment, segment))
                if length_sq <= 1e-12:
                    return float(np.linalg.norm(point - start))
                t = float(np.dot(point - start, segment) / length_sq)
                nearest = start + np.clip(t, 0.0, 1.0) * segment
                return float(np.linalg.norm(point - nearest))

            click = np.array([canvas_x, canvas_y], dtype=np.float64)

            for idx, cam_data in enumerate(self._cameras_data):
                world_points = [cam_data["center"], *self._world_corners_for(cam_data)]
                screen_points = [project(point) for point in world_points]
                if screen_points[0] is None:
                    continue

                # Camera origin plus the eight visible wireframe segments:
                # four rays from the origin and four image-plane edges.
                distances = [float(np.linalg.norm(click - screen_points[0]))]
                segment_indices = [
                    (0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 2), (2, 3), (3, 4), (4, 1),
                ]
                for start_idx, end_idx in segment_indices:
                    start = screen_points[start_idx]
                    end = screen_points[end_idx]
                    if start is not None and end is not None:
                        distances.append(point_segment_distance(click, start, end))

                dist = min(distances)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            return best_idx
        except Exception:
            return -1

    def select_camera(self, index: int):
        """Highlight camera at *index* with a cyan outline.  Pass -1 to clear."""
        self._selected_index = index
        self._clear_highlight_visual()
        if 0 <= index < len(self._cameras_data):
            self._build_highlight(index)

    def get_camera(self, index: int) -> dict:
        if 0 <= index < len(self._cameras_data):
            return self._cameras_data[index]
        return {}

    @property
    def camera_count(self) -> int:
        return len(self._cameras_data)

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def _frustum_dims(self):
        s = self._scale
        return _DEFAULT_DEPTH * s, _DEFAULT_HALF_W * s, _DEFAULT_HALF_H * s

    def _local_corners(self):
        d, w, h = self._frustum_dims()
        # In flipped photogrammetry space, -Z is the forward viewing direction toward the scene
        return np.array([
            [-w, -h, -d],  # 0: Bottom-Left
            [ w, -h, -d],  # 1: Bottom-Right
            [ w,  h, -d],  # 2: Top-Right
            [-w,  h, -d],  # 3: Top-Left
        ], dtype=np.float32)

    def _world_corners_for(self, cam: dict):
        C = cam["center"]
        R_cw = cam["R"].T
        return [R_cw @ c + C for c in self._local_corners()]

    def _build_frustums(self):
        from vispy import scene
        verts = []
        for cam in self._cameras_data:
            C = cam["center"]
            wc = self._world_corners_for(cam)
            for corner in wc:
                verts.append(C)
                verts.append(corner)
            for i in range(4):
                verts.append(wc[i])
                verts.append(wc[(i + 1) % 4])

        pos = np.array(verts, dtype=np.float32)
        self._frustums_visual = scene.visuals.Line(
            pos=pos,
            color="#FFFFFF",
            width=1.2,
            connect="segments",
            parent=self._view.scene,
        )
        self._frustums_visual.visible = self._frustums_visible

    def _build_planes(self):
        if self._atlas is None or len(self._uv_list) == 0:
            return

        from vispy import scene
        from vispy.visuals.filters import TextureFilter

        all_verts = []
        all_faces = []
        all_texcoords = []
        base_vert = 0

        for idx, cam in enumerate(self._cameras_data):
            if idx >= len(self._uv_list):
                break
            wc = self._world_corners_for(cam)
            u0, v0, u1, v1 = self._uv_list[idx]

            all_verts.extend(wc)
            # UV: bottom-left, bottom-right, top-right, top-left
            all_texcoords.extend([
                [u0, v1],
                [u1, v1],
                [u1, v0],
                [u0, v0],
            ])
            b = base_vert
            all_faces.append([b, b + 1, b + 2])
            all_faces.append([b, b + 2, b + 3])
            # Double-sided rendering so images are visible from both front and back
            all_faces.append([b, b + 2, b + 1])
            all_faces.append([b, b + 3, b + 2])
            base_vert += 4

        if not all_verts:
            return

        vertices = np.array(all_verts, dtype=np.float32)
        faces = np.array(all_faces, dtype=np.uint32)
        texcoords = np.array(all_texcoords, dtype=np.float32)

        vertex_colors = np.ones((len(vertices), 4), dtype=np.float32)
        vertex_colors[:, 3] = self._opacity

        self._planes_visual = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            parent=self._view.scene,
        )
        self._planes_visual.visible = self._planes_visible

        try:
            tex_filter = TextureFilter(self._atlas, texcoords)
            self._planes_visual.attach(tex_filter)
        except Exception:
            pass

    def _build_highlight(self, index: int):
        from vispy import scene
        cam = self._cameras_data[index]
        C = cam["center"]
        wc = self._world_corners_for(cam)

        verts = []
        for corner in wc:
            verts.append(C)
            verts.append(corner)
        for i in range(4):
            verts.append(wc[i])
            verts.append(wc[(i + 1) % 4])

        pos = np.array(verts, dtype=np.float32)
        self._highlight_visual = scene.visuals.Line(
            pos=pos,
            color="#00E676",
            width=2.5,
            connect="segments",
            parent=self._view.scene,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _clear_frustum_visual(self):
        if self._frustums_visual is not None:
            self._frustums_visual.parent = None
            self._frustums_visual = None

    def _clear_planes_visual(self):
        if self._planes_visual is not None:
            self._planes_visual.parent = None
            self._planes_visual = None

    def _clear_highlight_visual(self):
        if self._highlight_visual is not None:
            self._highlight_visual.parent = None
            self._highlight_visual = None

    def _clear_all_visuals(self):
        self._clear_frustum_visual()
        self._clear_planes_visual()
        self._clear_highlight_visual()
