import os
import sys
import subprocess
import ctypes
import json
import time
from typing import Optional
import numpy as np
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QPushButton, QProgressBar, QRadioButton, QButtonGroup,
        QFrame, QFileDialog, QTextEdit, QStackedWidget, QComboBox,
        QScrollArea, QTabWidget, QGridLayout, QCheckBox, QSlider,
        QMessageBox, QDialog, QColorDialog, QMenu, QSizePolicy, QInputDialog,
        QLineEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox
    )
except ModuleNotFoundError as e:
    missing_mod = getattr(e, 'name', 'PySide6')
    sys.exit(
        f"\n[ERROR] Missing required dependency: '{missing_mod}'\n\n"
        "To install all required dependencies, run:\n"
        "    pip install -r requirements.txt\n\n"
        "If you are using a virtual environment, activate it first:\n"
        "    source venv/bin/activate  # (Linux / macOS)\n"
        "    venv\\Scripts\\activate   # (Windows)\n"
        "    pip install -r requirements.txt\n"
    )

from vispy import app, scene
app.use_app("pyside6")

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

os.environ["U2NET_HOME"] = os.path.join(get_base_dir(), "models")

def get_reconstruction_out_dir():
    base_dir = get_base_dir()
    # Try writing a dummy file to check permissions
    try:
        test_file = os.path.join(base_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        # Base directory is writable, we can use it
        return os.path.join(base_dir, "reconstruction_out")
    except (IOError, OSError, PermissionError):
        # Base directory is read-only (e.g. Program Files). Fallback to AppData/Local/Proximap
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            out_dir = os.path.join(local_appdata, "Proximap", "reconstruction_out")
        else:
            # Fallback to user home
            out_dir = os.path.join(os.path.expanduser("~"), ".proximap", "reconstruction_out")
        return out_dir


def get_backup_dir():
    user_home = os.path.expanduser("~")
    backup_dir = os.path.join(user_home, ".proximap", "backup")
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir

def get_backup_metadata_path():
    return os.path.join(get_backup_dir(), "session_metadata.json")

def get_app_settings_path():
    user_home = os.path.expanduser("~")
    settings_dir = os.path.join(user_home, ".proximap")
    os.makedirs(settings_dir, exist_ok=True)
    return os.path.join(settings_dir, "app_settings.json")

def load_app_settings() -> dict:
    path = get_app_settings_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"dont_ask_recovery_on_startup": False}

def save_app_settings(settings: dict):
    path = get_app_settings_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[SETTINGS] Failed to save app settings: {e}")

def save_session_metadata(metadata: dict):
    path = get_backup_metadata_path()
    try:
        metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[BACKUP] Failed to save session metadata: {e}")

def load_session_metadata() -> Optional[dict]:
    path = get_backup_metadata_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[BACKUP] Failed to read session metadata: {e}")
    return None

def clear_backup_dir():
    import shutil
    bdir = get_backup_dir()
    if os.path.exists(bdir):
        for item in os.listdir(bdir):
            item_path = os.path.join(bdir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
            except Exception as e:
                print(f"[BACKUP] Warning: Could not clear backup item {item_path}: {e}")

def is_session_backup_valid() -> bool:
    """Returns True only if a session backup exists AND all required files (images/models) exist on disk."""
    meta = load_session_metadata()
    if not meta:
        return False

    # Standalone point clouds do not create or participate in session recovery
    if meta.get("scan_type") == "point_cloud" or meta.get("is_point_cloud", False):
        return False

    step = meta.get("last_completed_step", "unknown")
    if step in ["point_cloud_imported", "standalone_point_cloud"]:
        return False

    backup_dir = get_backup_dir()
    out_dir = get_reconstruction_out_dir()
    image_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    # If the step requires images to proceed, verify that image files actually exist on disk
    if step in ["images_imported", "features_extracted", "sparse_reconstruction"]:
        has_images = False
        candidates = [
            os.path.join(backup_dir, "images"),
            os.path.join(out_dir, "input_images"),
            os.path.join(out_dir, "extracted_frames")
        ]
        for cand in candidates:
            if os.path.exists(cand):
                for root, _, files in os.walk(cand):
                    if any(f.lower().endswith(image_exts) for f in files):
                        has_images = True
                        break
            if has_images:
                break
        if not has_images:
            return False

    # If the step is dense or mesh reconstruction, verify that model files exist
    if step in ["dense_reconstruction", "mesh_reconstruction"]:
        has_model = False
        for folder in [os.path.join(backup_dir, "mvs"), os.path.join(out_dir, "mvs")]:
            if os.path.exists(folder) and len(os.listdir(folder)) > 0:
                has_model = True
                break
        if not has_model:
            return False

    return True



from PySide6.QtCore import Qt, QSize, Signal, QTimer, QThread, QRectF
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QFont, QWindow, QPixmap, QImage, QPainter, QColor, QPen

import hardware_profiler


# Deferred imports for faster startup: MeshEditorWidget, PipelineWorker, BackgroundRemovalWorker

import http.server
import socketserver
import threading
import webbrowser

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.heic', '.heif', '.webp')
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.m4v')

DEFAULT_CAMERA_CONTROLS = (
    "<b>3D Viewport Controls:</b><br>"
    "• Left Click + Drag: Orbit Scene<br>"
    "• Right Click + Drag / Shift + Left Drag: Pan Scene<br>"
    "• Mouse Scroll: Zoom In / Out"
)

from vispy.scene.cameras import TurntableCamera
from vispy.util import keys


class ProximapTurntableCamera(TurntableCamera):
    """
    Custom turntable camera with increased panning speed and intuitive
    controls matching industry-standard 3D tools:
      - Left Drag: Orbit
      - Right Drag / Middle Drag / Shift+Left Drag: Pan (fast and responsive)
      - Mouse Scroll / Ctrl+Right Drag: Zoom
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translate_speed = 3.0

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return

        from vispy.scene.cameras.perspective import PerspectiveCamera
        PerspectiveCamera.viewbox_mouse_event(self, event)

        if event.type == 'mouse_release':
            self._event_value = None
        elif event.type == 'mouse_press':
            event.handled = True
        elif event.type == 'mouse_move':
            if event.press_event is None:
                return
            if 1 in event.buttons and 2 in event.buttons:
                return

            modifiers = event.mouse_event.modifiers
            p1 = event.mouse_event.press_event.pos
            p2 = event.mouse_event.pos
            d = p2 - p1

            if 1 in event.buttons and not modifiers:
                # Orbit
                self._update_rotation(event)
            elif (1 in event.buttons and keys.SHIFT in modifiers) or (2 in event.buttons and not modifiers) or (3 in event.buttons):
                # Panning with increased speed
                norm = np.mean(self._viewbox.size)
                if self._event_value is None or len(self._event_value) == 2:
                    self._event_value = self.center
                dist = (p1 - p2) / norm * self._scale_factor
                dist[1] *= -1
                dx, dy, dz = self._dist_to_trans(dist)
                ff = self._flip_factors
                up, forward, right = self._get_dim_vectors()
                dx, dy, dz = right * dx + forward * dy + up * dz
                dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
                c = self._event_value
                self.center = c[0] + dx, c[1] + dy, c[2] + dz
            elif 2 in event.buttons and keys.SHIFT in modifiers:
                # FOV adjustment
                if self._event_value is None:
                    self._event_value = self._fov
                fov = self._event_value - d[1] / 5.0
                self.fov = min(180.0, max(0.0, fov))
            elif 2 in event.buttons and (keys.CONTROL in modifiers or keys.ALT in modifiers):
                # Zoom
                if self._event_value is None:
                    self._event_value = (self._scale_factor, self._distance)
                zoomy = (1 + self.zoom_factor) ** d[1]
                self.scale_factor = self._event_value[0] * zoomy
                if self._distance is not None:
                    self._distance = self._event_value[1] * zoomy
                self.view_changed()


# ---------------------------------------------------------------------------
# Stepper Widget: Wraps a SpinBox with dedicated '-' and '+' buttons
# ---------------------------------------------------------------------------
class SpinBoxStepper(QWidget):
    """
    Wraps a QSpinBox or QDoubleSpinBox with dedicated '-' and '+' push buttons.
    Hides native arrow buttons to provide reliable, touch/click-friendly targets.
    """
    def __init__(self, spinbox, parent=None):
        super().__init__(parent)
        self.spin = spinbox
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setAlignment(Qt.AlignCenter)
        self.spin.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                background-color: #1E1E1E;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 2px 4px;
                font-size: 11px;
                min-height: 20px;
                max-width: 90px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        btn_style = """
            QPushButton {
                background-color: #2D2D2D;
                color: #00E676;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #444444;
                border-radius: 3px;
                min-width: 22px;
                max-width: 22px;
                min-height: 22px;
                max-height: 22px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #383838;
                border-color: #00E676;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #00E676;
                color: #121212;
            }
            QPushButton:disabled {
                background-color: #1C1C1C;
                color: #555555;
                border-color: #2A2A2A;
            }
        """

        self.btn_minus = QPushButton("-", self)
        self.btn_minus.setCursor(Qt.PointingHandCursor)
        self.btn_minus.setStyleSheet(btn_style)
        self.btn_minus.clicked.connect(self._step_down)

        self.btn_plus = QPushButton("+", self)
        self.btn_plus.setCursor(Qt.PointingHandCursor)
        self.btn_plus.setStyleSheet(btn_style)
        self.btn_plus.clicked.connect(self._step_up)

        layout.addWidget(self.btn_minus)
        layout.addWidget(self.spin, stretch=1)
        layout.addWidget(self.btn_plus)

    def _step_down(self):
        self.spin.stepBy(-1)

    def _step_up(self):
        self.spin.stepBy(1)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self.spin.setEnabled(enabled)
        self.btn_minus.setEnabled(enabled)
        self.btn_plus.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Background worker: loads a point cloud file off the UI thread on import
# ---------------------------------------------------------------------------
class CloudImportWorker(QThread):
    """Loads a point cloud in the background so the UI stays responsive."""
    finished = Signal(object, str)   # (LoadResult, file_path)
    error    = Signal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        try:
            import point_cloud_io
            res = point_cloud_io.load_point_cloud(self.file_path)
            self.finished.emit(res, self.file_path)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Background worker: parses geometry (PLY / Open3D) off the UI thread
# so _render_in_vispy can receive ready arrays and only do the fast GPU upload
# ---------------------------------------------------------------------------
class ViewerLoadWorker(QThread):
    """Parses point cloud / mesh data in a background thread."""
    # Emits (points_f32, colors_u8_or_None, faces_or_None, texcoords_or_None, texture_path_or_None)
    finished = Signal(object, object, object, object, object)
    error    = Signal(str)

    def __init__(self, file_path: str, mode: int, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.mode = mode

    def run(self):
        import numpy as np
        try:
            file_path = self.file_path
            mode      = self.mode
            points = colors = faces = texcoords = texture_path = None

            if mode == 0:
                # Sparse Point Cloud (decimated raw cloud for imported clouds or scene.ply)
                ply_path = file_path.replace(".mvs", ".ply")
                if os.path.exists(ply_path):
                    file_path = ply_path
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.ply':
                    points, colors, _ = _read_ply_static(file_path)
                else:
                    import point_cloud_io
                    res = point_cloud_io.load_point_cloud(file_path)
                    if res.success and res.cloud is not None:
                        points = np.asarray(res.cloud.points, dtype=np.float32)
                        colors = (np.asarray(res.cloud.colors) * 255).astype(np.uint8) \
                                 if res.has_colors else np.full((len(points), 3), 180, np.uint8)
                
                # Decimate raw points for sparse representation
                if points is not None and len(points) > 0:
                    stride = max(2, len(points) // 25000) if len(points) > 1000 else 1
                    points = points[::stride]
                    if colors is not None and len(colors) > 0:
                        colors = colors[::stride]

            elif mode == 1:
                # Dense Point Cloud
                ply_path = file_path.replace(".mvs", ".ply")
                if os.path.exists(ply_path):
                    file_path = ply_path
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.ply':
                    import struct
                    points, colors, _ = _read_ply_static(file_path)
                else:
                    import point_cloud_io
                    res = point_cloud_io.load_point_cloud(file_path)
                    if res.success and res.cloud is not None:
                        points = np.asarray(res.cloud.points, dtype=np.float32)
                        colors = (np.asarray(res.cloud.colors) * 255).astype(np.uint8) \
                                 if res.has_colors else np.full((len(points), 3), 180, np.uint8)

            elif mode == 2:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.ply':
                    points, colors, faces = _read_ply_static(file_path)
                elif ext not in ('.obj', '.ply'):
                    import point_cloud_io
                    res = point_cloud_io.load_point_cloud(file_path)
                    if res.success and res.cloud is not None:
                        points = np.asarray(res.cloud.points, dtype=np.float32)
                        colors = (np.asarray(res.cloud.colors) * 255).astype(np.uint8) \
                                 if res.has_colors else np.full((len(points), 3), 180, np.uint8)

            self.finished.emit(points, colors, faces, texcoords, texture_path)
        except Exception as e:
            self.error.emit(str(e))


def _read_ply_static(path):
    """Module-level PLY reader used by ViewerLoadWorker (no 'self' needed)."""
    import numpy as np, struct
    if not os.path.exists(path):
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.uint8), None
    try:
        with open(path, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            num_vertices = num_faces = 0
            format_type = None
            vertex_properties = []
            element_type = None
            for line in header_lines:
                parts = line.split()
                if not parts: continue
                if parts[0] == 'format':  format_type = parts[1]
                elif parts[0] == 'element':
                    element_type = parts[1]
                    if element_type == 'vertex': num_vertices = int(parts[2])
                    elif element_type == 'face': num_faces    = int(parts[2])
                elif parts[0] == 'property' and element_type == 'vertex':
                    if parts[1] == 'list':
                        vertex_properties.append((parts[4], 'list', True, parts[2], parts[3]))
                    else:
                        vertex_properties.append((parts[2], parts[1], False, None, None))
            type_map = {
                'char':(np.int8,1),'uchar':(np.uint8,1),'short':(np.int16,2),'ushort':(np.uint16,2),
                'int':(np.int32,4),'uint':(np.uint32,4),'float':(np.float32,4),'double':(np.float64,8),
                'int8':(np.int8,1),'uint8':(np.uint8,1),'int16':(np.int16,2),'uint16':(np.uint16,2),
                'int32':(np.int32,4),'uint32':(np.uint32,4),'float32':(np.float32,4),'float64':(np.float64,8)
            }
            type_char_map = {
                'char': 'b', 'uchar': 'B', 'short': 'h', 'ushort': 'H',
                'int': 'i', 'uint': 'I', 'float': 'f', 'double': 'd',
                'int8': 'b', 'uint8': 'B', 'int16': 'h', 'uint16': 'H',
                'int32': 'i', 'uint32': 'I', 'float32': 'f', 'float64': 'd'
            }
            type_sizes = {'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'f': 4, 'd': 8}

            has_list = any(p[2] for p in vertex_properties)
            points = np.zeros((num_vertices, 3), dtype=np.float32)
            colors = None
            faces  = None
            prop_names = [p[0] for p in vertex_properties]
            has_color  = all(c in prop_names for c in ('red','green','blue')) or all(c in prop_names for c in ('r','g','b'))
            if has_color:
                colors = np.zeros((num_vertices, 3), dtype=np.uint8)

            if 'binary' in (format_type or ''):
                if has_list:
                    fixed_properties = []
                    list_properties = []
                    for p in vertex_properties:
                        if p[2]: list_properties.append(p)
                        else:
                            if not list_properties: fixed_properties.append(p)

                    fmt_chars = [type_char_map[t] for name, t, _, _, _ in fixed_properties if t in type_char_map]
                    fixed_size = sum(type_sizes[c] for c in fmt_chars)
                    endian_flag = '>' if 'big' in format_type else '<'
                    fixed_struct = struct.Struct(endian_flag + ''.join(fmt_chars))

                    names = [p[0] for p in fixed_properties]
                    x_idx = names.index('x') if 'x' in names else -1
                    y_idx = names.index('y') if 'y' in names else -1
                    z_idx = names.index('z') if 'z' in names else -1

                    r_name = 'red' if 'red' in names else ('r' if 'r' in names else None)
                    g_name = 'green' if 'green' in names else ('g' if 'g' in names else None)
                    b_name = 'blue' if 'blue' in names else ('b' if 'b' in names else None)

                    r_idx = names.index(r_name) if r_name else -1
                    g_idx = names.index(g_name) if g_name else -1
                    b_idx = names.index(b_name) if b_name else -1

                    data = f.read()
                    offset = 0

                    for i in range(num_vertices):
                        val = fixed_struct.unpack_from(data, offset)
                        if x_idx != -1: points[i, 0] = val[x_idx]
                        if y_idx != -1: points[i, 1] = val[y_idx]
                        if z_idx != -1: points[i, 2] = val[z_idx]

                        if has_color and colors is not None:
                            if r_idx != -1: colors[i, 0] = val[r_idx]
                            if g_idx != -1: colors[i, 1] = val[g_idx]
                            if b_idx != -1: colors[i, 2] = val[b_idx]

                        offset += fixed_size

                        for name, _, _, count_type, item_type in list_properties:
                            c_char = type_char_map[count_type]
                            c_size = type_sizes[c_char]
                            count = struct.unpack_from(endian_flag + c_char, data, offset)[0]
                            offset += c_size

                            i_char = type_char_map[item_type]
                            i_size = type_sizes[i_char]
                            offset += count * i_size

                    if num_faces > 0 and len(data) > offset:
                        try:
                            faces_list = []
                            count_fmt = endian_flag + 'B'
                            while offset < len(data) and len(faces_list) < num_faces:
                                cnt = struct.unpack_from(count_fmt, data, offset)[0]
                                offset += 1
                                idxs = list(struct.unpack_from(f'{endian_flag}{cnt}I', data, offset))
                                offset += 4 * cnt
                                if len(idxs) == 3: faces_list.append(idxs)
                            if faces_list: faces = np.array(faces_list, dtype=np.int32)
                        except Exception: pass
                else:
                    fixed_props = [p for p in vertex_properties if not p[2]]
                    stride = sum(type_map[p[1]][1] for p in fixed_props if p[1] in type_map)
                    raw = f.read(stride * num_vertices)
                    offset = 0
                    col_offsets = {}
                    for p in fixed_props:
                        if p[1] in type_map:
                            col_offsets[p[0]] = offset
                            offset += type_map[p[1]][1]
                    dt_fields = []
                    for p in fixed_props:
                        if p[1] in type_map:
                            dt_fields.append((p[0], type_map[p[1]][0]))
                    if dt_fields:
                        dt = np.dtype(dt_fields)
                        arr = np.frombuffer(raw, dtype=dt)
                        if 'x' in arr.dtype.names: points[:,0] = arr['x'].astype(np.float32)
                        if 'y' in arr.dtype.names: points[:,1] = arr['y'].astype(np.float32)
                        if 'z' in arr.dtype.names: points[:,2] = arr['z'].astype(np.float32)
                        if has_color and colors is not None:
                            r_key = 'red' if 'red' in arr.dtype.names else 'r'
                            g_key = 'green' if 'green' in arr.dtype.names else 'g'
                            b_key = 'blue' if 'blue' in arr.dtype.names else 'b'
                            if r_key in arr.dtype.names: colors[:,0] = arr[r_key]
                            if g_key in arr.dtype.names: colors[:,1] = arr[g_key]
                            if b_key in arr.dtype.names: colors[:,2] = arr[b_key]
                    if num_faces > 0:
                        faces_list = []
                        count_fmt = '>B' if 'big' in (format_type or '') else '<B'
                        idx_fmt   = '>I' if 'big' in (format_type or '') else '<I'
                        for _ in range(num_faces):
                            cnt = struct.unpack(count_fmt, f.read(1))[0]
                            idxs = list(struct.unpack(f'<{cnt}I', f.read(4*cnt)))
                            if len(idxs) == 3: faces_list.append(idxs)
                        if faces_list: faces = np.array(faces_list, dtype=np.int32)
            else:
                for i in range(num_vertices):
                    vals = f.readline().decode('utf-8', errors='ignore').split()
                    if len(vals) >= 3:
                        points[i] = [float(vals[0]), float(vals[1]), float(vals[2])]
                        if has_color and colors is not None and len(vals) >= 6:
                            ri = prop_names.index('red') if 'red' in prop_names else (prop_names.index('r') if 'r' in prop_names else 3)
                            gi = prop_names.index('green') if 'green' in prop_names else (prop_names.index('g') if 'g' in prop_names else 4)
                            bi = prop_names.index('blue') if 'blue' in prop_names else (prop_names.index('b') if 'b' in prop_names else 5)
                            try: colors[i] = [int(vals[ri]), int(vals[gi]), int(vals[bi])]
                            except: pass
                if num_faces > 0:
                    faces_list = []
                    for _ in range(num_faces):
                        vals = f.readline().decode('utf-8', errors='ignore').split()
                        if vals and int(vals[0]) == 3 and len(vals) >= 4:
                            faces_list.append([int(vals[1]), int(vals[2]), int(vals[3])])
                    if faces_list: faces = np.array(faces_list, dtype=np.int32)
        if points is not None and len(points) > 0:
            from point_cloud_io import apply_photogrammetry_coordinate_flip
            points, _, _, _ = apply_photogrammetry_coordinate_flip(points=points)
        return points, colors, faces
    except Exception as e:
        print(f"[ERROR] Failed to parse PLY in _read_ply_static ({path}): {e}")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), None


def _weld_exact_duplicate_vertices(points, faces, colors=None):
    """Restore triangle adjacency lost when textured OBJ corners were expanded."""
    points_array = np.asarray(points)
    if faces is None or len(faces) == 0 or len(points_array) == 0:
        return points_array, faces, colors

    faces_array = np.asarray(faces, dtype=np.int64)
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        return points_array, faces, colors

    valid = np.all((faces_array >= 0) & (faces_array < len(points_array)), axis=1)
    faces_array = faces_array[valid]

    unique_points, first_indices, inverse = np.unique(
        points_array, axis=0, return_index=True, return_inverse=True
    )
    welded_faces = inverse[faces_array]
    nondegenerate = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
    )
    welded_faces = welded_faces[nondegenerate].astype(np.int32, copy=False)

    welded_colors = colors
    if colors is not None and len(colors) == len(points_array):
        welded_colors = np.asarray(colors)[first_indices]

    return unique_points, welded_faces, welded_colors


class ModelServerHandler(http.server.BaseHTTPRequestHandler):

    model_path = ""
    
    def log_message(self, format, *args):
        # Suppress standard logging to console for clean output
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
        
    def do_HEAD(self):
        if not os.path.exists(self.model_path):
            self.send_response(404)
            self.end_headers()
            return
            
        try:
            file_size = os.path.getsize(self.model_path)
            self.send_response(200)
            self.send_header('Content-Type', 'model/gltf-binary')
            self.send_header('Content-Length', str(file_size))
            self.end_headers()
        except Exception:
            self.send_response(500)
            self.end_headers()

    def do_GET(self):
        if not os.path.exists(self.model_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Model not found")
            return
            
        try:
            with open(self.model_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'model/gltf-binary')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Server error: {e}".encode())

class LoopbackServerThread(threading.Thread):
    def __init__(self, file_path, port=53120):
        super().__init__()
        self.file_path = file_path
        self.port = port
        self.daemon = True
        self.httpd = None
        
    def run(self):
        # We need a unique handler class instance since model_path is a class attribute
        class CustomHandler(ModelServerHandler):
            model_path = self.file_path

        # Try finding a free port starting at 53120
        while self.port < 53200:
            try:
                self.httpd = socketserver.TCPServer(("127.0.0.1", self.port), CustomHandler)
                break
            except OSError:
                self.port += 1
                
        if self.httpd:
            self.httpd.serve_forever()
            
    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
class DragDropArea(QFrame):
    """
    Custom widget designed as a prominent drag-and-drop landing container.
    """
    images_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("DragDropArea")
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        self.icon_label = QLabel("📥", self)
        self.icon_label.setStyleSheet("font-size: 64px; margin-bottom: 15px;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        
        self.instruction_label = QLabel("Drag images/videos or folder here to start", self)
        self.instruction_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #b3b3b3;")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        
        self.sub_label = QLabel("Supports JPG, PNG, TIFF, MP4, MOV, AVI, MKV", self)
        self.sub_label.setStyleSheet("font-size: 12px; color: #737373;")
        self.sub_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.icon_label)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.sub_label)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            self.setStyleSheet("border: 2px dashed #00E676; background-color: #213328;")
            event.acceptProposedAction()
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet("")
        
    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("")
        files = []
        ignored = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if os.path.isdir(local_path):
                # Scan folder for images/videos
                for root, _, filenames in os.walk(local_path):
                    for filename in filenames:
                        fp = os.path.join(root, filename)
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in IMAGE_EXTS or ext in VIDEO_EXTS or ext == '.ply':
                            files.append(os.path.normpath(fp))
                        else:
                            ignored.append(filename)
            elif os.path.isfile(local_path):
                ext = os.path.splitext(local_path)[1].lower()
                if ext in IMAGE_EXTS or ext in VIDEO_EXTS or ext == '.ply':
                    files.append(os.path.normpath(local_path))
                else:
                    ignored.append(os.path.basename(local_path))
                    
        if ignored:
            from PySide6.QtWidgets import QMessageBox
            msg = "The following files were ignored because they are not supported images, videos, or .ply point clouds:\n\n"
            if len(ignored) > 10:
                msg += "\n".join(ignored[:10]) + f"\n... and {len(ignored) - 10} more files."
            else:
                msg += "\n".join(ignored)
            QMessageBox.warning(self, "Unsupported Files Ignored", msg)

        if files:
            self.images_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()





class ViewerWrapperWidget(QFrame):
    """
    Widget wrapper that hosts the native VisPy 3D scene canvas
    and provides a control bar to reload, switch cameras, or change MVS scene modes.
    Now also acts as the main drag-and-drop landing area!
    """
    images_dropped = Signal(list)
    reload_requested = Signal(str)  # Emits target file path to reload
    camera_changed = Signal(int)  # Emits selected camera index (0: Arcball, 1: Turntable)
    selection_mode_changed = Signal(str)  # Emits mode: 'none', 'box', 'lasso', 'crop_box'
    remove_outside_requested = Signal()
    reset_crop_requested = Signal()
    finalize_crop_requested = Signal()
    delete_selection_requested = Signal()
    clear_selection_requested = Signal()
    invert_selection_requested = Signal()
    retexture_mesh_tool_requested = Signal()
    transform_cloud_requested = Signal()
    shading_mode_changed = Signal(str)  # Emits 'wireframe' or 'solid'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setAcceptDrops(True)
        self.setObjectName("ViewerWrapperWidget")
        self.setStyleSheet("background-color: #1A1A1A; border: 1px solid #2B2B2B; border-radius: 8px;")
        
        self._current_selection_mode = 'none'

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header/Control bar
        self.control_bar = QFrame(self)
        self.control_bar.setFixedHeight(50)
        self.control_bar.setStyleSheet("background-color: #242424; border-bottom: 1px solid #2B2B2B; border-top-left-radius: 8px; border-top-right-radius: 8px;")
        control_layout = QHBoxLayout(self.control_bar)
        control_layout.setContentsMargins(10, 5, 10, 5)
        
        # Dropdown File Menu
        self.file_menu_btn = QPushButton("File", self.control_bar)
        self.file_menu_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                padding: 4px 10px;
                font-weight: normal;
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                margin-left: 0px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00E676;
            }
            QPushButton::menu-indicator {
                image: none;
                width: 0px;
            }
        """)
        self.file_menu = QMenu(self)
        self.file_menu.setStyleSheet("""
            QMenu {
                background-color: #242424;
                color: #ffffff;
                border: 1px solid #2B2B2B;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #00E676;
                color: #121212;
            }
            QMenu::item:disabled {
                color: #555555;
            }
        """)
        self.action_new = self.file_menu.addAction("New Project")
        self.action_new.setShortcut("Ctrl+N")
        self.file_menu.addSeparator()
        self.action_save = self.file_menu.addAction("Save Project (.pxm)")
        self.action_load = self.file_menu.addAction("Load Project (.pxm)")
        self.action_recover = self.file_menu.addAction("Recover Last Session")
        self.file_menu.addSeparator()
        
        # Unified sub-menu for importing media, archives, & point clouds
        self.import_menu = QMenu("Import", self.file_menu)
        self.import_menu.setStyleSheet(self.file_menu.styleSheet())
        
        self.action_import_media = self.import_menu.addAction("Media Files (Images/Videos)...")
        self.action_import_dir = self.import_menu.addAction("Media Directory / Folder...")
        self.action_import_zip = self.import_menu.addAction("ZIP Archive (.zip)...")
        self.import_menu.addSeparator()
        self.action_import_mobile = self.import_menu.addAction("From Mobile Device (Local Network)...")
        self.action_import_point_cloud = self.import_menu.addAction("Point Cloud (.ply)...")
        
        # Backward compatibility aliases
        self.action_import_standalone = self.action_import_point_cloud
        self.action_mobile_import = self.action_import_mobile

        self.file_menu.addMenu(self.import_menu)
        self.file_menu.addSeparator()

        self.action_export_dense = self.file_menu.addAction("Export Dense")
        self.action_export_sparse = self.file_menu.addAction("Export Sparse")
        
        # Unified sub-menu for exporting consolidated meshes (format-specific sub-actions)
        self.export_mesh_menu = QMenu("Export Mesh", self.file_menu)
        self.export_mesh_menu.setStyleSheet(self.file_menu.styleSheet())
        
        self.action_export_glb = self.export_mesh_menu.addAction("GLB (.glb)")
        self.action_export_obj = self.export_mesh_menu.addAction("OBJ (.obj)")
        self.action_export_usdz = self.export_mesh_menu.addAction("USDZ (.usdz)")
        
        self.file_menu.addMenu(self.export_mesh_menu)
        self.file_menu.addSeparator()
        self.action_mobile_export = self.file_menu.addAction("Send 3D Model to Mobile")
        self.file_menu.addSeparator()
        self.action_upload_proximap = self.file_menu.addAction("Upload to Proximap")
        
        self.action_export_dense.setEnabled(False)
        self.action_export_sparse.setEnabled(False)
        self.action_export_glb.setEnabled(False)
        self.action_export_obj.setEnabled(False)
        self.action_export_usdz.setEnabled(False)
        self.action_mobile_export.setEnabled(False)
        self.action_upload_proximap.setEnabled(False)
        
        self.file_menu_btn.setMenu(self.file_menu)


        # "Show Controls" toggle checkbox (right side of toolbar)
        self.show_controls_cb = QCheckBox("Show Controls", self.control_bar)
        self.show_controls_cb.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                font-size: 11px;
                margin-left: 10px;
                margin-right: 10px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
        """)

        # View mode selector (Sparse / Dense / Textured Mesh)
        self.mode_select = QComboBox(self.control_bar)
        self.mode_select.setMinimumWidth(200)
        self.mode_select.addItems([
            "Sparse Point Cloud & Cameras",
            "Dense Point Cloud",
            "Textured Mesh",
        ])

        # Background colour picker button
        self.bg_btn = QPushButton("BG Color", self.control_bar)
        self.bg_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                padding: 4px 8px;
                font-weight: normal;
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00E676;
            }
        """)


        # View Shading segmented toggle (Wireframe vs Solid)
        self.shading_frame = QFrame(self.control_bar)
        self.shading_frame.setObjectName("ShadingFrame")
        self.shading_frame.setStyleSheet("""
            QFrame#ShadingFrame {
                background-color: #242424;
                border: 1px solid #3A3A3A;
                border-radius: 6px;
            }
        """)
        shading_layout = QHBoxLayout(self.shading_frame)
        shading_layout.setContentsMargins(2, 2, 2, 2)
        shading_layout.setSpacing(2)

        self.shading_group = QButtonGroup(self.control_bar)
        self.shading_group.setExclusive(True)

        public_dir = os.path.join(get_base_dir(), "public")
        wf_icon_path = os.path.join(public_dir, "proximap wireframe.png")
        solid_icon_path = os.path.join(public_dir, "proximap solid.png")

        shading_btn_style = """
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #444444;
            }
            QPushButton:checked {
                background-color: #1B382B;
                border: 1px solid #00E676;
            }
        """

        self.btn_wireframe_mode = QPushButton(self.shading_frame)
        self.btn_wireframe_mode.setCheckable(True)
        self.btn_wireframe_mode.setFixedSize(28, 24)
        self.btn_wireframe_mode.setToolTip("Wireframe Mode")
        self.btn_wireframe_mode.setIconSize(QSize(20, 20))
        if os.path.exists(wf_icon_path):
            self.btn_wireframe_mode.setIcon(QIcon(wf_icon_path))
        self.btn_wireframe_mode.setStyleSheet(shading_btn_style)

        self.btn_solid_mode = QPushButton(self.shading_frame)
        self.btn_solid_mode.setCheckable(True)
        self.btn_solid_mode.setChecked(True)
        self.btn_solid_mode.setFixedSize(28, 24)
        self.btn_solid_mode.setToolTip("Solid Surface Mode")
        self.btn_solid_mode.setIconSize(QSize(20, 20))
        if os.path.exists(solid_icon_path):
            self.btn_solid_mode.setIcon(QIcon(solid_icon_path))
        self.btn_solid_mode.setStyleSheet(shading_btn_style)

        self.shading_group.addButton(self.btn_wireframe_mode, 0)
        self.shading_group.addButton(self.btn_solid_mode, 1)

        shading_layout.addWidget(self.btn_wireframe_mode)
        shading_layout.addWidget(self.btn_solid_mode)

        self.btn_wireframe_mode.clicked.connect(lambda: self.shading_mode_changed.emit("wireframe"))
        self.btn_solid_mode.clicked.connect(lambda: self.shading_mode_changed.emit("solid"))

        # --- Assemble toolbar ---
        control_layout.addWidget(self.file_menu_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.show_controls_cb)
        control_layout.addWidget(self.mode_select)
        control_layout.addWidget(self.bg_btn)
        control_layout.addWidget(self.shading_frame)
        
        layout.addWidget(self.control_bar)
        
        # Container for the embedded window
        self.container_area = QWidget(self)
        self.container_area_layout = QVBoxLayout(self.container_area)
        self.container_area_layout.setContentsMargins(0, 0, 0, 0)
        self.container_area_layout.setSpacing(0)
        
        # A simple fallback label when no viewer is running
        self.fallback_label = QLabel("Drag Images/Videos Here or Process to View 3D Scene", self.container_area)
        self.fallback_label.setAlignment(Qt.AlignCenter)
        self.fallback_label.setStyleSheet("color: #737373; font-size: 14px;")
        self.container_area_layout.addWidget(self.fallback_label)
        
        layout.addWidget(self.container_area)
        
        # Setup actions
        self.mode_select.currentIndexChanged.connect(self._on_mode_changed)
        
        self.current_mvs_dir = None

    def set_tool_modals_busy(self, busy: bool):
        """Enables/disables buttons and controls on tool modals during async tasks."""
        pass

    def set_selection_mode(self, mode_name: str):
        self._current_selection_mode = mode_name
        self.selection_mode_changed.emit(mode_name)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        main_win = self._get_main_window()
        if main_win and hasattr(main_win, '_position_overlay'):
            main_win._position_overlay()
        w = self.width()
            
        if w < 600:
            self.show_controls_cb.setText("Controls")
            self.mode_select.setMinimumWidth(140)
            self.bg_btn.setText("Color")
        else:
            self.show_controls_cb.setText("Show Controls")
            self.mode_select.setMinimumWidth(200)
            self.bg_btn.setText("BG Color")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            self.setStyleSheet("background-color: #213328; border: 2px dashed #00E676; border-radius: 8px;")
            event.acceptProposedAction()
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet("background-color: #1A1A1A; border: 1px solid #2B2B2B; border-radius: 8px;")
        
    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("background-color: #1A1A1A; border: 1px solid #2B2B2B; border-radius: 8px;")
        files = []
        ignored = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if os.path.isdir(local_path):
                # Scan folder for images/videos
                for root, _, filenames in os.walk(local_path):
                    for filename in filenames:
                        fp = os.path.join(root, filename)
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in IMAGE_EXTS or ext in VIDEO_EXTS or ext == '.ply':
                             files.append(os.path.normpath(fp))
                        else:
                             ignored.append(filename)
            elif os.path.isfile(local_path):
                ext = os.path.splitext(local_path)[1].lower()
                if ext in IMAGE_EXTS or ext in VIDEO_EXTS or ext == '.ply':
                    files.append(os.path.normpath(local_path))
                else:
                    ignored.append(os.path.basename(local_path))
                    
        if ignored:
            from PySide6.QtWidgets import QMessageBox
            msg = "The following files were ignored because they are not supported images, videos, or .ply point clouds:\n\n"
            if len(ignored) > 10:
                msg += "\n".join(ignored[:10]) + f"\n... and {len(ignored) - 10} more files."
            else:
                msg += "\n".join(ignored)
            QMessageBox.warning(self, "Unsupported Files Ignored", msg)

        if files:
            self.images_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()

    def set_mvs_directory(self, mvs_dir: str):
        self.current_mvs_dir = mvs_dir

    def _get_main_window(self):
        if hasattr(self, 'main_window') and self.main_window is not None:
            return self.main_window
        p = self.parent()
        while p:
            if hasattr(p, 'standalone_cloud_path'):
                return p
            p = p.parent() if hasattr(p, 'parent') else None
        return None

    def get_selected_file_path(self) -> str:
        main_win = self._get_main_window()
        if main_win and getattr(main_win, 'standalone_cloud_path', None):
            standalone_path = main_win.standalone_cloud_path
            mvs_dir = self.current_mvs_dir if self.current_mvs_dir else os.path.join(get_reconstruction_out_dir(), "mvs")
            has_reconstruction = os.path.exists(os.path.join(mvs_dir, "scene_dense_mesh_refine.ply")) or \
                                 os.path.exists(os.path.join(mvs_dir, "scene_dense_mesh.ply"))
            index = self.mode_select.currentIndex()
            if index in (0, 1) or not has_reconstruction:
                return standalone_path

        if not self.current_mvs_dir:
            return None
            
        index = self.mode_select.currentIndex()
        if index == 0:
            return os.path.join(self.current_mvs_dir, "scene.mvs")
        elif index == 1:
            return os.path.join(self.current_mvs_dir, "scene_dense.mvs")
        elif index == 2:
            # We want to load the textured mesh.
            for candidate in [
                "scene_dense_mesh_texture.obj",
                "scene_dense_mesh_texture.ply",
                "scene_dense_mesh_refine.ply",
                "scene_dense_mesh.ply",
                "scene_mesh.ply",
                "scene_dense_mesh_texture.glb",
                "scene_dense_mesh_texture.mvs",
                "scene_dense_mesh_refine.mvs",
                "scene_dense.mvs"
            ]:
                path = os.path.join(self.current_mvs_dir, candidate)
                if os.path.exists(path):
                    return path
            return os.path.join(self.current_mvs_dir, "scene_dense_mesh_texture.obj")
        return None

    def _on_back_clicked(self):
        self.back_requested.emit()


    def set_shading_mode(self, mode_name: str):
        if hasattr(self, 'btn_wireframe_mode') and hasattr(self, 'btn_solid_mode'):
            if mode_name == "wireframe":
                self.btn_wireframe_mode.setChecked(True)
            else:
                self.btn_solid_mode.setChecked(True)

    def update_crop_box_state(self):
        is_textured_mesh = (self.mode_select.currentIndex() == 2)
        if hasattr(self, 'shading_frame'):
            self.shading_frame.setEnabled(is_textured_mesh)
        if not is_textured_mesh and getattr(self, '_current_selection_mode', None) in ['crop_box', 'box', 'lasso']:
            self.set_selection_mode('none')

    def _on_mode_changed(self, index):
        self.update_crop_box_state()
        main_win = self._get_main_window()
        if main_win and hasattr(main_win, '_update_sidebar_tools_visibility'):
            main_win._update_sidebar_tools_visibility(index)
        path = self.get_selected_file_path()
        if path:
            self.reload_requested.emit(path)

class ProjectProgressDialog(QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent, Qt.WindowTitleHint)
        self.setWindowTitle(title)
        self.setFixedSize(300, 120)
        self.setModal(True)
        if parent:
            self.setStyleSheet(parent.styleSheet())
            
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        self.msg_label = QLabel(message, self)
        self.msg_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        self.msg_label.setAlignment(Qt.AlignCenter)
        
        # A simple infinite progress bar to act as a loading indicator/spinner
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0) # Indeterminate mode
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3A3A3A;
                background-color: #222222;
                height: 6px;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #00E676;
                border-radius: 3px;
            }
        """)
        
        layout.addWidget(self.msg_label)
        layout.addWidget(self.progress)


class SaveProjectDialog(QDialog):
    """
    Dialog for saving a Proximap project (.pxm).
    Offers target path selection and an 'Include Images' toggle with dynamic warning.
    """
    def __init__(self, default_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Project (.pxm)")
        self.setMinimumWidth(480)
        self.setModal(True)
        self.selected_file_path = default_path
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 11px;
            }
            QLineEdit {
                background-color: #121212;
                border: 1px solid #333333;
                border-radius: 4px;
                color: #ffffff;
                padding: 6px 10px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #00E676;
            }
            QPushButton {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #383838;
                border-color: #555555;
            }
            QPushButton#primaryBtn {
                background-color: #00C853;
                color: #121212;
                border: 1px solid #00C853;
            }
            QPushButton#primaryBtn:hover {
                background-color: #00E676;
            }
            QCheckBox {
                color: #e0e0e0;
                font-size: 11px;
                font-weight: 500;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #161616;
            }
            QCheckBox::indicator:checked {
                background-color: #00E676;
                border-color: #00E676;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Title
        title_lbl = QLabel("Save Proximap Project Archive", self)
        title_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title_lbl)

        # Destination path row
        path_label = QLabel("Project File Destination:", self)
        layout.addWidget(path_label)

        path_row = QHBoxLayout()
        path_row.setSpacing(8)

        self.path_edit = QLineEdit(self)
        self.path_edit.setText(default_path)
        path_row.addWidget(self.path_edit, stretch=1)

        self.browse_btn = QPushButton("Browse...", self)
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        path_row.addWidget(self.browse_btn)

        layout.addLayout(path_row)

        # Include images checkbox
        self.include_images_check = QCheckBox("Include reconstruction images in project file", self)
        self.include_images_check.setChecked(False)
        self.include_images_check.setCursor(Qt.PointingHandCursor)
        self.include_images_check.toggled.connect(self._on_include_images_toggled)
        layout.addWidget(self.include_images_check)

        # Warning banner widget (Amber/Orange accent callout)
        self.warning_banner = QWidget(self)
        self.warning_banner.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 171, 0, 0.12);
                border: 1px solid #FFAB00;
                border-radius: 5px;
            }
        """)
        wb_layout = QHBoxLayout(self.warning_banner)
        wb_layout.setContentsMargins(12, 8, 12, 8)
        wb_layout.setSpacing(8)

        self.warning_text = QLabel(
            "⚠️ Note: Including images will significantly increase the .pxm file size and consume more disk space.",
            self.warning_banner
        )
        self.warning_text.setWordWrap(True)
        self.warning_text.setStyleSheet("color: #FFD54F; font-size: 11px; font-weight: 500; border: none; background: transparent;")
        wb_layout.addWidget(self.warning_text)

        self.warning_banner.setVisible(False)
        layout.addWidget(self.warning_banner)

        layout.addSpacing(6)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.save_btn = QPushButton("Save Project", self)
        self.save_btn.setObjectName("primaryBtn")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self._on_save_clicked)
        btn_row.addWidget(self.save_btn)

        layout.addLayout(btn_row)

    def _on_include_images_toggled(self, checked: bool):
        self.warning_banner.setVisible(checked)
        self.adjustSize()

    def _on_browse_clicked(self):
        cur_path = self.path_edit.text().strip()
        init_dir = os.path.dirname(cur_path) if cur_path else ""
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Proximap Project",
            cur_path or init_dir,
            "Proximap Project (*.pxm)"
        )
        if chosen:
            if not chosen.lower().endswith(".pxm"):
                chosen += ".pxm"
            self.path_edit.setText(chosen)

    def _on_save_clicked(self):
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Invalid Destination", "Please specify a destination file path.")
            return
        if not path.lower().endswith(".pxm"):
            path += ".pxm"
        self.selected_file_path = path
        self.accept()

    def get_result(self):
        return self.selected_file_path, self.include_images_check.isChecked()


class SaveWorker(QThread):
    finished = Signal(bool, str) # Emits (success, message)
    
    def __init__(self, mvs_dir, file_path, include_images=False, colmap_images_dir=None, image_list=None):
        super().__init__()
        self.mvs_dir = mvs_dir
        self.file_path = file_path
        self.include_images = include_images
        self.colmap_images_dir = colmap_images_dir
        self.image_list = image_list or []
        
    def run(self):
        import zipfile
        try:
            with zipfile.ZipFile(self.file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if self.include_images:
                    # Write reconstruction assets under mvs/
                    for root, _, files in os.walk(self.mvs_dir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, self.mvs_dir)
                            zip_path = os.path.join("mvs", rel_path)
                            zipf.write(full_path, zip_path)
                            
                    # Write calibrated images under colmap/images/
                    packed_images = 0
                    if self.colmap_images_dir and os.path.isdir(self.colmap_images_dir):
                        for root, _, files in os.walk(self.colmap_images_dir):
                            for file in files:
                                full_path = os.path.join(root, file)
                                rel_path = os.path.relpath(full_path, self.colmap_images_dir)
                                zip_path = os.path.join("colmap", "images", rel_path)
                                zipf.write(full_path, zip_path)
                                packed_images += 1
                    elif self.image_list:
                        for img_path in self.image_list:
                            if os.path.isfile(img_path):
                                fname = os.path.basename(img_path)
                                zip_path = os.path.join("colmap", "images", fname)
                                zipf.write(img_path, zip_path)
                                packed_images += 1
                    self.finished.emit(True, f"Project saved successfully with {packed_images} images included.")
                else:
                    for root, _, files in os.walk(self.mvs_dir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, self.mvs_dir)
                            zipf.write(full_path, rel_path)
                    self.finished.emit(True, "Project saved successfully.")
        except Exception as e:
            self.finished.emit(False, str(e))


class LoadWorker(QThread):
    finished = Signal(bool, str, str) # Emits (success, mvs_dir, message)
    
    def __init__(self, file_path, temp_root):
        super().__init__()
        self.file_path = file_path
        self.temp_root = temp_root
        
    def run(self):
        import zipfile
        import uuid
        try:
            extract_dir = os.path.join(self.temp_root, f"proximap_project_{uuid.uuid4()}")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(self.file_path, 'r') as zipf:
                zipf.extractall(extract_dir)
                
            # Locate mvs assets
            mvs_candidate = os.path.join(extract_dir, "mvs")
            if os.path.isdir(mvs_candidate):
                mvs_dir = mvs_candidate
            else:
                mvs_dir = extract_dir
                
            found_asset = False
            for root, _, files in os.walk(mvs_dir):
                if any(f.endswith((".mvs", ".ply", ".obj", ".glb", ".gltf")) for f in files):
                    found_asset = True
                    mvs_dir = root
                    break
                    
            if not found_asset:
                for root, _, files in os.walk(extract_dir):
                    if any(f.endswith((".mvs", ".ply", ".obj", ".glb", ".gltf")) for f in files):
                        found_asset = True
                        mvs_dir = root
                        break
                        
            if not found_asset:
                self.finished.emit(False, "", "Invalid project file: No 3D model or reconstruction assets found in archive.")
                return
            
            self.finished.emit(True, mvs_dir, "Project loaded successfully.")
        except Exception as e:
            self.finished.emit(False, "", str(e))


def create_point_cloud_thumbnail(size, filename=""):
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Background rounded rect with emerald border and dark bg
    painter.setBrush(QColor(22, 34, 28))
    painter.setPen(QPen(QColor(0, 230, 118, 200), 1.5))
    margin = 4
    painter.drawRoundedRect(margin, margin, size - margin * 2, size - margin * 2, 6, 6)
    
    # Text badge
    painter.setPen(QColor("#00E676"))
    font = painter.font()
    font.setPointSize(max(10, size // 6))
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(QRectF(0, size * 0.22, size, size * 0.35), Qt.AlignCenter, "PLY")
    
    font.setPointSize(max(7, size // 11))
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor("#A5D6A7"))
    painter.drawText(QRectF(0, size * 0.58, size, size * 0.25), Qt.AlignCenter, "Point Cloud")
    
    painter.end()
    return pix


class ThumbnailWorker(QThread):
    """
    Background worker that loads and scales images to QImage asynchronously.
    """
    thumbnail_loaded = Signal(str, QImage)  # Emits (file_path, scaled_qimage)
    finished_loading = Signal()

    def __init__(self, file_paths, target_size):
        super().__init__()
        self.file_paths = file_paths
        self.target_size = target_size
        self._is_running = True

    def run(self):
        for path in self.file_paths:
            if not self._is_running:
                break
            if path.lower().endswith('.ply'):
                pix = create_point_cloud_thumbnail(self.target_size, os.path.basename(path))
                self.thumbnail_loaded.emit(path, pix.toImage())
                continue
            # Load the image using QImage (which is thread-safe for background loading/scaling)
            image = QImage(path)
            if not image.isNull():
                scaled_image = image.scaled(
                    self.target_size, self.target_size,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.thumbnail_loaded.emit(path, scaled_image)
            else:
                self.thumbnail_loaded.emit(path, QImage())
        self.finished_loading.emit()

    def stop(self):
        self._is_running = False


class PhotoItemWidget(QWidget):
    """
    Individual photo / dataset thumbnail card display with a selection checkbox.
    Supports a placeholder initially and lazy updates.
    """
    def __init__(self, file_path, size, pixmap=None, parent=None, parent_grid=None):
        super().__init__(parent)
        self.file_path = file_path
        self.size = size
        self.selected = False
        self.pixmap = pixmap
        self.parent_grid = parent_grid
        self.init_ui()
        
    def init_ui(self):
        self.setFixedSize(self.size + 16, self.size + 40)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Container for the thumbnail image
        self.image_container = QFrame(self)
        self.image_container.setObjectName("ImageContainer")
        self.image_container.setFixedSize(self.size + 8, self.size + 8)
        self.image_container.setStyleSheet("""
            QFrame#ImageContainer {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: #1A1A1A;
            }
            QFrame#ImageContainer:hover {
                border-color: #00E676;
            }
        """)
        
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(2, 2, 2, 2)
        
        self.image_label = QLabel(self.image_container)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        if self.pixmap is not None:
            self.image_label.setPixmap(self.pixmap)
        elif self.file_path.lower().endswith('.ply'):
            pix = create_point_cloud_thumbnail(self.size, os.path.basename(self.file_path))
            self.image_label.setPixmap(pix)
        else:
            # Show a loading placeholder state
            self.image_label.setText("⏳")
            self.image_label.setStyleSheet("font-size: 20px; color: #888888;")
            
        container_layout.addWidget(self.image_label)
        
        # Checkbox & Name layout
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(2, 0, 2, 0)
        bottom_layout.setSpacing(4)
        
        self.checkbox = QCheckBox(self)
        self.checkbox.setFixedWidth(16)
        self.checkbox.clicked.connect(self._on_checkbox_clicked)
        
        self.name_label = QLabel(os.path.basename(self.file_path), self)
        self.name_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        self.name_label.setToolTip(self.file_path)
        
        # Elide text if too long
        metrics = self.name_label.fontMetrics()
        elided = metrics.elidedText(os.path.basename(self.file_path), Qt.ElideRight, self.size - 10)
        self.name_label.setText(elided)
        
        bottom_layout.addWidget(self.checkbox)
        bottom_layout.addWidget(self.name_label)
        
        layout.addWidget(self.image_container)
        layout.addLayout(bottom_layout)
        
    def _on_checkbox_clicked(self, checked):
        from PySide6.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()
        if hasattr(self, 'parent_grid') and self.parent_grid:
            self.parent_grid.handle_item_clicked(self, modifiers, target_checked=checked)
        else:
            self.set_checked(checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if hasattr(self, 'parent_grid') and self.parent_grid:
                self.parent_grid.handle_item_clicked(self, event.modifiers())
            else:
                self.set_checked(not self.selected)
        super().mousePressEvent(event)
        
    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        if not pixmap.isNull():
            self.image_label.setText("")
            self.image_label.setStyleSheet("")
            self.image_label.setPixmap(pixmap)
        elif self.file_path.lower().endswith('.ply'):
            pix = create_point_cloud_thumbnail(self.size, os.path.basename(self.file_path))
            self.image_label.setPixmap(pix)
        else:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("⚠️")
            self.image_label.setStyleSheet("font-size: 20px; color: #ff1744;")
            
    def _update_style(self):
        if self.selected:
            self.image_container.setStyleSheet("QFrame#ImageContainer { border: 2px solid #00E676; border-radius: 4px; background-color: #213328; }")
        else:
            self.image_container.setStyleSheet("""
                QFrame#ImageContainer {
                    border: 1px solid #333333;
                    border-radius: 4px;
                    background-color: #1A1A1A;
                }
                QFrame#ImageContainer:hover {
                    border-color: #00E676;
                }
            """)

    def set_checked(self, checked):
        self.selected = bool(checked)
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(self.selected)
        self.checkbox.blockSignals(False)
        self._update_style()


class PhotosGridWidget(QWidget):
    """
    Grid container that dynamically arranges PhotoItemWidgets depending on container width.
    Supports Shift+Click range selection and single-item toggles.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.image_items = []
        self.image_paths = []
        self.thumbnail_size = 100
        self.item_widgets = {}  # Map path -> PhotoItemWidget for dynamic updates
        self.current_cols = 0
        self.last_clicked_index = None
        
    def set_images(self, image_paths):
        self.image_paths = image_paths
        self.rebuild_grid(force=True)

    def clear_grid(self):
        while self.layout.count() > 0:
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.image_items.clear()
        self.item_widgets.clear()
        self.last_clicked_index = None
        
    def handle_item_clicked(self, item_widget, modifiers, target_checked=None):
        try:
            current_idx = self.image_items.index(item_widget)
        except ValueError:
            return

        if (modifiers & Qt.ShiftModifier) and self.last_clicked_index is not None and 0 <= self.last_clicked_index < len(self.image_items):
            start = min(self.last_clicked_index, current_idx)
            end = max(self.last_clicked_index, current_idx)
            for i in range(start, end + 1):
                self.image_items[i].set_checked(True)
        else:
            new_state = target_checked if target_checked is not None else (not item_widget.selected)
            item_widget.set_checked(new_state)
            self.last_clicked_index = current_idx

    def rebuild_grid(self, force=False):
        if not self.image_paths:
            self.clear_grid()
            self.current_cols = 0
            self.last_clicked_index = None
            return
            
        width = self.width()
        if width < 100:
            width = 400  # Fallback minimum width estimation
            
        col_width = self.thumbnail_size + 20
        cols = max(1, width // col_width)
        
        # If the number of columns hasn't changed and we aren't forcing a rebuild, do nothing
        if not force and cols == self.current_cols:
            return
            
        self.current_cols = cols
        
        # Check if we can reuse the existing widgets to avoid recreating them
        can_reuse = (not force and 
                     len(self.image_items) == len(self.image_paths) and
                     all(w.file_path == p for w, p in zip(self.image_items, self.image_paths)))
                     
        if can_reuse:
            # Just rearrange the existing widgets
            for item in self.image_items:
                self.layout.removeWidget(item)
                
            for idx, item_widget in enumerate(self.image_items):
                row = idx // cols
                col = idx % cols
                self.layout.addWidget(item_widget, row, col)
        else:
            # Rebuild from scratch
            self.clear_grid()
            cache = getattr(self.tab_widget, "thumbnail_cache", {})
            
            for idx, path in enumerate(self.image_paths):
                pixmap = cache.get(path)
                item_widget = PhotoItemWidget(path, self.thumbnail_size, pixmap, self, parent_grid=self)
                self.image_items.append(item_widget)
                self.item_widgets[path] = item_widget
                
                row = idx // cols
                col = idx % cols
                self.layout.addWidget(item_widget, row, col)
                
        self.layout.invalidate()
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rebuild_grid(force=False)


class PhotosTabWidget(QWidget):
    """
    Tab widget containing the Photos toolbar and dynamic photo grid area.
    Loads images asynchronously using a background thread and caches thumbnails.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_list = []
        self.thumbnail_cache = {}  # Map path -> QPixmap
        self.loader_thread = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        self.toolbar = QFrame(self)
        self.toolbar.setFixedHeight(38)
        self.toolbar.setStyleSheet("""
            QFrame {
                background-color: #1A1A1A;
                border-bottom: 1px solid #2B2B2B;
            }
            QToolTip {
                background-color: #242424;
                color: #FFFFFF;
                border: 1px solid #4D4D4D;
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 11px;
                font-weight: 500;
            }
        """)
        toolbar_layout = QHBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(10, 4, 10, 4)
        toolbar_layout.setSpacing(6)
        
        # Buttons matching reference UI functionality
        public_dir = os.path.join(get_base_dir(), "public")
        
        self.btn_select_all = QPushButton("Select All", self.toolbar)
        self.btn_select_all.setToolTip("Select All Photos")
        self.btn_select_all.setStyleSheet("""
            QPushButton {
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
                background-color: #242424;
                color: #e0e0e0;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #00E676;
                color: #ffffff;
            }
        """)

        self.btn_deselect_all = QPushButton("Deselect All", self.toolbar)
        self.btn_deselect_all.setToolTip("Deselect All Photos")
        self.btn_deselect_all.setStyleSheet("""
            QPushButton {
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
                background-color: #242424;
                color: #e0e0e0;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #00E676;
                color: #ffffff;
            }
        """)
        
        self.btn_remove_selected = QPushButton("", self.toolbar)
        self.btn_remove_selected.setIcon(QIcon(os.path.join(public_dir, "trash.png")))
        self.btn_remove_selected.setToolTip("Remove Selected")
        self.btn_remove_selected.setStyleSheet("QPushButton { padding: 4px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
        add_photos_icon_path = os.path.join(get_base_dir(), "interface element", "add photos.svg")
        self.btn_add_photos = QPushButton("", self.toolbar)
        self.btn_add_photos.setIcon(QIcon(add_photos_icon_path))
        self.btn_add_photos.setIconSize(QSize(20, 20))
        self.btn_add_photos.setToolTip("Add Files to Dataset (Images, Videos, Point Cloud)")
        self.btn_add_photos.setStyleSheet("QPushButton { padding: 4px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
        self.btn_bg_remove = QPushButton("Remove BG", self.toolbar)
        self.btn_bg_remove.setToolTip("Remove image backgrounds offline model")
        self.btn_bg_remove.setEnabled(False)
        self.btn_bg_remove.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
                font-size: 11px;
                font-weight: bold;
                background-color: #1a3325;
                color: #00E676;
                border: 1px solid #00E676;
                border-radius: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #00E676;
                color: #121212;
                border-color: #00E676;
            }
            QPushButton:pressed:enabled {
                background-color: #00c853;
                color: #121212;
                border-color: #00c853;
            }
            QPushButton:disabled {
                background-color: #202020;
                color: #555555;
                border: 1px solid #2D2D2D;
            }
        """)

        # Thumbnail size slider
        self.size_label = QLabel("Size:", self.toolbar)
        self.size_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.size_slider = QSlider(Qt.Horizontal, self.toolbar)
        self.size_slider.setRange(60, 200)
        self.size_slider.setValue(100)
        self.size_slider.setFixedWidth(80)
        self.size_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3A3A3A;
                height: 4px;
                background: #2D2D2D;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00E676;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
        """)
        
        toolbar_layout.addWidget(self.btn_select_all)
        toolbar_layout.addWidget(self.btn_deselect_all)
        toolbar_layout.addWidget(self.btn_remove_selected)
        toolbar_layout.addWidget(self.btn_add_photos)
        toolbar_layout.addWidget(self.btn_bg_remove)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.size_label)
        toolbar_layout.addWidget(self.size_slider)
        
        layout.addWidget(self.toolbar)
        
        # Scroll Area for Grid
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { background-color: #121212; border: none; }")
        
        self.grid_widget = PhotosGridWidget(self.scroll_area)
        self.grid_widget.tab_widget = self
        self.scroll_area.setWidget(self.grid_widget)
        
        layout.addWidget(self.scroll_area)
        
        # Connections
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_deselect_all.clicked.connect(self.deselect_all)
        self.size_slider.valueChanged.connect(self.change_thumbnail_size)
        
    def set_dataset_controls_enabled(self, enabled: bool):
        """Enables or disables dataset modifying buttons (Delete, Add Photos, Remove BG)."""
        self._dataset_controls_locked = not enabled
        if hasattr(self, 'btn_remove_selected'):
            self.btn_remove_selected.setEnabled(enabled)
        if hasattr(self, 'btn_add_photos'):
            self.btn_add_photos.setEnabled(enabled)
        if hasattr(self, 'btn_bg_remove'):
            if enabled:
                image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
                has_2d = any(isinstance(p, str) and p.lower().endswith(image_exts) for p in (self.image_list or []))
                self.btn_bg_remove.setEnabled(bool(self.image_list) and has_2d)
            else:
                self.btn_bg_remove.setEnabled(False)

    def set_images(self, image_paths):
        self.image_list = image_paths
        if hasattr(self, 'btn_bg_remove'):
            if not getattr(self, '_dataset_controls_locked', False):
                image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
                has_2d_images = any(isinstance(p, str) and p.lower().endswith(image_exts) for p in (image_paths or []))
                self.btn_bg_remove.setEnabled(bool(image_paths) and has_2d_images)
            else:
                self.btn_bg_remove.setEnabled(False)
        
        # 1. Stop any current loader thread
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.stop()
            self.loader_thread.wait()
            
        # 2. Clear cache keys of images that were removed
        current_set = set(image_paths)
        removed_keys = [k for k in self.thumbnail_cache.keys() if k not in current_set]
        for k in removed_keys:
            del self.thumbnail_cache[k]
            
        # 3. Find files that are not yet cached
        uncached_paths = [p for p in image_paths if p not in self.thumbnail_cache]
        
        # 4. Refresh grid immediately with placeholders or cached items
        self.grid_widget.set_images(image_paths)
        
        # 5. Start background thread loader for uncached paths
        if uncached_paths:
            self.loader_thread = ThumbnailWorker(uncached_paths, self.grid_widget.thumbnail_size)
            self.loader_thread.thumbnail_loaded.connect(self.on_thumbnail_loaded)
            self.loader_thread.start()
            
    def on_thumbnail_loaded(self, path, scaled_image):
        # Convert QImage to QPixmap in the GUI thread
        if not scaled_image.isNull():
            pixmap = QPixmap.fromImage(scaled_image)
        else:
            pixmap = QPixmap()
            
        # Add to memory cache
        self.thumbnail_cache[path] = pixmap
        
        # Update the live widget in grid if it is still displayed
        if path in self.grid_widget.item_widgets:
            self.grid_widget.item_widgets[path].set_pixmap(pixmap)
            
    def select_all(self):
        for item in self.grid_widget.image_items:
            item.set_checked(True)
            
    def deselect_all(self):
        for item in self.grid_widget.image_items:
            item.set_checked(False)
        self.grid_widget.last_clicked_index = None
            
    def get_selected_images(self):
        selected = []
        for item in self.grid_widget.image_items:
            if item.selected:
                selected.append(item.file_path)
        return selected
        
    def change_thumbnail_size(self, value):
        self.grid_widget.thumbnail_size = value
        
        # Reset the thumbnail cache entirely because size changed!
        self.thumbnail_cache.clear()
        
        # Reload images with the new size
        self.set_images(self.image_list)
        
    def closeEvent(self, event):
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.stop()
            self.loader_thread.wait()
        super().closeEvent(event)


class UploadProgressDialog(QDialog):
    """
    Loading modal dialog indicating that a model is being uploaded to Proximap cloud.
    """
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle("Proximap Cloud Upload")
        self.setFixedSize(380, 190)
        self.setModal(True)
        
        # Inherit styling for consistent UI aesthetics
        self.setStyleSheet(parent.styleSheet() if parent else "")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)
        
        self.title_label = QLabel("Uploading 3D Model...", self)
        self.title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        self.info_label = QLabel(
            "Please check your web browser window.\n"
            "Your model is currently uploading from your local workspace.\n"
            "Keep this dialog open until the browser confirms completion.", 
            self
        )
        self.info_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        
        self.done_btn = QPushButton("Done", self)
        self.done_btn.setStyleSheet("""
            QPushButton {
                background-color: #00E676;
                color: #121212;
                font-weight: bold;
                padding: 8px 20px;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #00FF87;
            }
            QPushButton:pressed {
                background-color: #00B35C;
            }
        """)
        self.done_btn.clicked.connect(self.accept)
        
        # Center the button horizontally
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.done_btn)
        btn_layout.addStretch()
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.info_label)
        layout.addLayout(btn_layout)


class SessionRecoveryDialog(QDialog):
    def __init__(self, metadata: dict, parent=None):
        super().__init__(parent)
        self.metadata = metadata
        self.setWindowTitle("Recover Previous Session")
        self.resize(520, 360)
        self.user_choice = "cancel"  # "resume", "discard", "cancel"

        self.setStyleSheet("""
            QDialog {
                background-color: #1A1A1A;
                color: #FFFFFF;
            }
            QLabel {
                color: #E0E0E0;
            }
            QCheckBox {
                color: #CCCCCC;
                font-size: 12px;
            }
            QPushButton {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #00E676;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Title
        title_lbl = QLabel("Previous Session Checkpoint Found", self)
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #00E676;")
        layout.addWidget(title_lbl)

        # Stage friendly names
        step_raw = metadata.get("last_completed_step", "unknown")
        step_map = {
            "images_imported": "Images Imported (Staged)",
            "features_extracted": "Extracted Features & Pair Matches",
            "sparse_reconstruction": "Sparse Point Cloud (Colmap SfM)",
            "dense_reconstruction": "Dense Point Cloud (OpenMVS)",
            "mesh_reconstruction": "Textured Mesh Reconstruction"
        }
        step_friendly = step_map.get(step_raw, step_raw)

        # Info Box
        info_frame = QFrame(self)
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #242424;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 12px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(6)

        info_layout.addWidget(QLabel(f"<b>Last Completed Stage:</b> <span style='color:#00E676;'>{step_friendly}</span>"))
        info_layout.addWidget(QLabel(f"<b>Timestamp:</b> {metadata.get('timestamp', 'N/A')}"))
        info_layout.addWidget(QLabel(f"<b>Image Count:</b> {metadata.get('image_count', 0)}"))
        info_layout.addWidget(QLabel(f"<b>Quality Preset:</b> {metadata.get('quality_preset', 'Medium').capitalize()}"))
        info_layout.addWidget(QLabel(f"<b>Mesh Mode:</b> {metadata.get('mesh_mode', 'Default').capitalize()}"))

        layout.addWidget(info_frame)

        prompt_lbl = QLabel("Would you like to resume this session from where it left off?")
        prompt_lbl.setStyleSheet("font-size: 13px; color: #CCCCCC;")
        layout.addWidget(prompt_lbl)

        # Don't ask again checkbox
        self.chk_dont_ask = QCheckBox("Don't prompt automatically on application startup", self)
        layout.addWidget(self.chk_dont_ask)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        
        btn_discard = QPushButton("Discard Backup", self)
        btn_discard.setStyleSheet("""
            QPushButton {
                background-color: #3A1A1A;
                color: #FF5252;
                border: 1px solid #FF5252;
            }
            QPushButton:hover {
                background-color: #4A1A1A;
            }
        """)
        btn_discard.clicked.connect(self._on_discard)

        btn_cancel = QPushButton("Later", self)
        btn_cancel.clicked.connect(self.reject)

        btn_resume = QPushButton("Resume Reconstruction", self)
        btn_resume.setStyleSheet("""
            QPushButton {
                background-color: #00E676;
                color: #121212;
                font-weight: bold;
                border: 1px solid #00E676;
            }
            QPushButton:hover {
                background-color: #00C853;
            }
        """)
        btn_resume.clicked.connect(self._on_resume)

        btn_layout.addWidget(btn_discard)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_resume)

        layout.addLayout(btn_layout)

    def _on_resume(self):
        self.user_choice = "resume"
        self._update_settings()
        self.accept()

    def _on_discard(self):
        self.user_choice = "discard"
        self._update_settings()
        self.accept()

    def _update_settings(self):
        if self.chk_dont_ask.isChecked():
            settings = load_app_settings()
            settings["dont_ask_recovery_on_startup"] = True
            save_app_settings(settings)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proximap 1.5.0")
        self.setMinimumSize(1100, 750)
        
        base_dir = get_base_dir()
        icon_path = os.path.join(base_dir, "app_icon.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(base_dir, "public", "app_icon.png")
        if os.path.exists(icon_path):
            from PySide6.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))
            
        self.image_list = []
        self.worker = None
        self._last_failed_stage = None
        
        # Load hardware properties
        self.total_ram_gb = hardware_profiler.get_total_memory() / (1024**3)
        self.available_ram_gb = hardware_profiler.get_available_memory() / (1024**3)
        self.dgpu_detected = not hardware_profiler.use_low_hardware_fallback
        
        # Initialize VisPy Canvas & Visual references
        self.canvas = None
        self.view = None
        self.markers_visual = None
        self.mesh_visual = None
        self.cameras_visual = None
        self.grid_visual = None
        self.origin_visual = None
        self.transform_gizmo = None
        self.crop_box = None
        self._last_points = None

        # Camera frustum / image-plane feature state
        self.frustum_manager = None
        self._camera_frustums_visible = True
        self._camera_photos_visible = True
        self._camera_frustum_scale = 1.0
        self._camera_photo_opacity = 0.85
        self._atlas_worker = None
        self.camera_preview_widget = None
        self._look_through_timer = None
        self._look_through_target = None   # (azimuth, elevation, center) to animate to
        self._look_through_saved = None    # saved orbit params before look-through
        self._look_through_badge = None    # QLabel shown in viewport top-centre
        self._look_through_steps = 0
        self._look_through_step = 0
        
        # Raw un-cropped geometry arrays (for Reset Crop)
        self._raw_points = None
        self._raw_colors = None
        self._raw_faces = None
        self._raw_texcoords = None
        self._raw_texture_path = None
        
        # Active geometry arrays (reflecting live crops and selections)
        self._current_points = None
        self._current_colors = None
        self._current_faces = None
        self._current_texcoords = None
        self._current_texture_path = None
        
        # 2D Screen-space selection states
        self._selected_vertex_indices = None
        self.selection_markers_visual = None
        self.selection_overlay = None
        self.nav_gizmo = None
        self.floating_toolbox = None
        self.editor_tool_host = None
        self.transform_tool_window = None
        self.mesh_cleanup_tool_window = None
        self._cumulative_mesh_transform = np.eye(4, dtype=np.float32)
        
        self.last_accessed_dir = os.path.expanduser("~")
        self.viewport_bg_color = '#0C0C0C'
        
        self._clear_reconstruction_out()
        self._init_ui()
        self._apply_styling()
        QTimer.singleShot(0, self._check_existing_scene)



    def _clear_reconstruction_out(self):
        """Clears temporary files in reconstruction_out on startup, but retains valid COLMAP database checkpoints."""
        import shutil
        out_dir = get_reconstruction_out_dir()
        db_checkpoint = os.path.join(out_dir, "colmap", "database.db")
        
        has_checkpoint = False
        if os.path.exists(db_checkpoint):
            try:
                import sqlite3
                conn = sqlite3.connect(db_checkpoint)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM images")
                n_imgs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0")
                n_pairs = cur.fetchone()[0]
                conn.close()
                if n_imgs >= 2 and n_pairs >= 1:
                    has_checkpoint = True
            except Exception:
                has_checkpoint = False

        if os.path.exists(out_dir):
            for item in os.listdir(out_dir):
                item_path = os.path.join(out_dir, item)
                try:
                    if item == "colmap" and has_checkpoint:
                        print("[INFO] Preserving valid database.db checkpoint in colmap/ directory on startup.")
                        colmap_dir = item_path
                        for sub_item in os.listdir(colmap_dir):
                            if sub_item == "database.db":
                                continue
                            sub_path = os.path.join(colmap_dir, sub_item)
                            if os.path.isfile(sub_path) or os.path.islink(sub_path):
                                os.unlink(sub_path)
                            elif os.path.isdir(sub_path):
                                shutil.rmtree(sub_path, ignore_errors=True)
                        continue

                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                except Exception as e:
                    print(f"[WARNING] Could not clear {item_path} on startup: {e}")

    def _init_ui(self):
        # Main Tabbed Interface
        self.main_tabs = QTabWidget(self)
        self.main_tabs.setObjectName("MainTabs")
        self.setCentralWidget(self.main_tabs)
        
        # 3D Reconstruction Tab
        reconstruction_tab = QWidget(self.main_tabs)
        main_layout = QHBoxLayout(reconstruction_tab)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Left Side Control Panel (Wizard Steps)
        sidebar = QFrame(self)
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(400)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # Fixed Title Bar at the top of the sidebar
        title_container = QWidget(sidebar)
        title_container.setStyleSheet("background-color: #1A1A1A; border-top-left-radius: 8px; border-top-right-radius: 8px;")
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(20, 20, 20, 10)
        
        title_label = QLabel("Workflow", title_container)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff; padding-bottom: 10px; border-bottom: 1px solid #3d3d3d;")
        title_layout.addWidget(title_label)
        sidebar_layout.addWidget(title_container)
        
        # Scroll Area for the steps
        scroll_area = QScrollArea(sidebar)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("QScrollArea { background-color: #1A1A1A; border: none; }")
        
        scroll_content = QWidget()
        scroll_content.setObjectName("ScrollContent")
        scroll_content.setStyleSheet("QWidget#ScrollContent { background-color: #1A1A1A; }")
        # Constrain scroll_content to exact sidebar width so no child widget can expand it
        scroll_content.setMaximumWidth(400)
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(12, 10, 12, 15)
        scroll_content_layout.setSpacing(16)
        
        # STEP 1: Import Images
        step1_box = QFrame(scroll_content)
        step1_box.setObjectName("StepBox")
        step1_layout = QVBoxLayout(step1_box)
        
        s1_title = QLabel("Step 1: Import Images", step1_box)
        s1_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #00E676;")
        
        self.img_count_label = QLabel("Images Loaded: 0", step1_box)
        self.camera_label = QLabel("Camera: Undetected", step1_box)
        self.camera_label.setTextFormat(Qt.PlainText)
        self.camera_label.setWordWrap(True)
        self.camera_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        
        # Hardware Status Badge
        self.badge = QLabel("Memory Check...", step1_box)
        self.badge.setObjectName("Badge")
        self.badge.setAlignment(Qt.AlignCenter)
        self._update_system_badge()
        
        self.browse_files_btn = QPushButton("Select Images/Videos", step1_box)
        self.browse_files_btn.clicked.connect(self._open_files_dialog)

        self.browse_btn = QPushButton("Select Images/Videos Directory", step1_box)
        self.browse_btn.clicked.connect(self._open_dir_dialog)
        
        self.mobile_import_btn = QPushButton("Import from Mobile Device", step1_box)
        self.mobile_import_btn.clicked.connect(self._on_import_from_mobile_clicked)
        
        # Add-on Panels Container (Step 1)
        self.addon_container = QWidget(step1_box)
        self.addon_container_layout = QVBoxLayout(self.addon_container)
        self.addon_container_layout.setContentsMargins(0, 0, 0, 0)
        self.addon_container_layout.setSpacing(4)
        
        # Standalone Cloud Import Container (Populated when importing via File Menu)
        self.standalone_cloud_container = QWidget(step1_box)
        standalone_cloud_layout = QHBoxLayout(self.standalone_cloud_container)
        standalone_cloud_layout.setContentsMargins(0, 0, 0, 0)
        standalone_cloud_layout.setSpacing(5)
        
        self.standalone_cloud_label = QLabel("", self.standalone_cloud_container)
        self.standalone_cloud_label.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold;")
        self.standalone_cloud_label.setTextFormat(Qt.PlainText)

        self.standalone_cloud_clear_btn = QPushButton("✕", self.standalone_cloud_container)
        self.standalone_cloud_clear_btn.setFixedSize(20, 20)
        self.standalone_cloud_clear_btn.setToolTip("Clear standalone cloud selection")
        self.standalone_cloud_clear_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #ff5252;
                border: 1px solid #ff5252;
                border-radius: 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #ff5252;
                color: #ffffff;
            }
        """)
        self.standalone_cloud_clear_btn.clicked.connect(self._clear_standalone_cloud_clicked)
        
        standalone_cloud_layout.addWidget(self.standalone_cloud_label, stretch=1)
        standalone_cloud_layout.addWidget(self.standalone_cloud_clear_btn)
        self.standalone_cloud_container.setVisible(False)
        
        self.standalone_cloud_path = None
        
        step1_layout.addWidget(s1_title)
        step1_layout.addWidget(self.img_count_label)
        step1_layout.addWidget(self.camera_label)
        step1_layout.addWidget(self.badge)
        step1_layout.addWidget(self.browse_files_btn)
        step1_layout.addWidget(self.browse_btn)
        step1_layout.addWidget(self.mobile_import_btn)
        step1_layout.addWidget(self.addon_container)

        scroll_content_layout.addWidget(step1_box)
        
        # STEP 2: Process
        step2_box = QFrame(scroll_content)
        step2_box.setObjectName("StepBox")
        step2_layout = QVBoxLayout(step2_box)
        step2_layout.setContentsMargins(8, 8, 8, 8)
        step2_layout.setSpacing(6)
        
        s2_title_row = QWidget(step2_box)
        s2_title_row_layout = QHBoxLayout(s2_title_row)
        s2_title_row_layout.setContentsMargins(0, 0, 0, 0)
        s2_title_row_layout.setSpacing(4)

        s2_title = QLabel("Step 2: Reconstruction", s2_title_row)
        s2_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #00E676;")
        s2_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.recon_mode_combo = QComboBox(s2_title_row)
        self.recon_mode_combo.addItems(["Simple", "Advanced"])
        self.recon_mode_combo.setCurrentIndex(0)
        self.recon_mode_combo.setFixedWidth(100)
        self.recon_mode_combo.setToolTip("Switch between Simple (guided) and Advanced (manual) reconstruction configuration.")

        s2_title_row_layout.addWidget(s2_title, 1)
        s2_title_row_layout.addWidget(self.recon_mode_combo, 0)
        
        self.quality_label = QLabel("Processing Quality:", step2_box)
        self.quality_combo = QComboBox(step2_box)
        self.quality_combo.addItems([
            "Preview (Fast, reduced density)",
            "Medium (Balanced — recommended)",
            "High (ULTRA features + full densification)",
            "Ultra (Maximum detail — very slow)"
        ])
        self.quality_combo.setCurrentIndex(1)  # Default to Medium
        
        self.gpu_label = QLabel("Hardware Acceleration:", step2_box)
        self.gpu_combo = QComboBox(step2_box)
        self.gpu_combo.addItems([
            "Auto-Detect",
            "Force GPU (CUDA)",
            "Force CPU Fallback"
        ])
        
        self.auto_cleanup_checkbox = QCheckBox("Auto Cleanup", step2_box)
        self.auto_cleanup_checkbox.setChecked(False)
        self.auto_cleanup_checkbox.setToolTip(
            "Automates mesh repair and quadric edge collapse decimation between refinement and texturing.\n"
            "Default: Off."
        )
        self.mc_enabled = self.auto_cleanup_checkbox

        self.manhattan_align_checkbox = QCheckBox("Manhattan Alignment (Auto-Level)", step2_box)
        self.manhattan_align_checkbox.setChecked(True)
        self.manhattan_align_checkbox.setVisible(False)
        self.manhattan_align_checkbox.setToolTip(
            "Automatically levels the 3D model and aligns ground/walls to the coordinate grid using COLMAP Manhattan-World vanishing point analysis.\n"
            "Default: On."
        )
        
        # Advanced Options Collapsible Panel
        self.advanced_toggle_btn = QPushButton("▸  Advanced Options", step2_box)
        self.advanced_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.advanced_toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #888888;
                border: none;
                text-align: left;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 0px;
            }
            QPushButton:hover {
                color: #00E676;
            }
        """)
        self.advanced_toggle_btn.setVisible(False)
        
        self.advanced_panel = QFrame(step2_box)
        self.advanced_panel.setStyleSheet("""
            QFrame {
                background-color: #1A1A1A;
                border: 1px solid #2D2D2D;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        advanced_layout = QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(6, 6, 6, 6)
        advanced_layout.setSpacing(4)
        
        self.img_max_res_label = QLabel("Image Max Resolution:", self.advanced_panel)
        self.img_max_res_label.setStyleSheet("font-size: 11px; color: #aaaaaa; border: none; background: transparent;")
        self.img_max_res_combo = QComboBox(self.advanced_panel)
        self.img_max_res_combo.addItems([
            "Unlimited (use original)",
            "3200px  (recommended)",
            "2400px",
            "1600px",
            "1200px",
            "800px   (fast preview)"
        ])
        self.img_max_res_combo.setCurrentIndex(1)  # Default: 3200px

        self.mapper_label = QLabel("SfM Mapper Algorithm:", self.advanced_panel)
        self.mapper_label.setStyleSheet("font-size: 11px; color: #aaaaaa; border: none; background: transparent;")
        self.mapper_combo = QComboBox(self.advanced_panel)
        self.mapper_combo.addItems([
            "COLMAP  —  Incremental (default, most robust)",
            "GLOMAP  —  Global (faster on large captures)"
        ])
        self.mapper_combo.setCurrentIndex(0)
        self.mapper_combo.currentIndexChanged.connect(self._on_mapper_combo_changed)
        
        self.mesh_mode_label = QLabel("Mesh Reconstruction Mode:", self.advanced_panel)
        self.mesh_mode_label.setStyleSheet("font-size: 11px; color: #aaaaaa; border: none; background: transparent;")
        self.mesh_mode_combo = QComboBox(self.advanced_panel)
        self.mesh_mode_combo.addItems([
            "Default  —  OpenMVS (Delaunay + Texture)",
            "Poisson  —  Open3D (smooth, watertight)"
        ])
        self.mesh_mode_combo.setCurrentIndex(0)
        self.mesh_mode_combo.currentIndexChanged.connect(self._on_mesh_mode_changed)

        self.poisson_widget = QWidget(self.advanced_panel)
        self.poisson_widget.setStyleSheet("border: none; background: transparent; padding: 0;")
        poisson_w_layout = QVBoxLayout(self.poisson_widget)
        poisson_w_layout.setContentsMargins(0, 0, 0, 0)
        poisson_w_layout.setSpacing(4)
        
        self.poisson_depth_label = QLabel("Poisson Depth: 9", self.poisson_widget)
        self.poisson_depth_label.setStyleSheet("font-size: 11px; color: #aaaaaa; border: none; background: transparent;")
        self.poisson_depth_slider = QSlider(Qt.Horizontal, self.poisson_widget)
        self.poisson_depth_slider.setRange(6, 12)
        self.poisson_depth_slider.setValue(9)
        self.poisson_depth_slider.valueChanged.connect(self._on_poisson_depth_changed)
        
        poisson_w_layout.addWidget(self.poisson_depth_label)
        poisson_w_layout.addWidget(self.poisson_depth_slider)
        self.poisson_widget.setVisible(False)

        # Custom Overrides Toggle (Hidden from UI, managed automatically by Simple/Advanced mode)
        self.custom_settings_toggle = QCheckBox("Enable Custom Parameter Overrides", self.advanced_panel)
        self.custom_settings_toggle.setStyleSheet("font-size: 11px; font-weight: bold; color: #aaaaaa;")
        self.custom_settings_toggle.setChecked(False)
        self.custom_settings_toggle.setVisible(False)
        self.custom_settings_toggle.toggled.connect(self._on_custom_settings_toggled)

        # Custom Settings Container Widget
        self.custom_settings_container = QWidget(self.advanced_panel)
        self.custom_settings_container.setStyleSheet("border: none; background: transparent; padding: 0;")
        self.custom_settings_container.setEnabled(False)
        
        custom_grid = QGridLayout(self.custom_settings_container)
        custom_grid.setContentsMargins(0, 4, 0, 0)
        custom_grid.setSpacing(6)
        # Col 0: labels — fixed-width; Col 1: inputs — fills remaining space
        custom_grid.setColumnStretch(0, 0)
        custom_grid.setColumnStretch(1, 1)
        custom_grid.setColumnMinimumWidth(0, 120)

        lbl_style = "font-size: 10px; color: #888888; border: none; background: transparent;"

        # COLMAP section
        colmap_sec = QLabel("COLMAP Settings", self.custom_settings_container)
        colmap_sec.setStyleSheet("font-size: 11px; font-weight: bold; color: #00E676; margin-top: 4px; border: none; background: transparent;")
        custom_grid.addWidget(colmap_sec, 0, 0, 1, 2)

        # Matcher Type
        lbl_matcher_type = QLabel("Matcher Type:", self.custom_settings_container)
        lbl_matcher_type.setStyleSheet(lbl_style)
        self.custom_matcher_combo = QComboBox(self.custom_settings_container)
        self.custom_matcher_combo.addItems([
            "Auto-Select (Hardware Profiler)",
            "Exhaustive (Full pair matching)",
            "Sequential (Ordered/Video frames)",
            "Vocabulary Tree (Large dataset scale)",
            "Spatial (GPS position based)"
        ])
        self.custom_matcher_combo.setCurrentIndex(0)
        self.custom_matcher_combo.setStyleSheet("background-color: #1E1E1E; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;")
        self.custom_matcher_combo.currentIndexChanged.connect(self._on_matcher_type_changed)
        custom_grid.addWidget(lbl_matcher_type, 1, 0)
        custom_grid.addWidget(self.custom_matcher_combo, 1, 1)

        # Vocab Tree File Selection Row (hidden by default)
        self.vocab_tree_widget = QWidget(self.custom_settings_container)
        self.vocab_tree_widget.setStyleSheet("border: none; background: transparent; padding: 0;")
        vocab_layout = QHBoxLayout(self.vocab_tree_widget)
        vocab_layout.setContentsMargins(0, 0, 0, 0)
        vocab_layout.setSpacing(4)

        self.vocab_path_edit = QLineEdit(self.vocab_tree_widget)
        from pipeline_manager import get_default_vocab_tree_path
        def_vocab = get_default_vocab_tree_path()
        if def_vocab:
            self.vocab_path_edit.setText(def_vocab)
        self.vocab_path_edit.setPlaceholderText("Path to vocab_tree.bin (leave empty for bundled default)...")
        self.vocab_path_edit.setStyleSheet("background-color: #1E1E1E; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px; font-size: 10px;")
        
        self.vocab_browse_btn = QPushButton("Browse...", self.vocab_tree_widget)
        self.vocab_browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                color: #CCCCCC;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
                color: #00E676;
            }
        """)
        self.vocab_browse_btn.clicked.connect(self._browse_vocab_tree_file)
        
        vocab_layout.addWidget(self.vocab_path_edit, stretch=1)
        vocab_layout.addWidget(self.vocab_browse_btn)

        self.lbl_vocab = QLabel("Vocab Tree File:", self.custom_settings_container)
        self.lbl_vocab.setStyleSheet(lbl_style)
        custom_grid.addWidget(self.lbl_vocab, 2, 0)
        custom_grid.addWidget(self.vocab_tree_widget, 2, 1)
        self.lbl_vocab.setVisible(False)
        self.vocab_tree_widget.setVisible(False)

        # SIFT Max Features
        lbl_features = QLabel("SIFT Max Features:", self.custom_settings_container)
        lbl_features.setStyleSheet(lbl_style)
        self.custom_features_spin = QSpinBox(self.custom_settings_container)
        self.custom_features_spin.setRange(1000, 65536)
        self.custom_features_spin.setSingleStep(1000)
        self.custom_features_spin.setValue(8000)
        self.custom_features_stepper = SpinBoxStepper(self.custom_features_spin, self.custom_settings_container)
        custom_grid.addWidget(lbl_features, 3, 0)
        custom_grid.addWidget(self.custom_features_stepper, 3, 1)

        # Exhaustive Max Matches
        lbl_matches = QLabel("Max Num Matches:", self.custom_settings_container)
        lbl_matches.setStyleSheet(lbl_style)
        self.custom_matches_spin = QSpinBox(self.custom_settings_container)
        self.custom_matches_spin.setRange(4096, 262144)
        self.custom_matches_spin.setSingleStep(4096)
        self.custom_matches_spin.setValue(16384)
        self.custom_matches_stepper = SpinBoxStepper(self.custom_matches_spin, self.custom_settings_container)
        custom_grid.addWidget(lbl_matches, 4, 0)
        custom_grid.addWidget(self.custom_matches_stepper, 4, 1)

        # Exhaustive Block Size
        self.lbl_block_size = QLabel("Exhaustive Block Size:", self.custom_settings_container)
        self.lbl_block_size.setStyleSheet(lbl_style)
        self.custom_block_size_spin = QSpinBox(self.custom_settings_container)
        self.custom_block_size_spin.setRange(5, 200)
        self.custom_block_size_spin.setSingleStep(5)
        self.custom_block_size_spin.setValue(50)
        self.custom_block_size_spin.setToolTip("Block size for COLMAP exhaustive_matcher. Lower values (e.g. 20) reduce RAM consumption during matrix matching.")
        self.custom_block_size_stepper = SpinBoxStepper(self.custom_block_size_spin, self.custom_settings_container)
        custom_grid.addWidget(self.lbl_block_size, 5, 0)
        custom_grid.addWidget(self.custom_block_size_stepper, 5, 1)

        # Guided Matching
        self.custom_guided_check = QCheckBox("Use Guided Matching", self.custom_settings_container)
        self.custom_guided_check.setStyleSheet("font-size: 10px; color: #aaaaaa; border: none; background: transparent;")
        self.custom_guided_check.setChecked(True)
        custom_grid.addWidget(self.custom_guided_check, 6, 0, 1, 2)

        # Bundle Adjuster
        self.custom_ba_check = QCheckBox("Run Extra Bundle Adjuster", self.custom_settings_container)
        self.custom_ba_check.setStyleSheet("font-size: 10px; color: #aaaaaa; border: none; background: transparent;")
        self.custom_ba_check.setChecked(False)
        custom_grid.addWidget(self.custom_ba_check, 7, 0, 1, 2)

        # Manhattan Alignment
        self.custom_manhattan_check = QCheckBox("Manhattan-World Alignment (Auto-Level)", self.custom_settings_container)
        self.custom_manhattan_check.setStyleSheet("font-size: 10px; color: #aaaaaa; border: none; background: transparent;")
        self.custom_manhattan_check.setChecked(True)
        custom_grid.addWidget(self.custom_manhattan_check, 8, 0, 1, 2)

        # OpenMVS section
        openmvs_sec = QLabel("OpenMVS Settings", self.custom_settings_container)
        openmvs_sec.setStyleSheet("font-size: 11px; font-weight: bold; color: #00E676; margin-top: 4px; border: none; background: transparent;")
        custom_grid.addWidget(openmvs_sec, 9, 0, 1, 2)

        # Densification Resolution
        lbl_densify_res = QLabel("Densification Res:", self.custom_settings_container)
        lbl_densify_res.setStyleSheet(lbl_style)
        self.custom_densify_res_combo = QComboBox(self.custom_settings_container)
        self.custom_densify_res_combo.addItems([
            "0 — Ultra (highest detail, slow)",
            "1 — High (half resolution)",
            "2 — Medium (quarter resolution)",
            "3 — Low (preview resolution)"
        ])
        self.custom_densify_res_combo.setCurrentIndex(1)
        self.custom_densify_res_combo.setStyleSheet("background-color: #1E1E1E; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;")
        custom_grid.addWidget(lbl_densify_res, 10, 0)
        custom_grid.addWidget(self.custom_densify_res_combo, 10, 1)

        # Max Views for Densification
        lbl_densify_views = QLabel("Max Densify Views:", self.custom_settings_container)
        lbl_densify_views.setStyleSheet(lbl_style)
        self.custom_densify_views_spin = QSpinBox(self.custom_settings_container)
        self.custom_densify_views_spin.setRange(1, 16)
        self.custom_densify_views_spin.setValue(4)
        self.custom_densify_views_stepper = SpinBoxStepper(self.custom_densify_views_spin, self.custom_settings_container)
        custom_grid.addWidget(lbl_densify_views, 11, 0)
        custom_grid.addWidget(self.custom_densify_views_stepper, 11, 1)

        # Mesh Refinement Scales
        lbl_refine_scales = QLabel("Mesh Refinement Scales:", self.custom_settings_container)
        lbl_refine_scales.setStyleSheet(lbl_style)
        self.custom_refine_scales_spin = QSpinBox(self.custom_settings_container)
        self.custom_refine_scales_spin.setRange(1, 5)
        self.custom_refine_scales_spin.setValue(2)
        self.custom_refine_scales_stepper = SpinBoxStepper(self.custom_refine_scales_spin, self.custom_settings_container)
        custom_grid.addWidget(lbl_refine_scales, 12, 0)
        custom_grid.addWidget(self.custom_refine_scales_stepper, 12, 1)

        # Texturing Resolution
        lbl_texture_res = QLabel("Texturing Res:", self.custom_settings_container)
        lbl_texture_res.setStyleSheet(lbl_style)
        self.custom_texture_res_combo = QComboBox(self.custom_settings_container)
        self.custom_texture_res_combo.addItems([
            "0 — Ultra (highest resolution)",
            "1 — High (half resolution)",
            "2 — Medium (quarter resolution)",
            "3 — Low (preview resolution)"
        ])
        self.custom_texture_res_combo.setCurrentIndex(1)
        self.custom_texture_res_combo.setStyleSheet("background-color: #1E1E1E; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;")
        custom_grid.addWidget(lbl_texture_res, 13, 0)
        custom_grid.addWidget(self.custom_texture_res_combo, 13, 1)
        
        advanced_layout.addWidget(self.img_max_res_label)
        advanced_layout.addWidget(self.img_max_res_combo)
        advanced_layout.addWidget(self.custom_settings_toggle)
        advanced_layout.addWidget(self.mapper_label)
        advanced_layout.addWidget(self.mapper_combo)
        advanced_layout.addWidget(self.mesh_mode_label)
        advanced_layout.addWidget(self.mesh_mode_combo)
        advanced_layout.addWidget(self.poisson_widget)
        advanced_layout.addWidget(self.custom_settings_container)
        self.advanced_panel.setVisible(False)
        
        # Initialize enabling states based on the default state of custom settings toggle
        self._on_custom_settings_toggled(self.custom_settings_toggle.isChecked())
        
        self.advanced_toggle_btn.clicked.connect(self._toggle_advanced_options)
        self.recon_mode_combo.currentIndexChanged.connect(self._on_recon_mode_changed)
        
        # Standalone Panel for Step 2 Settings
        self.standalone_panel = QFrame(step2_box)
        self.standalone_panel.setStyleSheet("""
            QFrame {
                background-color: #1A1A1A;
                border: 1px solid #2D2D2D;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        standalone_layout = QVBoxLayout(self.standalone_panel)
        standalone_layout.setContentsMargins(6, 6, 6, 6)
        standalone_layout.setSpacing(6)
        
        self.standalone_poisson_label = QLabel("Poisson Depth: 9", self.standalone_panel)
        self.standalone_poisson_label.setStyleSheet("font-size: 11px; color: #aaaaaa; border: none; background: transparent;")
        self.standalone_poisson_slider = QSlider(Qt.Horizontal, self.standalone_panel)
        self.standalone_poisson_slider.setRange(6, 12)
        self.standalone_poisson_slider.setValue(9)
        self.standalone_poisson_slider.valueChanged.connect(self._on_standalone_poisson_depth_changed)
        
        self.vertex_color_toggle = QCheckBox("Include vertex colors in final mesh", self.standalone_panel)
        self.vertex_color_toggle.setStyleSheet("font-size: 11px; font-weight: bold; color: #aaaaaa; border: none; background: transparent;")
        self.vertex_color_toggle.setChecked(True)
        
        standalone_layout.addWidget(self.standalone_poisson_label)
        standalone_layout.addWidget(self.standalone_poisson_slider)
        standalone_layout.addWidget(self.vertex_color_toggle)
        self.standalone_panel.setVisible(False)
        
        step2_layout.addWidget(s2_title_row)
        step2_layout.addWidget(self.quality_label)
        step2_layout.addWidget(self.quality_combo)
        step2_layout.addWidget(self.gpu_label)
        step2_layout.addWidget(self.gpu_combo)
        step2_layout.addWidget(self.auto_cleanup_checkbox)
        step2_layout.addWidget(self.manhattan_align_checkbox)

        # Dynamic Mesh Cleanup Parameters sub-panel (visible only when Auto Cleanup is checked)
        self.mc_options_container = QFrame(step2_box)
        self.mc_options_container.setObjectName("MeshCleanupOptionsContainer")
        self.mc_options_container.setStyleSheet("""
            QFrame#MeshCleanupOptionsContainer {
                background-color: #1A1A1A;
                border: 1px solid #2D2D2D;
                border-radius: 4px;
                padding: 6px;
                margin-top: 4px;
                margin-bottom: 4px;
            }
        """)
        mc_grid = QGridLayout(self.mc_options_container)
        mc_grid.setContentsMargins(6, 6, 6, 6)
        mc_grid.setSpacing(6)
        
        lbl_mc_style = "font-size: 11px; color: #aaaaaa; border: none; background: transparent;"
        
        self.mc_enable_reduction_check = QCheckBox("Enable Face Reduction", self.mc_options_container)
        self.mc_enable_reduction_check.setStyleSheet("font-size: 11px; color: #cccccc; font-weight: bold; border: none; background: transparent;")
        self.mc_enable_reduction_check.setChecked(True)
        mc_grid.addWidget(self.mc_enable_reduction_check, 0, 0, 1, 2)

        lbl_reduction = QLabel("Face Reduction (%):", self.mc_options_container)
        lbl_reduction.setStyleSheet(lbl_mc_style)
        self.mc_reduction_spin = QSpinBox(self.mc_options_container)
        self.mc_reduction_spin.setRange(5, 95)
        self.mc_reduction_spin.setSingleStep(5)
        self.mc_reduction_spin.setSuffix("%")
        self.mc_reduction_spin.setValue(50)
        self.mc_reduction_spin.setToolTip("Target face reduction percentage for mesh decimation (default: 50%).")
        self.mc_reduction_stepper = SpinBoxStepper(self.mc_reduction_spin, self.mc_options_container)
        mc_grid.addWidget(lbl_reduction, 1, 0)
        mc_grid.addWidget(self.mc_reduction_stepper, 1, 1)

        self.mc_enable_reduction_check.toggled.connect(self.mc_reduction_stepper.setEnabled)
        self.mc_enable_reduction_check.toggled.connect(lbl_reduction.setEnabled)
        
        lbl_max_hole = QLabel("Max Hole Size (faces):", self.mc_options_container)
        lbl_max_hole.setStyleSheet(lbl_mc_style)
        self.mc_max_hole_spin = QSpinBox(self.mc_options_container)
        self.mc_max_hole_spin.setRange(5, 5000)
        self.mc_max_hole_spin.setSingleStep(10)
        self.mc_max_hole_spin.setValue(30)
        self.mc_max_hole_spin.setToolTip("Maximum hole size in faces to automatically close during mesh repair.")
        self.mc_max_hole_stepper = SpinBoxStepper(self.mc_max_hole_spin, self.mc_options_container)
        mc_grid.addWidget(lbl_max_hole, 2, 0)
        mc_grid.addWidget(self.mc_max_hole_stepper, 2, 1)
        
        self.mc_remove_dups_check = QCheckBox("Remove Duplicate Faces / Vertices", self.mc_options_container)
        self.mc_remove_dups_check.setStyleSheet("font-size: 11px; color: #cccccc; border: none; background: transparent;")
        self.mc_remove_dups_check.setChecked(True)
        mc_grid.addWidget(self.mc_remove_dups_check, 3, 0, 1, 2)
        
        self.mc_repair_nm_check = QCheckBox("Repair Non-Manifold Edges", self.mc_options_container)
        self.mc_repair_nm_check.setStyleSheet("font-size: 11px; color: #cccccc; border: none; background: transparent;")
        self.mc_repair_nm_check.setChecked(True)
        mc_grid.addWidget(self.mc_repair_nm_check, 4, 0, 1, 2)
        
        self.mc_close_holes_check = QCheckBox("Close Mesh Holes", self.mc_options_container)
        self.mc_close_holes_check.setStyleSheet("font-size: 11px; color: #cccccc; border: none; background: transparent;")
        self.mc_close_holes_check.setChecked(True)
        mc_grid.addWidget(self.mc_close_holes_check, 5, 0, 1, 2)
        
        self.mc_options_container.setVisible(False)
        self.auto_cleanup_checkbox.toggled.connect(self._on_auto_cleanup_toggled)
        self.manhattan_align_checkbox.toggled.connect(self._sync_manhattan_to_custom)
        self.custom_manhattan_check.toggled.connect(self._sync_custom_to_manhattan)
        
        step2_layout.addWidget(self.mc_options_container)
        step2_layout.addWidget(self.advanced_toggle_btn)
        step2_layout.addWidget(self.advanced_panel)
        step2_layout.addWidget(self.standalone_panel)

        self.process_btn = QPushButton("Start Reconstruction", step2_box)
        self.process_btn.setObjectName("ProcessBtn")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._start_processing)

        self.resume_hint_label = QLabel("", step2_box)
        self.resume_hint_label.setStyleSheet("color: #FFB300; font-size: 11px; font-weight: bold; margin-top: 2px;")
        self.resume_hint_label.setAlignment(Qt.AlignCenter)
        self.resume_hint_label.setWordWrap(True)
        self.resume_hint_label.setVisible(False)

        self.start_fresh_btn = QPushButton("Or start fresh", step2_box)
        self.start_fresh_btn.setCursor(Qt.PointingHandCursor)
        self.start_fresh_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #888888;
                font-size: 11px;
                text-decoration: underline;
                padding: 2px;
                margin-bottom: 2px;
            }
            QPushButton:hover {
                color: #00E676;
            }
        """)
        self.start_fresh_btn.setVisible(False)
        self.start_fresh_btn.clicked.connect(self._start_fresh)

        self.progress_bar = QProgressBar(step2_box)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.status_label = QLabel("Status: Idle", step2_box)
        self.status_label.setStyleSheet("color: #a3a3a3; font-style: italic;")

        step2_layout.addWidget(self.process_btn)
        step2_layout.addWidget(self.resume_hint_label)
        step2_layout.addWidget(self.start_fresh_btn)
        step2_layout.addWidget(self.progress_bar)
        step2_layout.addWidget(self.status_label)
        scroll_content_layout.addWidget(step2_box)

        scroll_content_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll_area)
        
        main_layout.addWidget(sidebar)
        
        # Right Side Display Panel
        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)
        
        # Central Widget (Just the Viewer now, since it handles drops)
        self.viewer_widget = ViewerWrapperWidget(self)
        self.viewer_widget.images_dropped.connect(self._on_files_dropped)
        self.viewer_widget.reload_requested.connect(self._reload_viewer)
        self.viewer_widget.action_new.triggered.connect(self._new_project)
        self.viewer_widget.action_save.triggered.connect(self._save_project)
        self.viewer_widget.action_load.triggered.connect(self._load_project)
        self.viewer_widget.action_recover.triggered.connect(self._retrieve_last_session)
        self.viewer_widget.action_export_dense.triggered.connect(lambda: self._export_mesh(".glb"))
        self.viewer_widget.action_export_sparse.triggered.connect(lambda: self._export_mesh(".ply"))
        self.viewer_widget.action_export_glb.triggered.connect(lambda: self._export_mesh(".glb"))
        self.viewer_widget.action_export_obj.triggered.connect(lambda: self._export_mesh(".obj"))
        self.viewer_widget.action_export_usdz.triggered.connect(lambda: self._export_mesh(".usdz"))
        self.viewer_widget.action_import_media.triggered.connect(self._open_files_dialog)
        self.viewer_widget.action_import_dir.triggered.connect(self._open_dir_dialog)
        self.viewer_widget.action_import_zip.triggered.connect(self._open_zip_dialog)
        self.viewer_widget.action_import_mobile.triggered.connect(self._on_import_from_mobile_clicked)
        self.viewer_widget.action_import_point_cloud.triggered.connect(self._import_standalone_cloud_clicked)
        self.viewer_widget.action_mobile_export.triggered.connect(self._on_send_to_mobile_clicked)
        self.viewer_widget.action_upload_proximap.triggered.connect(self._upload_to_proximap)
        self.viewer_widget.selection_mode_changed.connect(self._on_selection_mode_changed)
        self.viewer_widget.remove_outside_requested.connect(self._apply_crop)
        self.viewer_widget.reset_crop_requested.connect(self._reset_crop)
        self.viewer_widget.finalize_crop_requested.connect(self._finalize_crop)
        self.viewer_widget.delete_selection_requested.connect(self._delete_selection)
        self.viewer_widget.clear_selection_requested.connect(self._clear_selection)
        self.viewer_widget.invert_selection_requested.connect(self._invert_selection)
        self.viewer_widget.retexture_mesh_tool_requested.connect(self._on_retexture_mesh_tool)
        self.viewer_widget.transform_cloud_requested.connect(self._open_point_cloud_transform_tool)
        self.viewer_widget.shading_mode_changed.connect(self._on_shading_mode_changed)

        self._current_shading_mode = "solid"
        self._wireframe_filter = None

        # Initialize VisPy Canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor=self.viewport_bg_color)
        if hasattr(self.canvas, '_keys_check') and 'escape' in self.canvas._keys_check:
            del self.canvas._keys_check['escape']
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = ProximapTurntableCamera(up='+y', elevation=30, azimuth=45)
        self.view.camera.translate_speed = 3.0
        
        self.canvas.events.mouse_press.connect(self._on_canvas_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_canvas_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_canvas_mouse_release)
        
        # Initialize fixed wide ground plane grid at Y=0
        self._init_fixed_ground_grid()
        
        # Add native VisPy canvas widget to the layout
        self.viewer_widget.container_area_layout.addWidget(self.canvas.native)
        self.viewer_widget.bg_btn.clicked.connect(self._choose_bg_color)

        from floating_camera_preview import FloatingCameraPreviewWidget

        # Initialise the floating PiP camera preview widget (hidden initially)
        self.camera_preview_widget = FloatingCameraPreviewWidget(
            self.viewer_widget.container_area
        )
        self.camera_preview_widget.look_through_requested.connect(self._on_look_through_camera)
        self.camera_preview_widget.closed.connect(self._on_camera_preview_closed)


        # Initialize 2D screen selection overlay for Box/Lasso tools
        from selection_overlay import SelectionOverlayWidget
        self.selection_overlay = SelectionOverlayWidget(self.viewer_widget.container_area, underlying_widget=self.canvas.native)
        self.selection_overlay.shape_changed.connect(self._on_selection_shape_changed)
        self.selection_overlay.setVisible(False)

        # Initialize 3D Navigation Orientation Gizmo for VisPy Viewport (Y-up coordinate system)
        from mesh_editor.nav_gizmo import NavGizmoWidget
        self.nav_gizmo = NavGizmoWidget(self.viewer_widget.container_area, coord_system="y-up")
        self.nav_gizmo.snap_requested.connect(self._on_vispy_nav_gizmo_snap)
        self.nav_gizmo.update_from_vispy(self.view.camera.azimuth, self.view.camera.elevation)
        self.nav_gizmo.show()

        if hasattr(self.view.camera, 'events') and hasattr(self.view.camera.events, 'transform_change'):
            self.view.camera.events.transform_change.connect(self._on_vispy_camera_transform_changed)

        # Initialize floating camera controls overlay
        self.overlay_label = QLabel(self.viewer_widget.container_area)
        self.overlay_label.setStyleSheet("""
            QLabel {
                background-color: rgba(20, 20, 20, 220);
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 6px;
                font-size: 11px;
                padding: 10px;
            }
        """)
        self.overlay_label.setVisible(False)
        self.viewer_widget.show_controls_cb.stateChanged.connect(self._on_show_controls_changed)

        # Initialize Unity-style EditorWindow host modal and movable floating toolbox
        from viewport_tool_system import (
            EditorWindowHostModal, FloatingToolboxWidget,
            TransformToolWindow, MeshCleanupToolWindow, MeshCutToolWindow,
            CameraRegistrationToolWindow
        )
        self.editor_tool_host = EditorWindowHostModal(
            self.viewer_widget.container_area,
            default_x=70,
            default_y=20,
            data_provider=lambda: bool(self._current_points is not None and len(self._current_points) > 0)
        )
        self.floating_toolbox = FloatingToolboxWidget(
            self.viewer_widget.container_area,
            host_modal=self.editor_tool_host,
            initial_x=16,
            initial_y=20
        )

        # Initialize interactive 3D Combined Transform Gizmo (Translation & Rotation)
        from viewport_gizmo import CombinedTransformGizmo
        self.transform_gizmo = CombinedTransformGizmo(parent_scene=self.view.scene)
        self.editor_tool_host.gizmo_toggled.connect(self._on_gizmo_toggled)

        self.camera_reg_tool_window = CameraRegistrationToolWindow()
        self.camera_reg_tool_window.cameras_toggled.connect(self._on_camera_frustums_toggled)
        self.camera_reg_tool_window.photos_toggled.connect(self._on_camera_photos_toggled)
        self.camera_reg_tool_window.scale_changed.connect(self._on_camera_scale_changed)
        self.camera_reg_tool_window.opacity_changed.connect(self._on_camera_opacity_changed)

        self.transform_tool_window = TransformToolWindow()
        self.transform_tool_window.transform_changed.connect(self._on_cloud_transform_preview)
        self.transform_tool_window.transform_applied.connect(self._on_cloud_transform_applied)
        self.transform_tool_window.transform_reset.connect(self._on_cloud_transform_reset)

        self.mesh_cleanup_tool_window = MeshCleanupToolWindow()
        self.mesh_cleanup_tool_window.apply_reduction_requested.connect(self._on_mesh_face_reduction_apply)
        self.mesh_cleanup_tool_window.revert_requested.connect(self._on_mesh_face_reduction_revert)
        self.mesh_cleanup_tool_window.apply_close_holes_requested.connect(self._on_mesh_close_holes_apply)
        self.mesh_cleanup_tool_window.revert_close_holes_requested.connect(self._on_mesh_close_holes_revert)
        self.mesh_cleanup_tool_window.apply_repair_nonmanifold_requested.connect(self._on_mesh_repair_nonmanifold_apply)
        self.mesh_cleanup_tool_window.revert_repair_nonmanifold_requested.connect(self._on_mesh_repair_nonmanifold_revert)
        self.mesh_cleanup_tool_window.apply_remove_duplicates_requested.connect(self._on_mesh_remove_duplicates_apply)
        self.mesh_cleanup_tool_window.revert_remove_duplicates_requested.connect(self._on_mesh_remove_duplicates_revert)
        self.mesh_cleanup_tool_window.apply_merge_vertices_requested.connect(self._on_mesh_merge_vertices_apply)
        self.mesh_cleanup_tool_window.revert_merge_vertices_requested.connect(self._on_mesh_merge_vertices_revert)
        self.mesh_cleanup_tool_window.apply_smooth_requested.connect(self._on_mesh_smooth_apply)
        self.mesh_cleanup_tool_window.revert_smooth_requested.connect(self._on_mesh_smooth_revert)
        self.mesh_cleanup_tool_window.retexture_requested.connect(self._on_retexture_mesh_tool)
        if hasattr(self, 'mc_max_hole_spin') and self.mc_max_hole_spin:
            self.mesh_cleanup_tool_window.initial_hole_size = self.mc_max_hole_spin.value()

        self.mesh_cut_tool_window = MeshCutToolWindow()
        self.mesh_cut_tool_window.mode_changed.connect(self._on_selection_mode_changed)
        self.mesh_cut_tool_window.crop_requested.connect(self._apply_crop)
        self.mesh_cut_tool_window.reset_crop_requested.connect(self._reset_crop)
        self.mesh_cut_tool_window.delete_selection_requested.connect(self._delete_selection)
        self.mesh_cut_tool_window.invert_selection_requested.connect(self._invert_selection)
        self.mesh_cut_tool_window.clear_selection_requested.connect(self._clear_selection)
        self.mesh_cut_tool_window.retexture_requested.connect(self._on_retexture_mesh_tool)

        self.floating_toolbox.register_tool(self.camera_reg_tool_window)
        self.floating_toolbox.register_tool(self.transform_tool_window)
        self.floating_toolbox.register_tool(self.mesh_cleanup_tool_window)
        self.floating_toolbox.register_tool(self.mesh_cut_tool_window)

        self._update_sidebar_tools_visibility()

        self.floating_toolbox.show()
        self.floating_toolbox.raise_()
        
        right_layout.addWidget(self.viewer_widget, stretch=4)
        
        # Tabbed panel containing Photos and Console
        self.bottom_tabs = QTabWidget(right_panel)
        self.bottom_tabs.setObjectName("BottomTabs")
        self.bottom_tabs.setTabPosition(QTabWidget.South)
        
        # Photos Tab
        self.photos_tab = PhotosTabWidget(self.bottom_tabs)
        self.photos_tab.btn_remove_selected.clicked.connect(self._remove_selected_photos)
        self.photos_tab.btn_add_photos.clicked.connect(self._add_photos_dialog)
        self.photos_tab.btn_bg_remove.clicked.connect(self._remove_backgrounds_clicked)
        
        # Console Tab
        self.console_frame = QFrame(self.bottom_tabs)
        self.console_frame.setObjectName("ConsoleFrame")
        console_layout = QVBoxLayout(self.console_frame)
        console_layout.setContentsMargins(10, 10, 10, 10)
        
        # Console Header Layout
        console_header_layout = QHBoxLayout()
        console_title = QLabel("System Output Log", self.console_frame)
        console_title.setStyleSheet("font-weight: bold; color: #888888; font-size: 11px; text-transform: uppercase;")
        
        self.clear_console_btn = QPushButton("Clear", self.console_frame)
        self.clear_console_btn.setCursor(Qt.PointingHandCursor)
        self.clear_console_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #aaaaaa;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #333333;
                color: #ffffff;
            }
        """)
        self.clear_console_btn.clicked.connect(lambda: self.console_text.clear())
        
        console_header_layout.addWidget(console_title)
        console_header_layout.addStretch()
        console_header_layout.addWidget(self.clear_console_btn)
        
        self.console_text = QTextEdit(self.console_frame)
        self.console_text.setReadOnly(True)
        self.console_text.setObjectName("Console")
        
        console_layout.addLayout(console_header_layout)
        console_layout.addWidget(self.console_text)
        
        # Add tabs
        self.bottom_tabs.addTab(self.photos_tab, "Dataset")
        self.bottom_tabs.addTab(self.console_frame, "Console")
        
        right_layout.addWidget(self.bottom_tabs, stretch=2)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        # Register tabs to MainTabs
        self.main_tabs.addTab(reconstruction_tab, "3D Reconstruction")
        
        # Lazy load the Mesh Editor tab to avoid importing trimesh at startup
        self.mesh_editor_tab = None
        self.mesh_editor_placeholder = QWidget(self.main_tabs)
        
        # Add styled loading UI to the placeholder
        placeholder_layout = QVBoxLayout(self.mesh_editor_placeholder)
        placeholder_layout.setContentsMargins(20, 20, 20, 20)
        
        loading_container = QWidget(self.mesh_editor_placeholder)
        loading_container.setFixedWidth(320)
        loading_layout = QVBoxLayout(loading_container)
        loading_layout.setSpacing(12)
        loading_layout.setAlignment(Qt.AlignCenter)
        
        self.loading_msg_label = QLabel("Opening Mesh Editor...", loading_container)
        self.loading_msg_label.setStyleSheet("color: #ffffff; font-size: 15px; font-weight: bold;")
        self.loading_msg_label.setAlignment(Qt.AlignCenter)
        
        self.loading_progress = QProgressBar(loading_container)
        self.loading_progress.setRange(0, 0)  # Indeterminate loading bar
        self.loading_progress.setTextVisible(False)
        self.loading_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3A3A3A;
                background-color: #222222;
                height: 6px;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #00E676;
                border-radius: 3px;
            }
        """)
        
        self.loading_sub_label = QLabel("Initializing 3D viewport and mesh utilities...", loading_container)
        self.loading_sub_label.setStyleSheet("color: #737373; font-size: 11px;")
        self.loading_sub_label.setAlignment(Qt.AlignCenter)
        
        loading_layout.addWidget(self.loading_msg_label)
        loading_layout.addWidget(self.loading_progress)
        loading_layout.addWidget(self.loading_sub_label)
        
        placeholder_layout.addStretch()
        placeholder_layout.addWidget(loading_container, 0, Qt.AlignCenter)
        placeholder_layout.addStretch()
        
        self.main_tabs.addTab(self.mesh_editor_placeholder, "Mesh Editor")
        self.main_tabs.currentChanged.connect(self._on_tab_changed)
        
        self._set_process_btn_state("idle")
        self._check_and_enable_cleanup_btn()

    def _on_tab_changed(self, index):
        if index == 1 and self.mesh_editor_tab is None:
            if not getattr(self, "_mesh_editor_loading", False):
                self._mesh_editor_loading = True
                QApplication.setOverrideCursor(Qt.WaitCursor)
                QTimer.singleShot(100, self._load_mesh_editor)

    def _load_mesh_editor(self):
        try:
            from mesh_editor import MeshEditorWidget
            self.mesh_editor_tab = MeshEditorWidget(self)
            self.mesh_editor_tab.action_upload_proximap.triggered.connect(self._upload_mesh_editor_scene)
            
            # Clear placeholder layout (loading UI) and swap in the actual mesh editor
            layout = self.mesh_editor_placeholder.layout()
            if layout:
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.mesh_editor_tab)
            else:
                layout = QVBoxLayout(self.mesh_editor_placeholder)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.mesh_editor_tab)
        except Exception as e:
            import traceback
            err_msg = f"Failed to load Mesh Editor:\n{str(e)}\n\n{traceback.format_exc()}"
            print(f"[ERROR] {err_msg}")
            
            layout = self.mesh_editor_placeholder.layout()
            if layout:
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            else:
                layout = QVBoxLayout(self.mesh_editor_placeholder)
            
            err_container = QWidget(self.mesh_editor_placeholder)
            err_container.setFixedWidth(540)
            err_layout = QVBoxLayout(err_container)
            err_layout.setSpacing(12)
            err_layout.setAlignment(Qt.AlignCenter)
            
            err_title = QLabel("Error Loading Mesh Editor", err_container)
            err_title.setStyleSheet("color: #FF5252; font-size: 16px; font-weight: bold;")
            err_title.setAlignment(Qt.AlignCenter)
            
            err_box = QTextEdit(err_container)
            err_box.setPlainText(f"{str(e)}\n\n{traceback.format_exc()}")
            err_box.setReadOnly(True)
            err_box.setMaximumHeight(180)
            err_box.setStyleSheet("""
                QTextEdit {
                    background-color: #1A1A1A;
                    color: #FF8A80;
                    border: 1px solid #FF5252;
                    border-radius: 6px;
                    font-family: monospace;
                    font-size: 11px;
                    padding: 8px;
                }
            """)
            
            retry_btn = QPushButton("Retry Loading", err_container)
            retry_btn.setCursor(Qt.PointingHandCursor)
            retry_btn.setStyleSheet("""
                QPushButton {
                    background-color: #00E676;
                    color: #121212;
                    font-weight: bold;
                    font-size: 13px;
                    border-radius: 4px;
                    padding: 8px 20px;
                }
                QPushButton:hover {
                    background-color: #00FF87;
                }
            """)
            retry_btn.clicked.connect(self._retry_load_mesh_editor)
            
            err_layout.addWidget(err_title)
            err_layout.addWidget(err_box)
            err_layout.addWidget(retry_btn, 0, Qt.AlignCenter)
            
            layout.addStretch()
            layout.addWidget(err_container, 0, Qt.AlignCenter)
            layout.addStretch()
        finally:
            QApplication.restoreOverrideCursor()
            self._mesh_editor_loading = False

    def _retry_load_mesh_editor(self):
        layout = self.mesh_editor_placeholder.layout()
        if layout:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            layout = QVBoxLayout(self.mesh_editor_placeholder)
            
        loading_container = QWidget(self.mesh_editor_placeholder)
        loading_container.setFixedWidth(320)
        loading_layout = QVBoxLayout(loading_container)
        loading_layout.setSpacing(12)
        loading_layout.setAlignment(Qt.AlignCenter)
        
        self.loading_msg_label = QLabel("Opening Mesh Editor...", loading_container)
        self.loading_msg_label.setStyleSheet("color: #ffffff; font-size: 15px; font-weight: bold;")
        self.loading_msg_label.setAlignment(Qt.AlignCenter)
        
        self.loading_progress = QProgressBar(loading_container)
        self.loading_progress.setRange(0, 0)
        self.loading_progress.setTextVisible(False)
        self.loading_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3A3A3A;
                background-color: #222222;
                height: 6px;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #00E676;
                border-radius: 3px;
            }
        """)
        
        self.loading_sub_label = QLabel("Initializing 3D viewport and mesh utilities...", loading_container)
        self.loading_sub_label.setStyleSheet("color: #737373; font-size: 11px;")
        self.loading_sub_label.setAlignment(Qt.AlignCenter)
        
        loading_layout.addWidget(self.loading_msg_label)
        loading_layout.addWidget(self.loading_progress)
        loading_layout.addWidget(self.loading_sub_label)
        
        layout.addStretch()
        layout.addWidget(loading_container, 0, Qt.AlignCenter)
        layout.addStretch()
        
        self._mesh_editor_loading = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QTimer.singleShot(100, self._load_mesh_editor)

    def _update_system_badge(self):
        """Calculates system resource quality badge and updates style dynamically."""
        try:
            budget = hardware_profiler.get_memory_budget()
            avail_gb = budget.available_gb
            swap_used = budget.swap_used_gb
            swap_total = budget.swap_total_gb
            
            if budget.pressure_level == "ok":
                status_text = f"SYSTEM READY ({avail_gb:.1f}GB RAM Free)"
                badge_color = "#00E676"  # Bright green
                text_color = "#121212"
            elif budget.pressure_level == "warn":
                status_text = f"SYSTEM WARN ({avail_gb:.1f}GB RAM Free)"
                badge_color = "#FFD700"  # Yellow
                text_color = "#121212"
            else:
                status_text = f"SYSTEM INSUFFICIENT ({avail_gb:.1f}GB RAM Free)"
                badge_color = "#D50000"  # Deep red
                text_color = "#ffffff"
                
            gpu_info = "dGPU Active" if self.dgpu_detected else "iGPU Fallback Active"
            if swap_used > 1.5 and swap_total > 0:
                gpu_info += f" · Swap: {swap_used:.1f}/{swap_total:.1f}GB"
                
            self.badge.setText(f"{status_text}\n{gpu_info}")
            self.badge.setStyleSheet(
                f"background-color: {badge_color}; color: {text_color}; "
                "font-weight: bold; border-radius: 4px; padding: 6px; font-size: 11px;"
            )
        except Exception:
            # Fallback if profiler not fully ready
            self.badge.setText(f"SYSTEM READY\n{'dGPU Active' if self.dgpu_detected else 'iGPU Fallback Active'}")
            self.badge.setStyleSheet(
                "background-color: #00E676; color: #121212; "
                "font-weight: bold; border-radius: 4px; padding: 6px; font-size: 11px;"
            )


    def _has_existing_mesh(self) -> bool:
        """Returns True if a reconstructable/cleanable PLY mesh is present in the active mvs directory."""
        mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
        if not os.path.exists(mvs_dir):
            return False
        candidates = ["scene_dense_mesh_texture.ply", "scene_dense_mesh_refine.ply", "scene_dense_mesh_refcloud.ply", "scene_dense_mesh.ply", "scene_mesh.ply", "scene_dense_mesh_cleaned.ply"]
        return any(os.path.exists(os.path.join(mvs_dir, c)) for c in candidates)

    def _check_and_enable_cleanup_btn(self):
        """Checks if a valid mesh exists and sets the cleanup button state accordingly."""
        if hasattr(self, 'cleanup_btn'):
            if self._has_existing_mesh():
                self._set_cleanup_btn_state("ready")
            else:
                self._set_cleanup_btn_state("idle")
        if hasattr(self, 'editor_tool_host') and self.editor_tool_host is not None:
            self.editor_tool_host.refresh_data_state()

    def _set_cleanup_btn_state(self, state: str):
        """
        Dynamically updates cleanup button colors, text, and enabled state.
        """
        if not hasattr(self, 'cleanup_btn'):
            return
        if state == "idle":
            self.cleanup_btn.setText("Run Mesh Cleanup")
            self.cleanup_btn.setEnabled(False)
            self.cleanup_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #202020;
                    color: #555555;
                    border: 1px solid #2D2D2D;
                }
            """)
        elif state == "ready":
            self.cleanup_btn.setText("Run Mesh Cleanup")
            self.cleanup_btn.setEnabled(True)
            self.cleanup_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #00E676;
                    color: #121212;
                    border: none;
                }
                QPushButton#ProcessBtn:hover {
                    background-color: #00FF87;
                    border: none;
                }
                QPushButton#ProcessBtn:pressed {
                    background-color: #00B35C;
                    border: none;
                }
            """)
        elif state == "progress":
            self.cleanup_btn.setText("Cleanup in Progress...")
            self.cleanup_btn.setEnabled(False)
            self.cleanup_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #FF9100;
                    color: #121212;
                    border: none;
                }
            """)
        elif state == "failed":
            self.cleanup_btn.setText("Retry Mesh Cleanup")
            self.cleanup_btn.setEnabled(True)
            self.cleanup_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #D50000;
                    color: #ffffff;
                    border: none;
                }
                QPushButton#ProcessBtn:hover {
                    background-color: #FF1744;
                    border: none;
                }
                QPushButton#ProcessBtn:pressed {
                    background-color: #B30000;
                    border: none;
                }
            """)

    def _get_stage_display_name(self, stage_key: str) -> str:
        step_map = {
            "image_preparation": "Image Preparation",
            "features_extracted": "SIFT Feature Extraction",
            "features_matched": "Feature Matching",
            "sparse_reconstruction": "Camera Poses & Sparse Cloud (SfM)",
            "dense_reconstruction": "Dense Point Cloud",
            "mesh_reconstructed": "Surface Mesh Reconstruction",
            "mesh_refined": "Mesh Geometry Refinement",
            "mesh_cleaned": "Mesh Auto Cleanup",
            "mesh_textured": "Texture Projection",
        }
        return step_map.get(stage_key, stage_key.replace("_", " ").title())

    def _retry_with_resume(self):
        if getattr(self, '_last_failed_stage', None):
            self.console_text.append(f"[RETRY] Resuming reconstruction from stage: '{self._last_failed_stage}'...")
            self._start_processing(resume_from_step=self._last_failed_stage)
        else:
            self.console_text.append("[RETRY] Retrying reconstruction from beginning...")
            self._start_processing(resume_from_step=None)

    def _start_fresh(self):
        self._last_failed_stage = None
        self.console_text.append("[START] Starting fresh reconstruction from beginning...")
        self._start_processing(resume_from_step=None)

    def _set_dataset_modification_controls_enabled(self, enabled: bool):
        """Enables or disables dataset modifying buttons (Delete, Add Photos, Remove BG) during reconstruction."""
        if hasattr(self, 'photos_tab') and self.photos_tab is not None:
            self.photos_tab.set_dataset_controls_enabled(enabled)

    def _update_sidebar_tools_visibility(self, mode_index: Optional[int] = None):
        """Updates tool icon button visibility in the floating sidebar based on active viewport mode."""
        if not hasattr(self, 'floating_toolbox') or self.floating_toolbox is None:
            return
        if mode_index is None:
            mode_index = self.viewer_widget.mode_select.currentIndex() if (hasattr(self, 'viewer_widget') and self.viewer_widget) else 0

        is_sparse = (mode_index == 0)
        is_textured_mesh = (mode_index == 2)

        self.floating_toolbox.set_tool_visible("camera_registration", is_sparse)
        self.floating_toolbox.set_tool_visible("transform", True)
        self.floating_toolbox.set_tool_visible("cleanup", is_textured_mesh)
        self.floating_toolbox.set_tool_visible("cut", is_textured_mesh)

    def _set_view_mode(self, index: int):
        """Programmatically sets the viewport view mode index and keeps sidebar visibility and crop state in sync."""
        if hasattr(self, 'viewer_widget') and self.viewer_widget is not None:
            self.viewer_widget.mode_select.blockSignals(True)
            self.viewer_widget.mode_select.setCurrentIndex(index)
            self.viewer_widget.mode_select.blockSignals(False)
            self.viewer_widget.update_crop_box_state()
        self._update_sidebar_tools_visibility(index)

    def _set_sidebar_and_tools_enabled(self, enabled: bool):
        """Enables or disables sidebar tool panels, floating toolbox, and tool windows during pipeline execution."""
        if hasattr(self, 'floating_toolbox') and self.floating_toolbox is not None:
            self.floating_toolbox.setEnabled(enabled)
        if hasattr(self, 'editor_tool_host') and self.editor_tool_host is not None:
            self.editor_tool_host.setEnabled(enabled)
        if hasattr(self, 'camera_reg_tool_window') and self.camera_reg_tool_window is not None:
            self.camera_reg_tool_window.set_busy(not enabled)
        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window is not None:
            self.mesh_cleanup_tool_window.set_busy(not enabled)
        if hasattr(self, 'mesh_cut_tool_window') and self.mesh_cut_tool_window is not None:
            self.mesh_cut_tool_window.set_busy(not enabled)
        if hasattr(self, 'transform_tool_window') and self.transform_tool_window is not None:
            if hasattr(self.transform_tool_window, 'apply_btn') and self.transform_tool_window.apply_btn is not None:
                self.transform_tool_window.apply_btn.setEnabled(enabled)
            if hasattr(self.transform_tool_window, 'reset_btn') and self.transform_tool_window.reset_btn is not None:
                self.transform_tool_window.reset_btn.setEnabled(enabled)
        if hasattr(self, 'viewer_widget') and self.viewer_widget is not None:
            self.viewer_widget.set_tool_modals_busy(not enabled)
            if hasattr(self.viewer_widget, 'file_menu_btn') and self.viewer_widget.file_menu_btn is not None:
                self.viewer_widget.file_menu_btn.setEnabled(enabled)

    def _set_process_btn_state(self, state: str):
        """
        Dynamically updates process button colors, text, and enabled state.
        """
        try:
            self.process_btn.clicked.disconnect()
        except RuntimeError:
            pass

        if state == "progress":
            self._set_dataset_modification_controls_enabled(False)
            self._set_sidebar_and_tools_enabled(False)
        else:
            self._set_dataset_modification_controls_enabled(True)
            self._set_sidebar_and_tools_enabled(True)

        if state == "idle":
            self.process_btn.setText("Start Reconstruction")
            self.process_btn.setEnabled(False)
            self.process_btn.clicked.connect(self._start_processing)
            if hasattr(self, 'resume_hint_label'):
                self.resume_hint_label.setVisible(False)
            if hasattr(self, 'start_fresh_btn'):
                self.start_fresh_btn.setVisible(False)
            self.process_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #202020;
                    color: #555555;
                    border: 1px solid #2D2D2D;
                }
            """)
        elif state == "ready":
            self.process_btn.setText("Start Reconstruction")
            self.process_btn.setEnabled(True)
            self.process_btn.clicked.connect(self._start_processing)
            if hasattr(self, 'resume_hint_label'):
                self.resume_hint_label.setVisible(False)
            if hasattr(self, 'start_fresh_btn'):
                self.start_fresh_btn.setVisible(False)
            self.process_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #00E676;
                    color: #121212;
                    border: none;
                }
                QPushButton#ProcessBtn:hover {
                    background-color: #00FF87;
                    border: none;
                }
                QPushButton#ProcessBtn:pressed {
                    background-color: #00B35C;
                    border: none;
                }
            """)
        elif state == "progress":
            self.process_btn.setText("Reconstruction in Progress...")
            self.process_btn.setEnabled(False)
            if hasattr(self, 'resume_hint_label'):
                self.resume_hint_label.setVisible(False)
            if hasattr(self, 'start_fresh_btn'):
                self.start_fresh_btn.setVisible(False)
            self.process_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #FF9100;
                    color: #121212;
                    border: none;
                }
            """)
        elif state == "failed":
            self.process_btn.setText("Retry Reconstruction")
            self.process_btn.setEnabled(True)
            self.process_btn.clicked.connect(self._retry_with_resume)

            if getattr(self, '_last_failed_stage', None):
                stage_display = self._get_stage_display_name(self._last_failed_stage)
                if hasattr(self, 'resume_hint_label'):
                    self.resume_hint_label.setText(f"Resuming from: {stage_display}")
                    self.resume_hint_label.setVisible(True)
                if hasattr(self, 'start_fresh_btn'):
                    self.start_fresh_btn.setVisible(True)
            else:
                if hasattr(self, 'resume_hint_label'):
                    self.resume_hint_label.setVisible(False)
                if hasattr(self, 'start_fresh_btn'):
                    self.start_fresh_btn.setVisible(False)

            self.process_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #D50000;
                    color: #ffffff;
                    border: none;
                }
                QPushButton#ProcessBtn:hover {
                    background-color: #FF1744;
                    border: none;
                }
                QPushButton#ProcessBtn:pressed {
                    background-color: #B30000;
                    border: none;
                }
            """)

    def _apply_styling(self):
        qss = """
            QMainWindow {
                background-color: #121212;
            }
            #Sidebar {
                background-color: #1A1A1A;
                border-right: 1px solid #2B2B2B;
                border-radius: 8px;
            }
            #StepBox {
                background-color: #242424;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 8px;
            }
            #StepBox QLabel {
                color: #e0e0e0;
                font-size: 13px;
                margin-bottom: 6px;
                padding-bottom: 2px;
            }
            #DragDropArea {
                background-color: #1A1A1A;
                border: 2px dashed #3A3A3A;
                border-radius: 8px;
            }
            #ConsoleFrame {
                background-color: #151515;
                border: 1px solid #282828;
                border-radius: 6px;
            }
            #Console {
                background-color: #0A0A0A;
                color: #00FF66;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: none;
            }
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00E676;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
            QPushButton:disabled {
                background-color: #202020;
                color: #555555;
                border-color: #2D2D2D;
            }
            QPushButton#ProcessBtn {
                background-color: #202020;
                color: #555555;
                border: 1px solid #2D2D2D;
            }
            QPushButton#ProcessBtn:enabled {
                background-color: #00E676;
                color: #121212;
                border: none;
            }
            QPushButton#ProcessBtn:hover:enabled {
                background-color: #00FF87;
                border: none;
            }
            QPushButton#ProcessBtn:pressed:enabled {
                background-color: #00B35C;
                border: none;
            }
            QPushButton#ProcessBtn:disabled {
                background-color: #202020;
                color: #555555;
                border-color: #2D2D2D;
            }
            QComboBox {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
                min-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1A1A1A;
                color: #ffffff;
                selection-background-color: #00E676;
                selection-color: #121212;
            }
            QScrollArea {
                background-color: #1A1A1A;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: #1A1A1A;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #2D2D2D;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00E676;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QProgressBar {
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                background-color: #222222;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #00E676;
                width: 10px;
            }
            QRadioButton {
                color: #cccccc;
                font-size: 12px;
                spacing: 8px;
                margin-top: 4px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QTabWidget#BottomTabs::pane {
                border: 1px solid #2B2B2B;
                background-color: #151515;
            }
            QTabWidget#BottomTabs QTabBar::tab {
                background-color: #242424;
                color: #aaaaaa;
                border: 1px solid #2B2B2B;
                padding: 6px 16px;
                font-weight: bold;
                font-size: 11px;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QTabWidget#BottomTabs QTabBar::tab:selected {
                background-color: #e0e0e0;
                color: #121212;
                border-top: none;
            }
            QTabWidget#BottomTabs QTabBar::tab:hover:!selected {
                background-color: #333333;
                color: #ffffff;
            }
            
            /* Main Window Tabs Styling */
            QTabWidget#MainTabs::pane {
                border: none;
                background-color: #121212;
            }
            QTabWidget#MainTabs QTabBar::tab {
                background-color: #1A1A1A;
                color: #aaaaaa;
                border: 1px solid #2B2B2B;
                border-bottom: none;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 12px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 4px;
            }
            QTabWidget#MainTabs QTabBar::tab:selected {
                background-color: #00E676;
                color: #121212;
                border-bottom: none;
            }
            QTabWidget#MainTabs QTabBar::tab:hover:!selected {
                background-color: #242424;
                color: #ffffff;
            }
            
            /* Viewport Toolbar Styling */
            QMenuBar#ViewportMenuBar {
                background-color: #1E1E1E;
                color: #e0e0e0;
                border-bottom: 1px solid #2D2D2D;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: normal;
            }
            QMenuBar#ViewportMenuBar::item {
                background-color: transparent;
                padding: 4px 12px;
                border-radius: 4px;
                margin-right: 4px;
            }
            QMenuBar#ViewportMenuBar::item:selected {
                background-color: #333333;
                color: #00E676;
            }
            QMenuBar#ViewportMenuBar::item:pressed {
                background-color: #00E676;
                color: #121212;
            }
            QMenu {
                background-color: #1A1A1A;
                color: #e0e0e0;
                border: 1px solid #2D2D2D;
                border-radius: 4px;
                padding: 4px 0px;
            }
            QMenu::item {
                padding: 6px 24px;
                background-color: transparent;
            }
            QMenu::item:selected {
                background-color: #00E676;
                color: #121212;
            }
            QMenu::separator {
                height: 1px;
                background-color: #2D2D2D;
                margin: 4px 0px;
            }
            
            QCheckBox {
                color: #cccccc;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
            QToolTip {
                background-color: #242424;
                color: #FFFFFF;
                border: 1px solid #4D4D4D;
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 11px;
                font-weight: 500;
            }
        """
        self.setStyleSheet(qss)
        if QApplication.instance():
            QApplication.instance().setStyleSheet("""
                QToolTip {
                    background-color: #242424;
                    color: #FFFFFF;
                    border: 1px solid #4D4D4D;
                    border-radius: 4px;
                    padding: 5px 8px;
                    font-size: 11px;
                    font-weight: 500;
                }
            """)

    def _detect_and_update_camera_info(self, image_list: list = None, video_list: list = None) -> str:
        """Scans image EXIF to extract camera make & model. Video files remain Undetected."""
        camera_name = "Undetected"
        imgs = image_list if image_list is not None else getattr(self, 'image_list', [])

        # Try reading EXIF from image files
        if imgs:
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                for path in imgs[:10]:
                    if not os.path.exists(path):
                        continue
                    with Image.open(path) as img:
                        exif = img.getexif()
                        if exif:
                            exif_dict = {TAGS.get(k, k): v for k, v in exif.items()}
                            make = str(exif_dict.get("Make", "")).strip()
                            model = str(exif_dict.get("Model", "")).strip()
                            if model:
                                if make and make.upper() not in model.upper():
                                    camera_name = f"{make} {model}"
                                else:
                                    camera_name = model
                                break
            except Exception:
                pass

        display_camera_name = self._camera_name_for_display(camera_name)
        self.camera_label.setText(f"Camera: {display_camera_name}")
        self.camera_label.setToolTip(camera_name if camera_name else "Undetected")
        return camera_name

    def _handle_dropped_images(self, files: list):
        if hasattr(self, 'standalone_cloud_path') and self.standalone_cloud_path:
            self._clear_standalone_cloud_clicked()
        self.image_list = files
        self.img_count_label.setText(f"Images Loaded: {len(files)}")
        if hasattr(self, 'photos_tab'):
            self.photos_tab.set_images(self.image_list)
        
        camera_name = self._detect_and_update_camera_info(files, getattr(self, 'current_video_list', []))
        if files:
            self.console_text.append(f"[INFO] Successfully imported {len(files)} files. Camera identified: {camera_name}")
            self._set_process_btn_state("ready")
            if hasattr(self, 'photos_tab') and hasattr(self.photos_tab, 'btn_bg_remove'):
                image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
                has_2d_images = any(isinstance(p, str) and p.lower().endswith(image_exts) for p in files)
                is_standalone = bool(getattr(self, 'standalone_cloud_path', None))
                self.photos_tab.btn_bg_remove.setEnabled(has_2d_images and not is_standalone)
        else:
            self.console_text.append("[INFO] Image list cleared.")
            self._set_process_btn_state("idle")
            if hasattr(self, 'photos_tab') and hasattr(self.photos_tab, 'btn_bg_remove'):
                self.photos_tab.btn_bg_remove.setEnabled(False)


    def _camera_name_for_display(self, camera_name: str) -> str:
        """Return a compact, printable camera name that cannot stretch the sidebar."""
        cleaned = "".join(ch if ch.isprintable() else " " for ch in str(camera_name or "Undetected"))
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return "Undetected"
        max_chars = 36
        if len(cleaned) <= max_chars:
            return cleaned
        return f"{cleaned[:max_chars - 1]}..."

    def _remove_selected_photos(self):
        selected = self.photos_tab.get_selected_images()
        if not selected:
            return
        
        selected_set = set(selected)
        if self.standalone_cloud_path and self.standalone_cloud_path in selected_set:
            self._clear_standalone_cloud_clicked()
            return
        
        # Filter out selected images
        self.image_list = [f for f in self.image_list if f not in selected_set]
        
        # Refresh UI
        self._handle_dropped_images(self.image_list)
        self.console_text.append(f"[INFO] Removed {len(selected)} selected file(s).")

    def _add_photos_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images/Videos or Point Cloud to Add", self.last_accessed_dir, 
            "Supported Files (*.png *.jpg *.jpeg *.tif *.tiff *.mp4 *.mov *.avi *.mkv *.ply);;Point Cloud Files (*.ply);;Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if files:
            self.last_accessed_dir = os.path.dirname(files[0])
            images = []
            videos = []
            ply_files = []
            ignored = []
            for f in files:
                normalized = os.path.normpath(f)
                ext = os.path.splitext(normalized)[1].lower()
                if ext in IMAGE_EXTS:
                    images.append(normalized)
                elif ext in VIDEO_EXTS:
                    videos.append(normalized)
                elif ext == '.ply':
                    ply_files.append(normalized)
                else:
                    ignored.append(os.path.basename(normalized))
            if ignored:
                self._warn_ignored_files(ignored)
            if ply_files:
                self._load_standalone_point_cloud(ply_files[0])
                if len(ply_files) > 1:
                    self.console_text.append(f"[INFO] Note: Multiple .ply files selected. Loaded first: {os.path.basename(ply_files[0])}")
            elif images or videos:
                self._route_import(images, videos, append_to_existing=True)

    def _open_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images/Videos or Point Cloud", self.last_accessed_dir, 
            "Supported Files (*.png *.jpg *.jpeg *.tif *.tiff *.mp4 *.mov *.avi *.mkv *.ply);;Point Cloud Files (*.ply);;Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if files:
            self.last_accessed_dir = os.path.dirname(files[0])
            images = []
            videos = []
            ply_files = []
            ignored = []
            for f in files:
                normalized = os.path.normpath(f)
                ext = os.path.splitext(normalized)[1].lower()
                if ext in IMAGE_EXTS:
                    images.append(normalized)
                elif ext in VIDEO_EXTS:
                    videos.append(normalized)
                elif ext == '.ply':
                    ply_files.append(normalized)
                else:
                    ignored.append(os.path.basename(normalized))
            if ignored:
                self._warn_ignored_files(ignored)
            if ply_files:
                self._load_standalone_point_cloud(ply_files[0])
                if len(ply_files) > 1:
                    self.console_text.append(f"[INFO] Note: Multiple .ply files selected. Loaded first: {os.path.basename(ply_files[0])}")
            elif images or videos:
                self._route_import(images, videos, append_to_existing=False)

    def _remove_backgrounds_clicked(self):
        if not self.image_list:
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Background Removal")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Remove Background from Imported Images?")
        msg_box.setInformativeText(
            "This process will remove the backgrounds from your images and add them to your dataset. "
            "Your original files will not be changed.\n\n"
            "Do you want to proceed?"
        )
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.Yes)
        
        # Apply the app stylesheet
        msg_box.setStyleSheet(self.styleSheet())
        
        ret = msg_box.exec()
        if ret == QMessageBox.Yes:
            self._start_background_removal()

    def _start_background_removal(self):
        if not self.image_list:
            return
            
        # Terminate any active viewer
        self._terminate_viewer()
        
        # Disable inputs to avoid modification during processing
        if hasattr(self, 'browse_files_btn'):
            self.browse_files_btn.setEnabled(False)
        if hasattr(self, 'browse_btn'):
            self.browse_btn.setEnabled(False)
        if hasattr(self, 'mobile_import_btn'):
            self.mobile_import_btn.setEnabled(False)
        if hasattr(self, 'photos_tab') and hasattr(self.photos_tab, 'btn_bg_remove'):
            self.photos_tab.btn_bg_remove.setEnabled(False)
        if hasattr(self, 'process_btn'):
            self.process_btn.setEnabled(False)
        if hasattr(self, 'step3_box'):
            self.step3_box.setEnabled(False)
        if hasattr(self, 'photos_tab'):
            self.photos_tab.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing background removal...")
        
        out_dir = os.path.join(get_reconstruction_out_dir(), "bg_removed_images")
        
        from pipeline_manager import BackgroundRemovalWorker
        self.bg_worker = BackgroundRemovalWorker(self.image_list, output_dir=out_dir, parent=self)
        self.bg_worker.progress_changed.connect(self.progress_bar.setValue)
        self.bg_worker.status_changed.connect(self.status_label.setText)
        self.bg_worker.log_message.connect(self._append_log)
        self.bg_worker.finished.connect(self._on_bg_removal_finished)
        
        self.console_text.append("[START] Initializing background removal worker thread...")
        self.bg_worker.start()

    def _on_bg_removal_finished(self, success: bool, updated_list: list, message: str):
        if hasattr(self, 'browse_files_btn'):
            self.browse_files_btn.setEnabled(True)
        if hasattr(self, 'browse_btn'):
            self.browse_btn.setEnabled(True)
        if hasattr(self, 'mobile_import_btn'):
            self.mobile_import_btn.setEnabled(True)
        if hasattr(self, 'photos_tab'):
            self.photos_tab.setEnabled(True)
        
        if success:
            self.console_text.append(f"[FINISHED] {message}")
            # Refresh photos list with the new files
            self._handle_dropped_images(updated_list)
        else:
            self.console_text.append(f"[FAILED] Background removal failed: {message}")
            # Re-enable controls with the current list
            self._handle_dropped_images(self.image_list)
            
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Idle")

    def _open_dir_dialog(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Images/Videos Folder", self.last_accessed_dir)
        if dir_path:
            self.last_accessed_dir = dir_path
            images = []
            videos = []
            for root, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    fp = os.path.join(root, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in IMAGE_EXTS:
                        images.append(os.path.normpath(fp))
                    elif ext in VIDEO_EXTS:
                        videos.append(os.path.normpath(fp))
            if images or videos:
                self._route_import(images, videos, append_to_existing=False)
            else:
                self.console_text.append("[WARNING] No valid images or videos found in selected folder.")

    def _open_zip_dialog(self):
        zip_path, _ = QFileDialog.getOpenFileName(
            self, "Select ZIP Archive", self.last_accessed_dir, "ZIP Archives (*.zip)"
        )
        if zip_path:
            self.last_accessed_dir = os.path.dirname(zip_path)
            extract_dir = os.path.join(
                get_reconstruction_out_dir(), "zip_imports", os.path.splitext(os.path.basename(zip_path))[0]
            )
            os.makedirs(extract_dir, exist_ok=True)
            
            images = []
            videos = []
            ignored = []
            
            import zipfile
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
                        clean_filename = os.path.basename(member.filename)
                        if not clean_filename or clean_filename.startswith("._") or "__MACOSX" in member.filename:
                            continue
                        
                        ext = os.path.splitext(clean_filename)[1].lower()
                        if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                            base, file_ext = os.path.splitext(clean_filename)
                            dest_path = os.path.join(extract_dir, clean_filename)
                            counter = 1
                            while os.path.exists(dest_path):
                                dest_path = os.path.join(extract_dir, f"{base}_{counter}{file_ext}")
                                counter += 1
                            
                            with zf.open(member) as src, open(dest_path, "wb") as dst:
                                dst.write(src.read())
                            
                            normalized = os.path.normpath(dest_path)
                            if ext in IMAGE_EXTS:
                                images.append(normalized)
                            else:
                                videos.append(normalized)
                        else:
                            ignored.append(clean_filename)
                
                if ignored:
                    self._warn_ignored_files(ignored[:10])
                
                if images or videos:
                    self.console_text.append(f"[INFO] Extracted {len(images)} image(s) and {len(videos)} video(s) from {os.path.basename(zip_path)}.")
                    self._route_import(images, videos, append_to_existing=False)
                else:
                    self.console_text.append("[WARNING] No valid supported media found in selected ZIP archive.")
            except Exception as e:
                self.console_text.append(f"[ERROR] Failed to open/extract ZIP archive: {e}")
                QMessageBox.critical(self, "ZIP Import Error", f"Failed to extract ZIP archive:\n{str(e)}")

    def _on_files_dropped(self, files: list):
        images = []
        videos = []
        ply_files = []
        for f in files:
            normalized = os.path.normpath(f)
            ext = os.path.splitext(normalized)[1].lower()
            if ext in IMAGE_EXTS:
                images.append(normalized)
            elif ext in VIDEO_EXTS:
                videos.append(normalized)
            elif ext == '.ply':
                ply_files.append(normalized)
        if ply_files:
            self._load_standalone_point_cloud(ply_files[0])
            if len(ply_files) > 1:
                self.console_text.append(f"[INFO] Note: Multiple .ply files dropped. Loaded first: {os.path.basename(ply_files[0])}")
        elif images or videos:
            self._route_import(images, videos, append_to_existing=False)

    def _warn_ignored_files(self, ignored: list):
        msg = "The following files were ignored because they are not supported images, videos, or .ply point clouds:\n\n"
        if len(ignored) > 10:
            msg += "\n".join(ignored[:10]) + f"\n... and {len(ignored) - 10} more files."
        else:
            msg += "\n".join(ignored)
        QMessageBox.warning(self, "Unsupported Files Ignored", msg)

    def _route_import(self, images: list, videos: list, append_to_existing: bool = False):
        if images:
            if append_to_existing:
                current_set = set(self.image_list)
                for img in images:
                    if img not in current_set:
                        self.image_list.append(img)
            else:
                self.image_list = images
            self._handle_dropped_images(self.image_list)
            
        if videos:
            dialog = VideoPresetModal(self)
            if dialog.exec() == QDialog.Accepted:
                name, desc, interval, blur = dialog.get_selected_preset()
                self.console_text.append(f"[VIDEO] Starting extraction using '{name}' preset (interval: {interval}s, blur threshold: {blur})...")
                self._start_video_extraction(videos, interval, blur)
            else:
                self.console_text.append("[VIDEO] Video import cancelled.")

    def _start_video_extraction(self, videos: list, interval: float, blur: float | None):
        self.current_video_list = list(videos)
        self.extraction_queue = list(videos)
        self.extracted_frames = []
        self.extraction_interval = interval
        self.extraction_blur = blur
        self.total_videos_to_extract = len(videos)
        self.progress_bar.setValue(0)
        
        self.browse_btn.setEnabled(False)
        self.mobile_import_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        
        self.process_btn.setText("Cancel Extraction")
        self.process_btn.setEnabled(True)
        try:
            self.process_btn.clicked.disconnect()
        except Exception:
            pass
        self.process_btn.clicked.connect(self._cancel_video_extraction)
        
        self._process_next_video()

    def _process_next_video(self):
        if not self.extraction_queue:
            self._on_all_extractions_finished()
            return
            
        video_path = self.extraction_queue.pop(0)
        video_name = os.path.basename(video_path)
        video_stem = os.path.splitext(video_name)[0]
        
        current_idx = self.total_videos_to_extract - len(self.extraction_queue)
        self.status_label.setText(f"Extracting {video_name} ({current_idx}/{self.total_videos_to_extract})...")
        
        out_dir = os.path.join(get_reconstruction_out_dir(), "extracted_frames", video_stem)
        os.makedirs(out_dir, exist_ok=True)
        
        self.extraction_worker = VideoExtractionWorker(
            video_path=video_path,
            output_dir=out_dir,
            interval_seconds=self.extraction_interval,
            blur_threshold=self.extraction_blur,
            parent=self
        )
        self.extraction_worker.progress.connect(self._on_extraction_progress)
        self.extraction_worker.finished.connect(self._on_video_extraction_finished)
        self.extraction_worker.error.connect(self._on_video_extraction_error)
        self.extraction_worker.start()

    def _on_extraction_progress(self, current, total):
        if total > 0 and self.total_videos_to_extract > 0:
            # 1-indexed video number (e.g. 1 out of 2)
            current_video_idx = max(self.total_videos_to_extract - len(self.extraction_queue), 1)
            
            # Clamp current video percentage to max 1.0 (prevents VFR overflow jumping)
            video_pct = min(max(current / total, 0.0), 1.0)
            
            # Calculate overall progress across all queued videos
            overall_pct = int(((current_video_idx - 1 + video_pct) / self.total_videos_to_extract) * 100)
            overall_pct = min(max(overall_pct, 0), 99)
            
            # Only allow progress bar to move forward — never backward
            if overall_pct > self.progress_bar.value():
                self.progress_bar.setValue(overall_pct)
            
    def _on_video_extraction_finished(self, result):
        import glob
        frames = glob.glob(os.path.join(result.output_dir, "*.jpg"))
        frames = [os.path.normpath(f) for f in frames]
        self.extracted_frames.extend(frames)
        self.console_text.append(f"[VIDEO] Extracted {result.frames_saved} frames from video (scanned: {result.total_frames_scanned}, rejected blur: {result.frames_rejected_blur}).")
        
        self._process_next_video()
        
    def _on_video_extraction_error(self, err_msg):
        self.console_text.append(f"[ERROR] Video extraction error: {err_msg}")
        QMessageBox.critical(self, "Video Extraction Error", f"An error occurred while extracting frames:\n\n{err_msg}")
        self._cleanup_extraction_ui()

    def _load_standalone_point_cloud(self, file_path: str):
        if not file_path or not os.path.exists(file_path):
            return

        self.last_accessed_dir = os.path.dirname(file_path)
        self._cumulative_mesh_transform = np.eye(4, dtype=np.float32)

        # Standalone point clouds are not photogrammetry datasets; clear any existing session backups
        clear_backup_dir()
        if hasattr(self, 'viewer_widget') and hasattr(self.viewer_widget, 'action_recover'):
            self.viewer_widget.action_recover.setEnabled(False)

        # 1. Instant header-peek for color detection (reads ~4 KB, never blocks)
        import point_cloud_io
        if not hasattr(point_cloud_io, "peek_has_colors"):
            import importlib
            importlib.reload(point_cloud_io)
        peek_fn = getattr(point_cloud_io, "peek_has_colors", lambda path: True)
        has_colors = peek_fn(file_path)

        # 2. Update UI immediately — place point cloud in Dataset tab
        self.standalone_cloud_path = file_path
        filename = os.path.basename(file_path)
        self.image_list = [file_path]
        if hasattr(self, 'photos_tab'):
            self.photos_tab.set_images([file_path])
        self._enter_standalone_mode(has_colors)

        # 3. Kick off viewer preview in background (non-blocking)
        self._set_view_mode(1)
        self._reload_viewer(file_path)

        # 4. Load full point cloud metadata in background (point count, normals, etc.)
        self.console_text.append(f"[STANDALONE] Loading point cloud metadata in background: {filename}...")
        self._cloud_import_worker = CloudImportWorker(file_path, parent=self)
        self._cloud_import_worker.finished.connect(self._on_cloud_import_done)
        self._cloud_import_worker.error.connect(
            lambda err: self.console_text.append(f"[WARNING] Background cloud load warning: {err}")
        )
        self._cloud_import_worker.start()

    def _import_standalone_cloud_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Point Cloud",
            self.last_accessed_dir,
            "Point Cloud Files (*.ply)"
        )
        if not file_path:
            return
        self._load_standalone_point_cloud(file_path)

    def _clear_standalone_cloud_clicked(self):
        self.standalone_cloud_path = None
        self.image_list = []
        if hasattr(self, 'photos_tab'):
            self.photos_tab.set_images([])
        self.console_text.append("[STANDALONE] Standalone cloud cleared.")
        self._exit_standalone_mode()

    def _enter_standalone_mode(self, has_colors: bool):
        # 1. Clear any loaded images / video frames
        self.extracted_frames = []
        filename = os.path.basename(self.standalone_cloud_path) if self.standalone_cloud_path else "Point Cloud"
        self.img_count_label.setText(f"Point Cloud: {filename}")
        self.camera_label.setText("Type: PLY Point Cloud (Direct Reconstruction)")
        
        # 2. Clear & disable reference cloud fusion
        # self._clear_reference_cloud_clicked()
        # self.ref_cloud_btn.setEnabled(False)
        
        # 3. Disable images directory browse/mobile import
        self.browse_files_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.mobile_import_btn.setEnabled(False)

        # 4. Hide photogrammetry options in Step 2 & Step 3
        self.quality_label.setVisible(False)
        self.quality_combo.setVisible(False)
        self.gpu_label.setVisible(False)
        self.gpu_combo.setVisible(False)
        if hasattr(self, 'recon_mode_combo'):
            self.recon_mode_combo.setVisible(False)
        if hasattr(self, 'auto_cleanup_row'):
            self.auto_cleanup_row.setVisible(False)
        else:
            self.auto_cleanup_checkbox.setVisible(False)
        if hasattr(self, 'manhattan_align_checkbox'):
            self.manhattan_align_checkbox.setVisible(False)
        self.advanced_toggle_btn.setVisible(False)
        self.advanced_panel.setVisible(False)
        if hasattr(self, 'step3_box'):
            self.step3_box.setVisible(False)

        # 5. Show standalone options
        self.standalone_panel.setVisible(True)
        self.vertex_color_toggle.setVisible(has_colors)
        self.vertex_color_toggle.setChecked(has_colors)
        if hasattr(self, 'photos_tab') and hasattr(self.photos_tab, 'btn_bg_remove'):
            self.photos_tab.btn_bg_remove.setEnabled(False)
        
        # 6. Enable process button
        self._set_process_btn_state("ready")
        self.console_text.append("[STANDALONE] UI transitioned to Standalone Reconstruction mode.")

    def _exit_standalone_mode(self):
        # 1. Enable main buttons again
        self.browse_files_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        
        self.img_count_label.setText("Images Loaded: 0")
        self.camera_label.setText("Camera: Undetected")
        
        # 2. Show photogrammetry options
        self.quality_label.setVisible(True)
        self.quality_combo.setVisible(True)
        self.gpu_label.setVisible(True)
        self.gpu_combo.setVisible(True)
        if hasattr(self, 'recon_mode_combo'):
            self.recon_mode_combo.setVisible(True)
            self._on_recon_mode_changed(self.recon_mode_combo.currentIndex())
        else:
            self.advanced_panel.setVisible(False)
        if hasattr(self, 'auto_cleanup_row'):
            self.auto_cleanup_row.setVisible(True)
        else:
            self.auto_cleanup_checkbox.setVisible(True)
        if hasattr(self, 'manhattan_align_checkbox'):
            self.manhattan_align_checkbox.setVisible(False)
        self.advanced_toggle_btn.setVisible(False)
        if hasattr(self, 'step3_box'):
            self.step3_box.setVisible(True)
            if hasattr(self, 'mc_enabled') and hasattr(self, 'mc_options_container'):
                self.mc_options_container.setEnabled(self.mc_enabled.isChecked())
        self._check_and_enable_cleanup_btn()

        # 3. Hide standalone panel
        self.standalone_panel.setVisible(False)
        
        # 4. Reset process button state
        self._set_process_btn_state("idle")
        if hasattr(self, 'photos_tab') and hasattr(self.photos_tab, 'btn_bg_remove'):
            image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
            has_2d_images = any(isinstance(p, str) and p.lower().endswith(image_exts) for p in (self.image_list or []))
            self.photos_tab.btn_bg_remove.setEnabled(has_2d_images)
        self.console_text.append("[STANDALONE] UI exited Standalone Reconstruction mode.")

    def _on_standalone_poisson_depth_changed(self, value):
        self.standalone_poisson_label.setText(f"Poisson Depth: {value}")

    def _cancel_video_extraction(self):
        self.console_text.append("[VIDEO] Cancelling video extraction...")
        if hasattr(self, 'extraction_worker') and self.extraction_worker:
            self.extraction_worker.cancel()
        self._cleanup_extraction_ui(cancelled=True)

    def _cleanup_extraction_ui(self, cancelled=False):
        self.browse_files_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        
        try:
            self.process_btn.clicked.disconnect()
        except Exception:
            pass
        self.process_btn.clicked.connect(self._start_processing)
        self._set_process_btn_state("ready" if len(self.image_list) > 0 else "idle")
        
        self.progress_bar.setValue(0)
        if cancelled:
            self.status_label.setText("Extraction cancelled.")
        else:
            self.status_label.setText("Extraction failed.")

    def _on_all_extractions_finished(self):
        self.browse_files_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        
        try:
            self.process_btn.clicked.disconnect()
        except Exception:
            pass
        self.process_btn.clicked.connect(self._start_processing)
        
        current_set = set(self.image_list)
        added_count = 0
        for f in self.extracted_frames:
            if f not in current_set:
                self.image_list.append(f)
                added_count += 1
        
        # Update Photos tab gallery view with extracted frames
        if hasattr(self, 'photos_tab'):
            self.photos_tab.set_images(self.image_list)

        # Detect camera info from video or extracted frames
        self._detect_and_update_camera_info(self.image_list, getattr(self, 'current_video_list', []))

        self.img_count_label.setText(f"Images Loaded: {len(self.image_list)}")
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Extraction complete. {len(self.image_list)} images ready.")
        self._set_process_btn_state("ready" if len(self.image_list) > 0 else "idle")
                


    def _find_available_session_images(self) -> list[str]:
        """Finds image files from backup/images, input_images, or extracted_frames."""
        found = []
        out_dir = get_reconstruction_out_dir()
        backup_dir = get_backup_dir()

        candidates = [
            os.path.join(backup_dir, "images"),
            os.path.join(out_dir, "input_images"),
            os.path.join(out_dir, "extracted_frames")
        ]
        image_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        for cand in candidates:
            if os.path.exists(cand):
                for root, _, files in os.walk(cand):
                    for f in files:
                        if f.lower().endswith(image_exts):
                            found.append(os.path.join(root, f))
                if found:
                    break
        return found

    def _stage_images_for_reconstruction(self) -> str | None:
        """
        Copy every path in self.image_list into a single flat staging directory
        so PipelineWorker can find all images (originals + extracted frames) in one place.

        Returns the staging directory path, or None if no images could be staged.
        """
        import shutil
        staging_dir = os.path.join(get_reconstruction_out_dir(), "input_images")
        
        image_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        valid_paths = [p for p in (self.image_list or []) if isinstance(p, str) and p.lower().endswith(image_exts)]
        if not valid_paths:
            self.image_list = self._find_available_session_images()
        else:
            self.image_list = valid_paths

        staged = []
        seen_names = {}   # Track duplicate basenames and rename to avoid collisions
        for path in self.image_list:
            basename = os.path.basename(path)
            # Handle duplicate filenames (e.g. frame_001.jpg from two different videos)
            if basename in seen_names:
                stem, ext = os.path.splitext(basename)
                seen_names[basename] += 1
                basename = f"{stem}_{seen_names[basename]}{ext}"
            else:
                seen_names[basename] = 0
            dest = os.path.join(staging_dir, basename)
            try:
                if os.path.abspath(path) != os.path.abspath(dest):
                    os.makedirs(staging_dir, exist_ok=True)
                    shutil.copy2(path, dest)
                staged.append(dest)
            except Exception as e:
                self.console_text.append(f"[WARNING] Could not stage {os.path.basename(path)}: {e}")

        # Fallback if self.image_list copy failed but staging_dir already has images
        if not staged and os.path.exists(staging_dir):
            staged = [
                os.path.join(staging_dir, f) for f in os.listdir(staging_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
            ]

        if not staged:
            return None

        # Update backup folder (~/.proximap/backup/images)
        try:
            backup_img_dir = os.path.join(get_backup_dir(), "images")
            clear_backup_dir()
            os.makedirs(backup_img_dir, exist_ok=True)
            for s_path in staged:
                try:
                    shutil.copy2(s_path, os.path.join(backup_img_dir, os.path.basename(s_path)))
                except Exception:
                    pass

            meta = {
                "scan_type": "photogrammetry",
                "last_completed_step": "images_imported",
                "image_count": len(staged),
                "quality_preset": self.quality_combo.currentText().lower(),
                "gpu_mode": self.gpu_combo.currentText().lower(),
                "has_plain_surfaces": (self.mapper_combo.currentIndex() == 1) if hasattr(self, 'mapper_combo') else False,
                "auto_cleanup": self.mc_enabled.isChecked() if hasattr(self, 'mc_enabled') else self.auto_cleanup_checkbox.isChecked(),
                "cleanup_params": {
                    "enable_reduction": self.mc_enable_reduction_check.isChecked() if hasattr(self, 'mc_enable_reduction_check') else True,
                    "target_reduction_pct": self.mc_reduction_spin.value() if hasattr(self, 'mc_reduction_spin') else 50,
                    "remove_duplicates": self.mc_remove_dups_check.isChecked() if hasattr(self, 'mc_remove_dups_check') else True,
                    "repair_nonmanifold": self.mc_repair_nm_check.isChecked() if hasattr(self, 'mc_repair_nm_check') else True,
                    "close_holes": self.mc_close_holes_check.isChecked() if hasattr(self, 'mc_close_holes_check') else True,
                    "max_hole_size": self.mc_max_hole_spin.value() if hasattr(self, 'mc_max_hole_spin') else 30
                },
                "mapper_mode": self.mapper_combo.currentText().lower() if hasattr(self, 'mapper_combo') else "incremental",
                "mesh_mode": "poisson" if hasattr(self, 'mesh_mode_combo') and self.mesh_mode_combo.currentIndex() == 1 else "default",
                "poisson_depth": self.poisson_depth_slider.value() if hasattr(self, 'poisson_depth_slider') else 9
            }
            save_session_metadata(meta)
        except Exception as e:
            self.console_text.append(f"[WARNING] Could not write backup metadata: {e}")

        self.console_text.append(f"[PREP] Staged {len(staged)} images for reconstruction → {staging_dir}")
        return staging_dir

    def _toggle_advanced_options(self):
        is_visible = not self.advanced_panel.isVisible()
        self.advanced_panel.setVisible(is_visible)
        self.advanced_toggle_btn.setText("▾  Advanced Options" if is_visible else "▸  Advanced Options")

    def _on_plain_surfaces_toggled(self, state):
        pass

    def _on_mapper_combo_changed(self, index: int):
        pass

    def _on_mc_enabled_toggled(self, checked):
        if hasattr(self, 'mc_options_container'):
            self.mc_options_container.setEnabled(checked)
        if hasattr(self, 'auto_cleanup_hint_label'):
            self.auto_cleanup_hint_label.setVisible(checked)

    def _sync_auto_cleanup_to_mc(self, checked):
        if hasattr(self, 'mc_enabled') and self.mc_enabled.isChecked() != checked:
            self.mc_enabled.blockSignals(True)
            self.mc_enabled.setChecked(checked)
            self.mc_enabled.blockSignals(False)
            self._on_mc_enabled_toggled(checked)

    def _sync_mc_to_auto_cleanup(self, checked):
        if hasattr(self, 'auto_cleanup_checkbox') and self.auto_cleanup_checkbox.isChecked() != checked:
            self.auto_cleanup_checkbox.blockSignals(True)
            self.auto_cleanup_checkbox.setChecked(checked)
            self.auto_cleanup_checkbox.blockSignals(False)
        self._on_mc_enabled_toggled(checked)

    def _sync_manhattan_to_custom(self, checked):
        if hasattr(self, 'custom_manhattan_check') and self.custom_manhattan_check.isChecked() != checked:
            self.custom_manhattan_check.blockSignals(True)
            self.custom_manhattan_check.setChecked(checked)
            self.custom_manhattan_check.blockSignals(False)

    def _sync_custom_to_manhattan(self, checked):
        if hasattr(self, 'manhattan_align_checkbox') and self.manhattan_align_checkbox.isChecked() != checked:
            self.manhattan_align_checkbox.blockSignals(True)
            self.manhattan_align_checkbox.setChecked(checked)
            self.manhattan_align_checkbox.blockSignals(False)


    def _on_mesh_mode_changed(self, index):
        self.poisson_widget.setVisible(index == 1)

    def _on_poisson_depth_changed(self, value):
        self.poisson_depth_label.setText(f"Poisson Depth: {value}")

    def _on_matcher_type_changed(self, index: int):
        is_exhaustive = (index in (0, 1))  # Auto-Select or Exhaustive
        is_vocab = (index == 3)
        if hasattr(self, 'lbl_block_size'):
            self.lbl_block_size.setVisible(is_exhaustive)
        if hasattr(self, 'custom_block_size_stepper'):
            self.custom_block_size_stepper.setVisible(is_exhaustive)
        elif hasattr(self, 'custom_block_size_spin'):
            self.custom_block_size_spin.setVisible(is_exhaustive)
        if hasattr(self, 'lbl_vocab'):
            self.lbl_vocab.setVisible(is_vocab)
        if hasattr(self, 'vocab_tree_widget'):
            self.vocab_tree_widget.setVisible(is_vocab)
            if is_vocab and hasattr(self, 'vocab_path_edit'):
                if not self.vocab_path_edit.text().strip():
                    from pipeline_manager import get_default_vocab_tree_path
                    def_path = get_default_vocab_tree_path()
                    if def_path:
                        self.vocab_path_edit.setText(def_path)

    def _browse_vocab_tree_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Vocabulary Tree File", self.last_accessed_dir,
            "Binary Vocab Tree (*.bin *.fbow);;All Files (*)"
        )
        if file_path:
            self.vocab_path_edit.setText(file_path)
            self.last_accessed_dir = os.path.dirname(file_path)

    def _on_custom_settings_toggled(self, checked):
        self.mapper_label.setEnabled(checked)
        self.mapper_combo.setEnabled(checked)
        self.mesh_mode_label.setEnabled(checked)
        self.mesh_mode_combo.setEnabled(checked)
        self.poisson_widget.setEnabled(checked)
        self.custom_settings_container.setEnabled(checked)
        self.custom_settings_container.setVisible(checked)
        if checked:
            self._prepopulate_custom_parameters_from_preset()

    def _prepopulate_custom_parameters_from_preset(self):
        quality_idx = self.quality_combo.currentIndex()
        if quality_idx == 0:  # Preview
            self.custom_features_spin.setValue(4096)
            self.custom_matches_spin.setValue(16384)
            self.custom_guided_check.setChecked(False)
            self.custom_ba_check.setChecked(False)
            self.custom_densify_res_combo.setCurrentIndex(2)
            self.custom_densify_views_spin.setValue(3)
            self.custom_refine_scales_spin.setValue(1)
            self.custom_texture_res_combo.setCurrentIndex(2)
        elif quality_idx == 1:  # Medium
            self.custom_features_spin.setValue(8192)
            self.custom_matches_spin.setValue(16384)
            self.custom_guided_check.setChecked(self.mapper_combo.currentIndex() == 1 if hasattr(self, 'mapper_combo') else False)
            self.custom_ba_check.setChecked(False)
            self.custom_densify_res_combo.setCurrentIndex(1)
            self.custom_densify_views_spin.setValue(4)
            self.custom_refine_scales_spin.setValue(2)
            self.custom_texture_res_combo.setCurrentIndex(1)
        elif quality_idx == 2:  # High
            self.custom_features_spin.setValue(12288)
            self.custom_matches_spin.setValue(32768)
            self.custom_guided_check.setChecked(True)
            self.custom_ba_check.setChecked(True)
            self.custom_densify_res_combo.setCurrentIndex(1)
            self.custom_densify_views_spin.setValue(5)
            self.custom_refine_scales_spin.setValue(2)
            self.custom_texture_res_combo.setCurrentIndex(1)
        elif quality_idx == 3:  # Ultra
            self.custom_features_spin.setValue(16384)
            self.custom_matches_spin.setValue(65536)
            self.custom_guided_check.setChecked(True)

    def _on_recon_mode_changed(self, index: int):
        is_advanced = (index == 1)
        if hasattr(self, 'quality_label'):
            self.quality_label.setVisible(not is_advanced)
        if hasattr(self, 'quality_combo'):
            self.quality_combo.setVisible(not is_advanced)
        if hasattr(self, 'advanced_panel'):
            self.advanced_panel.setVisible(is_advanced)
        if hasattr(self, 'custom_settings_toggle'):
            self.custom_settings_toggle.setChecked(is_advanced)
        if hasattr(self, 'advanced_toggle_btn'):
            self.advanced_toggle_btn.setVisible(False)

    def _on_auto_cleanup_toggled(self, checked):
        if hasattr(self, 'mc_options_container'):
            self.mc_options_container.setVisible(checked)

    def _on_mc_enabled_toggled(self, checked):
        self._on_auto_cleanup_toggled(checked)

    def _sync_auto_cleanup_to_mc(self, checked):
        self._on_auto_cleanup_toggled(checked)

    def _sync_mc_to_auto_cleanup(self, checked):
        self._on_auto_cleanup_toggled(checked)

    def _start_processing(self, resume_from_step: str = None):
        if not self.standalone_cloud_path and not self.image_list:
            self.image_list = self._find_available_session_images()
        if not self.standalone_cloud_path and not self.image_list:
            QMessageBox.warning(self, "No Images Found", "No images or video frames were found for this reconstruction session. Please import images or videos to proceed.")
            return

        # Check for unapplied point cloud / mesh transform changes
        if hasattr(self, 'transform_tool_window') and self.transform_tool_window is not None:
            if self._current_points is not None and len(self._current_points) > 0 and self.transform_tool_window.has_unapplied_changes():
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Unapplied Transform Detected")
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setText("You have modified the point cloud transform (Position / Orientation) without applying it.")
                msg_box.setInformativeText("Would you like to apply the transformation before starting reconstruction?")
                
                apply_btn = msg_box.addButton("Apply Transform", QMessageBox.AcceptRole)
                discard_btn = msg_box.addButton("Discard Changes", QMessageBox.RejectRole)
                cancel_btn = msg_box.addButton("Cancel", QMessageBox.ActionRole)
                
                msg_box.setDefaultButton(apply_btn)
                msg_box.exec()
                
                clicked = msg_box.clickedButton()
                if clicked == apply_btn:
                    self.transform_tool_window.apply_transform()
                    self.console_text.append("[TOOLS] Applied unapplied transform before starting reconstruction.")
                elif clicked == cancel_btn:
                    self.console_text.append("[TOOLS] Reconstruction canceled by user.")
                    return
                else:
                    self.transform_tool_window.reset_all()
                    self.console_text.append("[TOOLS] Discarded unapplied transform.")

        # Pre-flight dynamic memory guard & advisory
        try:
            budget = hardware_profiler.get_memory_budget()
            if budget.available_gb < 1.5 and budget.swap_used_gb > (budget.swap_total_gb * 0.85):
                QMessageBox.critical(
                    self,
                    "Critically Low System Memory",
                    f"Available System RAM ({budget.available_gb:.1f} GB) and Swap Memory are critically low.\n\n"
                    "Please close background applications (such as browsers or video editors) to free memory before starting reconstruction."
                )
                return
            
            n_images = len(self.image_list)
            if not self.custom_settings_toggle.isChecked() and n_images > 80 and budget.available_gb < 6.0:
                reply = QMessageBox.information(
                    self,
                    "Memory Safeguard Advisory",
                    f"Your scan contains {n_images} images with {budget.available_gb:.1f} GB available RAM.\n\n"
                    "Proximap will automatically activate Sequential Feature Matching and dynamic thread management "
                    "to protect system stability and prevent memory crashes.\n\n"
                    "Do you wish to proceed?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply != QMessageBox.Yes:
                    return
        except Exception:
            pass
            
        # Terminate any active viewer to prevent lock conflict on MVS files during reconstruction
        self._terminate_viewer()
        self._cumulative_mesh_transform = np.eye(4, dtype=np.float32)

        
        self._set_process_btn_state("progress")
        if hasattr(self, 'cleanup_btn'):
            self._set_cleanup_btn_state("idle")
        self.browse_files_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.mobile_import_btn.setEnabled(False)
        self._set_dataset_modification_controls_enabled(False)
        self._set_export_actions_enabled(False)
        # self.ref_cloud_btn.setEnabled(False)
        # self.ref_cloud_clear_btn.setEnabled(False)
        self.quality_combo.setEnabled(False)
        self.gpu_combo.setEnabled(False)
        if hasattr(self, 'recon_mode_combo'):
            self.recon_mode_combo.setEnabled(False)
        self.auto_cleanup_checkbox.setEnabled(False)
        if hasattr(self, 'manhattan_align_checkbox'):
            self.manhattan_align_checkbox.setEnabled(False)
        if hasattr(self, 'mc_enabled'):
            self.mc_enabled.setEnabled(False)
        if hasattr(self, 'mc_options_container'):
            self.mc_options_container.setEnabled(False)
        self.mapper_combo.setEnabled(False)
        self.mesh_mode_combo.setEnabled(False)
        self.poisson_depth_slider.setEnabled(False)
        self.custom_settings_toggle.setEnabled(False)
        self.custom_settings_container.setEnabled(False)
        self.advanced_toggle_btn.setEnabled(False)
        
        # Temp output dir inside the workspace or local appdata if not writable
        output_dir = get_reconstruction_out_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        if self.standalone_cloud_path:
            # Standalone reconstruction mode execution path
            self.viewer_widget.action_import_standalone.setEnabled(False)
            self.standalone_cloud_clear_btn.setEnabled(False)
            self.standalone_poisson_slider.setEnabled(False)
            self.vertex_color_toggle.setEnabled(False)
            
            try:
                from standalone_reconstruction import StandaloneReconstructionWorker
                include_colors = self.vertex_color_toggle.isChecked()
                poisson_depth = self.standalone_poisson_slider.value()
                
                self.worker = StandaloneReconstructionWorker(
                    self.standalone_cloud_path,
                    output_dir,
                    include_colors=include_colors,
                    poisson_depth=poisson_depth,
                    parent=self
                )
                self.worker.progress_changed.connect(self._on_progress_changed)
                self.worker.status_changed.connect(self.status_label.setText)
                self.worker.log_message.connect(self._append_log)
                self.worker.finished.connect(self._on_pipeline_finished)
                
                self.console_text.append("[START] Initializing asynchronous standalone point cloud reconstruction task thread...")
                self._reconstruction_heartbeat = QTimer(self)
                self._reconstruction_heartbeat.timeout.connect(lambda: QApplication.processEvents())
                self._reconstruction_heartbeat.start(200)
                self.worker.start()
                self._update_file_menu_states()
            except Exception as e:
                self.console_text.append(f"[ERROR] Failed to start standalone reconstruction: {e}")
                self._set_process_btn_state("ready")
                self.viewer_widget.action_import_standalone.setEnabled(True)
                self.standalone_cloud_clear_btn.setEnabled(True)
                self.standalone_poisson_slider.setEnabled(True)
                self.vertex_color_toggle.setEnabled(True)
            return
        
        # stage all images into one flat directory
        image_dir = self._stage_images_for_reconstruction()
        if image_dir is None:
            self.console_text.append("[ERROR] Could not stage images. Aborting reconstruction.")
            self._set_process_btn_state("ready")
            self.browse_files_btn.setEnabled(True)
            self.browse_btn.setEnabled(True)
            self.mobile_import_btn.setEnabled(True)
            self.gpu_combo.setEnabled(True)
            if hasattr(self, 'recon_mode_combo'):
                self.recon_mode_combo.setEnabled(True)
            self.custom_settings_toggle.setEnabled(True)
            self._on_custom_settings_toggled(self.custom_settings_toggle.isChecked())
            self.advanced_toggle_btn.setEnabled(True)
            return

        # Extract quality, gpu mode, and mapper mode
        quality_presets = ["preview", "medium", "high", "ultra"]
        gpu_modes = ["auto", "force_gpu", "force_cpu"]
        mapper_modes = ["incremental", "global"]
        quality_preset = quality_presets[self.quality_combo.currentIndex()]
        gpu_mode = gpu_modes[self.gpu_combo.currentIndex()]
        mapper_mode = mapper_modes[self.mapper_combo.currentIndex()]
        has_plain = (self.mapper_combo.currentIndex() == 1) if hasattr(self, 'mapper_combo') else False
        auto_cleanup = self.mc_enabled.isChecked() if hasattr(self, 'mc_enabled') else self.auto_cleanup_checkbox.isChecked()
        manhattan_align = self.custom_manhattan_check.isChecked() if (hasattr(self, 'custom_manhattan_check') and self.custom_settings_toggle.isChecked()) else (self.manhattan_align_checkbox.isChecked() if hasattr(self, 'manhattan_align_checkbox') else True)
        mesh_mode = "poisson" if self.mesh_mode_combo.currentIndex() == 1 else "default"
        poisson_depth = self.poisson_depth_slider.value()

        res_map = [0, 3200, 2400, 1600, 1200, 800]
        selected_max_res = res_map[self.img_max_res_combo.currentIndex()] if hasattr(self, 'img_max_res_combo') else 3200

        matcher_type_map = {
            0: "auto",
            1: "exhaustive",
            2: "sequential",
            3: "vocab_tree",
            4: "spatial"
        }
        selected_matcher = matcher_type_map.get(self.custom_matcher_combo.currentIndex(), "auto")

        cleanup_params = {
            "enable_cleanup": auto_cleanup,
            "enable_reduction": self.mc_enable_reduction_check.isChecked() if hasattr(self, 'mc_enable_reduction_check') else True,
            "target_reduction_pct": self.mc_reduction_spin.value() if hasattr(self, 'mc_reduction_spin') else 50,
            "remove_duplicates": self.mc_remove_dups_check.isChecked() if hasattr(self, 'mc_remove_dups_check') else True,
            "repair_nonmanifold": self.mc_repair_nm_check.isChecked() if hasattr(self, 'mc_repair_nm_check') else True,
            "close_holes": self.mc_close_holes_check.isChecked() if hasattr(self, 'mc_close_holes_check') else True,
            "max_hole_size": self.mc_max_hole_spin.value() if hasattr(self, 'mc_max_hole_spin') else 30
        }

        custom_params = None
        if self.custom_settings_toggle.isChecked():
            custom_params = {
                "image_max_resolution": selected_max_res,
                "colmap_matcher_type": selected_matcher,
                "vocab_tree_path": self.vocab_path_edit.text().strip(),
                "colmap_max_num_features": self.custom_features_spin.value(),
                "colmap_max_num_matches": self.custom_matches_spin.value(),
                "colmap_block_size": self.custom_block_size_spin.value(),
                "guided_matching": "1" if self.custom_guided_check.isChecked() else "0",
                "run_bundle_adjuster": self.custom_ba_check.isChecked(),
                "manhattan_align": manhattan_align,
                "densify_res": str(self.custom_densify_res_combo.currentIndex()),
                "densify_views": str(self.custom_densify_views_spin.value()),
                "refine_scales": str(self.custom_refine_scales_spin.value()),
                "texture_res": str(self.custom_texture_res_combo.currentIndex()),
                "cleanup_params": cleanup_params
            }
        else:
            custom_params = {
                "image_max_resolution": selected_max_res,
                "manhattan_align": manhattan_align,
                "cleanup_params": cleanup_params
            }

        from pipeline_manager import PipelineWorker
        self.worker = PipelineWorker(
            image_dir, 
            output_dir, 
            quality_preset=quality_preset, 
            gpu_mode=gpu_mode, 
            has_plain_surfaces=has_plain,
            auto_cleanup=auto_cleanup,
            manhattan_align=manhattan_align,
            mapper_mode=mapper_mode,
            mesh_mode=mesh_mode,
            poisson_depth=poisson_depth,
            custom_params=custom_params,
            resume_from_step=resume_from_step,
            parent=self
        )
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.log_message.connect(self._append_log)
        self.worker.finished.connect(self._on_pipeline_finished)
        
        self.console_text.append("[START] Initializing asynchronous reconstruction task thread...")
        self.worker.start()
        self._update_file_menu_states()

    def _append_log(self, text: str):
        if text:
            self.console_text.append(text)

    def _on_progress_changed(self, value: int):
        self.progress_bar.setValue(value)
        
        # At Step 6/10 (progress=70), the scene.mvs is exported from OpenMVG.
        # Auto-switch the viewer to show the sparse cloud + camera orientations.
        if value == 70:
            mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
            scene_mvs = os.path.join(mvs_dir, "scene.mvs")
            if os.path.exists(scene_mvs):
                self.viewer_widget.set_mvs_directory(mvs_dir)
                self._set_view_mode(0)
                self._reload_viewer(scene_mvs)

    def _on_pipeline_finished(self, success: bool, msg: str):
        if hasattr(self, '_reconstruction_heartbeat') and self._reconstruction_heartbeat is not None:
            self._reconstruction_heartbeat.stop()
            self._reconstruction_heartbeat.deleteLater()
            self._reconstruction_heartbeat = None
        if hasattr(self, 'standalone_cloud_path') and self.standalone_cloud_path:
            # Standalone reconstruction mode finished cleanup
            self.viewer_widget.action_import_standalone.setEnabled(True)
            self.standalone_cloud_clear_btn.setEnabled(True)
            self.standalone_poisson_slider.setEnabled(True)
            self.vertex_color_toggle.setEnabled(True)
            
            if success:
                self._set_process_btn_state("ready")
                self.console_text.append(f"[FINISHED] {msg}")
                self._update_upload_button_state()
                
                mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
                self.viewer_widget.set_mvs_directory(mvs_dir)
                self._set_view_mode(2)
                
                mesh_path = self.viewer_widget.get_selected_file_path()
                if mesh_path:
                    self._reload_viewer(mesh_path)
            else:
                self._set_process_btn_state("failed")
                self.console_text.append(f"[FAILED] Reconstruction failed: {msg}")
            self._update_file_menu_states()
            return

        self.browse_files_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        self._set_dataset_modification_controls_enabled(True)
        self.gpu_combo.setEnabled(True)
        if hasattr(self, 'recon_mode_combo'):
            self.recon_mode_combo.setEnabled(True)
        self.auto_cleanup_checkbox.setEnabled(True)
        if hasattr(self, 'manhattan_align_checkbox'):
            self.manhattan_align_checkbox.setEnabled(True)
        if hasattr(self, 'mc_enabled'):
            self.mc_enabled.setEnabled(True)
        if hasattr(self, 'mc_options_container'):
            self.mc_options_container.setEnabled(self.mc_enabled.isChecked())
        self.custom_settings_toggle.setEnabled(True)
        self._on_custom_settings_toggled(self.custom_settings_toggle.isChecked())
        self.advanced_toggle_btn.setEnabled(True)
        
        if success:
            self._last_failed_stage = None
            self._set_process_btn_state("ready")
            self.console_text.append(f"[FINISHED] {msg}")
            self._update_upload_button_state()
            
            mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
            self.viewer_widget.set_mvs_directory(mvs_dir)
            
            # Pick best available viewer mode
            mesh_exists = False
            for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_texture.obj", "scene_dense_mesh_refine.ply", "scene_dense_mesh_refcloud.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
                if os.path.exists(os.path.join(mvs_dir, candidate)):
                    mesh_exists = True
                    break
            
            dense_exists = os.path.exists(os.path.join(mvs_dir, "scene_dense.mvs"))
            
            if mesh_exists:
                self._set_view_mode(2)
            elif dense_exists:
                self._set_view_mode(1)
            else:
                self._set_view_mode(0)
            
            mesh_path = self.viewer_widget.get_selected_file_path()
            if mesh_path:
                self._reload_viewer(mesh_path)
        else:
            meta = load_session_metadata() or {}
            last_completed = meta.get("last_completed_step")
            
            auto_cleanup = self.mc_enabled.isChecked() if hasattr(self, 'mc_enabled') else self.auto_cleanup_checkbox.isChecked()
            resume_map = {
                "image_preparation": "features_extracted",
                "images_imported": "features_extracted",
                "features_extracted": "features_matched",
                "features_matched": "sparse_reconstruction",
                "sparse_reconstruction": "dense_reconstruction",
                "dense_reconstruction": "mesh_reconstructed",
                "mesh_reconstructed": "mesh_refined",
                "mesh_refined": "mesh_cleaned" if auto_cleanup else "mesh_textured",
                "mesh_cleaned": "mesh_textured",
            }
            if last_completed in resume_map:
                self._last_failed_stage = resume_map[last_completed]
            else:
                self._last_failed_stage = None

            self._set_process_btn_state("failed")
            self.console_text.append(f"[FAILED] Reconstruction failed: {msg}")
        self._update_file_menu_states()
        self._check_and_enable_cleanup_btn()

    def _start_cleanup_only(self):
        """
        Executes Mesh Cleanup + TextureMesh standalone on an existing reconstructed mesh.
        """
        if not self._has_existing_mesh():
            QMessageBox.warning(
                self,
                "No Mesh Available",
                "No reconstructed mesh was found to clean.\n\nPlease run reconstruction first (Step 2) before running mesh cleanup."
            )
            return

        self._terminate_viewer()
        
        self._set_process_btn_state("idle")
        self._set_cleanup_btn_state("progress")
        self.browse_files_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.mobile_import_btn.setEnabled(False)
        self._set_export_actions_enabled(False)
        self.quality_combo.setEnabled(False)
        self.gpu_combo.setEnabled(False)
        self.auto_cleanup_checkbox.setEnabled(False)
        if hasattr(self, 'mc_enabled'):
            self.mc_enabled.setEnabled(False)
        if hasattr(self, 'mc_options_container'):
            self.mc_options_container.setEnabled(False)
        self.custom_settings_toggle.setEnabled(False)
        self.custom_settings_container.setEnabled(False)
        self.advanced_toggle_btn.setEnabled(False)

        output_dir = get_reconstruction_out_dir()
        
        quality_presets = ["preview", "medium", "high", "ultra"]
        quality_preset = quality_presets[self.quality_combo.currentIndex()]

        cleanup_params = {
            "enable_cleanup": True,
            "enable_reduction": self.mc_enable_reduction_check.isChecked() if hasattr(self, 'mc_enable_reduction_check') else True,
            "target_reduction_pct": self.mc_reduction_spin.value() if hasattr(self, 'mc_reduction_spin') else 50,
            "remove_duplicates": self.mc_remove_dups_check.isChecked() if hasattr(self, 'mc_remove_dups_check') else True,
            "repair_nonmanifold": self.mc_repair_nm_check.isChecked() if hasattr(self, 'mc_repair_nm_check') else True,
            "close_holes": self.mc_close_holes_check.isChecked() if hasattr(self, 'mc_close_holes_check') else True,
            "max_hole_size": self.mc_max_hole_spin.value() if hasattr(self, 'mc_max_hole_spin') else 30
        }

        custom_params = {}
        if self.custom_settings_toggle.isChecked():
            custom_params["texture_res"] = str(self.custom_texture_res_combo.currentIndex())
        custom_params["cleanup_params"] = cleanup_params

        from pipeline_manager import CleanupAndTextureWorker
        self.worker = CleanupAndTextureWorker(
            output_dir=output_dir,
            cleanup_params=cleanup_params,
            custom_params=custom_params,
            quality_preset=quality_preset,
            parent=self
        )
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.log_message.connect(self._append_log)
        self.worker.finished.connect(self._on_cleanup_finished)

        self.console_text.append("[START] Initializing standalone mesh cleanup and texturing task thread...")
        self.worker.start()
        self._update_file_menu_states()

    def _on_cleanup_finished(self, success: bool, msg: str):
        """
        Handles completion of standalone mesh cleanup and texturing.
        """
        self.browse_files_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        self._set_dataset_modification_controls_enabled(True)
        self.gpu_combo.setEnabled(True)
        self.auto_cleanup_checkbox.setEnabled(True)
        if hasattr(self, 'manhattan_align_checkbox'):
            self.manhattan_align_checkbox.setEnabled(True)
        if hasattr(self, 'mc_enabled'):
            self.mc_enabled.setEnabled(True)
        if hasattr(self, 'mc_options_container'):
            self.mc_options_container.setEnabled(self.mc_enabled.isChecked())
        self.custom_settings_toggle.setEnabled(True)
        self._on_custom_settings_toggled(self.custom_settings_toggle.isChecked())
        self.advanced_toggle_btn.setEnabled(True)

        self._set_process_btn_state("ready" if len(self.image_list) > 0 else "idle")
        if success:
            self._set_cleanup_btn_state("ready")
            self.console_text.append(f"[FINISHED] {msg}")
            self._update_upload_button_state()
            
            mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
            self.viewer_widget.set_mvs_directory(mvs_dir)
            self._set_view_mode(2) # Mesh mode
            
            mesh_path = self.viewer_widget.get_selected_file_path()
            if mesh_path:
                self._reload_viewer(mesh_path)
        else:
            self._set_cleanup_btn_state("failed")
            self.console_text.append(f"[FAILED] Mesh cleanup failed: {msg}")
        self._update_file_menu_states()

    def _export_mesh(self, fmt):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Final Reconstruction Mesh", os.path.join(self.last_accessed_dir, f"reconstructed_mesh{fmt}"), f"Mesh Files (*{fmt})"
        )
        if not file_path:
            return
            
        self.last_accessed_dir = os.path.dirname(file_path)
            
        mvs_out = self._get_active_mvs_dir()
        
        if hasattr(self, 'standalone_cloud_path') and self.standalone_cloud_path:
            # Standalone reconstruction export logic
            src_ply = None
            for candidate in ["scene_dense_mesh_refine.ply", "scene_dense_mesh.ply"]:
                path = os.path.join(mvs_out, candidate)
                if os.path.exists(path):
                    src_ply = path
                    break
            
            if not src_ply:
                self.console_text.append(f"[ERROR] Could not find reconstructed standalone PLY file in {mvs_out}")
                QMessageBox.critical(self, "Export Error", f"Could not find reconstructed PLY file in {mvs_out}")
                return

            try:
                import shutil
                if fmt == ".ply":
                    shutil.copy2(src_ply, file_path)
                    self.console_text.append(f"[EXPORT] Standalone PLY mesh successfully written to {file_path}")
                elif fmt in [".obj", ".glb"]:
                    import trimesh
                    self.console_text.append(f"[INFO] Converting and exporting standalone mesh as {fmt.upper()}...")
                    mesh = trimesh.load(src_ply, force="mesh")
                    mesh.export(file_path, file_type=fmt[1:])
                    self.console_text.append(f"[EXPORT] Standalone {fmt.upper()} mesh successfully written to {file_path}")
                elif fmt == ".usdz":
                    import trimesh
                    from mesh_editor.scene import _export_usdz_from_trimesh
                    self.console_text.append("[INFO] Converting and exporting standalone mesh as USDZ...")
                    mesh = trimesh.load(src_ply, force="mesh")
                    _export_usdz_from_trimesh(mesh, file_path)
                    self.console_text.append(f"[EXPORT] Standalone USDZ mesh successfully written to {file_path}")
                else:
                    self.console_text.append(f"[ERROR] Unsupported export format: {fmt}")
            except Exception as e:
                self.console_text.append(f"[ERROR] Failed to export standalone mesh: {e}")
                QMessageBox.critical(self, "Export Error", f"Failed to export mesh:\n\n{e}")
            return

        import shutil
        try:
            if fmt == ".ply":
                src_ply = None
                for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
                    path = os.path.join(mvs_out, candidate)
                    if os.path.exists(path):
                        src_ply = path
                        break
                
                if src_ply:
                    shutil.copy2(src_ply, file_path)
                    self.console_text.append(f"[EXPORT] PLY mesh successfully written to {file_path}")
                else:
                    self.console_text.append(f"[ERROR] Could not find reconstructed PLY file in {mvs_out}")
            elif fmt == ".obj":
                src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
                src_mtl = os.path.join(mvs_out, "scene_dense_mesh_texture.mtl")
                
                if os.path.exists(src_obj):
                    shutil.copy2(src_obj, file_path)
                    dest_dir = os.path.dirname(file_path)
                    
                    if os.path.exists(src_mtl):
                        try:
                            # The MTL file might reference a different filename, so we keep the original name
                            shutil.copy2(src_mtl, os.path.join(dest_dir, "scene_dense_mesh_texture.mtl"))
                            
                            # Parse the MTL to find the texture image(s) and copy them
                            with open(src_mtl, 'r') as f:
                                for line in f:
                                    if line.strip().startswith("map_Kd "):
                                        parts = line.strip().split(" ", 1)
                                        if len(parts) > 1:
                                            tex_filename = parts[1].strip()
                                            src_tex = os.path.join(mvs_out, tex_filename)
                                            if os.path.exists(src_tex):
                                                shutil.copy2(src_tex, os.path.join(dest_dir, tex_filename))
                        except Exception as tex_err:
                            self.console_text.append(f"[WARNING] Failed to copy OBJ textures or material file: {tex_err}")
                        
                    self.console_text.append(f"[EXPORT] OBJ mesh and textures successfully written to {dest_dir}")
                else:
                    self.console_text.append(f"[ERROR] Could not find reconstructed OBJ file at {src_obj}")
            elif fmt == ".glb":
                src_glb = os.path.join(mvs_out, "scene_dense_mesh_texture.glb")
                src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
                
                if os.path.exists(src_glb):
                    shutil.copy2(src_glb, file_path)
                    self.console_text.append(f"[EXPORT] GLB mesh successfully written to {file_path}")
                elif os.path.exists(src_obj):
                    self.console_text.append("[INFO] Converting OBJ to GLB using trimesh...")
                    try:
                        import trimesh
                        mesh_obj = trimesh.load(src_obj)
                        mesh_obj.export(file_path, file_type="glb")
                        self.console_text.append(f"[EXPORT] GLB mesh successfully written to {file_path}")
                    except Exception as e:
                        self.console_text.append(f"[ERROR] Failed to convert OBJ to GLB: {e}")
                else:
                    src_ply = None
                    for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
                        path = os.path.join(mvs_out, candidate)
                        if os.path.exists(path):
                            src_ply = path
                            break
                    if src_ply:
                        self.console_text.append("[INFO] Converting PLY mesh to GLB using trimesh...")
                        try:
                            import trimesh
                            mesh_ply = trimesh.load(src_ply, force="mesh")
                            mesh_ply.export(file_path, file_type="glb")
                            self.console_text.append(f"[EXPORT] GLB mesh successfully written to {file_path}")
                        except Exception as e:
                            self.console_text.append(f"[ERROR] Failed to convert PLY to GLB: {e}")
                    else:
                        self.console_text.append(f"[ERROR] Could not find any reconstructed mesh files at {mvs_out}")
            elif fmt == ".usdz":
                src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
                src_glb = os.path.join(mvs_out, "scene_dense_mesh_texture.glb")
                
                source_path = None
                if os.path.exists(src_glb):
                    source_path = src_glb
                elif os.path.exists(src_obj):
                    source_path = src_obj
                else:
                    for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
                        path = os.path.join(mvs_out, candidate)
                        if os.path.exists(path):
                            source_path = path
                            break

                if source_path:
                    self.console_text.append("[INFO] Loading source mesh to export as USDZ...")
                    try:
                        import trimesh
                        from mesh_editor.scene import _export_usdz_from_trimesh
                        mesh = trimesh.load(source_path, force="mesh")
                        _export_usdz_from_trimesh(mesh, file_path)
                        self.console_text.append(f"[EXPORT] USDZ mesh successfully written to {file_path}")
                    except Exception as e:
                        self.console_text.append(f"[ERROR] Failed to export to USDZ: {e}")
                else:
                    self.console_text.append(f"[ERROR] Could not find any reconstructed source mesh files in {mvs_out} to export as USDZ.")
        except Exception as e:
            self.console_text.append(f"[ERROR] Failed to export mesh: {e}")

    def _get_active_mvs_dir(self):
        viewer_mvs = self.viewer_widget.current_mvs_dir
        if viewer_mvs and os.path.exists(viewer_mvs):
            return viewer_mvs
        output_dir = get_reconstruction_out_dir()
        return os.path.join(output_dir, "mvs")

    def _set_export_actions_enabled(self, enabled: bool):
        self.viewer_widget.action_export_dense.setEnabled(enabled)
        self.viewer_widget.action_export_sparse.setEnabled(enabled)
        self.viewer_widget.action_export_glb.setEnabled(enabled)
        self.viewer_widget.action_export_obj.setEnabled(enabled)
        self.viewer_widget.action_export_usdz.setEnabled(enabled)
        self.viewer_widget.action_mobile_export.setEnabled(enabled)
        self.viewer_widget.action_upload_proximap.setEnabled(enabled)

    def _update_upload_button_state(self):
        mvs_out = self._get_active_mvs_dir()
        if hasattr(self, 'standalone_cloud_path') and self.standalone_cloud_path:
            has_model = os.path.exists(os.path.join(mvs_out, "scene_dense_mesh_refine.ply")) or \
                        os.path.exists(os.path.join(mvs_out, "scene_dense_mesh.ply"))
        else:
            src_glb = os.path.join(mvs_out, "scene_dense_mesh_texture.glb")
            src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
            has_model = os.path.exists(src_glb) or os.path.exists(src_obj)
        self._set_export_actions_enabled(has_model)

    def _check_existing_scene(self):
        """Checks if a valid, recoverable reconstruction checkpoint exists in ~/.proximap/backup/."""
        has_backup = is_session_backup_valid()
        self.viewer_widget.action_recover.setEnabled(has_backup)
        if has_backup:
            meta = load_session_metadata()
            step = meta.get("last_completed_step", "unknown") if meta else "unknown"
            self.console_text.append(f"[INFO] Backup session found from previous run (Stage: {step}). Select File → Recover Last Session to load.")

    def _check_startup_recovery(self):
        """Checks on application initialization if an automatic recovery prompt should be displayed."""
        if not is_session_backup_valid():
            return

        meta = load_session_metadata()
        if not meta:
            return
        
        settings = load_app_settings()
        if settings.get("dont_ask_recovery_on_startup", False):
            return
        
        dlg = SessionRecoveryDialog(meta, self)
        if dlg.exec() == QDialog.Accepted:
            if dlg.user_choice == "resume":
                self._retrieve_last_session()
            elif dlg.user_choice == "discard":
                clear_backup_dir()
                self._check_existing_scene()

    def _retrieve_last_session(self):
        """Restores checkpoint from ~/.proximap/backup/, loads settings, updates UI and viewer."""
        import shutil
        if not is_session_backup_valid():
            QMessageBox.warning(
                self,
                "Incomplete Session Backup",
                "The previous session backup is missing required image or reconstruction files and cannot be recovered.\n\n"
                "You can discard the backup to start a fresh reconstruction."
            )
            return

        meta = load_session_metadata()

        backup_dir = get_backup_dir()
        out_dir = get_reconstruction_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        # Copy backup folders (colmap, mvs) to reconstruction_out
        for folder in ["colmap", "mvs"]:
            src_f = os.path.join(backup_dir, folder)
            dst_f = os.path.join(out_dir, folder)
            if os.path.exists(src_f):
                if os.path.exists(dst_f):
                    shutil.rmtree(dst_f, ignore_errors=True)
                try:
                    shutil.copytree(src_f, dst_f)
                except Exception as e:
                    self.console_text.append(f"[WARNING] Could not copy backup folder {folder}: {e}")

        # Restore images list if present in backup/images or input_images/extracted_frames
        restored_imgs = self._find_available_session_images()
        if restored_imgs:
            self.image_list = restored_imgs
            self.img_count_label.setText(f"Images Loaded: {len(self.image_list)}")
            if hasattr(self, 'photos_tab'):
                self.photos_tab.set_images(self.image_list)
            self._detect_and_update_camera_info(self.image_list)

        # Restore UI controls from metadata
        if meta:
            if "quality_preset" in meta:
                preset_name = meta["quality_preset"].capitalize()
                idx = self.quality_combo.findText(preset_name)
                if idx >= 0:
                    self.quality_combo.setCurrentIndex(idx)
            if "gpu_mode" in meta:
                gpu_name = meta["gpu_mode"].capitalize()
                idx = self.gpu_combo.findText(gpu_name)
                if idx >= 0:
                    self.gpu_combo.setCurrentIndex(idx)
            if "has_plain_surfaces" in meta and meta["has_plain_surfaces"] and hasattr(self, 'mapper_combo'):
                self.mapper_combo.setCurrentIndex(1)
            if "auto_cleanup" in meta:
                val = bool(meta["auto_cleanup"])
                self.auto_cleanup_checkbox.setChecked(val)
                if hasattr(self, 'mc_enabled'):
                    self.mc_enabled.setChecked(val)
                self._on_mc_enabled_toggled(val)
            
            clean_p = meta.get("cleanup_params")
            if clean_p and isinstance(clean_p, dict):
                if "enable_reduction" in clean_p and hasattr(self, 'mc_enable_reduction_check'):
                    self.mc_enable_reduction_check.setChecked(bool(clean_p["enable_reduction"]))
                if "target_reduction_pct" in clean_p and hasattr(self, 'mc_reduction_spin'):
                    self.mc_reduction_spin.setValue(int(clean_p["target_reduction_pct"]))
                if "max_hole_size" in clean_p and hasattr(self, 'mc_max_hole_spin'):
                    self.mc_max_hole_spin.setValue(int(clean_p["max_hole_size"]))
                if "remove_duplicates" in clean_p and hasattr(self, 'mc_remove_dups_check'):
                    self.mc_remove_dups_check.setChecked(bool(clean_p["remove_duplicates"]))
                if "repair_nonmanifold" in clean_p and hasattr(self, 'mc_repair_nm_check'):
                    self.mc_repair_nm_check.setChecked(bool(clean_p["repair_nonmanifold"]))
                if "close_holes" in clean_p and hasattr(self, 'mc_close_holes_check'):
                    self.mc_close_holes_check.setChecked(bool(clean_p["close_holes"]))
            if "mapper_mode" in meta and hasattr(self, 'mapper_combo'):
                mapper_str = str(meta["mapper_mode"]).lower()
                mapper_idx = 1 if mapper_str == "global" else 0
                self.mapper_combo.setCurrentIndex(mapper_idx)
            if "mesh_mode" in meta and hasattr(self, 'mesh_mode_combo'):
                mesh_idx = self.mesh_mode_combo.findText(str(meta["mesh_mode"]).capitalize())
                if mesh_idx >= 0:
                    self.mesh_mode_combo.setCurrentIndex(mesh_idx)
            if "poisson_depth" in meta:
                if hasattr(self, 'poisson_depth_slider'):
                    self.poisson_depth_slider.setValue(int(meta["poisson_depth"]))
                elif hasattr(self, 'poisson_depth_spin'):
                    self.poisson_depth_spin.setValue(int(meta["poisson_depth"]))

            # Restore custom parameters UI controls if present
            cparams = meta.get("custom_params")
            if cparams and isinstance(cparams, dict):
                if hasattr(self, 'custom_settings_toggle'):
                    self.custom_settings_toggle.setChecked(True)
                    if hasattr(self, 'custom_settings_container'):
                        self.custom_settings_container.setVisible(True)

                if "colmap_matcher_type" in cparams and hasattr(self, 'custom_matcher_combo'):
                    reverse_matcher_map = {"auto": 0, "exhaustive": 1, "sequential": 2, "vocab_tree": 3, "spatial": 4}
                    m_idx = reverse_matcher_map.get(cparams["colmap_matcher_type"], 0)
                    self.custom_matcher_combo.setCurrentIndex(m_idx)

                if "vocab_tree_path" in cparams and hasattr(self, 'vocab_path_edit'):
                    self.vocab_path_edit.setText(str(cparams["vocab_tree_path"]))

                if "colmap_max_num_features" in cparams and hasattr(self, 'custom_features_spin'):
                    self.custom_features_spin.setValue(int(cparams["colmap_max_num_features"]))

                if "colmap_max_num_matches" in cparams and hasattr(self, 'custom_matches_spin'):
                    self.custom_matches_spin.setValue(int(cparams["colmap_max_num_matches"]))

                if "colmap_block_size" in cparams and hasattr(self, 'custom_block_size_spin'):
                    self.custom_block_size_spin.setValue(int(cparams["colmap_block_size"]))

                if "guided_matching" in cparams and hasattr(self, 'custom_guided_check'):
                    self.custom_guided_check.setChecked(str(cparams["guided_matching"]) == "1")

                if "run_bundle_adjuster" in cparams and hasattr(self, 'custom_ba_check'):
                    self.custom_ba_check.setChecked(bool(cparams["run_bundle_adjuster"]))

                if "densify_res" in cparams and hasattr(self, 'custom_densify_res_combo'):
                    self.custom_densify_res_combo.setCurrentIndex(int(cparams["densify_res"]))

                if "densify_views" in cparams and hasattr(self, 'custom_densify_views_spin'):
                    self.custom_densify_views_spin.setValue(int(cparams["densify_views"]))

                if "refine_scales" in cparams and hasattr(self, 'custom_refine_scales_spin'):
                    self.custom_refine_scales_spin.setValue(int(cparams["refine_scales"]))

                if "texture_res" in cparams and hasattr(self, 'custom_texture_res_combo'):
                    self.custom_texture_res_combo.setCurrentIndex(int(cparams["texture_res"]))

                if "cleanup_params" in cparams and isinstance(cparams["cleanup_params"], dict):
                    clean_p = cparams["cleanup_params"]
                    if "enable_reduction" in clean_p and hasattr(self, 'mc_enable_reduction_check'):
                        self.mc_enable_reduction_check.setChecked(bool(clean_p["enable_reduction"]))
                    if "target_reduction_pct" in clean_p and hasattr(self, 'mc_reduction_spin'):
                        self.mc_reduction_spin.setValue(int(clean_p["target_reduction_pct"]))
                    if "max_hole_size" in clean_p and hasattr(self, 'mc_max_hole_spin'):
                        self.mc_max_hole_spin.setValue(int(clean_p["max_hole_size"]))
                    if "remove_duplicates" in clean_p and hasattr(self, 'mc_remove_dups_check'):
                        self.mc_remove_dups_check.setChecked(bool(clean_p["remove_duplicates"]))
                    if "repair_nonmanifold" in clean_p and hasattr(self, 'mc_repair_nm_check'):
                        self.mc_repair_nm_check.setChecked(bool(clean_p["repair_nonmanifold"]))
                    if "close_holes" in clean_p and hasattr(self, 'mc_close_holes_check'):
                        self.mc_close_holes_check.setChecked(bool(clean_p["close_holes"]))

            if "cumulative_mesh_transform" in meta:
                try:
                    self._cumulative_mesh_transform = np.array(meta["cumulative_mesh_transform"], dtype=np.float32)
                except Exception:
                    self._cumulative_mesh_transform = np.eye(4, dtype=np.float32)
            else:
                self._cumulative_mesh_transform = np.eye(4, dtype=np.float32)

        # Enable view scene button and set mode
        mvs_dir = os.path.join(out_dir, "mvs")
        self.viewer_widget.set_mvs_directory(mvs_dir)
        self._update_upload_button_state()

        mesh_exists = False
        for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_texture.obj", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
            if os.path.exists(os.path.join(mvs_dir, candidate)):
                mesh_exists = True
                break

        dense_exists = os.path.exists(os.path.join(mvs_dir, "scene_dense.mvs"))

        if mesh_exists:
            self._set_view_mode(2)
        elif dense_exists:
            self._set_view_mode(1)
        else:
            self._set_view_mode(0)

        path = self.viewer_widget.get_selected_file_path()
        if path:
            self._reload_viewer(path)

        self._update_file_menu_states()
        self._check_and_enable_cleanup_btn()

        # Prompt user if they want to resume remaining reconstruction steps
        step = meta.get("last_completed_step", "unknown") if meta else "unknown"
        if step in ["images_imported", "features_extracted", "sparse_reconstruction", "dense_reconstruction"]:
            reply = QMessageBox.question(
                self,
                "Continue Reconstruction?",
                f"Session restored to checkpoint stage '{step}'.\nWould you like to resume and execute the remaining reconstruction steps?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._start_processing(resume_from_step=step)
            else:
                self._set_process_btn_state("ready" if len(self.image_list) > 0 else "idle")
        else:
            self._set_process_btn_state("ready" if len(self.image_list) > 0 else "idle")

    def _save_project(self):
        mvs_dir = self.viewer_widget.current_mvs_dir
        if not mvs_dir:
            mvs_dir = self._get_active_mvs_dir()
        if not mvs_dir or not os.path.exists(mvs_dir):
            self.console_text.append("[ERROR] No active 3D reconstruction session to save.")
            return
            
        default_file = os.path.join(self.last_accessed_dir or os.path.expanduser("~"), "project.pxm")
        save_dlg = SaveProjectDialog(default_file, self)
        if save_dlg.exec_() != QDialog.Accepted:
            return
            
        file_path, include_images = save_dlg.get_result()
        if not file_path:
            return
            
        if not file_path.lower().endswith(".pxm"):
            file_path += ".pxm"
            
        self.last_accessed_dir = os.path.dirname(file_path)
        
        # Locate colmap/images directory
        colmap_images_dir = os.path.join(os.path.dirname(mvs_dir), "colmap", "images")
        if not os.path.isdir(colmap_images_dir):
            rec_out = get_reconstruction_out_dir()
            candidate = os.path.join(rec_out, "colmap", "images")
            if os.path.isdir(candidate):
                colmap_images_dir = candidate
            else:
                colmap_images_dir = None
                
        status_msg = "Compressing assets and images into project file..." if include_images else "Compressing assets and saving project file..."
        self.console_text.append(f"[SAVE] Packing reconstruction assets (images={'ON' if include_images else 'OFF'}) from {mvs_dir} to {file_path}...")
        
        self.save_dialog = ProjectProgressDialog("Saving Project", status_msg, self)
        self.save_dialog.show()
        
        self.save_worker = SaveWorker(
            mvs_dir=mvs_dir,
            file_path=file_path,
            include_images=include_images,
            colmap_images_dir=colmap_images_dir,
            image_list=getattr(self, 'image_list', [])
        )
        self.save_worker.finished.connect(self._on_save_finished)
        self.save_worker.start()

    def _on_save_finished(self, success, message):
        if hasattr(self, 'save_dialog') and self.save_dialog:
            self.save_dialog.accept()
            self.save_dialog = None
            
        if success:
            self.console_text.append(f"[SAVE SUCCESS] {message}")
        else:
            self.console_text.append(f"[ERROR] Failed to save project: {message}")

    def _load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project File",
            self.last_accessed_dir,
            "Proximap Project (*.pxm)"
        )
        if not file_path:
            return
            
        self.last_accessed_dir = os.path.dirname(file_path)
        self.console_text.append(f"[LOAD] Unpacking project archive: {file_path}...")
        
        self.load_dialog = ProjectProgressDialog("Loading Project", "Extracting project assets...", self)
        self.load_dialog.show()
        
        import tempfile
        temp_root = tempfile.gettempdir()
        
        self.load_worker = LoadWorker(file_path, temp_root)
        self.load_worker.finished.connect(self._on_load_finished)
        self.load_worker.start()

    def _on_load_finished(self, success, mvs_dir, message):
        if hasattr(self, 'load_dialog') and self.load_dialog:
            self.load_dialog.accept()
            self.load_dialog = None
            
        if not success:
            self.console_text.append(f"[ERROR] Failed to load project: {message}")
            return
            
        self.console_text.append(f"[LOAD] {message} Cache directory: {mvs_dir}")
        
        # Update viewer state
        self.viewer_widget.set_mvs_directory(mvs_dir)
        self._update_upload_button_state()
        
        # Determine the best view mode and load it immediately
        mesh_exists = False
        for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_texture.obj", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
            if os.path.exists(os.path.join(mvs_dir, candidate)):
                mesh_exists = True
                break
        
        dense_exists = os.path.exists(os.path.join(mvs_dir, "scene_dense.mvs"))
        
        if mesh_exists:
            self._set_view_mode(2)
        elif dense_exists:
            self._set_view_mode(1)
        else:
            self._set_view_mode(0)
        
        path = self.viewer_widget.get_selected_file_path()
        if path:
            self._reload_viewer(path)
            
        # Check if project included unpacked images
        parent_dir = os.path.dirname(mvs_dir)
        possible_img_dirs = [
            os.path.join(parent_dir, "colmap", "images"),
            os.path.join(mvs_dir, "colmap", "images"),
            os.path.join(mvs_dir, "images")
        ]
        found_imgs = []
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        for d in possible_img_dirs:
            if os.path.isdir(d):
                imgs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(image_exts)]
                if imgs:
                    found_imgs = imgs
                    break
        if found_imgs:
            self.image_list = found_imgs
            self.img_count_label.setText(f"Images Loaded: {len(self.image_list)}")
            if hasattr(self, 'photos_tab'):
                self.photos_tab.set_images(self.image_list)
            self._detect_and_update_camera_info(self.image_list)
            self._set_process_btn_state("ready")
            self.console_text.append(f"[LOAD] Loaded {len(found_imgs)} reconstruction photos into project.")

        self._update_file_menu_states()

    def _new_project(self):
        """Resets the current project session: clears loaded photos, standalone cloud, viewer, and UI state."""
        has_content = bool(self.image_list or self.standalone_cloud_path or self.viewer_widget.current_mvs_dir)
        if has_content:
            reply = QMessageBox.question(
                self,
                "New Project",
                "Start a new project? This will clear all loaded photos, standalone point clouds, and current 3D viewer content.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # 1. Clear standalone point cloud state if active
        if self.standalone_cloud_path:
            self.standalone_cloud_path = None
            self.standalone_cloud_label.setText("")
            self.standalone_cloud_container.setVisible(False)

        # 2. Reset images and video lists & photos tab
        self.image_list = []
        self.extracted_frames = []
        if hasattr(self, 'photos_tab'):
            self.photos_tab.set_images([])

        # 3. Terminate & clear 3D viewer
        self.viewer_widget.current_mvs_dir = None
        self._terminate_viewer()

        # 4. Re-enable Step 1 buttons and restore photogrammetry settings panel
        self._exit_standalone_mode()

        # 5. Reset progress and status bar
        self._set_process_btn_state("idle")
        self.progress_bar.setValue(0)
        self.status_label.setText("Step 1/5: Waiting for images...")

        # 6. Update menu states and log
        self._update_file_menu_states()
        self.console_text.append("[PROJECT] Created a new project session. Workspace and viewer cleared.")

    def _update_file_menu_states(self):
        # Is reconstruction running?
        is_running = (self.worker is not None and self.worker.isRunning())
        
        if is_running:
            self.viewer_widget.action_new.setEnabled(False)
            self.viewer_widget.action_save.setEnabled(False)
            self.viewer_widget.action_load.setEnabled(False)
            self.viewer_widget.action_recover.setEnabled(False)
            self.viewer_widget.import_menu.setEnabled(False)
            return

        self.viewer_widget.action_new.setEnabled(True)
        self.viewer_widget.import_menu.setEnabled(True)
            
        # We can save if we have a valid MVS directory containing reconstruction files/models
        mvs_dir = self.viewer_widget.current_mvs_dir
        if not mvs_dir:
            mvs_dir = self._get_active_mvs_dir()
            
        has_assets = False
        if mvs_dir and os.path.exists(mvs_dir):
            for root, _, files in os.walk(mvs_dir):
                if any(f.endswith((".mvs", ".ply", ".obj", ".glb", ".gltf")) for f in files):
                    has_assets = True
                    break
        self.viewer_widget.action_save.setEnabled(has_assets)
        
        # We can load at any time when not running
        self.viewer_widget.action_load.setEnabled(True)
        
        # We can recover if there's an existing scene in the base reconstruction directory
        output_dir = get_reconstruction_out_dir()
        mvs_out = os.path.join(output_dir, "mvs")
        has_recoverable = os.path.exists(os.path.join(mvs_out, "scene.mvs")) or \
                          os.path.exists(os.path.join(mvs_out, "scene_dense_mesh_refine.ply")) or \
                          os.path.exists(os.path.join(mvs_out, "scene_dense_mesh.ply"))
        self.viewer_widget.action_recover.setEnabled(has_recoverable)

    def _toggle_viewer_mode(self):
        """Reloads the embedded 3D viewer."""
        path = self.viewer_widget.get_selected_file_path()
        if path:
            self._reload_viewer(path)

    def _on_shading_mode_changed(self, mode_name: str):
        """Switches between Solid and Wireframe shading modes in the 3D viewport."""
        self._current_shading_mode = mode_name
        self._apply_shading_mode_to_mesh()
        self.console_text.append(f"[VIEWPORT] Viewport shading set to {mode_name.capitalize()} mode.")

    def _apply_shading_mode_to_mesh(self):
        """Applies the active shading mode (Wireframe vs Solid) to the active VisPy MeshVisual.
        Wireframe is rendered as a Line-edge overlay — compatible with TextureFilter, no shader issues.
        """
        import numpy as np
        from vispy import scene

        # Remove any existing wireframe overlay first
        if hasattr(self, '_wireframe_visual') and self._wireframe_visual is not None:
            self._wireframe_visual.parent = None
            self._wireframe_visual = None

        if not (hasattr(self, 'mesh_visual') and self.mesh_visual is not None):
            return

        if getattr(self, '_current_shading_mode', 'solid') == "wireframe":
            # Build wireframe line segments from mesh faces
            vertices = getattr(self, '_last_wf_vertices', None)
            faces = getattr(self, '_last_wf_faces', None)

            if vertices is None or faces is None:
                self.canvas.update()
                return

            # Build unique edge pairs from triangles
            edges = np.concatenate([
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [0, 2]],
            ], axis=0)
            edges = np.sort(edges, axis=1)
            edges = np.unique(edges, axis=0)

            # Interleave: [v0, v1, v0, v1, ...] as line segments
            line_pts = np.empty((len(edges) * 2, 3), dtype=np.float32)
            line_pts[0::2] = vertices[edges[:, 0]]
            line_pts[1::2] = vertices[edges[:, 1]]

            self._wireframe_visual = scene.visuals.Line(
                pos=line_pts,
                color=(0.0, 0.902, 0.463, 0.85),  # #00E676 + slight alpha
                connect='segments',
                width=1.0,
                parent=self.view.scene
            )
            self.mesh_visual.visible = False
        else:
            # Solid: restore mesh, ensure no overlay
            if hasattr(self, 'mesh_visual') and self.mesh_visual is not None:
                self.mesh_visual.visible = True

        self.canvas.update()

    def _clear_visuals(self):
        if hasattr(self, 'markers_visual') and self.markers_visual is not None:
            self.markers_visual.parent = None
            self.markers_visual = None
        if hasattr(self, 'mesh_visual') and self.mesh_visual is not None:
            self.mesh_visual.parent = None
            self.mesh_visual = None
        if hasattr(self, 'cameras_visual') and self.cameras_visual is not None:
            self.cameras_visual.parent = None
            self.cameras_visual = None
        if hasattr(self, 'frustum_manager') and self.frustum_manager is not None:
            self.frustum_manager.clear()
        if hasattr(self, 'selection_markers_visual') and self.selection_markers_visual is not None:
            self.selection_markers_visual.parent = None
            self.selection_markers_visual = None
        if hasattr(self, '_wireframe_visual') and self._wireframe_visual is not None:
            self._wireframe_visual.parent = None
            self._wireframe_visual = None
        self._last_wf_vertices = None
        self._last_wf_faces = None
        self._wireframe_filter = None  # kept for compat
        self._selected_vertex_indices = None
        self._last_points = None
        # Hide camera preview card on scene clear
        if hasattr(self, 'camera_preview_widget') and self.camera_preview_widget is not None:
            self.camera_preview_widget.hide_card()
        if hasattr(self, '_look_through_badge') and self._look_through_badge is not None:
            self._look_through_badge.hide()
        if hasattr(self, 'editor_tool_host') and self.editor_tool_host is not None:
            self.editor_tool_host.refresh_data_state()
        self._init_fixed_ground_grid()

    def _init_fixed_ground_grid(self):
        """Creates or ensures a wide fixed ground plane grid with origin indicator at (0,0,0) in the 3D viewport."""
        import numpy as np
        from vispy import scene
        if not hasattr(self, 'view') or self.view is None or not hasattr(self.view, 'scene'):
            return

        # 1. Base Ground Grid (Excluding center axes)
        if not hasattr(self, 'grid_visual') or self.grid_visual is None or self.grid_visual.parent is None:
            half_side = 25.0
            num_divs = 50
            x_ticks = np.linspace(-half_side, half_side, num_divs + 1)
            z_ticks = np.linspace(-half_side, half_side, num_divs + 1)
            
            line_vertices = []
            for x in x_ticks:
                if abs(x) > 1e-4:  # Skip x=0 (highlighted by origin indicator)
                    line_vertices.append([x, 0.0, -half_side])
                    line_vertices.append([x, 0.0, half_side])
                
            for z in z_ticks:
                if abs(z) > 1e-4:  # Skip z=0 (highlighted by origin indicator)
                    line_vertices.append([-half_side, 0.0, z])
                    line_vertices.append([half_side, 0.0, z])
                
            pos = np.array(line_vertices, dtype=np.float32)
            color = (0.0, 0.45, 0.25, 0.35)
            
            self.grid_visual = scene.visuals.Line(
                pos=pos,
                color=color,
                connect='segments',
                method='gl',
                parent=self.view.scene
            )

        # 2. Origin Indicator (X: Red, Z: Blue, Y: Green)
        if not hasattr(self, 'origin_visual') or self.origin_visual is None or self.origin_visual.parent is None:
            half_side = 25.0
            axis_vertices = []
            axis_colors = []

            # Ground X-Axis (Red: #FF5252)
            axis_vertices.append([-half_side, 0.0, 0.0])
            axis_vertices.append([half_side, 0.0, 0.0])
            axis_colors.extend([[1.0, 0.32, 0.32, 0.85], [1.0, 0.32, 0.32, 0.85]])

            # Ground Z-Axis (Blue: #448AFF)
            axis_vertices.append([0.0, 0.0, -half_side])
            axis_vertices.append([0.0, 0.0, half_side])
            axis_colors.extend([[0.27, 0.55, 1.0, 0.85], [0.27, 0.55, 1.0, 0.85]])

            # Vertical Y-Axis Stem at Origin (Green: #00E676)
            axis_vertices.append([0.0, 0.0, 0.0])
            axis_vertices.append([0.0, 2.0, 0.0])
            axis_colors.extend([[0.0, 0.9, 0.46, 0.95], [0.0, 0.9, 0.46, 0.95]])

            self.origin_visual = scene.visuals.Line(
                pos=np.array(axis_vertices, dtype=np.float32),
                color=np.array(axis_colors, dtype=np.float32),
                connect='segments',
                width=2.0,
                method='gl',
                parent=self.view.scene
            )

    def _update_ground_grid(self, points=None):
        """Ensures the fixed ground plane is active. Does not resize or move with loaded geometry."""
        self._init_fixed_ground_grid()

    def _read_points3d_binary(self, path_to_model_file):
        import struct
        import numpy as np
        from point_cloud_io import apply_photogrammetry_coordinate_flip
        points = []
        colors = []
        if not os.path.exists(path_to_model_file):
            return None, None
        try:
            with open(path_to_model_file, "rb") as fid:
                num_points = struct.unpack("<Q", fid.read(8))[0]
                for _ in range(num_points):
                    binary_point_properties = struct.unpack("<QdddBBBd", fid.read(43))
                    x, y, z = binary_point_properties[1:4]
                    r, g, b = binary_point_properties[4:7]
                    track_len = struct.unpack("<Q", fid.read(8))[0]
                    fid.read(track_len * 8)
                    points.append((x, y, z))
                    colors.append((r, g, b))
        except Exception as e:
            self.console_text.append(f"[WARNING] Failed to parse points3D.bin: {e}")
        
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        pts_arr = np.array(points, dtype=np.float32)
        pts_arr, _, _, _ = apply_photogrammetry_coordinate_flip(points=pts_arr)
        return pts_arr, np.array(colors, dtype=np.uint8)

    def _read_images_binary(self, path_to_model_file):
        import struct
        import numpy as np
        from point_cloud_io import apply_photogrammetry_coordinate_flip
        images_data = []
        if not os.path.exists(path_to_model_file):
            return images_data

        # Build a fast lookup: basename -> absolute path from self.image_list
        image_name_to_path = {}
        for p in getattr(self, 'image_list', []):
            if isinstance(p, str):
                image_name_to_path[os.path.basename(p).lower()] = p
        # Also check the staging directory and reconstruction output
        for search_dir in [
            getattr(self, 'mvs_dir', None),
            getattr(self, 'output_dir', None),
            os.path.join(getattr(self, 'output_dir', ''), 'colmap', 'images'),
        ]:
            if search_dir and os.path.isdir(search_dir):
                for fname in os.listdir(search_dir):
                    key = fname.lower()
                    if key not in image_name_to_path:
                        image_name_to_path[key] = os.path.join(search_dir, fname)

        try:
            with open(path_to_model_file, "rb") as fid:
                num_reg_images = struct.unpack("<Q", fid.read(8))[0]
                for _ in range(num_reg_images):
                    binary_image_properties = struct.unpack("<IdddddddI", fid.read(64))
                    image_id = binary_image_properties[0]
                    qvec = np.array(binary_image_properties[1:5])
                    tvec = np.array(binary_image_properties[5:8])

                    # Read image name (null-terminated string)
                    image_name = b""
                    while True:
                        char = fid.read(1)
                        if char == b"\x00" or not char:
                            break
                        image_name += char
                    image_name = image_name.decode("utf-8", errors="ignore")

                    num_points2D = struct.unpack("<Q", fid.read(8))[0]
                    fid.read(num_points2D * 24)

                    qw, qx, qy, qz = qvec
                    R = np.array([
                        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                    ])

                    # Apply Similarity Transform: R' = F R F^T, T' = F T
                    _, _, R_prime, tvec_prime = apply_photogrammetry_coordinate_flip(camera_R=R, camera_T=tvec)
                    camera_center = -R_prime.T @ tvec_prime

                    # Resolve the source image file path
                    resolved_path = image_name_to_path.get(image_name.lower(), "")

                    images_data.append({
                        "image_id":   image_id,
                        "image_name": image_name,
                        "image_path": resolved_path,
                        "center":     camera_center,
                        "R":          R_prime,
                    })
        except Exception as e:
            self.console_text.append(f"[WARNING] Failed to parse images.bin: {e}")
        return images_data

    def _read_ply(self, path):
        import numpy as np
        import struct
        if not os.path.exists(path):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), None
            
        try:
            with open(path, 'rb') as f:
                header_lines = []
                while True:
                    line = f.readline().decode('utf-8', errors='ignore').strip()
                    header_lines.append(line)
                    if line == 'end_header':
                        break
                        
                # Parse header
                num_vertices = 0
                num_faces = 0
                format_type = None
                vertex_properties = []
                element_type = None
                
                for line in header_lines:
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == 'format':
                        format_type = parts[1]
                    elif parts[0] == 'element':
                        element_type = parts[1]
                        if element_type == 'vertex':
                            num_vertices = int(parts[2])
                        elif element_type == 'face':
                            num_faces = int(parts[2])
                    elif parts[0] == 'property':
                        if element_type == 'vertex':
                            if parts[1] == 'list':
                                # List property under vertex element (e.g. property list uint8 uint32 view_indices)
                                vertex_properties.append((parts[4], 'list', True, parts[2], parts[3]))
                            else:
                                vertex_properties.append((parts[2], parts[1], False, None, None))
                                
                type_map = {
                    'char': (np.int8, 1), 'uchar': (np.uint8, 1),
                    'short': (np.int16, 2), 'ushort': (np.uint16, 2),
                    'int': (np.int32, 4), 'uint': (np.uint32, 4),
                    'float': (np.float32, 4), 'double': (np.float64, 8),
                    'int8': (np.int8, 1), 'uint8': (np.uint8, 1),
                    'int16': (np.int16, 2), 'uint16': (np.uint16, 2),
                    'int32': (np.int32, 4), 'uint32': (np.uint32, 4),
                    'float32': (np.float32, 4), 'float64': (np.float64, 8)
                }
                
                has_list = any(p[2] for p in vertex_properties)
                
                if 'binary' in format_type:
                    if has_list:
                        # Extract fixed size properties before the first list property
                        fixed_properties = []
                        list_properties = []
                        for p in vertex_properties:
                            if p[2]:
                                list_properties.append(p)
                            else:
                                if not list_properties:
                                    fixed_properties.append(p)
                                    
                        # Build struct character mapping
                        fmt_chars = []
                        type_char_map = {
                            'char': 'b', 'uchar': 'B',
                            'short': 'h', 'ushort': 'H',
                            'int': 'i', 'uint': 'I',
                            'float': 'f', 'double': 'd',
                            'int8': 'b', 'uint8': 'B',
                            'int16': 'h', 'uint16': 'H',
                            'int32': 'i', 'uint32': 'I',
                            'float32': 'f', 'float64': 'd'
                        }
                        
                        fixed_size = 0
                        type_sizes = {
                            'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'f': 4, 'd': 8
                        }
                        
                        for name, t, _, _, _ in fixed_properties:
                            c = type_char_map[t]
                            fmt_chars.append(c)
                            fixed_size += type_sizes[c]
                            
                        fixed_format = '<' + ''.join(fmt_chars)
                        fixed_struct = struct.Struct(fixed_format)
                        
                        points = np.zeros((num_vertices, 3), dtype=np.float32)
                        colors = np.ones((num_vertices, 3), dtype=np.uint8) * 255
                        
                        names = [p[0] for p in fixed_properties]
                        x_idx = names.index('x') if 'x' in names else -1
                        y_idx = names.index('y') if 'y' in names else -1
                        z_idx = names.index('z') if 'z' in names else -1
                        
                        r_name = 'red' if 'red' in names else ('r' if 'r' in names else None)
                        g_name = 'green' if 'green' in names else ('g' if 'g' in names else None)
                        b_name = 'blue' if 'blue' in names else ('b' if 'b' in names else None)
                        
                        r_idx = names.index(r_name) if r_name else -1
                        g_idx = names.index(g_name) if g_name else -1
                        b_idx = names.index(b_name) if b_name else -1
                        
                        data = f.read()
                        offset = 0
                        
                        for i in range(num_vertices):
                            val = fixed_struct.unpack_from(data, offset)
                            if x_idx != -1: points[i, 0] = val[x_idx]
                            if y_idx != -1: points[i, 1] = val[y_idx]
                            if z_idx != -1: points[i, 2] = val[z_idx]
                            
                            if r_idx != -1: colors[i, 0] = val[r_idx]
                            if g_idx != -1: colors[i, 1] = val[g_idx]
                            if b_idx != -1: colors[i, 2] = val[b_idx]
                            
                            offset += fixed_size
                            
                            # Skip list properties dynamically
                            for name, _, _, count_type, item_type in list_properties:
                                c_char = type_char_map[count_type]
                                c_size = type_sizes[c_char]
                                count = struct.unpack_from('<' + c_char, data, offset)[0]
                                offset += c_size
                                
                                i_char = type_char_map[item_type]
                                i_size = type_sizes[i_char]
                                offset += count * i_size
                                
                        faces = None
                        if num_faces > 0:
                            try:
                                face_bytes = data[offset:]
                                if len(face_bytes) >= num_faces * 13:
                                    dt = np.dtype([('count', np.uint8), ('indices', np.int32, 3)])
                                    face_data = np.frombuffer(face_bytes[:num_faces * 13], dtype=dt)
                                    faces = face_data['indices'].copy()
                            except Exception as face_err:
                                self.console_text.append(f"[WARNING] Failed to parse PLY faces: {face_err}")
                                
                        return points, colors, faces
                    else:
                        vertex_dtype = []
                        for name, t, _, _, _ in vertex_properties:
                            dtype_t, _ = type_map[t]
                            vertex_dtype.append((name, dtype_t))
                            
                        vertex_struct_dtype = np.dtype(vertex_dtype)
                        vertex_data = np.frombuffer(f.read(num_vertices * vertex_struct_dtype.itemsize), dtype=vertex_struct_dtype)
                        
                        points = np.zeros((num_vertices, 3), dtype=np.float32)
                        points[:, 0] = vertex_data['x']
                        points[:, 1] = vertex_data['y']
                        points[:, 2] = vertex_data['z']
                        
                        colors = np.ones((num_vertices, 3), dtype=np.uint8) * 255
                        color_keys = [k for k in ['red', 'green', 'blue', 'r', 'g', 'b'] if k in vertex_data.dtype.names]
                        if len(color_keys) >= 3:
                            r_key = 'red' if 'red' in vertex_data.dtype.names else 'r'
                            g_key = 'green' if 'green' in vertex_data.dtype.names else 'g'
                            b_key = 'blue' if 'blue' in vertex_data.dtype.names else 'b'
                            colors[:, 0] = vertex_data[r_key]
                            colors[:, 1] = vertex_data[g_key]
                            colors[:, 2] = vertex_data[b_key]
                            
                        faces = None
                        if num_faces > 0:
                            try:
                                face_bytes = f.read()
                                if len(face_bytes) >= num_faces * 13:
                                    dt = np.dtype([('count', np.uint8), ('indices', np.int32, 3)])
                                    face_data = np.frombuffer(face_bytes[:num_faces * 13], dtype=dt)
                                    faces = face_data['indices'].copy()
                            except Exception as face_err:
                                self.console_text.append(f"[WARNING] Failed to parse PLY faces: {face_err}")
                                
                        return points, colors, faces
                else:
                    # ASCII format
                    lines = f.read().decode('utf-8', errors='ignore').splitlines()
                    points = []
                    colors = []
                    faces = []
                    
                    for i in range(num_vertices):
                        parts = lines[i].split()
                        if len(parts) >= 3:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            if len(parts) >= 6:
                                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
                            else:
                                colors.append([255, 255, 255])
                                 
                    start_face_idx = num_vertices
                    for i in range(num_faces):
                        if (start_face_idx + i) < len(lines):
                            parts = lines[start_face_idx + i].split()
                            if len(parts) >= 4 and int(parts[0]) == 3:
                                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                                 
                    points = np.array(points, dtype=np.float32)
                    colors = np.array(colors, dtype=np.uint8)
                    faces = np.array(faces, dtype=np.int32) if faces else None
                    if len(points) > 0:
                        from point_cloud_io import apply_photogrammetry_coordinate_flip
                        points, _, _, _ = apply_photogrammetry_coordinate_flip(points=points)
                    return points, colors, faces
        except Exception as e:
            self.console_text.append(f"[WARNING] Failed to parse PLY file: {e}")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), None

    def _read_obj(self, obj_path):
        import numpy as np
        temp_v = []
        temp_vt = []
        unpacked_map = {}
        unpacked_v = []
        unpacked_vt = []
        faces = []
        texture_filename = None
        
        # Parse companion MTL for texture filename
        mtl_path = obj_path.replace('.obj', '.mtl')
        if os.path.exists(mtl_path):
            try:
                with open(mtl_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('map_Kd'):
                            parts = line.strip().split(None, 1)
                            if len(parts) > 1:
                                texture_filename = parts[1].strip()
                                break
            except Exception as e:
                self.console_text.append(f"[WARNING] Failed to parse MTL file: {e}")
                
        # Parse OBJ file
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == 'v':
                        temp_v.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == 'vt':
                        temp_vt.append([float(parts[1]), float(parts[2])])
                    elif parts[0] == 'f':
                        face_indices = []
                        for part in parts[1:4]:
                            subparts = part.split('/')
                            v_idx = int(subparts[0]) - 1
                            vt_idx = int(subparts[1]) - 1 if len(subparts) > 1 and subparts[1] else -1
                            
                            key = (v_idx, vt_idx)
                            if key not in unpacked_map:
                                new_idx = len(unpacked_v)
                                unpacked_map[key] = new_idx
                                unpacked_v.append(temp_v[v_idx])
                                if vt_idx != -1 and vt_idx < len(temp_vt):
                                    unpacked_vt.append(temp_vt[vt_idx])
                                else:
                                    unpacked_vt.append([0.0, 0.0])
                            face_indices.append(unpacked_map[key])
                        faces.append(face_indices)
        except Exception as e:
            self.console_text.append(f"[WARNING] Failed to parse OBJ file: {e}")
            
        vertices = np.array(unpacked_v, dtype=np.float32)
        texcoords = np.array(unpacked_vt, dtype=np.float32)
        if len(texcoords) > 0:
            # Flip V coordinate for OpenGL/VisPy compatibility
            texcoords[:, 1] = 1.0 - texcoords[:, 1]
        faces = np.array(faces, dtype=np.int32)
        
        # Locate texture file
        texture_path = None
        if texture_filename:
            potential_paths = [
                os.path.join(os.path.dirname(obj_path), texture_filename),
                os.path.join(os.path.dirname(obj_path), os.path.basename(texture_filename))
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    texture_path = path
                    break
        else:
            dirname = os.path.dirname(obj_path)
            if os.path.exists(dirname):
                for filename in os.listdir(dirname):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and 'texture' in filename.lower():
                        texture_path = os.path.join(dirname, filename)
                        break
                         
        if len(vertices) > 0:
            from point_cloud_io import apply_photogrammetry_coordinate_flip
            vertices, _, _, _ = apply_photogrammetry_coordinate_flip(points=vertices)
        return vertices, texcoords, faces, texture_path

    def _draw_cameras(self, cameras_data):
        """Build camera frustum wireframes and start the thumbnail atlas worker.
        Delegates geometry ownership to CameraFrustumManager.
        """
        if not cameras_data:
            return

        # Initialise the manager (or reuse if already created)
        if self.frustum_manager is None:
            from camera_frustum_manager import CameraFrustumManager
            self.frustum_manager = CameraFrustumManager(self.view)
        else:
            self.frustum_manager.clear()

        # Apply stored camera visualization settings
        self.frustum_manager.set_scale(self._camera_frustum_scale)
        self.frustum_manager.set_opacity(self._camera_photo_opacity)
        self.frustum_manager.set_frustums_visible(self._camera_frustums_visible)
        self.frustum_manager.set_planes_visible(self._camera_photos_visible)

        # Build wireframe frustums immediately
        self.frustum_manager.load(cameras_data)
        # Keep backward-compat reference so _clear_visuals still works
        self.cameras_visual = self.frustum_manager._frustums_visual

        # Connect canvas mouse click for picking (once)
        if not getattr(self, '_camera_pick_connected', False):
            self.canvas.events.mouse_press.connect(self._on_viewer_mouse_clicked)
            self._camera_pick_connected = True

        # Start background thumbnail atlas builder
        from camera_frustum_manager import CameraThumbnailAtlasWorker
        if self._atlas_worker is not None and self._atlas_worker.isRunning():
            self._atlas_worker.quit()
            self._atlas_worker.wait()
        self._atlas_worker = CameraThumbnailAtlasWorker(cameras_data, tile_size=256, parent=self)
        self._atlas_worker.finished.connect(self._on_camera_atlas_ready)
        self._atlas_worker.error.connect(
            lambda e: self.console_text.append(f"[WARNING] Camera atlas builder failed: {e}")
        )
        self._atlas_worker.start()
        self.console_text.append(
            f"[INFO] Building camera image planes for {len(cameras_data)} cameras…"
        )

    # ------------------------------------------------------------------
    # Camera feature handlers
    # ------------------------------------------------------------------

    def _on_camera_atlas_ready(self, atlas, uv_list, cameras_data):
        """UI-thread slot: texture atlas is ready, hand it to the manager."""
        if self.frustum_manager is None:
            return
        self.frustum_manager.apply_atlas(atlas, uv_list, cameras_data)
        self.console_text.append(
            f"[INFO] Camera image planes ready ({len(cameras_data)} cameras)."
        )
        if self.canvas:
            self.canvas.update()

    def _on_camera_frustums_toggled(self, visible: bool):
        self._camera_frustums_visible = visible
        if self.frustum_manager is not None:
            self.frustum_manager.set_frustums_visible(visible)
            if self.canvas:
                self.canvas.update()

    def _on_camera_photos_toggled(self, visible: bool):
        self._camera_photos_visible = visible
        if self.frustum_manager is not None:
            self.frustum_manager.set_planes_visible(visible)
            if self.canvas:
                self.canvas.update()

    def _on_camera_scale_changed(self, scale: float):
        self._camera_frustum_scale = scale
        if self.frustum_manager is not None:
            self.frustum_manager.set_scale(scale)
            if self.canvas:
                self.canvas.update()

    def _on_camera_opacity_changed(self, opacity: float):
        self._camera_photo_opacity = opacity
        if self.frustum_manager is not None:
            self.frustum_manager.set_opacity(opacity)
            if self.canvas:
                self.canvas.update()

    def _on_viewer_mouse_clicked(self, event):
        """VisPy canvas mouse press event: try to pick a camera frustum."""
        if self.frustum_manager is None or self.frustum_manager.camera_count == 0:
            return
        if event.button != 1:  # left button only
            return

        # Pixel coordinates of the click
        x, y = event.pos
        w = self.canvas.size[0]
        h = self.canvas.size[1]

        idx = self.frustum_manager.pick_camera(x, y, w, h, radius_px=20.0)
        if idx < 0:
            return

        # Highlight the selected frustum
        self.frustum_manager.select_camera(idx)
        if self.canvas:
            self.canvas.update()

        # Open the PiP preview card
        cam = self.frustum_manager.get_camera(idx)
        if self.camera_preview_widget is not None:
            self.camera_preview_widget.show_camera(idx, cam)

        # Sync to the Photos tab
        self._sync_photos_tab_to_camera(cam)

    def _sync_photos_tab_to_camera(self, cam: dict):
        """Deselect all photos then select the one matching the clicked camera."""
        if not hasattr(self, 'photos_tab') or self.photos_tab is None:
            return
        image_path = cam.get("image_path", "")
        if not image_path:
            return
        # Deselect all, then select the matching item
        self.photos_tab.deselect_all()
        grid = self.photos_tab.grid_widget
        item = grid.item_widgets.get(image_path)
        if item is not None:
            item.set_checked(True)
            # Scroll the bottom tab into view
            if hasattr(self, 'bottom_tabs'):
                # Switch to Photos tab (index 0 typically)
                for i in range(self.bottom_tabs.count()):
                    if self.bottom_tabs.widget(i) is self.photos_tab:
                        self.bottom_tabs.setCurrentIndex(i)
                        break

    def _on_camera_preview_closed(self):
        """Deselect camera highlight when the PiP card is closed."""
        if self.frustum_manager is not None:
            self.frustum_manager.select_camera(-1)
        if self.canvas:
            self.canvas.update()

    def _on_look_through_camera(self, camera_index: int):
        """Smoothly animate the 3D viewport to look through the selected camera."""
        if self.frustum_manager is None:
            return
        cam = self.frustum_manager.get_camera(camera_index)
        if not cam:
            return

        import numpy as np
        # Save current orbit state for exit
        vc = self.view.camera
        self._look_through_saved = {
            "azimuth":   getattr(vc, 'azimuth',   45.0),
            "elevation": getattr(vc, 'elevation',  30.0),
            "center":    list(getattr(vc, 'center', [0.0, 0.0, 0.0])),
            "distance":  getattr(vc, 'distance',   5.0),
        }

        # Forward direction in world space: R^T . [0, 0, -1]
        R_cw = cam["R"].T
        forward = R_cw @ np.array([0.0, 0.0, -1.0])
        norm = np.linalg.norm(forward)
        if norm > 1e-6:
            forward = forward / norm
        else:
            forward = np.array([0.0, 0.0, -1.0])
        C = np.asarray(cam["center"], dtype=np.float64)

        # In VisPy TurntableCamera(up='+y'):
        # view_dir = [cos(el)*cos(az), -sin(el), -cos(el)*sin(az)]
        # Therefore:
        # el = arcsin(-forward[1])
        # az = atan2(-forward[2], forward[0])
        el = float(np.degrees(np.arcsin(np.clip(-forward[1], -1.0, 1.0))))
        az = float(np.degrees(np.arctan2(-forward[2], forward[0])))

        # Unwrap azimuth so animation smoothly takes the shortest path
        saved_az = self._look_through_saved["azimuth"]
        diff_az = (az - saved_az + 180.0) % 360.0 - 180.0
        target_az = saved_az + diff_az

        # Place target center ahead along forward ray so camera eye sits exactly at C
        look_dist = 2.0
        target_center = (C + forward * look_dist).tolist()

        self._look_through_target = {
            "azimuth":   target_az,
            "elevation": el,
            "center":    target_center,
            "distance":  look_dist,
        }
        self._look_through_steps = 20
        self._look_through_step  = 0

        # Show exit badge
        self._show_look_through_badge(cam.get("image_name", f"Camera {camera_index + 1}"))

        # Start animation timer
        if self._look_through_timer is None:
            self._look_through_timer = QTimer(self)
            self._look_through_timer.timeout.connect(self._on_look_through_tick)
        self._look_through_timer.start(16)  # ~60 fps

    def _on_look_through_tick(self):
        """Advance the look-through camera animation one frame."""
        if self._look_through_saved is None or self._look_through_target is None:
            if self._look_through_timer:
                self._look_through_timer.stop()
            return

        t = (self._look_through_step + 1) / self._look_through_steps
        # Smoothstep easing
        t = t * t * (3.0 - 2.0 * t)
        t = min(t, 1.0)

        saved  = self._look_through_saved
        target = self._look_through_target
        vc = self.view.camera

        def _lerp(a, b, t_):
            return a + (b - a) * t_

        try:
            vc.azimuth   = _lerp(saved["azimuth"],   target["azimuth"],   t)
            vc.elevation = _lerp(saved["elevation"],  target["elevation"], t)
            vc.distance  = _lerp(saved["distance"],   target["distance"],  t)
            sc = saved["center"]
            tc = target["center"]
            vc.center = [_lerp(sc[i], tc[i], t) for i in range(3)]
        except Exception:
            pass

        self.canvas.update()
        self._look_through_step += 1
        if self._look_through_step >= self._look_through_steps:
            self._look_through_timer.stop()

    def _show_look_through_badge(self, camera_name: str):
        """Show a floating badge at the top-centre of the viewport with an exit button."""
        if self._look_through_badge is not None:
            self._look_through_badge.deleteLater()

        badge = QWidget(self.viewer_widget.container_area)
        badge.setStyleSheet("""
            QWidget {
                background-color: rgba(22, 22, 22, 220);
                border: 1px solid #00E676;
                border-radius: 6px;
            }
        """)
        badge_layout = QHBoxLayout(badge)
        badge_layout.setContentsMargins(10, 6, 10, 6)
        badge_layout.setSpacing(10)

        lbl = QLabel(f"Camera View: {camera_name}")
        lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold; background: transparent; border: none;")
        badge_layout.addWidget(lbl)

        exit_btn = QPushButton("Exit Camera View")
        exit_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                padding: 3px 8px;
                background-color: #00E676;
                color: #121212;
                border: 1px solid #00E676;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00C853;
                border-color: #00C853;
            }
        """)
        exit_btn.clicked.connect(self._on_exit_look_through)
        badge_layout.addWidget(exit_btn)

        badge.adjustSize()
        # Centre horizontally at top
        container_w = self.viewer_widget.container_area.width()
        badge.move((container_w - badge.width()) // 2, 10)
        badge.show()
        badge.raise_()
        self._look_through_badge = badge

    def _on_exit_look_through(self):
        """Restore the orbit camera to the state before Look Through Camera."""
        if self._look_through_timer is not None:
            self._look_through_timer.stop()

        if self._look_through_saved is not None:
            saved = self._look_through_saved
            vc = self.view.camera
            try:
                vc.azimuth   = saved["azimuth"]
                vc.elevation = saved["elevation"]
                vc.distance  = saved["distance"]
                vc.center    = saved["center"]
            except Exception:
                pass
            self._look_through_saved  = None
            self._look_through_target = None

        if self._look_through_badge is not None:
            self._look_through_badge.hide()
            self._look_through_badge.deleteLater()
            self._look_through_badge = None

        if self.canvas:
            self.canvas.update()

    def _render_in_vispy_from_data(self, points, colors, faces, texcoords, texture_path, mode, reset_camera=True):
        """Upload pre-parsed geometry arrays to the VisPy scene (UI thread only).
        Called by _on_viewer_data_ready after ViewerLoadWorker finishes background parsing.
        """
        import numpy as np
        from PIL import Image
        self._clear_visuals()

        if points is None or len(points) == 0:
            self.canvas.native.hide()
            self.viewer_widget.fallback_label.setText("No valid 3D points or faces could be parsed.")
            self.viewer_widget.fallback_label.show()
            return

        # Track active geometry arrays
        self._current_points = points
        self._current_colors = colors
        self._current_faces = faces
        self._current_texcoords = texcoords
        self._current_texture_path = texture_path
        self._last_points = points

        # Store raw checkpoint for Reset Crop if not already recorded
        if self._raw_points is None:
            self._raw_points = np.copy(points) if points is not None else None
            self._raw_colors = np.copy(colors) if colors is not None else None
            self._raw_faces = np.copy(faces) if faces is not None else None
            self._raw_texcoords = np.copy(texcoords) if texcoords is not None else None
            self._raw_texture_path = texture_path

        if mode == 2 and faces is not None and len(faces) > 0:
            mesh_colors = None
            if colors is not None and len(colors) > 0:
                mesh_colors = colors.astype(np.float32) / 255.0
            self.mesh_visual = scene.visuals.Mesh(
                vertices=points, faces=faces,
                vertex_colors=mesh_colors, color='white',
                parent=self.view.scene
            )
            if texture_path and texcoords is not None and len(texcoords) > 0:
                try:
                    texture_image = np.array(Image.open(texture_path))
                    from vispy.visuals.filters import TextureFilter
                    self.mesh_visual.attach(TextureFilter(texture_image, texcoords))
                except Exception as tex_err:
                    self.console_text.append(f"[WARNING] Could not apply texture filter: {tex_err}")
            # Store geometry so wireframe overlay can build edges
            self._last_wf_vertices = points.astype(np.float32)
            self._last_wf_faces = faces.astype(np.uint32)
            self._apply_shading_mode_to_mesh()
        else:
            marker_colors = 'white'
            if colors is not None and len(colors) > 0:
                mc = colors.astype(np.float32) / 255.0
                if mc.shape[1] == 3:
                    mc = np.hstack([mc, np.ones((mc.shape[0], 1), dtype=np.float32)])
                marker_colors = mc
            self.markers_visual = scene.visuals.Markers(parent=self.view.scene)
            point_size = 4 if mode == 0 else 2
            self.markers_visual.set_data(pos=points, face_color=marker_colors, size=point_size, edge_width=0)

        self.canvas.native.show()
        self.viewer_widget.fallback_label.hide()

        if reset_camera:
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            center = (bbox_min + bbox_max) / 2.0
            scale  = np.max(bbox_max - bbox_min)
            self.view.camera.center   = center
            self.view.camera.distance = max(0.1, scale * 1.5)
            self.view.camera.elevation = 30
            self.view.camera.azimuth   = 45
            self.view.camera.up        = '+y'

        # Update auto-scaling ground plane grid
        self._update_ground_grid(points)

        # Update bounding box diagonal for mesh cleanup tools (e.g., Merge Vertices)
        if points is not None and len(points) > 0:
            b_min = np.min(points, axis=0)
            b_max = np.max(points, axis=0)
            b_diag = float(np.linalg.norm(b_max - b_min))
            if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                self.mesh_cleanup_tool_window.set_bbox_diagonal(b_diag)

        self.canvas.update()
        if hasattr(self, 'editor_tool_host') and self.editor_tool_host is not None:
            self.editor_tool_host.refresh_data_state()
        self._sync_active_transform_to_viewport()

    def _on_selection_mode_changed(self, mode_name: str):
        """Triggered when user selects a cut / selection mode from the Cut & Select tool in the sidebar."""
        from crop_box import CropBoxOverlay

        self.viewer_widget._current_selection_mode = mode_name
        self._current_selection_mode = mode_name

        if mode_name == 'crop_box':
            if hasattr(self, 'selection_overlay') and self.selection_overlay is not None:
                self.selection_overlay.set_mode('none')
            self._clear_selection()

            if self.crop_box is None:
                self.crop_box = CropBoxOverlay(parent_scene=self.view.scene)
            else:
                self.crop_box.set_parent(self.view.scene)

            ref_pts = self._current_points if self._current_points is not None else self._last_points
            self.crop_box.fit_to_points(ref_pts)
            self.crop_box.set_visible(True)
            self.canvas.update()
            self.console_text.append("[CROP] Activated 3D Bounding Box Crop Overlay (RealityScan style).")
        elif mode_name in ['box', 'lasso']:
            if self.crop_box is not None:
                self.crop_box.is_dragging = False
                self.crop_box.set_visible(False)
                self.view.camera.interactive = True
                self.canvas.update()

            if hasattr(self, 'selection_overlay') and self.selection_overlay is not None:
                self.selection_overlay.set_mode(mode_name)
                self.selection_overlay.setGeometry(0, 0, self.viewer_widget.container_area.width(), self.viewer_widget.container_area.height())
                self.selection_overlay.raise_()
                if hasattr(self, 'floating_toolbox') and self.floating_toolbox is not None:
                    self.floating_toolbox.raise_()
                if hasattr(self, 'editor_tool_host') and self.editor_tool_host is not None:
                    self.editor_tool_host.raise_()
                if hasattr(self, 'overlay_label'):
                    self.overlay_label.raise_()

            tool_name = "Box Select" if mode_name == 'box' else "Lasso Select"
            self.console_text.append(f"[SELECT] Activated {tool_name} tool. Drag across viewport to select vertices.")
        else:
            if self.crop_box is not None:
                self.crop_box.is_dragging = False
                self.crop_box.set_visible(False)
                self.view.camera.interactive = True
                self.canvas.update()

            if hasattr(self, 'selection_overlay') and self.selection_overlay is not None:
                self.selection_overlay.set_mode('none')
            self._clear_selection()

    def _on_selection_shape_changed(self, shape_data):
        if getattr(self.viewer_widget, '_current_selection_mode', None) not in ['box', 'lasso']:
            return
        if self._current_points is None or len(self._current_points) == 0:
            return

        import numpy as np
        import matplotlib.path as mpath

        shape_type, data = shape_data
        points_3d = self._current_points

        visual = self.mesh_visual if (hasattr(self, 'mesh_visual') and self.mesh_visual is not None) else getattr(self, 'markers_visual', None)
        if visual is None:
            return

        try:
            # Map 3D visual coordinates to 2D canvas pixels via visual transform chain
            tr = visual.transforms.get_transform('visual', 'canvas')
            mapped = tr.map(points_3d)
            w = mapped[:, 3:4]
            valid_w = (w[:, 0] > 1e-4)

            screen_pts = np.zeros((len(points_3d), 2), dtype=np.float32)
            screen_pts[valid_w] = mapped[valid_w, :2] / w[valid_w]

            if shape_type == 'box':
                x0, y0, x1, y1 = data
                min_x, max_x = min(x0, x1), max(x0, x1)
                min_y, max_y = min(y0, y1), max(y0, y1)
                inside = valid_w & (screen_pts[:, 0] >= min_x) & (screen_pts[:, 0] <= max_x) & \
                                   (screen_pts[:, 1] >= min_y) & (screen_pts[:, 1] <= max_y)
                selected_indices = np.where(inside)[0]
            elif shape_type == 'lasso':
                poly_pts = np.array(data, dtype=np.float32)
                if len(poly_pts) < 3:
                    selected_indices = np.empty(0, dtype=np.int32)
                else:
                    path = mpath.Path(poly_pts)
                    inside = valid_w & path.contains_points(screen_pts)
                    selected_indices = np.where(inside)[0]
            else:
                selected_indices = np.empty(0, dtype=np.int32)

            self._selected_vertex_indices = selected_indices
            self._update_selection_highlight()

            n_sel = len(selected_indices)
            self.console_text.append(f"[SELECT] Selected {n_sel:,} vertices on mesh.")
        except Exception as e:
            self.console_text.append(f"[WARNING] Selection projection failed: {e}")

    def _update_selection_highlight(self):
        import numpy as np
        if self._selected_vertex_indices is None or len(self._selected_vertex_indices) == 0 or self._current_points is None:
            if hasattr(self, 'selection_markers_visual') and self.selection_markers_visual is not None:
                self.selection_markers_visual.set_data(pos=np.empty((0, 3), dtype=np.float32))
                self.canvas.update()
            return

        sel_pts = self._current_points[self._selected_vertex_indices]
        if not hasattr(self, 'selection_markers_visual') or self.selection_markers_visual is None:
            from vispy import scene
            self.selection_markers_visual = scene.visuals.Markers(parent=self.view.scene)
            self.selection_markers_visual.set_gl_state('translucent', depth_test=False)

        # Vivid glowing orange-red highlight markers
        n_sel = len(sel_pts)
        colors = np.zeros((n_sel, 4), dtype=np.float32)
        colors[:, 0] = 1.0   # R
        colors[:, 1] = 0.25  # G
        colors[:, 2] = 0.15  # B
        colors[:, 3] = 0.95  # A

        self.selection_markers_visual.set_data(
            pos=sel_pts,
            face_color=colors,
            edge_color=[1.0, 0.6, 0.2, 1.0],
            edge_width=1,
            size=6
        )
        self.selection_markers_visual.parent = self.view.scene
        self.canvas.update()

    def _clear_selection(self):
        import numpy as np
        self._selected_vertex_indices = np.empty(0, dtype=np.int32)
        self._update_selection_highlight()
        if hasattr(self, 'selection_overlay') and self.selection_overlay is not None:
            self.selection_overlay.clear()

    def _invert_selection(self):
        if self._current_points is None or len(self._current_points) == 0:
            return
        import numpy as np
        total_n = len(self._current_points)
        curr_set = set(self._selected_vertex_indices.tolist()) if self._selected_vertex_indices is not None else set()
        all_set = set(range(total_n))
        inverted_set = all_set - curr_set
        self._selected_vertex_indices = np.array(sorted(list(inverted_set)), dtype=np.int32)
        self._update_selection_highlight()

    def _delete_selection(self):
        if self._selected_vertex_indices is None or len(self._selected_vertex_indices) == 0:
            self.console_text.append("[WARNING] No vertices currently selected to delete.")
            return

        if self._current_points is None or len(self._current_points) == 0:
            return

        import numpy as np

        points = self._current_points
        faces = self._current_faces
        colors = self._current_colors
        texcoords = self._current_texcoords
        texture_path = self._current_texture_path

        n_pts = len(points)
        delete_mask = np.zeros(n_pts, dtype=bool)
        delete_mask[self._selected_vertex_indices] = True
        keep_vertex_mask = ~delete_mask

        n_del = len(self._selected_vertex_indices)
        self.console_text.append(f"[DELETE] Deleting {n_del:,} selected vertices from mesh...")

        if faces is not None and len(faces) > 0:
            f_v0 = keep_vertex_mask[faces[:, 0]]
            f_v1 = keep_vertex_mask[faces[:, 1]]
            f_v2 = keep_vertex_mask[faces[:, 2]]
            keep_face_mask = f_v0 & f_v1 & f_v2

            sub_faces = faces[keep_face_mask]
            if len(sub_faces) == 0:
                self.console_text.append("[WARNING] Deletion resulted in empty mesh geometry. Operation cancelled.")
                return

            unique_v_idx, new_faces = np.unique(sub_faces, return_inverse=True)
            new_faces = new_faces.reshape(sub_faces.shape).astype(np.int32)

            new_points = points[unique_v_idx]
            new_colors = colors[unique_v_idx] if colors is not None and len(colors) == len(points) else colors
            new_texcoords = texcoords[unique_v_idx] if texcoords is not None and len(texcoords) == len(points) else texcoords
        else:
            new_points = points[keep_vertex_mask]
            new_colors = colors[keep_vertex_mask] if colors is not None and len(colors) == len(points) else colors
            new_faces = None
            new_texcoords = None

        if len(new_points) == 0:
            self.console_text.append("[WARNING] Deletion resulted in empty geometry. Operation cancelled.")
            return

        # Update in-memory state
        self._current_points = new_points
        self._current_colors = new_colors
        self._current_faces = new_faces
        self._current_texcoords = new_texcoords

        # Update raw checkpoints
        self._raw_points = np.copy(new_points) if new_points is not None else None
        self._raw_colors = np.copy(new_colors) if new_colors is not None else None
        self._raw_faces = np.copy(new_faces) if new_faces is not None else None
        self._raw_texcoords = np.copy(new_texcoords) if new_texcoords is not None else None
        self._raw_texture_path = texture_path

        # Clear selection highlights
        self._clear_selection()

        # Re-render in VisPy without resetting camera
        mode = self.viewer_widget.mode_select.currentIndex()
        self._render_in_vispy_from_data(new_points, new_colors, new_faces, new_texcoords, texture_path, mode, reset_camera=False)

        # Destructively save updated mesh to disk
        self._save_active_mesh_to_disk()

    def _save_active_mesh_to_disk(self):
        """Helper to save current in-memory points, faces, and UVs destructively to disk (.obj / .ply)."""
        import os
        import numpy as np
        if self._current_points is None or len(self._current_points) == 0:
            return

        mesh_path = self.viewer_widget.get_selected_file_path()
        if not mesh_path or not os.path.exists(mesh_path):
            mvs_dir = self.viewer_widget.current_mvs_dir if self.viewer_widget.current_mvs_dir else os.path.join(get_reconstruction_out_dir(), "mvs")
            mesh_path = os.path.join(mvs_dir, "scene_dense_mesh_texture.obj")

        try:
            from point_cloud_io import apply_photogrammetry_coordinate_flip
            save_pts, _, _, _ = apply_photogrammetry_coordinate_flip(points=self._current_points)
            save_uvs = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
            if save_uvs is not None and len(save_uvs) > 0:
                save_uvs[:, 1] = 1.0 - save_uvs[:, 1]

            # Save OBJ
            if mesh_path.lower().endswith('.obj') or not mesh_path.lower().endswith('.ply'):
                obj_target = mesh_path if mesh_path.lower().endswith('.obj') else os.path.splitext(mesh_path)[0] + ".obj"
                base_name = os.path.splitext(os.path.basename(obj_target))[0]
                mtl_filename = f"{base_name}.mtl"
                mtl_path = os.path.join(os.path.dirname(obj_target), mtl_filename)

                has_uv = save_uvs is not None and len(save_uvs) == len(save_pts)
                tex_path = self._current_texture_path

                if tex_path and os.path.exists(tex_path):
                    tex_filename = os.path.basename(tex_path)
                    with open(mtl_path, 'w') as fm:
                        fm.write("# Material file generated by Proximap\n")
                        fm.write("newmtl material_0\n")
                        fm.write("Ka 1.000 1.000 1.000\n")
                        fm.write("Kd 1.000 1.000 1.000\n")
                        fm.write("Ks 0.000 0.000 0.000\n")
                        fm.write("d 1.0\n")
                        fm.write("illum 1\n")
                        fm.write(f"map_Kd {tex_filename}\n")

                with open(obj_target, 'w') as f:
                    f.write("# Wavefront OBJ file generated by Proximap (Destructive Edit)\n")
                    if tex_path and os.path.exists(tex_path):
                        f.write(f"mtllib {mtl_filename}\n")
                        f.write("usemtl material_0\n")

                    for v in save_pts:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                    if has_uv:
                        for vt in save_uvs:
                            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")

                    if self._current_faces is not None and len(self._current_faces) > 0:
                        for face in self._current_faces:
                            i0, i1, i2 = face[0] + 1, face[1] + 1, face[2] + 1
                            if has_uv:
                                f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
                            else:
                                f.write(f"f {i0} {i1} {i2}\n")

            # Save PLY if target is PLY
            if mesh_path.lower().endswith('.ply'):
                with open(mesh_path, 'w') as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(save_pts)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    if self._current_colors is not None and len(self._current_colors) == len(save_pts):
                        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write(f"element face {len(self._current_faces) if self._current_faces is not None else 0}\n")
                    f.write("property list uchar int vertex_indices\nend_header\n")
                    for idx, v in enumerate(save_pts):
                        if self._current_colors is not None and len(self._current_colors) == len(save_pts):
                            c = self._current_colors[idx]
                            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
                        else:
                            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    if self._current_faces is not None:
                        for face in self._current_faces:
                            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

            rem_v = len(save_pts)
            rem_f = len(self._current_faces) if self._current_faces is not None else 0
            self.console_text.append(f"[SAVE] Geometry saved destructively to disk ({os.path.basename(mesh_path)} - Vertices: {rem_v:,}, Faces: {rem_f:,}).")
        except Exception as err:
            self.console_text.append(f"[ERROR] Failed to save geometry to disk: {err}")

    def _on_gizmo_toggled(self, checked: bool):
        """Enables/disables the 3D combined transform gizmo in the viewport."""
        if not hasattr(self, 'transform_gizmo') or self.transform_gizmo is None:
            return
        if checked:
            import numpy as np
            if self._current_points is not None and len(self._current_points) > 0:
                pivot = np.mean(self._current_points, axis=0)
            else:
                pivot = np.zeros(3, dtype=np.float32)
            self.transform_gizmo.set_pivot(pivot)
            if hasattr(self, 'transform_tool_window') and self.transform_tool_window is not None:
                if self.transform_tool_window.position_row and self.transform_tool_window.rotation_row:
                    tx, ty, tz = self.transform_tool_window.position_row.get_values()
                    rx, ry, rz = self.transform_tool_window.rotation_row.get_values()
                    self.transform_gizmo.set_transform(position=(tx, ty, tz), rotation=(rx, ry, rz))
            self.transform_gizmo.set_visible(True)
        else:
            self.transform_gizmo.set_visible(False)
        self.canvas.update()

    def _on_canvas_mouse_press(self, event):
        """Detect handle clicks on the Combined Transform Gizmo or 3D Crop Box."""
        # 1. Test 3D Combined Transform Gizmo handles (Translation & Rotation)
        if event.button == 1 and hasattr(self, 'transform_gizmo') and self.transform_gizmo is not None and self.transform_gizmo.visible:
            try:
                if self.transform_gizmo.trans_lines is not None:
                    tr = self.transform_gizmo.trans_lines.transforms.get_transform('visual', 'canvas')
                    hit = self.transform_gizmo.hit_test(event.pos, tr)
                    if hit:
                        self.transform_gizmo.start_drag(hit, event.pos, tr)
                        self.view.camera.interactive = False
                        self.canvas.update()
                        return
            except Exception:
                pass

        # 2. Test 3D Crop Box handles
        if event.button != 1 or self.crop_box is None or not self.crop_box.visible:
            return
        if self.crop_box.handle_markers is None:
            return

        try:
            import numpy as np
            handles_3d = self.crop_box.get_handle_positions()
            tr = self.crop_box.handle_markers.transforms.get_transform('visual', 'canvas')
            mapped = tr.map(handles_3d)
            handles_2d = mapped[:, :2] / mapped[:, 3:4]

            mouse_xy = np.array(event.pos, dtype=np.float32)
            dists = np.linalg.norm(handles_2d - mouse_xy, axis=1)

            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= 25.0:  # 25 pixel handle pick tolerance
                self.crop_box.active_handle_idx = min_idx
                self.crop_box.is_dragging = True
                self.crop_box.drag_start_pos = mouse_xy
                # Lock camera rotation during handle drag!
                self.view.camera.interactive = False
        except Exception:
            pass

    def _on_canvas_mouse_move(self, event):
        """Update 3D transform gizmo or crop box bounds during mouse drag."""
        # 1. Combined Transform Gizmo drag & hover
        if hasattr(self, 'transform_gizmo') and self.transform_gizmo is not None and self.transform_gizmo.visible:
            try:
                if self.transform_gizmo.trans_lines is not None:
                    tr = self.transform_gizmo.trans_lines.transforms.get_transform('visual', 'canvas')
                    if self.transform_gizmo.is_dragging:
                        new_pos, new_rot = self.transform_gizmo.update_drag(event.pos, tr, camera=self.view.camera)
                        if new_pos is not None and hasattr(self, 'transform_tool_window') and self.transform_tool_window is not None:
                            self.transform_tool_window.set_transform_values(pos=new_pos, rot=new_rot)
                        self.canvas.update()
                        return
                    else:
                        hovered = self.transform_gizmo.hit_test(event.pos, tr)
                        self.transform_gizmo.set_hover(hovered)
            except Exception:
                pass

        # 2. Crop Box drag
        if self.crop_box is None or not self.crop_box.is_dragging or self.crop_box.active_handle_idx is None:
            return

        try:
            import numpy as np
            mouse_xy = np.array(event.pos, dtype=np.float32)
            last_xy = self.crop_box.drag_start_pos
            if last_xy is None:
                return

            dx_scr = mouse_xy[0] - last_xy[0]
            dy_scr = mouse_xy[1] - last_xy[1]

            if abs(dx_scr) < 1e-4 and abs(dy_scr) < 1e-4:
                return

            idx = self.crop_box.active_handle_idx
            axis_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
            axis = axis_map[idx]

            handles_3d = self.crop_box.get_handle_positions()
            hp_3d = handles_3d[idx]

            unit_v = np.zeros(3, dtype=np.float32)
            unit_v[axis] = 1.0
            hp_step_3d = hp_3d + unit_v

            tr = self.crop_box.handle_markers.transforms.get_transform('visual', 'canvas')
            m0 = tr.map(hp_3d.reshape(1, 3))
            m1 = tr.map(hp_step_3d.reshape(1, 3))

            p0 = (m0[:, :2] / m0[:, 3:4])[0]
            p1 = (m1[:, :2] / m1[:, 3:4])[0]

            scr_dir = p1 - p0
            len_sq = np.dot(scr_dir, scr_dir)
            if len_sq > 1e-6:
                delta_3d = (dx_scr * scr_dir[0] + dy_scr * scr_dir[1]) / len_sq

                b_min, b_max = self.crop_box.get_bounds()
                if idx == 0:    # +X
                    b_max[0] = max(b_min[0] + 0.01, b_max[0] + delta_3d)
                elif idx == 1:  # -X
                    b_min[0] = min(b_max[0] - 0.01, b_min[0] + delta_3d)
                elif idx == 2:  # +Y
                    b_max[1] = max(b_min[1] + 0.01, b_max[1] + delta_3d)
                elif idx == 3:  # -Y
                    b_min[1] = min(b_max[1] - 0.01, b_min[1] + delta_3d)
                elif idx == 4:  # +Z
                    b_max[2] = max(b_min[2] + 0.01, b_max[2] + delta_3d)
                elif idx == 5:  # -Z
                    b_min[2] = min(b_max[2] - 0.01, b_min[2] + delta_3d)

                self.crop_box.set_bounds(b_min, b_max)
                self.canvas.update()
                self.crop_box.drag_start_pos = mouse_xy
        except Exception:
            pass
        self._on_vispy_camera_transform_changed()

    def _on_canvas_mouse_release(self, event):
        """Release handle drag and unlock camera rotation."""
        if hasattr(self, 'transform_gizmo') and self.transform_gizmo is not None and self.transform_gizmo.is_dragging:
            self.transform_gizmo.end_drag()
            self.view.camera.interactive = True

        if self.crop_box is not None and self.crop_box.is_dragging:
            self.crop_box.is_dragging = False
            self.crop_box.active_handle_idx = None
            self.crop_box.drag_start_pos = None
            # Unlock camera rotation on mouse release
            self.view.camera.interactive = True
        self._on_vispy_camera_transform_changed()

    def _apply_crop(self):
        """RealityScan-style crop operation: trims mesh vertices/faces outside the 3D crop box."""
        if self.crop_box is None or not self.crop_box.visible:
            self.console_text.append("[WARNING] Bounding box crop overlay is not currently active.")
            return

        points = self._current_points if self._current_points is not None else self._last_points
        faces = self._current_faces
        colors = self._current_colors
        texcoords = self._current_texcoords
        texture_path = self._current_texture_path

        if points is None or len(points) == 0:
            self.console_text.append("[WARNING] No 3D geometry available in scene to crop.")
            return

        mode = self.viewer_widget.mode_select.currentIndex()
        self.console_text.append("[CROP] Executing 'Remove Outside' on 3D geometry...")

        new_pts, new_cls, new_fcs, new_tcs = self.crop_box.crop_points_and_mesh(
            points, colors, faces, texcoords, keep_inside=True
        )

        if len(new_pts) == 0:
            self.console_text.append("[WARNING] Crop operation resulted in empty geometry. Operation cancelled.")
            return

        self._current_points = new_pts
        self._current_colors = new_cls
        self._current_faces = new_fcs
        self._current_texcoords = new_tcs

        # Re-render updated geometry in VisPy scene without resetting camera
        self._render_in_vispy_from_data(new_pts, new_cls, new_fcs, new_tcs, texture_path, mode, reset_camera=False)

        # Re-fit crop box to remaining geometry
        if self.crop_box is not None:
            self.crop_box.fit_to_points(new_pts)

        rem_v = len(new_pts)
        rem_f = len(new_fcs) if new_fcs is not None else 0
        self.console_text.append(f"[CROP] Crop applied. Geometry updated (Vertices: {rem_v:,}, Faces: {rem_f:,}). Click 'Finalize Crop' to save to disk.")

    def _reset_crop(self):
        """Restores original un-cropped mesh geometry from checkpoint."""
        if self._raw_points is None:
            self.console_text.append("[WARNING] No original un-cropped checkpoint found to restore.")
            return

        import numpy as np
        self._current_points = np.copy(self._raw_points) if self._raw_points is not None else None
        self._current_colors = np.copy(self._raw_colors) if self._raw_colors is not None else None
        self._current_faces = np.copy(self._raw_faces) if self._raw_faces is not None else None
        self._current_texcoords = np.copy(self._raw_texcoords) if self._raw_texcoords is not None else None
        self._current_texture_path = self._raw_texture_path

        mode = self.viewer_widget.mode_select.currentIndex()
        self._render_in_vispy_from_data(
            self._current_points, self._current_colors, self._current_faces,
            self._current_texcoords, self._current_texture_path, mode, reset_camera=False
        )

        if self.crop_box is not None and self._current_points is not None:
            self.crop_box.fit_to_points(self._current_points)

        self.console_text.append("[CROP] Reset crop completed. Restored original mesh geometry.")

    def _finalize_crop(self):
        """Destructively and permanently crops the mesh and overwrites the mesh on disk."""
        import numpy as np
        if self._current_points is None or len(self._current_points) == 0:
            self.console_text.append("[WARNING] No 3D geometry available to finalize crop.")
            return

        # If bounding box is visible, execute remove outside to ensure latest box bounds are applied
        if self.crop_box is not None and self.crop_box.visible:
            self._apply_crop()

        # Update raw checkpoint to point to cropped geometry permanently
        self._raw_points = np.copy(self._current_points) if self._current_points is not None else None
        self._raw_colors = np.copy(self._current_colors) if self._current_colors is not None else None
        self._raw_faces = np.copy(self._current_faces) if self._current_faces is not None else None
        self._raw_texcoords = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
        self._raw_texture_path = self._current_texture_path

        # Save to disk
        self._save_active_mesh_to_disk()

        # Exit bounding box crop mode and revert to default navigation
        self.viewer_widget.set_selection_mode('none')

    def _export_temp_mesh_ply(self, points, faces, colors, file_path):
        """Write current geometry to native-oriented PLY on disk for worker processing."""
        import numpy as np, os
        from point_cloud_io import apply_photogrammetry_coordinate_flip
        save_pts, _, _, _ = apply_photogrammetry_coordinate_flip(points=points)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Filter valid non-degenerate faces (each vertex distinct and within bounds)
        valid_faces = []
        if faces is not None:
            n_pts = len(save_pts)
            for face in faces:
                if len(face) == 3:
                    v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
                    if v0 != v1 and v1 != v2 and v0 != v2:
                        if 0 <= v0 < n_pts and 0 <= v1 < n_pts and 0 <= v2 < n_pts:
                            valid_faces.append((v0, v1, v2))
                            
        with open(file_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(save_pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            has_color = colors is not None and len(colors) == len(save_pts)
            if has_color:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(f"element face {len(valid_faces)}\n")
            f.write("property list uchar int vertex_indices\nend_header\n")
            for idx, v in enumerate(save_pts):
                if has_color:
                    c = colors[idx]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
                else:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in valid_faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def _open_point_cloud_transform_tool(self):
        """Opens the Unity-style Transform tool in the attached floating toolbox."""
        if hasattr(self, 'editor_tool_host') and hasattr(self, 'transform_tool_window'):
            self.editor_tool_host.open_tool(self.transform_tool_window)
            self._sync_active_transform_to_viewport()
            self.console_text.append("[TOOLS] Opened Transform tool window.")

    def _get_point_cloud_marker_colors(self):
        """Returns RGBA float32 array normalized to [0, 1] for VisPy marker coloring."""
        if self._current_colors is not None and len(self._current_colors) > 0:
            mc = self._current_colors.astype(np.float32)
            if mc.max() > 1.0:
                mc = mc / 255.0
            if mc.shape[1] == 3:
                mc = np.hstack([mc, np.ones((mc.shape[0], 1), dtype=np.float32)])
            return mc
        return 'white'

    def _on_cloud_transform_preview(self, T_mat: np.ndarray):
        """Live updates VisPy point cloud or mesh visualization based on transform matrix."""
        if self._current_points is None or len(self._current_points) == 0:
            return
        
        R = T_mat[:3, :3]
        t = T_mat[:3, 3]
        transformed_pts = (self._current_points @ R.T) + t
        
        if hasattr(self, 'markers_visual') and self.markers_visual is not None:
            mode = self.viewer_widget.mode_select.currentIndex() if hasattr(self, 'viewer_widget') else 1
            point_size = 4 if mode == 0 else 2
            self.markers_visual.set_data(
                pos=transformed_pts.astype(np.float32),
                face_color=self._get_point_cloud_marker_colors(),
                size=point_size,
                edge_width=0
            )

        if hasattr(self, 'mesh_visual') and self.mesh_visual is not None and self._current_faces is not None:
            mesh_colors = None
            if self._current_colors is not None and len(self._current_colors) > 0:
                mc = self._current_colors.astype(np.float32)
                if mc.max() > 1.0:
                    mc = mc / 255.0
                mesh_colors = mc
            self.mesh_visual.set_data(
                vertices=transformed_pts.astype(np.float32),
                faces=self._current_faces.astype(np.uint32),
                vertex_colors=mesh_colors,
                color='white'
            )
            self._last_wf_vertices = transformed_pts.astype(np.float32)
            self._apply_shading_mode_to_mesh()
            
        # Keep gizmo visual transform in sync if visible and not dragging
        if hasattr(self, 'transform_gizmo') and self.transform_gizmo is not None and self.transform_gizmo.visible:
            if not self.transform_gizmo.is_dragging and hasattr(self, 'transform_tool_window') and self.transform_tool_window is not None:
                if self.transform_tool_window.position_row and self.transform_tool_window.rotation_row:
                    pos = self.transform_tool_window.position_row.get_values()
                    rot = self.transform_tool_window.rotation_row.get_values()
                    self.transform_gizmo.set_transform(position=pos, rotation=rot)

        self._update_ground_grid(transformed_pts)
        self.canvas.update()

    def _sync_active_transform_to_viewport(self):
        """Immediately syncs the active Transform tool values (position/rotation) to the current viewport visual."""
        if self._current_points is not None and len(self._current_points) > 0:
            pivot = np.mean(self._current_points, axis=0)
            if hasattr(self, 'transform_gizmo') and self.transform_gizmo is not None:
                self.transform_gizmo.set_pivot(pivot)

            if hasattr(self, 'transform_tool_window') and self.transform_tool_window is not None:
                if self.transform_tool_window.has_unapplied_changes():
                    T_mat = self.transform_tool_window.get_transform_matrix()
                    self._on_cloud_transform_preview(T_mat)

    def _on_cloud_transform_reset(self):
        """Reverts point cloud or mesh viewport display back to original untransformed coordinates."""
        if self._current_points is None or len(self._current_points) == 0:
            return
            
        if hasattr(self, 'markers_visual') and self.markers_visual is not None:
            self.markers_visual.set_data(
                pos=self._current_points.astype(np.float32),
                face_color=self._get_point_cloud_marker_colors(),
                size=2,
                edge_width=0
            )

        if hasattr(self, 'mesh_visual') and self.mesh_visual is not None and self._current_faces is not None:
            mesh_colors = None
            if self._current_colors is not None and len(self._current_colors) > 0:
                mc = self._current_colors.astype(np.float32)
                if mc.max() > 1.0:
                    mc = mc / 255.0
                mesh_colors = mc
            self.mesh_visual.set_data(
                vertices=self._current_points.astype(np.float32),
                faces=self._current_faces.astype(np.uint32),
                vertex_colors=mesh_colors,
                color='white'
            )
            
        if hasattr(self, 'transform_gizmo') and self.transform_gizmo is not None:
            self.transform_gizmo.set_transform(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))

        self._update_ground_grid(self._current_points)
        self.canvas.update()

    def _on_cloud_transform_closed(self):
        self._on_cloud_transform_reset()

    def _on_cloud_transform_applied(self, T_mat: np.ndarray):
        """Bakes the transformation into active scene data (points/mesh) in memory and writes to PLY files."""
        if self._current_points is None or len(self._current_points) == 0:
            return
            
        R = T_mat[:3, :3]
        t = T_mat[:3, 3]

        # Track cumulative transformation relative to reconstruction camera coordinates
        if not hasattr(self, '_cumulative_mesh_transform') or self._cumulative_mesh_transform is None:
            self._cumulative_mesh_transform = np.eye(4, dtype=np.float32)
        self._cumulative_mesh_transform = T_mat @ self._cumulative_mesh_transform
        
        # Bake transformation
        self._current_points = (self._current_points @ R.T) + t
        self._last_points = self._current_points
        if self._raw_points is not None:
            self._raw_points = (self._raw_points @ R.T) + t
            
        if hasattr(self, 'markers_visual') and self.markers_visual is not None:
            mode = self.viewer_widget.mode_select.currentIndex() if hasattr(self, 'viewer_widget') else 1
            point_size = 4 if mode == 0 else 2
            self.markers_visual.set_data(
                pos=self._current_points.astype(np.float32),
                face_color=self._get_point_cloud_marker_colors(),
                size=point_size,
                edge_width=0
            )

        if hasattr(self, 'mesh_visual') and self.mesh_visual is not None and self._current_faces is not None:
            mesh_colors = None
            if self._current_colors is not None and len(self._current_colors) > 0:
                mc = self._current_colors.astype(np.float32)
                if mc.max() > 1.0:
                    mc = mc / 255.0
                mesh_colors = mc
            self.mesh_visual.set_data(
                vertices=self._current_points.astype(np.float32),
                faces=self._current_faces.astype(np.uint32),
                vertex_colors=mesh_colors,
                color='white'
            )
            self._last_wf_vertices = self._current_points.astype(np.float32)
            self._apply_shading_mode_to_mesh()
            
        if hasattr(self, 'transform_gizmo') and self.transform_gizmo is not None:
            pivot = np.mean(self._current_points, axis=0) if self._current_points is not None and len(self._current_points) > 0 else np.zeros(3)
            self.transform_gizmo.set_pivot(pivot)
            self.transform_gizmo.set_transform(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))
            
        self._update_ground_grid(self._current_points)
        self.canvas.update()
        
        # Save to disk
        from point_cloud_io import save_transformed_point_cloud
        saved_target = None
        mvs_dir = self.viewer_widget.current_mvs_dir if hasattr(self, 'viewer_widget') else None
        if not mvs_dir:
            mvs_dir = self._get_active_mvs_dir()

        if hasattr(self, 'standalone_cloud_path') and self.standalone_cloud_path and os.path.isfile(self.standalone_cloud_path):
            save_transformed_point_cloud(self.standalone_cloud_path, self._current_points, colors=self._current_colors)
            saved_target = os.path.basename(self.standalone_cloud_path)
        elif mvs_dir and os.path.isdir(mvs_dir):
            dense_path = os.path.join(mvs_dir, "scene_dense.ply")
            if os.path.exists(dense_path):
                save_transformed_point_cloud(dense_path, self._current_points, colors=self._current_colors)
                saved_target = "scene_dense.ply"

            # Transform active textured mesh files on disk in-place so disk models match the viewport
            obj_path = os.path.join(mvs_dir, "scene_dense_mesh_texture.obj")
            if os.path.exists(obj_path):
                self._apply_transform_to_obj_file(obj_path, T_mat)
            ply_tex_path = os.path.join(mvs_dir, "scene_dense_mesh_texture.ply")
            if os.path.exists(ply_tex_path):
                self._apply_transform_to_ply_file(ply_tex_path, T_mat)

            # Invalidate stale GLB cache so next export re-converts the transformed OBJ
            glb_path = os.path.join(mvs_dir, "scene_dense_mesh_texture.glb")
            if os.path.exists(glb_path):
                try:
                    os.remove(glb_path)
                except Exception:
                    pass

        # Update cleanup backup data and file on disk so future Revert actions respect the new position
        if mvs_dir and os.path.isdir(mvs_dir):
            backup_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_backup.ply")
            if hasattr(self, '_cleanup_backup_points') and self._cleanup_backup_points is not None:
                self._cleanup_backup_points = (self._cleanup_backup_points @ R.T) + t
                if os.path.exists(backup_ply):
                    try:
                        self._export_temp_mesh_ply(
                            self._cleanup_backup_points,
                            self._cleanup_backup_faces,
                            self._cleanup_backup_colors,
                            backup_ply
                        )
                    except Exception as bkp_err:
                        self.console_text.append(f"[WARNING] Could not update backup PLY coordinates: {bkp_err}")
            elif os.path.exists(backup_ply) and self._current_points is not None and self._current_faces is not None:
                try:
                    self._export_temp_mesh_ply(
                        self._current_points,
                        self._current_faces,
                        self._current_colors,
                        backup_ply
                    )
                except Exception:
                    pass

            # Also update legacy preop backup if present
            preop_backup = os.path.join(mvs_dir, "scene_dense_mesh_preop_backup.ply")
            if os.path.exists(preop_backup) and self._current_points is not None and self._current_faces is not None:
                try:
                    self._export_temp_mesh_ply(
                        self._current_points,
                        self._current_faces,
                        self._current_colors,
                        preop_backup
                    )
                except Exception:
                    pass

            # Update session recovery backup if working in the active reconstruction session
            try:
                import shutil
                active_rec_mvs = os.path.normpath(os.path.join(get_reconstruction_out_dir(), "mvs"))
                if os.path.normpath(mvs_dir) == active_rec_mvs:
                    backup_mvs = os.path.join(get_backup_dir(), "mvs")
                    if os.path.isdir(backup_mvs):
                        dense_src = os.path.join(mvs_dir, "scene_dense.ply")
                        if os.path.isfile(dense_src):
                            shutil.copy2(dense_src, os.path.join(backup_mvs, "scene_dense.ply"))
                        obj_src = os.path.join(mvs_dir, "scene_dense_mesh_texture.obj")
                        if os.path.isfile(obj_src):
                            shutil.copy2(obj_src, os.path.join(backup_mvs, "scene_dense_mesh_texture.obj"))
                        ply_tex_src = os.path.join(mvs_dir, "scene_dense_mesh_texture.ply")
                        if os.path.isfile(ply_tex_src):
                            shutil.copy2(ply_tex_src, os.path.join(backup_mvs, "scene_dense_mesh_texture.ply"))
                        backup_glb = os.path.join(backup_mvs, "scene_dense_mesh_texture.glb")
                        if os.path.exists(backup_glb):
                            try:
                                os.remove(backup_glb)
                            except Exception:
                                pass

                        meta = load_session_metadata() or {}
                        meta["cumulative_mesh_transform"] = self._cumulative_mesh_transform.tolist()
                        save_session_metadata(meta)
                        self.console_text.append("[BACKUP] Synced transformed model to session recovery backup.")
            except Exception as e:
                self.console_text.append(f"[WARNING] Could not sync transform to session backup: {e}")

        if saved_target:
            self.console_text.append(f"[SUCCESS] Model transformed and saved to '{saved_target}'.")
            self.status_label.setText(f"Transformed & saved to {saved_target}")
        else:
            self.console_text.append("[SUCCESS] Model transformation applied in viewport.")

        # Close the transform tool window once transform is baked
        if hasattr(self, 'editor_tool_host') and self.editor_tool_host is not None:
            self.editor_tool_host.close_tool()

    def _on_mesh_face_reduction_apply(self, reduction_pct: float):
        """Executes Face Reduction via PyMeshLab (with Open3D fallback) and updates the active mesh in VisPy."""
        if self._current_points is None or len(self._current_points) == 0 or self._current_faces is None or len(self._current_faces) == 0:
            QMessageBox.warning(self, "No Mesh Available", "Please load or reconstruct a 3D textured mesh before applying Face Reduction.")
            return

        mvs_dir = self.viewer_widget.current_mvs_dir or os.path.join(get_reconstruction_out_dir(), "mvs")
        os.makedirs(mvs_dir, exist_ok=True)

        tool_in_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_in.ply")
        tool_out_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_out.ply")
        backup_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_backup.ply")

        # Create pre-operation backup for instant revert
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, backup_ply)
        self._cleanup_backup_points = np.copy(self._current_points) if self._current_points is not None else None
        self._cleanup_backup_faces = np.copy(self._current_faces) if self._current_faces is not None else None
        self._cleanup_backup_colors = np.copy(self._current_colors) if self._current_colors is not None else None
        self._cleanup_backup_texcoords = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
        self._cleanup_backup_texture_path = self._current_texture_path

        # Export working mesh to in_ply
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, tool_in_ply)

        # Show non-blocking progress dialog
        self._cleanup_progress_dialog = ProjectProgressDialog(
            "Mesh Face Reduction",
            f"Reducing mesh faces by {int(reduction_pct)}%...\nPlease wait.",
            self
        )
        self._cleanup_progress_dialog.show()

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(True)

        cleanup_params = {
            "enable_reduction": True,
            "target_reduction_pct": reduction_pct,
            "remove_duplicates": False,
            "repair_nonmanifold": False,
            "close_holes": False
        }

        from pipeline_manager import MeshOperationWorker
        self._cleanup_decimation_worker = MeshOperationWorker("cleanup", tool_in_ply, tool_out_ply, cleanup_params, parent=self)
        self._cleanup_decimation_worker.log_message.connect(self._append_log)
        self._cleanup_decimation_worker.finished.connect(
            lambda ok, out_path, msg: self._on_mesh_face_reduction_finished(ok, out_path, msg, reduction_pct)
        )
        self._cleanup_decimation_worker.start()

    def _on_mesh_face_reduction_finished(self, success: bool, output_ply: str, msg: str, reduction_pct: float):
        """Called when Mesh Face Reduction worker completes."""
        if hasattr(self, '_cleanup_progress_dialog') and self._cleanup_progress_dialog:
            self._cleanup_progress_dialog.accept()
            self._cleanup_progress_dialog = None

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(False)

        if success and output_ply and os.path.isfile(output_ply):
            pts, cls, fcs = _read_ply_static(output_ply)
            if pts is not None and len(pts) > 0 and fcs is not None and len(fcs) > 0:
                init_faces = len(self._current_faces) if self._current_faces is not None else 0
                new_faces = len(fcs)
                init_verts = len(self._current_points) if self._current_points is not None else 0
                new_verts = len(pts)

                self._current_points = pts
                self._current_faces = fcs
                self._current_colors = cls
                self._last_points = pts

                # Re-render in VisPy without resetting camera position
                self._render_in_vispy_from_data(
                    pts, cls, fcs,
                    self._current_texcoords, self._current_texture_path,
                    mode=2, reset_camera=False
                )
                self._sync_active_transform_to_viewport()

                if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                    self.mesh_cleanup_tool_window.set_revert_enabled(True)

                pct_achieved = ((init_faces - new_faces) / init_faces * 100.0) if init_faces > 0 else reduction_pct
                status_msg = f"[CLEANUP] Face Reduction complete: {init_faces:,} → {new_faces:,} faces ({pct_achieved:.1f}% reduction, {new_verts:,} vertices)."
                self.console_text.append(status_msg)
                self.status_label.setText(f"Face Reduction: {new_faces:,} faces ({pct_achieved:.1f}% reduced)")
            else:
                self.console_text.append("[ERROR] Face reduction produced empty geometry.")
                QMessageBox.warning(self, "Face Reduction Failed", "Mesh decimation did not produce valid geometry.")
        else:
            self.console_text.append(f"[ERROR] Face reduction failed: {msg}")
            QMessageBox.warning(self, "Face Reduction Failed", f"Decimation failed:\n{msg}")

    def _on_mesh_face_reduction_revert(self):
        """Restores original pre-reduction mesh geometry."""
        if hasattr(self, '_cleanup_backup_points') and self._cleanup_backup_points is not None:
            self._current_points = np.copy(self._cleanup_backup_points)
            self._current_faces = np.copy(self._cleanup_backup_faces) if self._cleanup_backup_faces is not None else None
            self._current_colors = np.copy(self._cleanup_backup_colors) if self._cleanup_backup_colors is not None else None
            self._current_texcoords = np.copy(self._cleanup_backup_texcoords) if self._cleanup_backup_texcoords is not None else None
            self._current_texture_path = self._cleanup_backup_texture_path
            self._last_points = self._current_points

            self._render_in_vispy_from_data(
                self._current_points, self._current_colors, self._current_faces,
                self._current_texcoords, self._current_texture_path,
                mode=2, reset_camera=False
            )
            self._sync_active_transform_to_viewport()

            if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                self.mesh_cleanup_tool_window.set_revert_enabled(False)

            num_f = len(self._current_faces) if self._current_faces is not None else 0
            num_v = len(self._current_points) if self._current_points is not None else 0
            self.console_text.append("[CLEANUP] Reverted mesh to original geometry ({num_v:,} vertices, {num_f:,} faces).")
            self.status_label.setText("Mesh restored to original state.")
        else:
            self.console_text.append("[WARNING] No previous mesh backup found to revert.")

    def _on_mesh_close_holes_apply(self, max_hole_size: int):
        """Executes Close Mesh Holes via PyMeshLab and updates the active mesh in VisPy."""
        if self._current_points is None or len(self._current_points) == 0 or self._current_faces is None or len(self._current_faces) == 0:
            QMessageBox.warning(self, "No Mesh Available", "Please load or reconstruct a 3D textured mesh before applying Close Holes.")
            return

        mvs_dir = self.viewer_widget.current_mvs_dir or os.path.join(get_reconstruction_out_dir(), "mvs")
        os.makedirs(mvs_dir, exist_ok=True)

        tool_in_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_in.ply")
        tool_out_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_out.ply")
        backup_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_backup.ply")

        # Create pre-operation backup for instant revert
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, backup_ply)
        self._cleanup_backup_points = np.copy(self._current_points) if self._current_points is not None else None
        self._cleanup_backup_faces = np.copy(self._current_faces) if self._current_faces is not None else None
        self._cleanup_backup_colors = np.copy(self._current_colors) if self._current_colors is not None else None
        self._cleanup_backup_texcoords = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
        self._cleanup_backup_texture_path = self._current_texture_path

        # Export working mesh to in_ply
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, tool_in_ply)

        # Show non-blocking progress dialog
        self._cleanup_progress_dialog = ProjectProgressDialog(
            "Close Mesh Holes",
            f"Closing mesh holes up to {int(max_hole_size)} faces...\nPlease wait.",
            self
        )
        self._cleanup_progress_dialog.show()

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(True)

        cleanup_params = {
            "enable_reduction": False,
            "target_reduction_pct": 0,
            "remove_duplicates": False,
            "repair_nonmanifold": False,
            "close_holes": True,
            "max_hole_size": int(max_hole_size)
        }

        # Keep tool menu spin box in sync if it exists
        if hasattr(self, 'mc_max_hole_spin') and self.mc_max_hole_spin:
            self.mc_max_hole_spin.blockSignals(True)
            self.mc_max_hole_spin.setValue(int(max_hole_size))
            self.mc_max_hole_spin.blockSignals(False)

        from pipeline_manager import MeshOperationWorker
        self._cleanup_holes_worker = MeshOperationWorker("cleanup", tool_in_ply, tool_out_ply, cleanup_params, parent=self)
        self._cleanup_holes_worker.log_message.connect(self._append_log)
        self._cleanup_holes_worker.finished.connect(
            lambda ok, out_path, msg: self._on_mesh_close_holes_finished(ok, out_path, msg, max_hole_size)
        )
        self._cleanup_holes_worker.start()

    def _on_mesh_close_holes_finished(self, success: bool, output_ply: str, msg: str, max_hole_size: int):
        """Called when Close Mesh Holes worker completes."""
        if hasattr(self, '_cleanup_progress_dialog') and self._cleanup_progress_dialog:
            self._cleanup_progress_dialog.accept()
            self._cleanup_progress_dialog = None

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(False)

        if success and output_ply and os.path.isfile(output_ply):
            pts, cls, fcs = _read_ply_static(output_ply)
            if pts is not None and len(pts) > 0 and fcs is not None and len(fcs) > 0:
                init_faces = len(self._current_faces) if self._current_faces is not None else 0
                new_faces = len(fcs)
                init_verts = len(self._current_points) if self._current_points is not None else 0
                new_verts = len(pts)

                self._current_points = pts
                self._current_faces = fcs
                self._current_colors = cls
                self._last_points = pts

                # Re-render in VisPy without resetting camera position
                self._render_in_vispy_from_data(
                    pts, cls, fcs,
                    self._current_texcoords, self._current_texture_path,
                    mode=2, reset_camera=False
                )
                self._sync_active_transform_to_viewport()

                if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                    self.mesh_cleanup_tool_window.set_holes_revert_enabled(True)

                faces_diff = new_faces - init_faces
                diff_str = f"+{faces_diff:,} faces added" if faces_diff >= 0 else f"{faces_diff:,} faces"
                status_msg = f"[CLEANUP] Close Holes complete: {init_faces:,} → {new_faces:,} faces ({diff_str}, {new_verts:,} vertices). Click 'Retexture' to project textures or 'Revert' to undo."
                self.console_text.append(status_msg)
                self.status_label.setText(f"Close Holes: {new_faces:,} faces ({diff_str})")
            else:
                self.console_text.append("[ERROR] Close holes produced empty geometry.")
                QMessageBox.warning(self, "Close Holes Failed", "Mesh hole closing did not produce valid geometry.")
        else:
            self.console_text.append(f"[ERROR] Close holes failed: {msg}")
            QMessageBox.warning(self, "Close Holes Failed", f"Close holes failed:\n{msg}")

    def _on_mesh_close_holes_revert(self):
        """Restores original pre-hole-closing mesh geometry."""
        if hasattr(self, '_cleanup_backup_points') and self._cleanup_backup_points is not None:
            self._current_points = np.copy(self._cleanup_backup_points)
            self._current_faces = np.copy(self._cleanup_backup_faces) if self._cleanup_backup_faces is not None else None
            self._current_colors = np.copy(self._cleanup_backup_colors) if self._cleanup_backup_colors is not None else None
            self._current_texcoords = np.copy(self._cleanup_backup_texcoords) if self._cleanup_backup_texcoords is not None else None
            self._current_texture_path = self._cleanup_backup_texture_path
            self._last_points = self._current_points

            self._render_in_vispy_from_data(
                self._current_points, self._current_colors, self._current_faces,
                self._current_texcoords, self._current_texture_path,
                mode=2, reset_camera=False
            )
            self._sync_active_transform_to_viewport()

            if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                self.mesh_cleanup_tool_window.set_holes_revert_enabled(False)

            num_f = len(self._current_faces) if self._current_faces is not None else 0
            num_v = len(self._current_points) if self._current_points is not None else 0
            self.console_text.append(f"[CLEANUP] Reverted mesh to previous geometry ({num_v:,} vertices, {num_f:,} faces).")
            self.status_label.setText("Mesh restored to previous state.")
        else:
            self.console_text.append("[WARNING] No previous mesh backup found to revert.")

    def _on_mesh_repair_nonmanifold_apply(self):
        """Executes Non-Manifold repair via PyMeshLab and updates the active mesh in VisPy."""
        if self._current_points is None or len(self._current_points) == 0 or self._current_faces is None or len(self._current_faces) == 0:
            QMessageBox.warning(self, "No Mesh Available", "Please load or reconstruct a 3D textured mesh before repairing non-manifold topology.")
            return

        mvs_dir = self.viewer_widget.current_mvs_dir or os.path.join(get_reconstruction_out_dir(), "mvs")
        os.makedirs(mvs_dir, exist_ok=True)

        tool_in_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_in.ply")
        tool_out_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_out.ply")
        backup_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_backup.ply")

        # Create pre-operation backup for instant revert
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, backup_ply)
        self._cleanup_backup_points = np.copy(self._current_points) if self._current_points is not None else None
        self._cleanup_backup_faces = np.copy(self._current_faces) if self._current_faces is not None else None
        self._cleanup_backup_colors = np.copy(self._current_colors) if self._current_colors is not None else None
        self._cleanup_backup_texcoords = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
        self._cleanup_backup_texture_path = self._current_texture_path

        # Export working mesh to in_ply
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, tool_in_ply)

        # Show non-blocking progress dialog
        self._cleanup_progress_dialog = ProjectProgressDialog(
            "Repair Non-Manifold",
            "Repairing non-manifold edges and vertices...\nPlease wait.",
            self
        )
        self._cleanup_progress_dialog.show()

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(True)

        cleanup_params = {
            "enable_reduction": False,
            "target_reduction_pct": 0,
            "remove_duplicates": False,
            "repair_nonmanifold": True,
            "close_holes": False
        }

        from pipeline_manager import MeshOperationWorker
        self._cleanup_nm_worker = MeshOperationWorker("cleanup", tool_in_ply, tool_out_ply, cleanup_params, parent=self)
        self._cleanup_nm_worker.log_message.connect(self._append_log)
        self._cleanup_nm_worker.finished.connect(
            lambda ok, out_path, msg: self._on_mesh_repair_nonmanifold_finished(ok, out_path, msg)
        )
        self._cleanup_nm_worker.start()

    def _on_mesh_repair_nonmanifold_finished(self, success: bool, output_ply: str, msg: str):
        """Called when Repair Non-Manifold worker completes."""
        if hasattr(self, '_cleanup_progress_dialog') and self._cleanup_progress_dialog:
            self._cleanup_progress_dialog.accept()
            self._cleanup_progress_dialog = None

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(False)

        if success and output_ply and os.path.isfile(output_ply):
            pts, cls, fcs = _read_ply_static(output_ply)
            if pts is not None and len(pts) > 0 and fcs is not None and len(fcs) > 0:
                init_faces = len(self._current_faces) if self._current_faces is not None else 0
                new_faces = len(fcs)
                init_verts = len(self._current_points) if self._current_points is not None else 0
                new_verts = len(pts)

                self._current_points = pts
                self._current_faces = fcs
                self._current_colors = cls
                self._last_points = pts

                # Cleanup changes vertex/face indexing; the PLY has no remapped UVs.
                self._current_texcoords = None
                self._current_texture_path = None

                # Re-render in VisPy without resetting camera position
                self._render_in_vispy_from_data(
                    pts, cls, fcs,
                    self._current_texcoords, self._current_texture_path,
                    mode=2, reset_camera=False
                )
                self._sync_active_transform_to_viewport()

                if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                    self.mesh_cleanup_tool_window.set_nonmanifold_revert_enabled(True)

                status_msg = f"[CLEANUP] Repaired non-manifold geometry: {init_faces:,} → {new_faces:,} faces, {init_verts:,} → {new_verts:,} vertices. Click 'Retexture' to project textures or 'Revert' to undo."
                self.console_text.append(status_msg)
                self.status_label.setText(f"Repair Non-Manifold: {new_faces:,} faces ({new_verts:,} vertices)")
            else:
                self.console_text.append("[ERROR] Repair non-manifold produced empty geometry.")
                QMessageBox.warning(self, "Repair Failed", "Mesh repair did not produce valid geometry.")
        else:
            self.console_text.append(f"[ERROR] Repair non-manifold failed: {msg}")
            QMessageBox.warning(self, "Repair Failed", f"Repair non-manifold failed:\n{msg}")

    def _on_mesh_repair_nonmanifold_revert(self):
        """Restores original pre-repair mesh geometry."""
        if hasattr(self, '_cleanup_backup_points') and self._cleanup_backup_points is not None:
            self._current_points = np.copy(self._cleanup_backup_points)
            self._current_faces = np.copy(self._cleanup_backup_faces) if self._cleanup_backup_faces is not None else None
            self._current_colors = np.copy(self._cleanup_backup_colors) if self._cleanup_backup_colors is not None else None
            self._current_texcoords = np.copy(self._cleanup_backup_texcoords) if self._cleanup_backup_texcoords is not None else None
            self._current_texture_path = self._cleanup_backup_texture_path
            self._last_points = self._current_points

            self._render_in_vispy_from_data(
                self._current_points, self._current_colors, self._current_faces,
                self._current_texcoords, self._current_texture_path,
                mode=2, reset_camera=False
            )
            self._sync_active_transform_to_viewport()

            if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                self.mesh_cleanup_tool_window.set_nonmanifold_revert_enabled(False)

            num_f = len(self._current_faces) if self._current_faces is not None else 0
            num_v = len(self._current_points) if self._current_points is not None else 0
            self.console_text.append(f"[CLEANUP] Reverted mesh to previous geometry ({num_v:,} vertices, {num_f:,} faces).")
            self.status_label.setText("Mesh restored to previous state.")
        else:
            self.console_text.append("[WARNING] No previous mesh backup found to revert.")

    def _on_mesh_remove_duplicates_apply(self):
        """Executes Duplicate removal via PyMeshLab and updates the active mesh in VisPy."""
        if self._current_points is None or len(self._current_points) == 0 or self._current_faces is None or len(self._current_faces) == 0:
            QMessageBox.warning(self, "No Mesh Available", "Please load or reconstruct a 3D textured mesh before removing duplicates.")
            return

        mvs_dir = self.viewer_widget.current_mvs_dir or os.path.join(get_reconstruction_out_dir(), "mvs")
        os.makedirs(mvs_dir, exist_ok=True)

        tool_in_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_in.ply")
        tool_out_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_out.ply")
        backup_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_backup.ply")

        # Create pre-operation backup for instant revert
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, backup_ply)
        self._cleanup_backup_points = np.copy(self._current_points) if self._current_points is not None else None
        self._cleanup_backup_faces = np.copy(self._current_faces) if self._current_faces is not None else None
        self._cleanup_backup_colors = np.copy(self._current_colors) if self._current_colors is not None else None
        self._cleanup_backup_texcoords = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
        self._cleanup_backup_texture_path = self._current_texture_path

        # Export working mesh to in_ply
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, tool_in_ply)

        # Show non-blocking progress dialog
        self._cleanup_progress_dialog = ProjectProgressDialog(
            "Remove Duplicates",
            "Removing duplicate faces and vertices...\nPlease wait.",
            self
        )
        self._cleanup_progress_dialog.show()

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(True)

        cleanup_params = {
            "enable_reduction": False,
            "target_reduction_pct": 0,
            "remove_duplicates": True,
            "repair_nonmanifold": False,
            "close_holes": False
        }

        from pipeline_manager import MeshOperationWorker
        self._cleanup_dup_worker = MeshOperationWorker("cleanup", tool_in_ply, tool_out_ply, cleanup_params, parent=self)
        self._cleanup_dup_worker.log_message.connect(self._append_log)
        self._cleanup_dup_worker.finished.connect(
            lambda ok, out_path, msg: self._on_mesh_remove_duplicates_finished(ok, out_path, msg)
        )
        self._cleanup_dup_worker.start()

    def _on_mesh_remove_duplicates_finished(self, success: bool, output_ply: str, msg: str):
        """Called when Remove Duplicates worker completes."""
        if hasattr(self, '_cleanup_progress_dialog') and self._cleanup_progress_dialog:
            self._cleanup_progress_dialog.accept()
            self._cleanup_progress_dialog = None

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(False)

        if success and output_ply and os.path.isfile(output_ply):
            pts, cls, fcs = _read_ply_static(output_ply)
            if pts is not None and len(pts) > 0 and fcs is not None and len(fcs) > 0:
                init_faces = len(self._current_faces) if self._current_faces is not None else 0
                new_faces = len(fcs)
                init_verts = len(self._current_points) if self._current_points is not None else 0
                new_verts = len(pts)

                self._current_points = pts
                self._current_faces = fcs
                self._current_colors = cls
                self._last_points = pts

                # Cleanup changes vertex/face indexing; the PLY has no remapped UVs.
                self._current_texcoords = None
                self._current_texture_path = None

                # Re-render in VisPy without resetting camera position
                self._render_in_vispy_from_data(
                    pts, cls, fcs,
                    self._current_texcoords, self._current_texture_path,
                    mode=2, reset_camera=False
                )
                self._sync_active_transform_to_viewport()

                if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                    self.mesh_cleanup_tool_window.set_duplicates_revert_enabled(True)

                rem_faces = init_faces - new_faces
                rem_verts = init_verts - new_verts
                status_msg = f"[CLEANUP] Remove Duplicates complete: {init_faces:,} → {new_faces:,} faces ({rem_faces:,} removed, {rem_verts:,} vertices removed). Click 'Retexture' to project textures or 'Revert' to undo."
                self.console_text.append(status_msg)
                self.status_label.setText(f"Remove Duplicates: {rem_faces:,} faces, {rem_verts:,} vertices removed")
            else:
                self.console_text.append("[ERROR] Remove duplicates produced empty geometry.")
                QMessageBox.warning(self, "Remove Duplicates Failed", "Mesh duplicate removal did not produce valid geometry.")
        else:
            self.console_text.append(f"[ERROR] Remove duplicates failed: {msg}")
            QMessageBox.warning(self, "Remove Duplicates Failed", f"Remove duplicates failed:\n{msg}")

    def _on_mesh_remove_duplicates_revert(self):
        """Restores original pre-duplicate-removal mesh geometry."""
        if hasattr(self, '_cleanup_backup_points') and self._cleanup_backup_points is not None:
            self._current_points = np.copy(self._cleanup_backup_points)
            self._current_faces = np.copy(self._cleanup_backup_faces) if self._cleanup_backup_faces is not None else None
            self._current_colors = np.copy(self._cleanup_backup_colors) if self._cleanup_backup_colors is not None else None
            self._current_texcoords = np.copy(self._cleanup_backup_texcoords) if self._cleanup_backup_texcoords is not None else None
            self._current_texture_path = self._cleanup_backup_texture_path
            self._last_points = self._current_points

            self._render_in_vispy_from_data(
                self._current_points, self._current_colors, self._current_faces,
                self._current_texcoords, self._current_texture_path,
                mode=2, reset_camera=False
            )
            self._sync_active_transform_to_viewport()

            if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                self.mesh_cleanup_tool_window.set_duplicates_revert_enabled(False)

            num_f = len(self._current_faces) if self._current_faces is not None else 0
            num_v = len(self._current_points) if self._current_points is not None else 0
            self.console_text.append(f"[CLEANUP] Reverted mesh to previous geometry ({num_v:,} vertices, {num_f:,} faces).")
            self.status_label.setText("Mesh restored to previous state.")
        else:
            self.console_text.append("[WARNING] No previous mesh backup found to revert.")

    def _on_mesh_merge_vertices_apply(self, threshold_pct: float, bbox_diagonal: float):
        """Executes Merge Vertices by Distance via PyMeshLab and updates the active mesh in VisPy."""
        if self._current_points is None or len(self._current_points) == 0 or self._current_faces is None or len(self._current_faces) == 0:
            QMessageBox.warning(self, "No Mesh Available", "Please load or reconstruct a 3D textured mesh before merging vertices.")
            return

        mvs_dir = self.viewer_widget.current_mvs_dir or os.path.join(get_reconstruction_out_dir(), "mvs")
        os.makedirs(mvs_dir, exist_ok=True)

        tool_in_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_in.ply")
        tool_out_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_out.ply")
        backup_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_backup.ply")

        # Create pre-operation backup for instant revert
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, backup_ply)
        self._cleanup_backup_points = np.copy(self._current_points) if self._current_points is not None else None
        self._cleanup_backup_faces = np.copy(self._current_faces) if self._current_faces is not None else None
        self._cleanup_backup_colors = np.copy(self._current_colors) if self._current_colors is not None else None
        self._cleanup_backup_texcoords = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
        self._cleanup_backup_texture_path = self._current_texture_path

        # Export working mesh to in_ply
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, tool_in_ply)

        # Show non-blocking progress dialog
        self._cleanup_progress_dialog = ProjectProgressDialog(
            "Merge Close Vertices",
            f"Merging vertices within {threshold_pct:.3f}% threshold...\nPlease wait.",
            self
        )
        self._cleanup_progress_dialog.show()

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(True)

        merge_params = {
            "threshold_pct": float(threshold_pct),
            "bbox_diagonal": float(bbox_diagonal)
        }

        from pipeline_manager import MeshOperationWorker
        self._cleanup_merge_worker = MeshOperationWorker("merge_by_distance", tool_in_ply, tool_out_ply, merge_params, parent=self)
        self._cleanup_merge_worker.log_message.connect(self._append_log)
        self._cleanup_merge_worker.finished.connect(
            lambda ok, out_path, msg: self._on_mesh_merge_vertices_finished(ok, out_path, msg)
        )
        self._cleanup_merge_worker.start()

    def _on_mesh_merge_vertices_finished(self, success: bool, output_ply: str, msg: str):
        """Called when Merge Vertices worker completes."""
        if hasattr(self, '_cleanup_progress_dialog') and self._cleanup_progress_dialog:
            self._cleanup_progress_dialog.accept()
            self._cleanup_progress_dialog = None

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(False)

        if success and output_ply and os.path.isfile(output_ply):
            pts, cls, fcs = _read_ply_static(output_ply)
            if pts is not None and len(pts) > 0 and fcs is not None and len(fcs) > 0:
                init_faces = len(self._current_faces) if self._current_faces is not None else 0
                new_faces = len(fcs)
                init_verts = len(self._current_points) if self._current_points is not None else 0
                new_verts = len(pts)

                self._current_points = pts
                self._current_faces = fcs
                self._current_colors = cls
                self._last_points = pts

                # Cleanup changes vertex/face indexing; the PLY has no remapped UVs.
                self._current_texcoords = None
                self._current_texture_path = None

                # Re-render in VisPy without resetting camera position
                self._render_in_vispy_from_data(
                    pts, cls, fcs,
                    self._current_texcoords, self._current_texture_path,
                    mode=2, reset_camera=False
                )
                self._sync_active_transform_to_viewport()

                if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                    self.mesh_cleanup_tool_window.set_merge_revert_enabled(True)

                rem_verts = init_verts - new_verts
                rem_faces = init_faces - new_faces
                status_msg = f"[CLEANUP] Merge Vertices complete: {init_verts:,} → {new_verts:,} vertices ({rem_verts:,} merged, {rem_faces:,} faces updated). Click 'Retexture' to project textures or 'Revert' to undo."
                self.console_text.append(status_msg)
                self.status_label.setText(f"Merge Vertices: {rem_verts:,} vertices merged ({new_verts:,} remaining)")
            else:
                self.console_text.append("[ERROR] Merge vertices produced empty geometry.")
                QMessageBox.warning(self, "Merge Vertices Failed", "Merge vertices did not produce valid geometry.")
        else:
            self.console_text.append(f"[ERROR] Merge vertices failed: {msg}")
            QMessageBox.warning(self, "Merge Vertices Failed", f"Merge vertices failed:\n{msg}")

    def _on_mesh_merge_vertices_revert(self):
        """Restores original pre-merge mesh geometry."""
        if hasattr(self, '_cleanup_backup_points') and self._cleanup_backup_points is not None:
            self._current_points = np.copy(self._cleanup_backup_points)
            self._current_faces = np.copy(self._cleanup_backup_faces) if self._cleanup_backup_faces is not None else None
            self._current_colors = np.copy(self._cleanup_backup_colors) if self._cleanup_backup_colors is not None else None
            self._current_texcoords = np.copy(self._cleanup_backup_texcoords) if self._cleanup_backup_texcoords is not None else None
            self._current_texture_path = self._cleanup_backup_texture_path
            self._last_points = self._current_points

            self._render_in_vispy_from_data(
                self._current_points, self._current_colors, self._current_faces,
                self._current_texcoords, self._current_texture_path,
                mode=2, reset_camera=False
            )
            self._sync_active_transform_to_viewport()

            if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                self.mesh_cleanup_tool_window.set_merge_revert_enabled(False)

            num_f = len(self._current_faces) if self._current_faces is not None else 0
            num_v = len(self._current_points) if self._current_points is not None else 0
            self.console_text.append(f"[CLEANUP] Reverted mesh to previous geometry ({num_v:,} vertices, {num_f:,} faces).")
            self.status_label.setText("Mesh restored to previous state.")
        else:
            self.console_text.append("[WARNING] No previous mesh backup found to revert.")

    def _on_mesh_smooth_apply(self, lambda_factor: float, iterations: int):
        """Executes Taubin Laplacian Mesh Smoothing via PyMeshLab and updates the active mesh in VisPy."""
        if self._current_points is None or len(self._current_points) == 0 or self._current_faces is None or len(self._current_faces) == 0:
            QMessageBox.warning(self, "No Mesh Available", "Please load or reconstruct a 3D textured mesh before smoothing.")
            return

        mvs_dir = self.viewer_widget.current_mvs_dir or os.path.join(get_reconstruction_out_dir(), "mvs")
        os.makedirs(mvs_dir, exist_ok=True)

        tool_in_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_in.ply")
        tool_out_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_out.ply")
        backup_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_backup.ply")

        # Create pre-operation backup for instant revert
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, backup_ply)
        self._cleanup_backup_points = np.copy(self._current_points) if self._current_points is not None else None
        self._cleanup_backup_faces = np.copy(self._current_faces) if self._current_faces is not None else None
        self._cleanup_backup_colors = np.copy(self._current_colors) if self._current_colors is not None else None
        self._cleanup_backup_texcoords = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
        self._cleanup_backup_texture_path = self._current_texture_path

        # Export working mesh to in_ply
        self._export_temp_mesh_ply(self._current_points, self._current_faces, self._current_colors, tool_in_ply)

        # Show non-blocking progress dialog
        self._cleanup_progress_dialog = ProjectProgressDialog(
            "Smooth Mesh",
            f"Applying Taubin smoothing (λ={lambda_factor:.2f}, {iterations} steps)...\nPlease wait.",
            self
        )
        self._cleanup_progress_dialog.show()

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(True)

        mu_factor = -(lambda_factor + 0.01)
        smooth_params = {
            "lambda_factor": float(lambda_factor),
            "mu_factor": float(mu_factor),
            "iterations": int(iterations)
        }

        from pipeline_manager import MeshOperationWorker
        self._cleanup_smooth_worker = MeshOperationWorker("smooth_taubin", tool_in_ply, tool_out_ply, smooth_params, parent=self)
        self._cleanup_smooth_worker.log_message.connect(self._append_log)
        self._cleanup_smooth_worker.finished.connect(
            lambda ok, out_path, msg: self._on_mesh_smooth_finished(ok, out_path, msg, lambda_factor, iterations)
        )
        self._cleanup_smooth_worker.start()

    def _on_mesh_smooth_finished(self, success: bool, output_ply: str, msg: str, lambda_factor: float, iterations: int):
        """Called when Taubin Smoothing worker completes."""
        if hasattr(self, '_cleanup_progress_dialog') and self._cleanup_progress_dialog:
            self._cleanup_progress_dialog.accept()
            self._cleanup_progress_dialog = None

        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(False)

        if success and output_ply and os.path.isfile(output_ply):
            pts, cls, fcs = _read_ply_static(output_ply)
            if pts is not None and len(pts) > 0 and fcs is not None and len(fcs) > 0:
                self._current_points = pts
                self._current_faces = fcs
                self._current_colors = cls
                self._last_points = pts

                # Re-render in VisPy without resetting camera position
                self._render_in_vispy_from_data(
                    pts, cls, fcs,
                    self._current_texcoords, self._current_texture_path,
                    mode=2, reset_camera=False
                )
                self._sync_active_transform_to_viewport()

                if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                    self.mesh_cleanup_tool_window.set_smooth_revert_enabled(True)

                num_v = len(pts)
                num_f = len(fcs)
                status_msg = f"[CLEANUP] Taubin Smoothing complete (λ={lambda_factor:.2f}, {iterations} passes, {num_v:,} vertices, {num_f:,} faces). Click 'Retexture' to project textures or 'Revert' to undo."
                self.console_text.append(status_msg)
                self.status_label.setText(f"Smooth Mesh complete ({num_v:,} vertices, {num_f:,} faces)")
            else:
                self.console_text.append("[ERROR] Mesh smoothing produced empty geometry.")
                QMessageBox.warning(self, "Smooth Mesh Failed", "Mesh smoothing did not produce valid geometry.")
        else:
            self.console_text.append(f"[ERROR] Mesh smoothing failed: {msg}")
            QMessageBox.warning(self, "Smooth Mesh Failed", f"Mesh smoothing failed:\n{msg}")

    def _on_mesh_smooth_revert(self):
        """Restores original pre-smoothing mesh geometry."""
        if hasattr(self, '_cleanup_backup_points') and self._cleanup_backup_points is not None:
            self._current_points = np.copy(self._cleanup_backup_points)
            self._current_faces = np.copy(self._cleanup_backup_faces) if self._cleanup_backup_faces is not None else None
            self._current_colors = np.copy(self._cleanup_backup_colors) if self._cleanup_backup_colors is not None else None
            self._current_texcoords = np.copy(self._cleanup_backup_texcoords) if self._cleanup_backup_texcoords is not None else None
            self._current_texture_path = self._cleanup_backup_texture_path
            self._last_points = self._current_points

            self._render_in_vispy_from_data(
                self._current_points, self._current_colors, self._current_faces,
                self._current_texcoords, self._current_texture_path,
                mode=2, reset_camera=False
            )
            self._sync_active_transform_to_viewport()

            if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                self.mesh_cleanup_tool_window.set_smooth_revert_enabled(False)

            num_f = len(self._current_faces) if self._current_faces is not None else 0
            num_v = len(self._current_points) if self._current_points is not None else 0
            self.console_text.append(f"[CLEANUP] Reverted mesh to previous geometry ({num_v:,} vertices, {num_f:,} faces).")
            self.status_label.setText("Mesh restored to previous state.")
        else:
            self.console_text.append("[WARNING] No previous mesh backup found to revert.")

    def _on_retexture_mesh_tool(self):
        """Reruns OpenMVS TextureMesh on current modified mesh using reconstruction scene.mvs."""
        import os
        mvs_dir = self.viewer_widget.current_mvs_dir or os.path.join(get_reconstruction_out_dir(), "mvs")
        
        # Verify scene.mvs candidates
        texture_input_scene = None
        for cand in ["scene_dense_mesh_refine.mvs", "scene_dense.mvs", "scene.mvs"]:
            if os.path.exists(os.path.join(mvs_dir, cand)):
                texture_input_scene = cand
                break

        if not texture_input_scene:
            QMessageBox.warning(
                self,
                "Retexture Unavailable",
                "Retexturing requires the reconstruction's project file ('scene.mvs') and source images, which were not found in this session.\n\n"
                "Note: Retexturing is only available for in-session reconstructions where project assets are intact."
            )
            self.console_text.append("[WARNING] Retexture failed: scene.mvs not found in session directory.")
            return

        # Save current preview mesh to scene_dense_mesh_modified.ply
        modified_ply = os.path.join(mvs_dir, "scene_dense_mesh_modified.ply")
        pts = self._current_points if self._current_points is not None else self._last_points
        fcs = self._current_faces
        cls = self._current_colors
        if pts is None or len(pts) == 0:
            QMessageBox.warning(self, "Retexture Unavailable", "No active mesh geometry found to retexture.")
            return

        # Reference camera mesh (ground truth from OpenMVS reconstruction)
        ref_ply = None
        for candidate in ["scene_dense_mesh_refine.ply", "scene_dense_mesh.ply"]:
            c_path = os.path.join(mvs_dir, candidate)
            if os.path.isfile(c_path):
                ref_ply = c_path
                break

        has_transform = (
            hasattr(self, '_cumulative_mesh_transform')
            and self._cumulative_mesh_transform is not None
            and not np.allclose(self._cumulative_mesh_transform, np.eye(4, dtype=np.float32), atol=1e-4)
        )

        self._retexture_restore_data = None

        if ref_ply and os.path.isfile(ref_ply):
            try:
                ref_pts, _, _ = _read_ply_static(ref_ply)
                if ref_pts is not None and len(ref_pts) > 0:
                    C_ref = np.mean(ref_pts, axis=0).astype(np.float64)
                    C_user = np.mean(pts, axis=0).astype(np.float64)

                    R_vp = np.eye(3, dtype=np.float64)
                    if has_transform:
                        R_vp = self._cumulative_mesh_transform[:3, :3].astype(np.float64)

                    # Align user mesh back to original camera pose for OpenMVS texturing
                    export_pts = (((pts.astype(np.float64) - C_user) @ R_vp) + C_ref).astype(np.float32)

                    # Compute restoration matrix for raw OpenMVS coordinates:
                    # Conversion from viewport space (photogrammetry coordinate flip: Y->-Y, Z->-Z)
                    F = np.diag([1.0, -1.0, -1.0])
                    C_ref_raw = C_ref @ F
                    C_user_raw = C_user @ F
                    R_raw = F @ R_vp @ F
                    t_raw = C_user_raw - (C_ref_raw @ R_raw.T)

                    M_raw = np.eye(4, dtype=np.float32)
                    M_raw[:3, :3] = R_raw.astype(np.float32)
                    M_raw[:3, 3] = t_raw.astype(np.float32)

                    self._retexture_restore_data = {
                        "M_raw": M_raw,
                        "C_user": C_user,
                        "C_ref": C_ref
                    }
                    self.console_text.append("[TOOLS] Mesh aligned with original reconstruction camera space for OpenMVS texturing.")
                else:
                    export_pts = pts
            except Exception as align_err:
                self.console_text.append(f"[WARNING] Reference alignment calculation failed: {align_err}")
                export_pts = pts
        elif has_transform:
            self.console_text.append("[TOOLS] Temporarily aligning mesh with original camera poses for OpenMVS texturing...")
            try:
                T_inv = np.linalg.inv(self._cumulative_mesh_transform)
                export_pts = (pts @ T_inv[:3, :3].T) + T_inv[:3, 3]
            except Exception as inv_err:
                self.console_text.append(f"[WARNING] Could not invert cumulative transform: {inv_err}")
                export_pts = pts
        else:
            export_pts = pts

        # Textured OBJ loaders commonly expand every UV corner into a separate
        # vertex. The geometry looks identical, but OpenMVS then sees every face
        # as a disconnected texture patch and atlas generation becomes extremely
        # slow. Retexturing discards the old UVs, so exact-position duplicates can
        # be safely welded to restore the original triangle adjacency.
        texture_pts, texture_faces, texture_colors = _weld_exact_duplicate_vertices(
            export_pts, fcs, cls
        )
        welded_count = len(export_pts) - len(texture_pts)
        if welded_count > 0:
            self.console_text.append(
                f"[TOOLS] Optimized mesh connectivity for texturing: "
                f"{len(export_pts):,} → {len(texture_pts):,} vertices "
                f"({welded_count:,} duplicate texture-corner vertices welded)."
            )

        self._export_temp_mesh_ply(
            texture_pts, texture_faces, texture_colors, modified_ply
        )

        # Update UI: Disable Start Reconstruction button, lock tool modals & animate shared progress bar
        self.process_btn.setEnabled(False)
        self.viewer_widget.set_tool_modals_busy(True)
        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(True)
        if hasattr(self, 'mesh_cut_tool_window') and self.mesh_cut_tool_window:
            self.mesh_cut_tool_window.set_busy(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Retexturing mesh...")
        self.console_text.append("[START] Running OpenMVS TextureMesh on modified mesh...")

        from pipeline_manager import RetextureOnlyWorker
        texture_res = str(self.custom_texture_res_combo.currentIndex()) if self.custom_settings_toggle.isChecked() else "1"
        quality_preset = ["preview", "medium", "high", "ultra"][self.quality_combo.currentIndex()] if hasattr(self, 'quality_combo') else "medium"

        self.retexture_worker = RetextureOnlyWorker(
            output_dir=mvs_dir,
            target_mesh_ply="scene_dense_mesh_modified.ply",
            custom_params={"texture_res": texture_res},
            quality_preset=quality_preset,
            parent=self
        )
        self.retexture_worker.log_message.connect(self._append_log)
        self.retexture_worker.status_changed.connect(self.status_label.setText)
        self.retexture_worker.finished.connect(self._on_mesh_tool_retexture_finished)
        self.retexture_worker.start()

    def _on_mesh_tool_retexture_finished(self, success: bool, msg: str):
        """Handles completion of retexturing worker and reloads the textured mesh into the viewport."""
        import os
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        self._set_process_btn_state("ready" if len(self.image_list) > 0 else "idle")
        self.viewer_widget.set_tool_modals_busy(False)
        if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
            self.mesh_cleanup_tool_window.set_busy(False)
        if hasattr(self, 'mesh_cut_tool_window') and self.mesh_cut_tool_window:
            self.mesh_cut_tool_window.set_busy(False)

        if success:
            self.status_label.setText("Retexturing Complete")
            self.console_text.append("[SUCCESS] Retexturing completed. Reloading textured mesh in viewport...")
            mvs_dir = self.viewer_widget.current_mvs_dir or os.path.join(get_reconstruction_out_dir(), "mvs")
            obj_path = os.path.join(mvs_dir, "scene_dense_mesh_texture.obj")
            ply_texture_path = os.path.join(mvs_dir, "scene_dense_mesh_texture.ply")

            # Restore user's custom position/rotation onto the newly textured model
            restore_data = getattr(self, '_retexture_restore_data', None)
            if restore_data is not None and "M_raw" in restore_data:
                M_raw = restore_data["M_raw"]
                if not np.allclose(M_raw, np.eye(4, dtype=np.float32), atol=1e-4):
                    self.console_text.append("[TOOLS] Restoring custom mesh position onto newly textured model...")
                    if os.path.exists(obj_path):
                        self._apply_transform_to_obj_file(obj_path, M_raw)
                    if os.path.exists(ply_texture_path):
                        self._apply_transform_to_ply_file(ply_texture_path, M_raw)
                # Invalidate stale GLB cache so next export re-converts the transformed OBJ
                glb_path = os.path.join(mvs_dir, "scene_dense_mesh_texture.glb")
                if os.path.exists(glb_path):
                    try:
                        os.remove(glb_path)
                    except Exception:
                        pass
                # Reset cumulative transform since the textured model geometry is now at the user position
                self._cumulative_mesh_transform = np.eye(4, dtype=np.float32)
                self._retexture_restore_data = None
            else:
                has_transform = (
                    hasattr(self, '_cumulative_mesh_transform')
                    and self._cumulative_mesh_transform is not None
                    and not np.allclose(self._cumulative_mesh_transform, np.eye(4, dtype=np.float32), atol=1e-4)
                )
                if has_transform:
                    self.console_text.append("[TOOLS] Restoring custom mesh transform onto newly textured model...")
                    if os.path.exists(obj_path):
                        self._apply_transform_to_obj_file(obj_path, self._cumulative_mesh_transform)
                    if os.path.exists(ply_texture_path):
                        self._apply_transform_to_ply_file(ply_texture_path, self._cumulative_mesh_transform)
                    glb_path = os.path.join(mvs_dir, "scene_dense_mesh_texture.glb")
                    if os.path.exists(glb_path):
                        try:
                            os.remove(glb_path)
                        except Exception:
                            pass
                    self._cumulative_mesh_transform = np.eye(4, dtype=np.float32)

            if os.path.exists(obj_path):
                self._reload_viewer(obj_path)
            else:
                mesh_path = self.viewer_widget.get_selected_file_path()
                if mesh_path:
                    self._reload_viewer(mesh_path)

            # Commit newly textured mesh as the active baseline backup on disk and in memory
            backup_ply = os.path.join(mvs_dir, "scene_dense_mesh_cleanup_backup.ply")
            if self._current_points is not None and len(self._current_points) > 0 and self._current_faces is not None:
                try:
                    self._export_temp_mesh_ply(
                        self._current_points, self._current_faces, self._current_colors, backup_ply
                    )
                except Exception as bkp_err:
                    self.console_text.append(f"[WARNING] Could not update cleanup backup PLY: {bkp_err}")

                self._cleanup_backup_points = np.copy(self._current_points)
                self._cleanup_backup_faces = np.copy(self._current_faces)
                self._cleanup_backup_colors = np.copy(self._current_colors) if self._current_colors is not None else None
                self._cleanup_backup_texcoords = np.copy(self._current_texcoords) if self._current_texcoords is not None else None
                self._cleanup_backup_texture_path = self._current_texture_path
                self.console_text.append("[CLEANUP] Backup updated to newly retextured mesh baseline.")

            # Also update legacy preop backup if present
            preop_backup_path = os.path.join(mvs_dir, "scene_dense_mesh_preop_backup.ply")
            if os.path.exists(preop_backup_path) and self._current_points is not None and self._current_faces is not None:
                try:
                    self._export_temp_mesh_ply(
                        self._current_points, self._current_faces, self._current_colors, preop_backup_path
                    )
                except Exception:
                    pass

            # Reset revert buttons in Mesh Cleanup tool window (new textured state is now the baseline)
            if hasattr(self, 'mesh_cleanup_tool_window') and self.mesh_cleanup_tool_window:
                self.mesh_cleanup_tool_window.set_revert_enabled(False)
                self.mesh_cleanup_tool_window.set_holes_revert_enabled(False)
                self.mesh_cleanup_tool_window.set_nonmanifold_revert_enabled(False)
                self.mesh_cleanup_tool_window.set_duplicates_revert_enabled(False)
        else:
            self.status_label.setText("Retexturing Failed")
            self.console_text.append(f"[ERROR] Retexturing failed: {msg}")

    def _apply_transform_to_obj_file(self, obj_path: str, T_mat: np.ndarray):
        """Transforms vertices and vertex normals in an OBJ file in-place, preserving UVs and materials."""
        if not os.path.isfile(obj_path) or np.allclose(T_mat, np.eye(4, dtype=np.float32), atol=1e-4):
            return
        try:
            R = T_mat[:3, :3]
            t = T_mat[:3, 3]

            with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        vx, vy, vz = float(parts[1]), float(parts[2]), float(parts[3])
                        nx = vx * R[0, 0] + vy * R[0, 1] + vz * R[0, 2] + t[0]
                        ny = vx * R[1, 0] + vy * R[1, 1] + vz * R[1, 2] + t[1]
                        nz = vx * R[2, 0] + vy * R[2, 1] + vz * R[2, 2] + t[2]
                        new_lines.append(f"v {nx:.6f} {ny:.6f} {nz:.6f}\n")
                    else:
                        new_lines.append(line)
                elif line.startswith('vn '):
                    parts = line.split()
                    if len(parts) >= 4:
                        vx, vy, vz = float(parts[1]), float(parts[2]), float(parts[3])
                        nx = vx * R[0, 0] + vy * R[0, 1] + vz * R[0, 2]
                        ny = vx * R[1, 0] + vy * R[1, 1] + vz * R[1, 2]
                        nz = vx * R[2, 0] + vy * R[2, 1] + vz * R[2, 2]
                        norm = (nx * nx + ny * ny + nz * nz) ** 0.5
                        if norm > 1e-6:
                            nx /= norm
                            ny /= norm
                            nz /= norm
                        new_lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            with open(obj_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            self.console_text.append(f"[WARNING] Failed to apply transform to OBJ file: {e}")

    def _apply_transform_to_ply_file(self, ply_path: str, T_mat: np.ndarray):
        """Transforms vertex coordinates in an OpenMVS binary/ASCII PLY file in-place."""
        if not os.path.isfile(ply_path) or np.allclose(T_mat, np.eye(4, dtype=np.float32), atol=1e-4):
            return
        try:
            R = T_mat[:3, :3]
            t = T_mat[:3, 3]

            with open(ply_path, 'rb') as f:
                data = f.read()

            end_hdr_idx = data.find(b"end_header\n")
            if end_hdr_idx == -1:
                end_hdr_idx = data.find(b"end_header\r\n")
                if end_hdr_idx == -1:
                    return
                hdr_len = end_hdr_idx + len(b"end_header\r\n")
            else:
                hdr_len = end_hdr_idx + len(b"end_header\n")

            header_str = data[:hdr_len].decode('latin1', errors='ignore')
            num_vertices = 0
            is_binary = "format binary_little_endian" in header_str

            for line in header_str.splitlines():
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[2])
                    break

            if num_vertices > 0 and is_binary:
                v_bytes = data[hdr_len : hdr_len + num_vertices * 12]
                if len(v_bytes) == num_vertices * 12:
                    pts = np.frombuffer(v_bytes, dtype=np.float32).reshape(-1, 3)
                    pts_trans = (pts @ R.T) + t
                    new_v_bytes = pts_trans.astype(np.float32).tobytes()
                    with open(ply_path, 'wb') as f:
                        f.write(data[:hdr_len])
                        f.write(new_v_bytes)
                        f.write(data[hdr_len + num_vertices * 12 :])
        except Exception as e:
            self.console_text.append(f"[WARNING] Failed to apply transform to PLY file: {e}")

    def _render_in_vispy(self, file_path, mode):
        import numpy as np
        from PIL import Image
        
        self._clear_visuals()
        
        points = None
        colors = None
        faces = None
        texcoords = None
        texture_path = None
        
        if mode == 0:
            # Sparse Point Cloud & Cameras
            mvs_dir = self.viewer_widget.current_mvs_dir
            if not mvs_dir and file_path:
                mvs_dir = os.path.dirname(file_path) if os.path.basename(file_path).endswith(('.mvs', '.ply', '.obj')) else file_path
            if not mvs_dir:
                mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
                
            output_dir = os.path.dirname(mvs_dir) if os.path.basename(mvs_dir) == 'mvs' else mvs_dir
            
            points_bin_candidates = [
                os.path.join(output_dir, "colmap", "sparse", "points3D.bin"),
                os.path.join(output_dir, "colmap", "sparse", "0", "points3D.bin"),
                os.path.join(mvs_dir, "points3D.bin"),
                os.path.join(output_dir, "points3D.bin"),
                os.path.join(output_dir, "colmap", "sparse", "points3D.txt"),
                os.path.join(output_dir, "colmap", "sparse", "0", "points3D.txt"),
            ]
            points_bin = None
            for p in points_bin_candidates:
                if os.path.exists(p):
                    points_bin = p
                    break
                    
            if points_bin and points_bin.endswith(".bin"):
                points, colors = self._read_points3d_binary(points_bin)
            else:
                scene_ply_candidates = [
                    file_path if (file_path and file_path.lower().endswith(('.ply', '.pcd', '.xyz'))) else None,
                    os.path.join(mvs_dir, "scene.ply"),
                    os.path.join(output_dir, "scene.ply"),
                    os.path.join(mvs_dir, "scene_dense.ply"),
                    os.path.join(output_dir, "scene_dense.ply"),
                ]
                for sp in scene_ply_candidates:
                    if sp and os.path.exists(sp):
                        pts, cls, _ = self._read_ply(sp)
                        if pts is not None and len(pts) > 0:
                            points, colors = pts, cls
                            break

            # Decimate raw point cloud for sparse view if not already from COLMAP binary
            if points is not None and len(points) > 0 and not (points_bin and points_bin.endswith(".bin")):
                stride = max(2, len(points) // 25000) if len(points) > 1000 else 1
                points = points[::stride]
                if colors is not None and len(colors) > 0:
                    colors = colors[::stride]
                    
            images_bin_candidates = [
                os.path.join(output_dir, "colmap", "sparse", "images.bin"),
                os.path.join(output_dir, "colmap", "sparse", "0", "images.bin"),
                os.path.join(mvs_dir, "images.bin"),
                os.path.join(output_dir, "images.bin"),
                os.path.join(output_dir, "colmap", "sparse", "images.txt"),
                os.path.join(output_dir, "colmap", "sparse", "0", "images.txt"),
            ]
            images_bin = None
            for i in images_bin_candidates:
                if os.path.exists(i):
                    images_bin = i
                    break
                    
            if images_bin and images_bin.endswith(".bin"):
                cameras_data = self._read_images_binary(images_bin)
                if cameras_data:
                    self._draw_cameras(cameras_data)
                    
        elif mode == 1:
            # Dense Point Cloud
            ply_path = file_path.replace(".mvs", ".ply")
            if not os.path.exists(ply_path):
                ply_path = file_path
            if os.path.exists(ply_path):
                ext = os.path.splitext(ply_path)[1].lower()
                if ext == ".ply":
                    points, colors, _ = self._read_ply(ply_path)
                else:
                    try:
                        import point_cloud_io
                        res = point_cloud_io.load_point_cloud(ply_path)
                        if res.success and res.cloud is not None:
                            points = np.asarray(res.cloud.points, dtype=np.float32)
                            if res.has_colors:
                                colors = (np.asarray(res.cloud.colors) * 255.0).astype(np.uint8)
                            else:
                                colors = np.ones((len(points), 3), dtype=np.uint8) * 180
                    except Exception as e:
                        self.console_text.append(f"[WARNING] Vispy fallback loader failed: {e}")
                
        elif mode == 2:
            # Textured Mesh
            if file_path.lower().endswith(".obj"):
                vertices, texcoords, faces, texture_path = self._read_obj(file_path)
                points = vertices
            elif file_path.lower().endswith(".ply"):
                points, colors, faces = self._read_ply(file_path)
            else:
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in [".obj", ".ply"]:
                    obj_cand = os.path.splitext(file_path)[0] + ".obj"
                    if os.path.exists(obj_cand):
                        vertices, texcoords, faces, texture_path = self._read_obj(obj_cand)
                        points = vertices
                    else:
                        points, colors, faces = self._read_ply(file_path)
                    
        if mode == 2 and faces is not None and len(faces) > 0:
            mesh_colors = None
            if colors is not None and len(colors) > 0:
                mesh_colors = colors.astype(np.float32) / 255.0
                
            self.mesh_visual = scene.visuals.Mesh(
                vertices=points,
                faces=faces,
                vertex_colors=mesh_colors,
                color='white',
                parent=self.view.scene
            )
            
            if texture_path and texcoords is not None and len(texcoords) > 0:
                try:
                    texture_image = np.array(Image.open(texture_path))
                    from vispy.visuals.filters import TextureFilter
                    tex_filter = TextureFilter(texture_image, texcoords)
                    self.mesh_visual.attach(tex_filter)
                except Exception as tex_err:
                    self.console_text.append(f"[WARNING] Could not apply texture filter: {tex_err}")
            # Store geometry so wireframe overlay can build edges
            self._last_wf_vertices = points.astype(np.float32)
            self._last_wf_faces = faces.astype(np.uint32)
            self._apply_shading_mode_to_mesh()
                    
        elif points is not None and len(points) > 0:
            marker_colors = None
            if colors is not None and len(colors) > 0:
                marker_colors = colors.astype(np.float32) / 255.0
                if marker_colors.shape[1] == 3:
                    alphas = np.ones((marker_colors.shape[0], 1), dtype=np.float32)
                    marker_colors = np.hstack([marker_colors, alphas])
            else:
                marker_colors = 'white'
                
            self.markers_visual = scene.visuals.Markers(parent=self.view.scene)
            point_size = 4 if mode == 0 else 2
            self.markers_visual.set_data(
                pos=points,
                face_color=marker_colors,
                size=point_size,
                edge_width=0
            )
            
        elif hasattr(self, 'cameras_visual') and self.cameras_visual is not None:
            # Cameras exist even if sparse points array is empty
            pass
        else:
            self.canvas.native.hide()
            self.viewer_widget.fallback_label.setText("No 3D data found to render.")
            self.viewer_widget.fallback_label.show()
            return
            
        self.canvas.native.show()
        self.viewer_widget.fallback_label.hide()
        
        # Track active geometry arrays
        self._current_points = points
        self._current_colors = colors
        self._current_faces = faces
        self._current_texcoords = texcoords
        self._current_texture_path = texture_path
        self._last_points = points
        
        # Store raw uncropped geometry arrays
        if self._raw_points is None:
            self._raw_points = np.copy(points) if points is not None else None
            self._raw_colors = np.copy(colors) if colors is not None else None
            self._raw_faces = np.copy(faces) if faces is not None else None
            self._raw_texcoords = np.copy(texcoords) if texcoords is not None else None
            self._raw_texture_path = texture_path
        
        if points is not None and len(points) > 0:
            # Calculate scene bounding box
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            center = (min_bound + max_bound) / 2.0
            scale = np.max(max_bound - min_bound)
            
            # Reset camera to fit model
            self.view.camera.center = center
            self.view.camera.distance = max(0.1, scale * 1.5)
            self.view.camera.elevation = 30
            self.view.camera.azimuth = 45
            self.view.camera.up = '+y'
            
            # Update auto-scaling ground plane grid
            self._update_ground_grid(points)
            
        elif hasattr(self, 'cameras_visual') and self.cameras_visual is not None:
            ref_pts = getattr(self, '_last_camera_positions', None)
            if ref_pts is None:
                ref_pts = np.array([[0, 0, 0]], dtype=np.float32)
            min_bound = np.min(ref_pts, axis=0)
            max_bound = np.max(ref_pts, axis=0)
            center = (min_bound + max_bound) / 2.0
            scale = np.max(max_bound - min_bound)
            
            self.view.camera.center = center
            self.view.camera.distance = max(1.0, scale * 2.0)
            self.view.camera.elevation = 30
            self.view.camera.azimuth = 45
            self.view.camera.up = '+y'
            
            # Update auto-scaling ground plane grid
            self._update_ground_grid(ref_pts)

        self.canvas.update()
        self._sync_active_transform_to_viewport()

    def _on_cloud_import_done(self, res, file_path):
        """Called on the UI thread when background cloud import finishes."""
        if res.success:
            self.console_text.append(
                f"[STANDALONE] Point cloud ready. Points: {res.point_count:,} | "
                f"Colors: {res.has_colors} | BBox diag: {res.bbox_diagonal:.2f}"
            )
            # Refine vertex color toggle now that we have the definitive answer
            self.vertex_color_toggle.setChecked(res.has_colors)
            self.vertex_color_toggle.setVisible(res.has_colors)
        else:
            warnings_str = ', '.join(res.warnings) if res.warnings else 'unknown error'
            self.console_text.append(f"[WARNING] Point cloud load issues: {warnings_str}")

    def _reload_viewer(self, file_path):
        self.viewer_widget.update_crop_box_state()
        if not os.path.exists(file_path):
            self.viewer_widget.fallback_label.setText(
                f"File not found: {os.path.basename(file_path)}\nRun reconstruction to generate this file first."
            )
            self.viewer_widget.fallback_label.show()
            self.console_text.append(f"[WARNING] 3D file not found: {file_path}")
            return

        self._raw_points = None
        self._raw_colors = None
        self._raw_faces = None
        self._raw_texcoords = None
        self._raw_texture_path = None

        mode = self.viewer_widget.mode_select.currentIndex()
        mode_names = ["Sparse Point Cloud", "Dense Point Cloud", "Textured Mesh"]
        mode_name = mode_names[mode] if mode < len(mode_names) else "3D Scene"

        self.viewer_widget.fallback_label.setText(f"Loading {mode_name}...\n(Parsing geometry in background…)")
        self.viewer_widget.fallback_label.show()
        if self.canvas and self.canvas.native:
            self.canvas.native.hide()
        QApplication.processEvents()

        # For mode 0 (sparse), keep existing sync path (reads COLMAP binary, fast)
        # For OBJ files (mode 2), keep sync path (OBJ reader is already needed on UI thread)
        ext = os.path.splitext(file_path)[1].lower()
        use_background = mode in (1, 2) and ext != '.obj'

        if use_background:
            # Spawn background parser — UI stays responsive
            self._viewer_load_worker = ViewerLoadWorker(file_path, mode, parent=self)
            self._viewer_load_worker.finished.connect(
                lambda pts, cols, fcs, tcs, tex: self._on_viewer_data_ready(pts, cols, fcs, tcs, tex, file_path, mode)
            )
            self._viewer_load_worker.error.connect(lambda err: (
                self.console_text.append(f"[ERROR] VisPy background load failed: {err}"),
                self.viewer_widget.fallback_label.setText(f"Rendering failed:\n{err}"),
                self.viewer_widget.fallback_label.show()
            ))
            self._viewer_load_worker.start()
        else:
            # Sync path for sparse (mode 0) and OBJ (mode 2)
            try:
                self._render_in_vispy(file_path, mode)
                self.console_text.append(f"[INFO] Successfully rendered {os.path.basename(file_path)} in VisPy canvas.")
            except Exception as e:
                self.console_text.append(f"[ERROR] VisPy rendering failed: {e}")
                self.viewer_widget.fallback_label.setText(f"Rendering failed:\n{e}")
                self.viewer_widget.fallback_label.show()
                if self.canvas and self.canvas.native:
                    self.canvas.native.hide()

    def _on_viewer_data_ready(self, points, colors, faces, texcoords, texture_path, file_path, mode):
        """Called on the UI thread once ViewerLoadWorker has finished parsing."""
        try:
            self._render_in_vispy_from_data(points, colors, faces, texcoords, texture_path, mode)
            self.console_text.append(f"[INFO] Successfully rendered {os.path.basename(file_path)} in VisPy canvas.")
        except Exception as e:
            self.console_text.append(f"[ERROR] VisPy rendering failed: {e}")
            self.viewer_widget.fallback_label.setText(f"Rendering failed:\n{e}")
            self.viewer_widget.fallback_label.show()
            if self.canvas and self.canvas.native:
                self.canvas.native.hide()

    def _terminate_viewer(self):
        self._clear_visuals()
        self.viewer_widget.fallback_label.setText("3D Viewer Idle")
        self.viewer_widget.fallback_label.show()
        if self.canvas and self.canvas.native:
            self.canvas.native.hide()

    def _choose_bg_color(self):
        from PySide6.QtGui import QColor
        current_color = QColor(self.viewport_bg_color)
        color = QColorDialog.getColor(current_color, self, "Select Viewport Background Color")
        if color.isValid():
            hex_color = color.name()
            self.viewport_bg_color = hex_color
            if self.canvas:
                self.canvas.bgcolor = hex_color
                self.canvas.update()

    def closeEvent(self, event):
        self._terminate_viewer()
        if hasattr(self, 'loopback_server') and self.loopback_server:
            try:
                self.loopback_server.stop()
            except Exception:
                pass
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self._position_overlay()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlay()

    def _position_overlay(self):
        container_w = self.viewer_widget.container_area.width()
        container_h = self.viewer_widget.container_area.height()
        if hasattr(self, 'selection_overlay') and self.selection_overlay is not None:
            self.selection_overlay.setGeometry(0, 0, container_w, container_h)
            if self.selection_overlay.isVisible():
                self.selection_overlay.raise_()
        if hasattr(self, 'nav_gizmo') and self.nav_gizmo is not None:
            margin = 12
            gx = container_w - self.nav_gizmo.width() - margin
            gy = margin
            self.nav_gizmo.move(max(margin, gx), gy)
            self.nav_gizmo.raise_()
        if hasattr(self, 'floating_toolbox') and self.floating_toolbox is not None:
            if self.floating_toolbox.isVisible():
                max_x = max(0, container_w - self.floating_toolbox.width())
                max_y = max(0, container_h - self.floating_toolbox.height())
                cur_x = min(self.floating_toolbox.x(), max_x)
                cur_y = min(self.floating_toolbox.y(), max_y)
                self.floating_toolbox.move(cur_x, cur_y)
                self.floating_toolbox.raise_()
        if hasattr(self, 'editor_tool_host') and self.editor_tool_host is not None:
            if self.editor_tool_host.isVisible():
                self.editor_tool_host.reposition()
                self.editor_tool_host.raise_()
        if hasattr(self, 'overlay_label') and self.overlay_label.isVisible():
            label_w = self.overlay_label.width()
            label_h = self.overlay_label.height()
            if label_w <= 16 or label_h <= 16:
                label_size = self.overlay_label.sizeHint()
                label_w = label_size.width()
                label_h = label_size.height()
                
            margin = 15
            x = container_w - label_w - margin
            y = container_h - label_h - margin
            self.overlay_label.setGeometry(x, y, label_w, label_h)
            self.overlay_label.raise_()

    def _on_vispy_camera_transform_changed(self, event=None):
        """Synchronizes the 3D navigation gizmo orientation when the VisPy turntable camera moves."""
        if hasattr(self, 'nav_gizmo') and self.nav_gizmo is not None and hasattr(self, 'view') and self.view.camera is not None:
            try:
                az = float(getattr(self.view.camera, 'azimuth', 45.0))
                el = float(getattr(self.view.camera, 'elevation', 30.0))
                self.nav_gizmo.update_from_vispy(az, el)
            except Exception:
                pass

    def _on_vispy_nav_gizmo_snap(self, view_name: str):
        """Snaps the VisPy TurntableCamera to canonical views when clicking nav gizmo axis dots."""
        if not hasattr(self, 'view') or self.view.camera is None:
            return
        
        _SNAP_ANGLES = {
            "front":  (0.0,   0.0),
            "back":   (180.0, 0.0),
            "right":  (90.0,  0.0),
            "left":   (-90.0, 0.0),
            "top":    (0.0,   89.9),
            "bottom": (0.0,  -89.9),
        }
        
        if view_name in _SNAP_ANGLES:
            az, el = _SNAP_ANGLES[view_name]
            try:
                self.view.camera.azimuth = az
                self.view.camera.elevation = el
                if hasattr(self.view.camera, 'roll'):
                    self.view.camera.roll = 0.0
                if hasattr(self, 'nav_gizmo') and self.nav_gizmo:
                    self.nav_gizmo.update_from_vispy(az, el)
                if self.canvas:
                    self.canvas.update()
            except Exception as err:
                print(f"[GIZMO] Error snapping camera: {err}")

    def _on_show_controls_changed(self, state):
        visible = (state == Qt.Checked.value or state == 2)
        self.overlay_label.setVisible(visible)
        if visible:
            self._update_overlay_content()
            self._position_overlay()

    def _update_overlay_content(self):
        self.overlay_label.setText(DEFAULT_CAMERA_CONTROLS)
        self.overlay_label.adjustSize()

    def _upload_to_proximap(self):
        mvs_out = self._get_active_mvs_dir()
        
        src_glb = os.path.join(mvs_out, "scene_dense_mesh_texture.glb")
        src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
        
        if not os.path.exists(src_glb):
            if os.path.exists(src_obj):
                self.console_text.append("[BRIDGE] Pre-converting reconstructed OBJ to GLB for upload...")
                try:
                    import trimesh
                    mesh_obj = trimesh.load(src_obj)
                    mesh_obj.export(src_glb, file_type="glb")
                except Exception as e:
                    self.console_text.append(f"[BRIDGE ERROR] Could not convert model to GLB: {e}")
                    return
            else:
                src_ply = None
                for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
                    path = os.path.join(mvs_out, candidate)
                    if os.path.exists(path):
                        src_ply = path
                        break
                if src_ply:
                    self.console_text.append("[BRIDGE] Pre-converting reconstructed PLY to GLB for upload...")
                    try:
                        import trimesh
                        mesh_ply = trimesh.load(src_ply, force="mesh")
                        mesh_ply.export(src_glb, file_type="glb")
                    except Exception as e:
                        self.console_text.append(f"[BRIDGE ERROR] Could not convert model to GLB: {e}")
                        return
                else:
                    self.console_text.append("[BRIDGE ERROR] No reconstructed mesh found. Please run reconstruction first.")
                    return

        self.console_text.append(f"[BRIDGE] Initializing local server to host model: {src_glb}")
        
        if hasattr(self, 'loopback_server') and self.loopback_server:
            try:
                self.loopback_server.stop()
            except Exception:
                pass
                
        import random
        port = random.randint(53120, 53200)
        self.loopback_server = LoopbackServerThread(src_glb, port=port)
        self.loopback_server.start()
        
        import time
        time.sleep(0.5)
        
        actual_port = self.loopback_server.port
        local_url = f"http://127.0.0.1:{actual_port}/model.glb"
        
        try:
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(mvs_out)))
        except Exception:
            folder_name = "Reconstructed_Space"
            
        model_name = folder_name if folder_name else "Reconstructed_Space"
        
        import urllib.parse
        encoded_url = urllib.parse.quote(local_url, safe='')
        encoded_name = urllib.parse.quote(model_name, safe='')
        bridge_url = f"https://proximap.space/upload-bridge?local_url={encoded_url}&name={encoded_name}"
        
        self.console_text.append(f"[BRIDGE] Directing system browser to: {bridge_url}")
        import webbrowser
        webbrowser.open(bridge_url)
        
        # Show progress dialog modally
        dialog = UploadProgressDialog(self)
        dialog.exec()
        
        # Stop loopback server when user clicks "Done"
        self.console_text.append("[BRIDGE] Upload dialog closed. Terminating local server...")
        if hasattr(self, 'loopback_server') and self.loopback_server:
            try:
                self.loopback_server.stop()
            except Exception:
                pass
            self.loopback_server = None

    def _upload_mesh_editor_scene(self):
        if not self.mesh_editor_tab or not self.mesh_editor_tab.viewport.scene.objects:
            QMessageBox.warning(
                self, "Upload Warning", "There are no objects in the scene to upload."
            )
            return
            
        # Create a temp path inside the reconstruction_out directory
        output_dir = get_reconstruction_out_dir()
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_glb_path = os.path.join(temp_dir, "temp_editor_baked.glb")
        
        # BAKE the transform by exporting to GLB
        from mesh_editor.scene import export_scene_to_file
        try:
            export_scene_to_file(self.mesh_editor_tab.viewport.scene, temp_glb_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Baking/Export Error", f"Failed to bake and prepare model for upload:\n{str(e)}"
            )
            return
            
        # Start local loopback server to host this temp baked GLB
        self.console_text.append(f"[BRIDGE] Initializing local server to host baked model: {temp_glb_path}")
        
        if hasattr(self, 'loopback_server') and self.loopback_server:
            try:
                self.loopback_server.stop()
            except Exception:
                pass
                
        import random
        port = random.randint(53120, 53200)
        self.loopback_server = LoopbackServerThread(temp_glb_path, port=port)
        self.loopback_server.start()
        
        import time
        time.sleep(0.5)
        
        actual_port = self.loopback_server.port
        local_url = f"http://127.0.0.1:{actual_port}/model.glb"
        
        model_name = "Mesh_Editor_Space"
        
        import urllib.parse
        encoded_url = urllib.parse.quote(local_url, safe='')
        encoded_name = urllib.parse.quote(model_name, safe='')
        bridge_url = f"https://proximap.space/upload-bridge?local_url={encoded_url}&name={encoded_name}"
        
        self.console_text.append(f"[BRIDGE] Directing system browser to: {bridge_url}")
        import webbrowser
        webbrowser.open(bridge_url)
        
        # Show progress dialog modally
        dialog = UploadProgressDialog(self)
        dialog.exec()
        
        # Stop loopback server when user clicks "Done"
        self.console_text.append("[BRIDGE] Upload dialog closed. Terminating local server...")
        if hasattr(self, 'loopback_server') and self.loopback_server:
            try:
                self.loopback_server.stop()
            except Exception:
                pass
        self.loopback_server = None
            
        # Clean up the temp file
        try:
            if os.path.exists(temp_glb_path):
                os.remove(temp_glb_path)
        except Exception:
            pass

    # =========================================================================
    # MOBILE DEVICE BRIDGE (IMPORT & EXPORT OVER LOCAL NETWORK)
    # =========================================================================
    def _on_import_from_mobile_clicked(self):
        default_save_path = os.path.join(get_reconstruction_out_dir(), "mobile_imports")
        os.makedirs(default_save_path, exist_ok=True)

        setup_dialog = MobileImportSetupDialog(default_path=default_save_path, parent=self)
        if setup_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        save_dir = setup_dialog.get_save_dir()
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)

        from mobile_bridge_server import MobileBridgeServer
        self.received_mobile_files = []
        self.mobile_server = MobileBridgeServer(save_dir=save_dir, mode="import", parent=self)
        
        self.mobile_server.file_received.connect(self._on_mobile_file_received)
        self.mobile_server.all_files_received.connect(self._on_mobile_all_files_received)
        self.mobile_server.error.connect(self._on_mobile_server_error)
        
        self.mobile_server.start()
        
        # Remove the 150ms sleep — port is now bound in __init__, no race condition
        urls = self.mobile_server.get_urls()
        
        self.console_text.append(f"[MOBILE BRIDGE] Server listening at: {', '.join(urls)}")
        
        self.mobile_qr_dialog = MobileQRDialog(urls=urls, mode="import", parent=self)
        
        if self.mobile_qr_dialog.exec() == QDialog.DialogCode.Rejected:
            if hasattr(self, 'mobile_server') and self.mobile_server:
                self.mobile_server.stop()
                self.mobile_server = None
            self.console_text.append("[MOBILE BRIDGE] Mobile import session cancelled.")

    def _on_mobile_file_received(self, file_path: str):
        self.console_text.append(f"[MOBILE BRIDGE] Received file: {os.path.basename(file_path)}")
        if not hasattr(self, 'received_mobile_files'):
            self.received_mobile_files = []
        self.received_mobile_files.append(file_path)
        
        if hasattr(self, 'mobile_qr_dialog') and self.mobile_qr_dialog:
            self.mobile_qr_dialog.status_lbl.setText(f"Receiving files... ({len(self.received_mobile_files)} uploaded)")

    def _on_mobile_all_files_received(self, files: list):
        self.console_text.append(f"[MOBILE BRIDGE] Transfer complete! Received {len(files)} media file(s).")
        
        if hasattr(self, 'mobile_qr_dialog') and self.mobile_qr_dialog:
            self.mobile_qr_dialog.accept()
            self.mobile_qr_dialog = None

        if hasattr(self, 'mobile_server') and self.mobile_server:
            self.mobile_server.stop()
            self.mobile_server = None
            
        if files:
            images = []
            videos = []
            ply_files = []
            for f in files:
                normalized = os.path.normpath(f)
                ext = os.path.splitext(normalized)[1].lower()
                if ext in IMAGE_EXTS:
                    images.append(normalized)
                elif ext in VIDEO_EXTS:
                    videos.append(normalized)
                elif ext == '.ply':
                    ply_files.append(normalized)
            
            if ply_files:
                self._load_standalone_point_cloud(ply_files[0])
                if len(ply_files) > 1:
                    self.console_text.append(f"[MOBILE BRIDGE] Note: Multiple .ply files received. Loaded first: {os.path.basename(ply_files[0])}")
            if images or videos:
                self._route_import(images, videos, append_to_existing=True)

    def _on_mobile_server_error(self, err_msg: str):
        self.console_text.append(f"[MOBILE BRIDGE ERROR] {err_msg}")
        QMessageBox.critical(self, "Mobile Bridge Error", f"Mobile bridge server error:\n{err_msg}")

    def _on_send_to_mobile_clicked(self):
        formats = [".glb", ".obj", ".usdz"]
        items = ["GLB (.glb)", "OBJ (.obj)", "USDZ (.usdz)"]
        item, ok = QInputDialog.getItem(self, "Send 3D Model to Mobile", "Select 3D Model Format:", items, 0, False)
        if not ok or not item:
            return
            
        fmt = formats[items.index(item)]
        mvs_out = self._get_active_mvs_dir()
        
        target_file = None
        if fmt == ".glb":
            target_file = os.path.join(mvs_out, "scene_dense_mesh_texture.glb")
            if not os.path.exists(target_file):
                src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
                src_ply = None
                for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
                    path = os.path.join(mvs_out, candidate)
                    if os.path.exists(path):
                        src_ply = path
                        break
                src = src_obj if os.path.exists(src_obj) else src_ply
                if src:
                    try:
                        self.console_text.append("[MOBILE BRIDGE] Pre-converting reconstructed mesh to GLB...")
                        import trimesh
                        mesh = trimesh.load(src, force="mesh") if src.endswith(".ply") else trimesh.load(src)
                        mesh.export(target_file, file_type="glb")
                    except Exception as e:
                        self.console_text.append(f"[ERROR] Failed to convert mesh to GLB for mobile export: {e}")
        elif fmt == ".obj":
            target_file = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
        elif fmt == ".usdz":
            candidate = os.path.join(mvs_out, "scene_dense_mesh_texture.usdz")
            if os.path.exists(candidate):
                target_file = candidate
            else:
                src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
                src_glb = os.path.join(mvs_out, "scene_dense_mesh_texture.glb")
                src = src_glb if os.path.exists(src_glb) else (src_obj if os.path.exists(src_obj) else None)
                if src:
                    try:
                        import trimesh
                        from mesh_editor.scene import _export_usdz_from_trimesh
                        mesh = trimesh.load(src, force="mesh")
                        _export_usdz_from_trimesh(mesh, candidate)
                        target_file = candidate
                    except Exception as e:
                        self.console_text.append(f"[ERROR] Failed to prepare USDZ for mobile: {e}")
                        
        if not target_file or not os.path.exists(target_file):
            QMessageBox.warning(self, "No Model Found", f"Could not find or generate a {fmt.upper()} model file for mobile export.")
            return

        from mobile_bridge_server import MobileBridgeServer
        self.mobile_server = MobileBridgeServer(mode="export", serve_file=target_file, parent=self)
        self.mobile_server.start()
        
        QThread.msleep(150)
        urls = self.mobile_server.get_urls()
        
        self.console_text.append(f"[MOBILE BRIDGE] Mobile model server running at: {', '.join(urls)}")
        
        dialog = MobileQRDialog(urls=urls, mode="export", parent=self)
        dialog.exec()
        
        if hasattr(self, 'mobile_server') and self.mobile_server:
            self.mobile_server.stop()
            self.mobile_server = None
            self.console_text.append("[MOBILE BRIDGE] Mobile model download session closed.")


class MobileQRDialog(QDialog):
    def __init__(self, urls: list, mode: str = "import", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.urls = urls
        self.setWindowTitle("Mobile Device Bridge — " + ("Import Media" if mode == "import" else "Download 3D Model"))
        self.setFixedSize(420, 560 if len(urls) > 1 else 520)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #121212;
                color: #e0e0e0;
                border: 1px solid #2B2B2B;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("Scan with your Mobile Phone", self)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00E676;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Get active network Wi-Fi / Hotspot SSID if possible
        from mobile_bridge_server import get_wifi_ssid
        wifi_ssid = get_wifi_ssid()

        if wifi_ssid:
            network_msg = f"\nMake sure your phone is connected to the same Wi-Fi: '{wifi_ssid}'"
        else:
            network_msg = "\nMake sure your phone is on the same network (Wi-Fi or Hotspot)."

        desc_text = (
            "Open your phone's camera app to scan the QR code below." + network_msg
            if mode == "import" else
            "Scan the QR code below on your mobile device\nto download the generated 3D model." + network_msg
        )
        desc = QLabel(desc_text, self)
        desc.setStyleSheet("font-size: 12px; color: #aaaaaa;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        # Network Selector Dropdown if multiple IPs available
        if len(urls) > 1:
            url_select_box = QFrame(self)
            url_select_box.setStyleSheet("background-color: #1E1E1E; border: 1px solid #333333; border-radius: 6px; padding: 4px 8px;")
            url_select_layout = QHBoxLayout(url_select_box)
            url_select_layout.setContentsMargins(4, 2, 4, 2)
            
            lbl_ip_select = QLabel("Network Adapter / IP:", url_select_box)
            lbl_ip_select.setStyleSheet("font-size: 11px; color: #888888;")
            url_select_layout.addWidget(lbl_ip_select)
            
            self.url_combo = QComboBox(url_select_box)
            self.url_combo.setStyleSheet("""
                QComboBox {
                    background-color: #292929;
                    color: #00E676;
                    font-weight: bold;
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 4px 8px;
                }
                QComboBox QAbstractItemView {
                    background-color: #1E1E1E;
                    color: #00E676;
                    selection-background-color: #333333;
                }
            """)
            for u in urls:
                self.url_combo.addItem(u)
            self.url_combo.currentTextChanged.connect(self._on_url_selected)
            url_select_layout.addWidget(self.url_combo, stretch=1)
            layout.addWidget(url_select_box)

        # QR Code Image Label
        self.qr_label = QLabel(self)
        self.qr_label.setFixedSize(220, 220)
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr_label.setStyleSheet("background-color: #ffffff; border-radius: 8px; padding: 10px;")
        
        primary_url = urls[0] if urls else "http://127.0.0.1"
        self._set_qr_url(primary_url)

        qr_container = QHBoxLayout()
        qr_container.addStretch()
        qr_container.addWidget(self.qr_label)
        qr_container.addStretch()
        layout.addLayout(qr_container)

        # URL Text Box for manual typing
        url_box = QFrame(self)
        url_box.setStyleSheet("background-color: #1E1E1E; border: 1px solid #333333; border-radius: 6px; padding: 8px;")
        url_layout = QVBoxLayout(url_box)
        url_layout.setContentsMargins(8, 4, 8, 4)
        
        lbl_type = QLabel("Or type this URL into your phone browser:", url_box)
        lbl_type.setStyleSheet("font-size: 11px; color: #888888;")
        self.lbl_url = QLabel(primary_url, url_box)
        self.lbl_url.setStyleSheet("font-size: 13px; font-weight: bold; color: #00E676;")
        self.lbl_url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        url_layout.addWidget(lbl_type)
        url_layout.addWidget(self.lbl_url)
        layout.addWidget(url_box)

        # Status text
        self.status_lbl = QLabel("Waiting for phone connection...", self)
        self.status_lbl.setStyleSheet("font-size: 12px; font-style: italic; color: #888888;")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_lbl)

        # Cancel/Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_text = "Cancel" if mode == "import" else "Done / Close"
        self.cancel_btn = QPushButton(btn_text, self)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #292929;
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px 24px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #333333; border-color: #00E676; }
        """)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _on_url_selected(self, new_url: str):
        if new_url:
            self._set_qr_url(new_url)
            self.lbl_url.setText(new_url)

    def _set_qr_url(self, url: str):
        try:
            import qrcode
            from io import BytesIO
            from PySide6.QtGui import QImage, QPixmap
            
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=6,
                border=2,
            )
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            qimg = QImage.fromData(buffer.getvalue(), "PNG")
            self.qr_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            self.qr_label.setText(f"QR Error:\n{e}")


class MobileImportSetupDialog(QDialog):
    def __init__(self, default_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mobile Import Setup")
        self.setFixedSize(450, 310)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #121212;
                color: #e0e0e0;
                border: 1px solid #2B2B2B;
            }
            QLabel { color: #e0e0e0; }
            QLineEdit {
                background-color: #1E1E1E;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                background-color: #292929;
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #333333; border-color: #00E676; }
            QPushButton#StartBtn {
                background-color: #00E676;
                color: #121212;
                border: none;
                font-weight: bold;
            }
            QPushButton#StartBtn:hover { background-color: #00c860; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        lbl = QLabel("Choose where to save imported media / point clouds on your PC:", self)
        lbl.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addWidget(lbl)

        # Path input row
        row = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.path_edit.setText(default_path)
        row.addWidget(self.path_edit)

        self.choose_btn = QPushButton("Choose...", self)
        self.choose_btn.clicked.connect(self._on_choose_clicked)
        row.addWidget(self.choose_btn)
        layout.addLayout(row)

        # Network Status Info Frame
        self.network_status_frame = QFrame(self)
        self.network_status_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border: 1px solid #2B2B2B;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        net_layout = QVBoxLayout(self.network_status_frame)
        net_layout.setContentsMargins(8, 8, 8, 8)
        net_layout.setSpacing(6)
        
        net_header = QHBoxLayout()
        net_header.setContentsMargins(0, 0, 0, 0)
        
        self.net_title_label = QLabel("📶 Local Network Check:", self.network_status_frame)
        self.net_title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #00E676; border: none; background: transparent; padding: 0;")
        net_header.addWidget(self.net_title_label, stretch=1)
        
        self.net_refresh_btn = QPushButton("Refresh", self.network_status_frame)
        self.net_refresh_btn.setCursor(Qt.PointingHandCursor)
        self.net_refresh_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #00E676;
                border: 1px solid #00E676;
                border-radius: 3px;
                font-size: 10px;
                padding: 2px 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(0, 230, 118, 0.1);
            }
        """)
        self.net_refresh_btn.clicked.connect(self.check_network)
        net_header.addWidget(self.net_refresh_btn)
        
        net_layout.addLayout(net_header)
        
        self.net_details_label = QLabel("Checking network connection...", self.network_status_frame)
        self.net_details_label.setStyleSheet("font-size: 11px; color: #aaaaaa; border: none; background: transparent; padding: 0;")
        self.net_details_label.setWordWrap(True)
        net_layout.addWidget(self.net_details_label)
        
        layout.addWidget(self.network_status_frame)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        self.start_btn = QPushButton("Start Server", self)
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.start_btn)
        
        layout.addLayout(btn_layout)

        # Trigger network check
        self.check_network()

    def check_network(self):
        import psutil
        import socket
        from mobile_bridge_server import get_wifi_ssid
        
        # 1. Get active network connections from psutil
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        active_nets = []
        
        for name, stat in stats.items():
            name_lower = name.lower()
            if name_lower in ("lo", "loopback"):
                continue
            if stat.isup and name in addrs:
                for addr in addrs[name]:
                    if addr.family == socket.AF_INET:
                        ip = addr.address
                        if ip and not ip.startswith("127.") and not ip.startswith("169.254."):
                            if any(x in name_lower for x in ["wi-fi", "wifi", "wlan", "wireless", "802.11"]):
                                conn_type = "Wi-Fi"
                            elif any(x in name_lower for x in ["local area connection*", "direct", "ap", "hotspot", "tether"]):
                                conn_type = "Wi-Fi Hotspot"
                            elif any(x in name_lower for x in ["ethernet", "lan", "eth", "en"]):
                                conn_type = "Ethernet"
                            else:
                                conn_type = "LAN"
                            active_nets.append((name, ip, conn_type))
                            
        # 2. Try to get Wi-Fi network / Hotspot name across OSes
        wifi_ssid = get_wifi_ssid()
                
        # 3. Format description message
        if not active_nets:
            self.net_title_label.setText("📶 Local Network Check: Disconnected")
            self.net_title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #ff5252; border: none; background: transparent; padding: 0;")
            self.net_details_label.setText(
                "⚠️ No active network detected! Please connect your PC to a Wi-Fi network or enable a mobile hotspot. "
                "Both your PC and phone MUST be on the same network to transfer files."
            )
            self.start_btn.setEnabled(False)
            self.start_btn.setToolTip("Please connect to a network first")
        else:
            self.net_title_label.setText("📶 Local Network Check: Connected")
            self.net_title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #00E676; border: none; background: transparent; padding: 0;")
            self.start_btn.setEnabled(True)
            self.start_btn.setToolTip("")
            
            lines = []
            for name, ip, conn_type in active_nets:
                net_name_str = f"'{wifi_ssid}'" if (wifi_ssid and conn_type in ["Wi-Fi", "Wi-Fi Hotspot"]) else f"({name})"
                lines.append(f"• {conn_type}: {net_name_str} — IP: <b>{ip}</b>")
                
            connections_str = "<br>".join(lines)
            self.net_details_label.setText(
                f"Your PC is connected to the following local network(s):<br>{connections_str}<br>"
                "⚠️ <b>Important:</b> Your mobile device <b>MUST</b> be connected to the exact same Wi-Fi/Hotspot network to upload."
            )

    def _on_choose_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self.path_edit.text())
        if folder:
            self.path_edit.setText(os.path.normpath(folder))

    def get_save_dir(self) -> str:
        return self.path_edit.text()


class VideoPresetModal(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Detected — Choose Frame Extraction Quality")
        self.setMinimumWidth(520)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #121212;
                color: #e0e0e0;
                border: 1px solid #2B2B2B;
            }
            QLabel {
                color: #e0e0e0;
            }
            QFrame#PresetCard {
                border-radius: 8px;
                padding: 12px;
            }
            QRadioButton {
                font-weight: bold;
                font-size: 13px;
                background: transparent;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #555555;
                border-radius: 9px;
                background-color: #222222;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #00E676;
                background-color: #00E676;
            }
            QPushButton#StartBtn {
                background-color: #00E676;
                color: #121212;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton#StartBtn:hover {
                background-color: #00FF87;
            }
            QPushButton#CancelBtn {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton#CancelBtn:hover {
                background-color: #444444;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)
        
        info_label = QLabel("Proximap detected one or more video files. Select a frame extraction preset below:", self)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 13px; color: #b3b3b3; margin-bottom: 5px;")
        main_layout.addWidget(info_label)

        self.btn_group = QButtonGroup(self)
        self.presets = [
            ("Quick", "Fastest — fewer frames, best for simple objects or quick previews", 1.0, None),
            ("Balanced (recommended)", "Good coverage for most scans — filters out blurry frames automatically", 0.5, 25.0),
            ("Detailed", "Maximum coverage — best for complex geometry, slower to process", 0.25, 20.0)
        ]
        
        self.radio_buttons = []
        self.cards = []
        for i, (name, desc, val_interval, val_blur) in enumerate(self.presets):
            card = QFrame(self)
            card.setObjectName("PresetCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(4)
            
            rb = QRadioButton(name, card)
            self.btn_group.addButton(rb, i)
            rb.toggled.connect(self.update_card_styles)
            self.radio_buttons.append(rb)
            self.cards.append(card)
            
            desc_label = QLabel(desc, card)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #aaaaaa; font-size: 11px; margin-left: 20px;")
            
            card_layout.addWidget(rb)
            card_layout.addWidget(desc_label)
            main_layout.addWidget(card)
            
        # Select Balanced by default
        self.radio_buttons[1].setChecked(True)
        self.update_card_styles()

        # Advanced Options Collapsible Section
        self.advanced_toggle_btn = QPushButton("▸ Advanced Options", self)
        self.advanced_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.advanced_toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #888888;
                border: none;
                text-align: left;
                font-size: 11px;
                font-weight: bold;
                padding: 4px 0px;
            }
            QPushButton:hover {
                color: #00E676;
            }
        """)
        self.advanced_toggle_btn.clicked.connect(self._toggle_advanced)
        main_layout.addWidget(self.advanced_toggle_btn)

        self.advanced_panel = QFrame(self)
        self.advanced_panel.setVisible(False)
        self.advanced_panel.setStyleSheet("""
            QFrame {
                background-color: #1A1A1A;
                border: 1px solid #2D2D2D;
                border-radius: 6px;
                padding: 8px;
            }
            QLabel {
                color: #aaaaaa;
                font-size: 11px;
            }
            QCheckBox {
                color: #e0e0e0;
                font-size: 11px;
            }
        """)
        adv_layout = QVBoxLayout(self.advanced_panel)
        adv_layout.setSpacing(8)

        self.cb_override_blur = QCheckBox("Override Blur Filter Threshold", self.advanced_panel)
        self.cb_override_blur.toggled.connect(self._on_override_toggled)
        adv_layout.addWidget(self.cb_override_blur)

        blur_controls_layout = QHBoxLayout()
        blur_controls_layout.setContentsMargins(15, 0, 0, 0)
        
        lbl_blur_thresh = QLabel("Blur Threshold:", self.advanced_panel)
        self.sp_blur_thresh = QDoubleSpinBox(self.advanced_panel)
        self.sp_blur_thresh.setRange(0.0, 500.0)
        self.sp_blur_thresh.setSingleStep(5.0)
        self.sp_blur_thresh.setValue(25.0)
        self.sp_blur_thresh.setEnabled(False)
        self.sp_blur_thresh.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2D2D2D;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px;
            }
            QDoubleSpinBox:disabled {
                background-color: #1A1A1A;
                color: #555555;
            }
        """)

        self.cb_disable_blur = QCheckBox("Disable blur filter entirely", self.advanced_panel)
        self.cb_disable_blur.setEnabled(False)
        self.cb_disable_blur.toggled.connect(self._on_disable_blur_toggled)

        blur_controls_layout.addWidget(lbl_blur_thresh)
        blur_controls_layout.addWidget(self.sp_blur_thresh)
        blur_controls_layout.addWidget(self.cb_disable_blur)
        adv_layout.addLayout(blur_controls_layout)

        main_layout.addWidget(self.advanced_panel)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.setObjectName("CancelBtn")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.start_btn = QPushButton("Start Extraction", self)
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.clicked.connect(self.accept)
        
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.start_btn)
        main_layout.addLayout(btn_layout)

    def _toggle_advanced(self):
        is_visible = not self.advanced_panel.isVisible()
        self.advanced_panel.setVisible(is_visible)
        self.advanced_toggle_btn.setText("▾ Advanced Options" if is_visible else "▸ Advanced Options")
        self.adjustSize()

    def _on_override_toggled(self, checked: bool):
        self.cb_disable_blur.setEnabled(checked)
        self.sp_blur_thresh.setEnabled(checked and not self.cb_disable_blur.isChecked())

    def _on_disable_blur_toggled(self, checked: bool):
        self.sp_blur_thresh.setEnabled(not checked and self.cb_override_blur.isChecked())

    def update_card_styles(self):
        for i, rb in enumerate(self.radio_buttons):
            card = self.cards[i]
            if rb.isChecked():
                card.setStyleSheet("QFrame#PresetCard { border: 2px solid #00E676; background-color: #1A2E24; }")
                rb.setStyleSheet("QRadioButton { color: #00E676; }")
            else:
                card.setStyleSheet("QFrame#PresetCard { border: 1px solid #333333; background-color: #242424; }")
                rb.setStyleSheet("QRadioButton { color: #ffffff; }")

    def get_selected_preset(self):
        idx = self.btn_group.checkedId()
        if idx == -1:
            idx = 1
        name, desc, interval, blur = self.presets[idx]

        if self.cb_override_blur.isChecked():
            if self.cb_disable_blur.isChecked():
                blur = None
            else:
                blur = float(self.sp_blur_thresh.value())

        return (name, desc, interval, blur)


class VideoExtractionWorker(QThread):
    progress = Signal(int, int)  # (current, total)
    finished = Signal(object)    # ExtractionResult
    error = Signal(str)          # error message

    def __init__(self, video_path: str, output_dir: str, interval_seconds: float, blur_threshold: float | None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval_seconds = interval_seconds
        self.blur_threshold = blur_threshold
        self._should_continue = True

    def cancel(self):
        self._should_continue = False

    def should_continue_callback(self):
        return self._should_continue

    def run(self):
        from video_extraction import extract_frames, VideoExtractionError
        
        def progress_callback(current, total):
            self.progress.emit(current, total)
            
        try:
            result = extract_frames(
                video_path=self.video_path,
                output_dir=self.output_dir,
                interval_seconds=self.interval_seconds,
                blur_threshold=self.blur_threshold,
                progress_callback=progress_callback,
                should_continue_cb=self.should_continue_callback
            )
            self.finished.emit(result)
        except VideoExtractionError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")


class HardwareInitWorker(QThread):

    def run(self):
        hardware_profiler.initialize()

        # Set up the PyMeshLab venv on first launch (pip installs the bundled .whl
        # into ~/.local/share/proximap/pymeshlab_venv using the bundled Python 3.10).
        # Subsequent launches skip this (sentinel file check) and take <1 second.
        try:
            from mesh_cleanup import ensure_pymeshlab_venv
            ensure_pymeshlab_venv()
        except Exception:
            pass  # Non-fatal: mesh cleanup falls back to Open3D if venv setup fails

class StartupManager:
    def __init__(self, splash, icon_path):
        self.splash = splash
        self.icon_path = icon_path
        self.worker = HardwareInitWorker()
        self.worker.finished.connect(self.on_init_finished)
        self.window = None

    def start(self):
        self.worker.start()

    def on_init_finished(self):
        self.window = MainWindow()
        if os.path.exists(self.icon_path):
            self.window.setWindowIcon(QIcon(self.icon_path))
        self.window.show()
        if self.splash:
            self.splash.finish(self.window)
        QTimer.singleShot(500, self.window._check_startup_recovery)


if __name__ == "__main__":
    # Fix taskbar icon grouping on Windows
    if sys.platform == 'win32':
        import ctypes
        myappid = 'proximaxr.proximap.photogrammetry.1.0'
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QToolTip {
            background-color: #242424;
            color: #FFFFFF;
            border: 1px solid #4D4D4D;
            border-radius: 4px;
            padding: 5px 8px;
            font-size: 11px;
            font-weight: 500;
        }
    """)
    
    # Resolve app icon path
    base_dir = get_base_dir()
    icon_path = os.path.join(base_dir, "app_icon.ico")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(base_dir, "public", "app_icon.png")
        
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
        
    # Create and show splash screen using the high-res PNG icon
    from PySide6.QtWidgets import QSplashScreen
    splash = None
    splash_path = os.path.join(base_dir, "public", "app_icon.png")
    if not os.path.exists(splash_path):
        splash_path = icon_path
        
    if os.path.exists(splash_path):
        pixmap = QPixmap(splash_path)
        # Scale to a standard splash size (e.g. 256x256) keeping aspect ratio
        scaled_pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(scaled_pixmap)
        splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        splash.show()
        splash.showMessage("Initializing hardware profile...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        
    # Start startup manager to handle background initialization
    manager = StartupManager(splash, icon_path)
    manager.start()
    
    sys.exit(app.exec())
