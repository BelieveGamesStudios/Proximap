import os
import sys
import subprocess
import ctypes
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QProgressBar, QRadioButton, QButtonGroup,
    QFrame, QFileDialog, QTextEdit, QStackedWidget, QComboBox,
    QScrollArea, QTabWidget, QGridLayout, QCheckBox, QSlider,
    QMessageBox, QDialog, QColorDialog, QMenu, QSizePolicy, QInputDialog,
    QLineEdit, QSpinBox, QDoubleSpinBox
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


from PySide6.QtCore import Qt, QSize, Signal, QTimer, QThread
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QFont, QWindow, QPixmap, QImage

import hardware_profiler

# Deferred imports for faster startup: MeshEditorWidget, PipelineWorker, BackgroundRemovalWorker

import http.server
import socketserver
import threading
import webbrowser

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv')

CAMERA_CONTROLS = {
    0: "<b>Arcball Camera Controls:</b><br>"
       "• Left Drag: Orbit camera<br>"
       "• Right Drag / Scroll: Zoom / Dolly<br>"
       "• Middle Drag / Shift+Left Drag: Pan",
    1: "<b>Turntable Camera Controls:</b><br>"
       "• Left Drag: Orbit (fixed Z-up)<br>"
       "• Right Drag / Scroll: Zoom / Dolly<br>"
       "• Middle Drag / Shift+Left Drag: Pan",
    2: "<b>Fly Camera Controls:</b><br>"
       "• Left Drag: Look around (pitch/yaw)<br>"
       "• WASD / Arrow Keys: Fly around<br>"
       "• Space / C: Fly up / down<br>"
       "• Q / E: Roll camera left / right<br>"
       "• Mouse Scroll: Adjust movement speed",
    3: "<b>Pan-Zoom Camera Controls:</b><br>"
       "• Left Drag: Pan 2D boundaries<br>"
       "• Right Drag / Scroll: Zoom in/out",
    4: "<b>Magnify Camera Controls:</b><br>"
       "• Left Drag: Pan 2D boundaries<br>"
       "• Scroll: Localized magnifying zoom<br>"
       "• Shift + Mouse Move: Adjust focus area"
}


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

            if mode == 1:
                # Dense Point Cloud
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
        has_list = any(p[2] for p in vertex_properties)
        points = np.zeros((num_vertices, 3), dtype=np.float32)
        colors = None
        faces  = None
        prop_names = [p[0] for p in vertex_properties]
        has_color  = all(c in prop_names for c in ('red','green','blue'))
        if has_color:
            colors = np.zeros((num_vertices, 3), dtype=np.uint8)
        if 'binary' in (format_type or ''):
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
                    if 'red'   in arr.dtype.names: colors[:,0] = arr['red']
                    if 'green' in arr.dtype.names: colors[:,1] = arr['green']
                    if 'blue'  in arr.dtype.names: colors[:,2] = arr['blue']
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
                        ri = prop_names.index('red')   if 'red'   in prop_names else 3
                        gi = prop_names.index('green') if 'green' in prop_names else 4
                        bi = prop_names.index('blue')  if 'blue'  in prop_names else 5
                        try: colors[i] = [int(vals[ri]), int(vals[gi]), int(vals[bi])]
                        except: pass
            if num_faces > 0:
                faces_list = []
                for _ in range(num_faces):
                    vals = f.readline().decode('utf-8', errors='ignore').split()
                    if vals and int(vals[0]) == 3 and len(vals) >= 4:
                        faces_list.append([int(vals[1]), int(vals[2]), int(vals[3])])
                if faces_list: faces = np.array(faces_list, dtype=np.int32)
    return points, colors, faces


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
                        if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                            files.append(os.path.normpath(fp))
                        else:
                            ignored.append(filename)
            elif os.path.isfile(local_path):
                ext = os.path.splitext(local_path)[1].lower()
                if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                    files.append(os.path.normpath(local_path))
                else:
                    ignored.append(os.path.basename(local_path))
                    
        if ignored:
            from PySide6.QtWidgets import QMessageBox
            msg = "The following files were ignored because they are not supported images or videos:\n\n"
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("ViewerWrapperWidget")
        self.setStyleSheet("background-color: #1A1A1A; border: 1px solid #2B2B2B; border-radius: 8px;")
        
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
        self.file_menu_btn = QPushButton("File ▾", self.control_bar)
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
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00E676;
            }
            QPushButton::menu-indicator {
                image: none;
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
        self.action_mobile_import = self.file_menu.addAction("Import Images/Videos from Mobile Device")
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
        
        # Dropdown to choose camera tracking style
        self.cam_select = QComboBox(self.control_bar)
        self.cam_select.setMinimumWidth(150)
        self.cam_select.addItems([
            "Arcball Camera",
            "Turntable Camera",
            "Fly Camera",
            "Pan-Zoom Camera",
            "Magnify Camera"
        ])
        
        # Checkbox to toggle controls display overlay
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
        
        # Dropdown to choose MVS scene mode
        self.mode_select = QComboBox(self.control_bar)
        self.mode_select.setMinimumWidth(200)
        self.mode_select.addItems([
            "Sparse Point Cloud & Cameras",
            "Dense Point Cloud",
            "Textured Mesh"
        ])
        
        # Action buttons
        self.bg_btn = QPushButton("BG Color", self.control_bar)
        self.bg_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                padding: 4px 8px;
                font-weight: normal;
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00E676;
            }
        """)
        
        self.reload_btn = QPushButton("Reload", self.control_bar)
        self.reload_btn.setStyleSheet("font-size: 11px; padding: 4px 8px; font-weight: normal;")
        
        control_layout.addWidget(self.file_menu_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.cam_select)
        control_layout.addWidget(self.show_controls_cb)
        control_layout.addWidget(self.mode_select)
        control_layout.addWidget(self.bg_btn)
        control_layout.addWidget(self.reload_btn)
        
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
        self.reload_btn.clicked.connect(self._on_reload_clicked)
        self.mode_select.currentIndexChanged.connect(self._on_mode_changed)
        self.cam_select.currentIndexChanged.connect(self.camera_changed.emit)
        
        self.current_mvs_dir = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
            
        if w < 600:
            self.show_controls_cb.setText("Controls")
            self.cam_select.setMinimumWidth(100)
            self.mode_select.setMinimumWidth(140)
            self.bg_btn.setText("Color")
        else:
            self.show_controls_cb.setText("Show Controls")
            self.cam_select.setMinimumWidth(150)
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
                        if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                             files.append(os.path.normpath(fp))
                        else:
                             ignored.append(filename)
            elif os.path.isfile(local_path):
                ext = os.path.splitext(local_path)[1].lower()
                if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                    files.append(os.path.normpath(local_path))
                else:
                    ignored.append(os.path.basename(local_path))
                    
        if ignored:
            from PySide6.QtWidgets import QMessageBox
            msg = "The following files were ignored because they are not supported images or videos:\n\n"
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

    def get_selected_file_path(self) -> str:
        parent = self.parent()
        if parent and hasattr(parent, 'standalone_cloud_path') and parent.standalone_cloud_path:
            mvs_dir = self.current_mvs_dir if self.current_mvs_dir else os.path.join(get_reconstruction_out_dir(), "mvs")
            has_reconstruction = os.path.exists(os.path.join(mvs_dir, "scene_dense_mesh_refine.ply")) or \
                                 os.path.exists(os.path.join(mvs_dir, "scene_dense_mesh.ply"))
            if not has_reconstruction:
                return parent.standalone_cloud_path

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

    def _on_reload_clicked(self):
        path = self.get_selected_file_path()
        if path:
            self.reload_requested.emit(path)

    def _on_mode_changed(self, index):
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


class SaveWorker(QThread):
    finished = Signal(bool, str) # Emits (success, message)
    
    def __init__(self, mvs_dir, file_path):
        super().__init__()
        self.mvs_dir = mvs_dir
        self.file_path = file_path
        
    def run(self):
        import zipfile
        try:
            with zipfile.ZipFile(self.file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
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
            mvs_dir = os.path.join(self.temp_root, f"proximap_project_{uuid.uuid4()}")
            os.makedirs(mvs_dir, exist_ok=True)
            
            with zipfile.ZipFile(self.file_path, 'r') as zipf:
                zipf.extractall(mvs_dir)
                
            # Verify if it extracted scene.mvs
            scene_mvs = os.path.join(mvs_dir, "scene.mvs")
            if not os.path.exists(scene_mvs):
                found_scene = None
                for root, _, files in os.walk(mvs_dir):
                    if "scene.mvs" in files:
                        found_scene = os.path.join(root, "scene.mvs")
                        mvs_dir = root
                        break
                if not found_scene:
                    self.finished.emit(False, "", "Invalid project file: 'scene.mvs' not found in archive.")
                    return
            
            self.finished.emit(True, mvs_dir, "Project loaded successfully.")
        except Exception as e:
            self.finished.emit(False, "", str(e))


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
    Individual photo thumbnail card display with a selection checkbox.
    Supports a placeholder initially and lazy updates.
    """
    def __init__(self, file_path, size, pixmap=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.size = size
        self.selected = False
        self.pixmap = pixmap
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
        self.checkbox.stateChanged.connect(self._on_check_changed)
        
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
        
    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        if not pixmap.isNull():
            self.image_label.setText("")
            self.image_label.setStyleSheet("")
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("⚠️")
            self.image_label.setStyleSheet("font-size: 20px; color: #ff1744;")
            
    def _on_check_changed(self, state):
        self.selected = (state == Qt.Checked.value or state == 2)
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
        self.checkbox.setChecked(checked)


class PhotosGridWidget(QWidget):
    """
    Grid container that dynamically arranges PhotoItemWidgets depending on container width.
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
        
    def rebuild_grid(self, force=False):
        if not self.image_paths:
            self.clear_grid()
            self.current_cols = 0
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
                item_widget = PhotoItemWidget(path, self.thumbnail_size, pixmap, self)
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
        self.toolbar.setStyleSheet("background-color: #1A1A1A; border-bottom: 1px solid #2B2B2B;")
        toolbar_layout = QHBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(10, 4, 10, 4)
        toolbar_layout.setSpacing(6)
        
        # Buttons matching reference UI functionality
        public_dir = os.path.join(get_base_dir(), "public")
        
        self.btn_select_all = QPushButton("", self.toolbar)
        self.btn_select_all.setIcon(QIcon(os.path.join(public_dir, "all.png")))
        self.btn_select_all.setToolTip("Select All")
        self.btn_select_all.setStyleSheet("QPushButton { padding: 4px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
        self.btn_deselect_all = QPushButton("", self.toolbar)
        self.btn_deselect_all.setIcon(QIcon(os.path.join(public_dir, "none.png")))
        self.btn_deselect_all.setToolTip("Deselect All")
        self.btn_deselect_all.setStyleSheet("QPushButton { padding: 4px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
        self.btn_remove_selected = QPushButton("", self.toolbar)
        self.btn_remove_selected.setIcon(QIcon(os.path.join(public_dir, "trash.png")))
        self.btn_remove_selected.setToolTip("Remove Selected")
        self.btn_remove_selected.setStyleSheet("QPushButton { padding: 4px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
        self.btn_add_photos = QPushButton("", self.toolbar)
        self.btn_add_photos.setIcon(QIcon(os.path.join(public_dir, "folder.png")))
        self.btn_add_photos.setToolTip("Add Photos")
        self.btn_add_photos.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
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
        
    def set_images(self, image_paths):
        self.image_list = image_paths
        
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proximap 1.4.0")
        self.setMinimumSize(1100, 750)
        self.image_list = []
        self.worker = None
        
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
        self._last_points = None
        self.last_accessed_dir = os.path.expanduser("~")
        self.viewport_bg_color = '#0C0C0C'
        
        self._init_ui()
        self._apply_styling()
        QTimer.singleShot(0, self._check_existing_scene)

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
        sidebar.setFixedWidth(360)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # Fixed Title Bar at the top of the sidebar
        title_container = QWidget(sidebar)
        title_container.setStyleSheet("background-color: #1A1A1A; border-top-left-radius: 8px; border-top-right-radius: 8px;")
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(20, 20, 20, 10)
        
        title_label = QLabel("Reconstruction Wizard", title_container)
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
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(20, 10, 20, 20)
        scroll_content_layout.setSpacing(20)
        
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
        
        self.browse_btn = QPushButton("Select Images/Videos Directory", step1_box)
        self.browse_btn.clicked.connect(self._open_dir_dialog)
        
        self.mobile_import_btn = QPushButton("Import from Mobile Device", step1_box)
        self.mobile_import_btn.clicked.connect(self._on_import_from_mobile_clicked)
        
        self.bg_remove_btn = QPushButton("Remove Image Background", step1_box)
        self.bg_remove_btn.setObjectName("BgRemoveBtn")
        self.bg_remove_btn.setEnabled(False)
        self.bg_remove_btn.clicked.connect(self._remove_backgrounds_clicked)

        # Reference Cloud Import (Optional)
        # self.ref_cloud_btn = QPushButton("Import Reference Cloud", step1_box)
        # self.ref_cloud_btn.setObjectName("RefCloudBtn")
        # self.ref_cloud_btn.clicked.connect(self._import_reference_cloud_clicked)
        # 
        # self.ref_cloud_container = QWidget(step1_box)
        # ref_cloud_layout = QHBoxLayout(self.ref_cloud_container)
        # ref_cloud_layout.setContentsMargins(0, 0, 0, 0)
        # ref_cloud_layout.setSpacing(5)
        # 
        # self.ref_cloud_label = QLabel("", self.ref_cloud_container)
        # self.ref_cloud_label.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold;")
        # self.ref_cloud_label.setTextFormat(Qt.PlainText)
        # 
        # self.ref_cloud_clear_btn = QPushButton("✕", self.ref_cloud_container)
        # self.ref_cloud_clear_btn.setFixedSize(20, 20)
        # self.ref_cloud_clear_btn.setToolTip("Clear reference cloud selection")
        # self.ref_cloud_clear_btn.setStyleSheet("""
        #     QPushButton {
        #         background: transparent;
        #         color: #ff5252;
        #         border: 1px solid #ff5252;
        #         border-radius: 10px;
        #         font-weight: bold;
        #         font-size: 10px;
        #     }
        #     QPushButton:hover {
        #         background: #ff5252;
        #         color: #ffffff;
        #     }
        # """)
        # self.ref_cloud_clear_btn.clicked.connect(self._clear_reference_cloud_clicked)
        # 
        # ref_cloud_layout.addWidget(self.ref_cloud_label, stretch=1)
        # ref_cloud_layout.addWidget(self.ref_cloud_clear_btn)
        # self.ref_cloud_container.setVisible(False)
        # 
        # self.ref_cloud_path = None
        
        # Standalone Cloud Import (Optional/Mutual Exclusive Mode)
        self.standalone_cloud_btn = QPushButton("Import Standalone Point Cloud", step1_box)
        self.standalone_cloud_btn.setObjectName("StandaloneCloudBtn")
        self.standalone_cloud_btn.clicked.connect(self._import_standalone_cloud_clicked)
        
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
        step1_layout.addWidget(self.browse_btn)
        step1_layout.addWidget(self.mobile_import_btn)
        step1_layout.addWidget(self.bg_remove_btn)
        # step1_layout.addWidget(self.ref_cloud_btn)
        # step1_layout.addWidget(self.ref_cloud_container)
        step1_layout.addWidget(self.standalone_cloud_btn)
        step1_layout.addWidget(self.standalone_cloud_container)

        scroll_content_layout.addWidget(step1_box)
        
        # STEP 2: Process
        step2_box = QFrame(scroll_content)
        step2_box.setObjectName("StepBox")
        step2_layout = QVBoxLayout(step2_box)
        
        s2_title = QLabel("Step 2: Run Reconstruction", step2_box)
        s2_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #00E676;")
        
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
        
        self.plain_surfaces_checkbox = QCheckBox("Surface is plain/smooth", step2_box)
        self.plain_surfaces_checkbox.setChecked(False)
        
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
        
        self.mapper_label = QLabel("SfM Mapper Algorithm:", self.advanced_panel)
        self.mapper_label.setStyleSheet("font-size: 11px; color: #aaaaaa; border: none; background: transparent;")
        self.mapper_combo = QComboBox(self.advanced_panel)
        self.mapper_combo.addItems([
            "COLMAP  —  Incremental (default, most robust)",
            "GLOMAP  —  Global (faster on large captures)"
        ])
        self.mapper_combo.setCurrentIndex(0)
        
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

        # Custom Overrides Toggle
        self.custom_settings_toggle = QCheckBox("Enable Custom Parameter Overrides", self.advanced_panel)
        self.custom_settings_toggle.setStyleSheet("font-size: 11px; font-weight: bold; color: #aaaaaa;")
        self.custom_settings_toggle.setChecked(False)
        self.custom_settings_toggle.toggled.connect(self._on_custom_settings_toggled)

        # Custom Settings Container Widget
        self.custom_settings_container = QWidget(self.advanced_panel)
        self.custom_settings_container.setStyleSheet("border: none; background: transparent; padding: 0;")
        self.custom_settings_container.setEnabled(False)
        
        custom_grid = QGridLayout(self.custom_settings_container)
        custom_grid.setContentsMargins(0, 4, 0, 0)
        custom_grid.setSpacing(6)

        lbl_style = "font-size: 10px; color: #888888; border: none; background: transparent;"

        # COLMAP section
        colmap_sec = QLabel("COLMAP Settings", self.custom_settings_container)
        colmap_sec.setStyleSheet("font-size: 11px; font-weight: bold; color: #00E676; margin-top: 4px; border: none; background: transparent;")
        custom_grid.addWidget(colmap_sec, 0, 0, 1, 2)

        # SIFT Max Features
        lbl_features = QLabel("SIFT Max Features:", self.custom_settings_container)
        lbl_features.setStyleSheet(lbl_style)
        self.custom_features_spin = QSpinBox(self.custom_settings_container)
        self.custom_features_spin.setRange(1000, 65536)
        self.custom_features_spin.setSingleStep(1000)
        self.custom_features_spin.setValue(8000)
        self.custom_features_spin.setStyleSheet("background-color: #1E1E1E; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;")
        custom_grid.addWidget(lbl_features, 1, 0)
        custom_grid.addWidget(self.custom_features_spin, 1, 1)

        # Exhaustive Max Matches
        lbl_matches = QLabel("Max Num Matches:", self.custom_settings_container)
        lbl_matches.setStyleSheet(lbl_style)
        self.custom_matches_spin = QSpinBox(self.custom_settings_container)
        self.custom_matches_spin.setRange(4096, 262144)
        self.custom_matches_spin.setSingleStep(4096)
        self.custom_matches_spin.setValue(16384)
        self.custom_matches_spin.setStyleSheet("background-color: #1E1E1E; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;")
        custom_grid.addWidget(lbl_matches, 2, 0)
        custom_grid.addWidget(self.custom_matches_spin, 2, 1)

        # Guided Matching
        self.custom_guided_check = QCheckBox("Use Guided Matching", self.custom_settings_container)
        self.custom_guided_check.setStyleSheet("font-size: 10px; color: #aaaaaa; border: none; background: transparent;")
        self.custom_guided_check.setChecked(True)
        custom_grid.addWidget(self.custom_guided_check, 3, 0, 1, 2)

        # Bundle Adjuster
        self.custom_ba_check = QCheckBox("Run Extra Bundle Adjuster", self.custom_settings_container)
        self.custom_ba_check.setStyleSheet("font-size: 10px; color: #aaaaaa; border: none; background: transparent;")
        self.custom_ba_check.setChecked(False)
        custom_grid.addWidget(self.custom_ba_check, 4, 0, 1, 2)

        # OpenMVS section
        openmvs_sec = QLabel("OpenMVS Settings", self.custom_settings_container)
        openmvs_sec.setStyleSheet("font-size: 11px; font-weight: bold; color: #00E676; margin-top: 4px; border: none; background: transparent;")
        custom_grid.addWidget(openmvs_sec, 5, 0, 1, 2)

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
        custom_grid.addWidget(lbl_densify_res, 6, 0)
        custom_grid.addWidget(self.custom_densify_res_combo, 6, 1)

        # Max Views for Densification
        lbl_densify_views = QLabel("Max Densify Views:", self.custom_settings_container)
        lbl_densify_views.setStyleSheet(lbl_style)
        self.custom_densify_views_spin = QSpinBox(self.custom_settings_container)
        self.custom_densify_views_spin.setRange(1, 16)
        self.custom_densify_views_spin.setValue(4)
        self.custom_densify_views_spin.setStyleSheet("background-color: #1E1E1E; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;")
        custom_grid.addWidget(lbl_densify_views, 7, 0)
        custom_grid.addWidget(self.custom_densify_views_spin, 7, 1)

        # Mesh Refinement Scales
        lbl_refine_scales = QLabel("Mesh Refinement Scales:", self.custom_settings_container)
        lbl_refine_scales.setStyleSheet(lbl_style)
        self.custom_refine_scales_spin = QSpinBox(self.custom_settings_container)
        self.custom_refine_scales_spin.setRange(1, 5)
        self.custom_refine_scales_spin.setValue(2)
        self.custom_refine_scales_spin.setStyleSheet("background-color: #1E1E1E; color: #ffffff; border: 1px solid #333333; border-radius: 3px; padding: 2px;")
        custom_grid.addWidget(lbl_refine_scales, 8, 0)
        custom_grid.addWidget(self.custom_refine_scales_spin, 8, 1)

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
        custom_grid.addWidget(lbl_texture_res, 9, 0)
        custom_grid.addWidget(self.custom_texture_res_combo, 9, 1)
        
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
        
        self.process_btn = QPushButton("▶  Start Processing", step2_box)
        self.process_btn.setObjectName("ProcessBtn")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._start_processing)
        
        self.progress_bar = QProgressBar(step2_box)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        
        self.status_label = QLabel("Status: Idle", step2_box)
        self.status_label.setStyleSheet("color: #a3a3a3; font-style: italic;")
        
        step2_layout.addWidget(s2_title)
        step2_layout.addWidget(self.quality_label)
        step2_layout.addWidget(self.quality_combo)
        step2_layout.addWidget(self.gpu_label)
        step2_layout.addWidget(self.gpu_combo)
        step2_layout.addWidget(self.plain_surfaces_checkbox)
        step2_layout.addWidget(self.advanced_toggle_btn)
        step2_layout.addWidget(self.advanced_panel)
        step2_layout.addWidget(self.standalone_panel)
        step2_layout.addWidget(self.process_btn)
        step2_layout.addWidget(self.progress_bar)
        step2_layout.addWidget(self.status_label)
        scroll_content_layout.addWidget(step2_box)
        
        # Step 3 Export Mesh option has been moved to the file header menu.
        
        # 3D Visualizer Toggle Button
        self.view_scene_btn = QPushButton("Show 3D Viewer", scroll_content)
        self.view_scene_btn.setEnabled(False)
        self.view_scene_btn.clicked.connect(self._toggle_viewer_mode)
        self.view_scene_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00E676;
            }
            QPushButton:disabled {
                background-color: #202020;
                color: #555555;
                border-color: #2D2D2D;
            }
        """)
        scroll_content_layout.addWidget(self.view_scene_btn)
        
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
        self.viewer_widget.camera_changed.connect(self._on_camera_changed)
        self.viewer_widget.action_new.triggered.connect(self._new_project)
        self.viewer_widget.action_save.triggered.connect(self._save_project)
        self.viewer_widget.action_load.triggered.connect(self._load_project)
        self.viewer_widget.action_recover.triggered.connect(self._retrieve_last_session)
        self.viewer_widget.action_export_dense.triggered.connect(lambda: self._export_mesh(".glb"))
        self.viewer_widget.action_export_sparse.triggered.connect(lambda: self._export_mesh(".ply"))
        self.viewer_widget.action_export_glb.triggered.connect(lambda: self._export_mesh(".glb"))
        self.viewer_widget.action_export_obj.triggered.connect(lambda: self._export_mesh(".obj"))
        self.viewer_widget.action_export_usdz.triggered.connect(lambda: self._export_mesh(".usdz"))
        self.viewer_widget.action_mobile_import.triggered.connect(self._on_import_from_mobile_clicked)
        self.viewer_widget.action_mobile_export.triggered.connect(self._on_send_to_mobile_clicked)
        self.viewer_widget.action_upload_proximap.triggered.connect(self._upload_to_proximap)
        
        # Initialize VisPy Canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor=self.viewport_bg_color)
        if hasattr(self.canvas, '_keys_check') and 'escape' in self.canvas._keys_check:
            del self.canvas._keys_check['escape']
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'arcball' # Default camera mode is Arcball
        
        # Add native VisPy canvas widget to the layout
        self.viewer_widget.container_area_layout.addWidget(self.canvas.native)
        self.viewer_widget.bg_btn.clicked.connect(self._choose_bg_color)
        
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
        
        right_layout.addWidget(self.viewer_widget, stretch=4)
        
        # Tabbed panel containing Photos and Console
        self.bottom_tabs = QTabWidget(right_panel)
        self.bottom_tabs.setObjectName("BottomTabs")
        self.bottom_tabs.setTabPosition(QTabWidget.South)
        
        # Photos Tab
        self.photos_tab = PhotosTabWidget(self.bottom_tabs)
        self.photos_tab.btn_remove_selected.clicked.connect(self._remove_selected_photos)
        self.photos_tab.btn_add_photos.clicked.connect(self._add_photos_dialog)
        
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
        self.bottom_tabs.addTab(self.photos_tab, "Photos")
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
        finally:
            QApplication.restoreOverrideCursor()
            self._mesh_editor_loading = False

    def _update_system_badge(self):
        """Calculates system resource quality badge and updates style dynamically."""
        if self.total_ram_gb >= 8.0:
            status_text = "SYSTEM READY (Optimal RAM)"
            badge_color = "#00E676"  # Bright green
            text_color = "#121212"
        elif self.total_ram_gb >= 4.0:
            status_text = "SYSTEM WARN (Low Memory Mode)"
            badge_color = "#FFD700"  # Yellow
            text_color = "#121212"
        else:
            status_text = "SYSTEM INSUFFICIENT (Below 4GB)"
            badge_color = "#D50000"  # Deep red
            text_color = "#ffffff"
            
        # Append GPU status details
        gpu_info = "dGPU Active" if self.dgpu_detected else "iGPU Fallback Active"
        self.badge.setText(f"{status_text}\n{gpu_info}")
        self.badge.setStyleSheet(
            f"background-color: {badge_color}; color: {text_color}; "
            "font-weight: bold; border-radius: 4px; padding: 6px; font-size: 11px;"
        )

    def _set_process_btn_state(self, state: str):
        """
        Dynamically updates process button colors, text, and enabled state.
        """
        if state == "idle":
            self.process_btn.setText("▶  Start Processing")
            self.process_btn.setEnabled(False)
            self.process_btn.setStyleSheet("""
                QPushButton#ProcessBtn {
                    background-color: #202020;
                    color: #555555;
                    border: 1px solid #2D2D2D;
                }
            """)
        elif state == "ready":
            self.process_btn.setText("▶  Start Processing")
            self.process_btn.setEnabled(True)
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
                padding: 12px;
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
            QPushButton#BgRemoveBtn {
                background-color: #2a1b40;
                color: #d8c8f0;
                border: 1px solid #5a3d8c;
            }
            QPushButton#BgRemoveBtn:hover:enabled {
                background-color: #3b275c;
                border-color: #00E676;
            }
            QPushButton#BgRemoveBtn:pressed:enabled {
                background-color: #1e1230;
            }
            QPushButton#BgRemoveBtn:disabled {
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
        """
        self.setStyleSheet(qss)

    def _handle_dropped_images(self, files: list):
        if hasattr(self, 'standalone_cloud_path') and self.standalone_cloud_path:
            self._clear_standalone_cloud_clicked()
        self.image_list = files
        self.img_count_label.setText(f"Images Loaded: {len(files)}")
        self.photos_tab.set_images(self.image_list)
        
        # Scan actual EXIF camera model using Pillow
        camera_name = "Undetected"
        if files:
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                with Image.open(files[0]) as img:
                    exif = img.getexif()
                    if exif:
                        exif_dict = {TAGS.get(k, k): v for k, v in exif.items()}
                        make = exif_dict.get("Make", "").strip()
                        model = exif_dict.get("Model", "").strip()
                        if model:
                            if make and make.upper() not in model.upper():
                                camera_name = f"{make} {model}"
                            else:
                                camera_name = model
            except Exception:
                pass
                
        display_camera_name = self._camera_name_for_display(camera_name)
        self.camera_label.setText(f"Camera: {display_camera_name}")
        self.camera_label.setToolTip(camera_name if camera_name else "Undetected")
        if files:
            self.console_text.append(f"[INFO] Successfully imported {len(files)} files. Camera identified: {camera_name}")
            self._set_process_btn_state("ready")
            self.bg_remove_btn.setEnabled(True)
        else:
            self.console_text.append("[INFO] Image list cleared.")
            self._set_process_btn_state("idle")
            self.bg_remove_btn.setEnabled(False)

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
        
        # Filter out selected images
        selected_set = set(selected)
        self.image_list = [f for f in self.image_list if f not in selected_set]
        
        # Refresh UI
        self._handle_dropped_images(self.image_list)
        self.console_text.append(f"[INFO] Removed {len(selected)} selected image(s).")

    def _add_photos_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images/Videos to Add", self.last_accessed_dir, 
            "Supported Files (*.png *.jpg *.jpeg *.tif *.tiff *.mp4 *.mov *.avi *.mkv);;Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if files:
            self.last_accessed_dir = os.path.dirname(files[0])
            images = []
            videos = []
            ignored = []
            for f in files:
                normalized = os.path.normpath(f)
                ext = os.path.splitext(normalized)[1].lower()
                if ext in IMAGE_EXTS:
                    images.append(normalized)
                elif ext in VIDEO_EXTS:
                    videos.append(normalized)
                else:
                    ignored.append(os.path.basename(normalized))
            if ignored:
                self._warn_ignored_files(ignored)
            if images or videos:
                self._route_import(images, videos, append_to_existing=True)

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

    def _on_files_dropped(self, files: list):
        images = []
        videos = []
        for f in files:
            normalized = os.path.normpath(f)
            ext = os.path.splitext(normalized)[1].lower()
            if ext in IMAGE_EXTS:
                images.append(normalized)
            elif ext in VIDEO_EXTS:
                videos.append(normalized)
        if images or videos:
            self._route_import(images, videos, append_to_existing=False)

    def _warn_ignored_files(self, ignored: list):
        msg = "The following files were ignored because they are not supported images or videos:\n\n"
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
        self.extraction_queue = list(videos)
        self.extracted_frames = []
        self.extraction_interval = interval
        self.extraction_blur = blur
        self.total_videos_to_extract = len(videos)
        self.progress_bar.setValue(0)
        
        self.browse_btn.setEnabled(False)
        self.mobile_import_btn.setEnabled(False)
        self.bg_remove_btn.setEnabled(False)
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

    # def _import_reference_cloud_clicked(self):
    #     file_path, _ = QFileDialog.getOpenFileName(
    #         self,
    #         "Import Reference Point Cloud",
    #         self.last_accessed_dir,
    #         "Point Cloud Files (*.ply *.las *.laz *.xyz *.pts *.txt)"
    #     )
    #     if not file_path:
    #         return
    # 
    #     self.last_accessed_dir = os.path.dirname(file_path)
    #     self.ref_cloud_path = file_path
    #     filename = os.path.basename(file_path)
    # 
    #     self.ref_cloud_label.setText(f"✓ {filename}")
    #     self.ref_cloud_container.setVisible(True)
    #     self.console_text.append(f"[REF_CLOUD] Selected external reference cloud: {file_path}")
    # 
    # def _clear_reference_cloud_clicked(self):
    #     self.ref_cloud_path = None
    #     self.ref_cloud_label.setText("")
    #     self.ref_cloud_container.setVisible(False)
    #     self.console_text.append("[REF_CLOUD] Reference cloud selection cleared.")

    def _import_standalone_cloud_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Standalone Point Cloud",
            self.last_accessed_dir,
            "Point Cloud Files (*.ply *.las *.laz *.xyz *.pts *.txt)"
        )
        if not file_path:
            return

        self.last_accessed_dir = os.path.dirname(file_path)

        # 1. Instant header-peek for color detection (reads ~4 KB, never blocks)
        import point_cloud_io
        if not hasattr(point_cloud_io, "peek_has_colors"):
            import importlib
            importlib.reload(point_cloud_io)
        peek_fn = getattr(point_cloud_io, "peek_has_colors", lambda path: True)
        has_colors = peek_fn(file_path)

        # 2. Update UI immediately — no freeze
        self.standalone_cloud_path = file_path
        filename = os.path.basename(file_path)
        self.standalone_cloud_label.setText(f"✓ {filename}")
        self.standalone_cloud_container.setVisible(True)
        self._enter_standalone_mode(has_colors)

        # 3. Kick off viewer preview in background (non-blocking)
        self.view_scene_btn.setEnabled(True)
        self.viewer_widget.mode_select.blockSignals(True)
        self.viewer_widget.mode_select.setCurrentIndex(1)
        self.viewer_widget.mode_select.blockSignals(False)
        self._reload_viewer(file_path)

        # 4. Load full point cloud metadata in background (point count, normals, etc.)
        self.console_text.append(f"[STANDALONE] Loading point cloud metadata in background: {filename}...")
        self._cloud_import_worker = CloudImportWorker(file_path, parent=self)
        self._cloud_import_worker.finished.connect(self._on_cloud_import_done)
        self._cloud_import_worker.error.connect(
            lambda err: self.console_text.append(f"[WARNING] Background cloud load warning: {err}")
        )
        self._cloud_import_worker.start()

    def _clear_standalone_cloud_clicked(self):
        self.standalone_cloud_path = None
        self.standalone_cloud_label.setText("")
        self.standalone_cloud_container.setVisible(False)
        self.console_text.append("[STANDALONE] Standalone cloud cleared.")
        self._exit_standalone_mode()

    def _enter_standalone_mode(self, has_colors: bool):
        # 1. Clear any loaded images
        self.image_list = []
        self.extracted_frames = []
        self.img_count_label.setText("Images Loaded: 0 (Standalone Mode)")
        self.camera_label.setText("Camera: N/A (Direct point cloud reconstruction)")
        self.bg_remove_btn.setEnabled(False)
        
        # 2. Clear & disable reference cloud fusion
        # self._clear_reference_cloud_clicked()
        # self.ref_cloud_btn.setEnabled(False)
        
        # 3. Disable images directory browse/mobile import
        self.browse_btn.setEnabled(False)
        self.mobile_import_btn.setEnabled(False)

        # 4. Hide photogrammetry options in Step 2
        self.quality_label.setVisible(False)
        self.quality_combo.setVisible(False)
        self.gpu_label.setVisible(False)
        self.gpu_combo.setVisible(False)
        self.plain_surfaces_checkbox.setVisible(False)
        self.advanced_toggle_btn.setVisible(False)
        self.advanced_panel.setVisible(False)

        # 5. Show standalone options
        self.standalone_panel.setVisible(True)
        self.vertex_color_toggle.setVisible(has_colors)
        self.vertex_color_toggle.setChecked(has_colors)
        
        # 6. Enable process button
        self._set_process_btn_state("ready")
        self.console_text.append("[STANDALONE] UI transitioned to Standalone Reconstruction mode.")

    def _exit_standalone_mode(self):
        # 1. Enable main buttons again
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        # self.ref_cloud_btn.setEnabled(True)
        
        self.img_count_label.setText("Images Loaded: 0")
        self.camera_label.setText("Camera: Undetected")
        
        # 2. Show photogrammetry options
        self.quality_label.setVisible(True)
        self.quality_combo.setVisible(True)
        self.gpu_label.setVisible(True)
        self.gpu_combo.setVisible(True)
        self.plain_surfaces_checkbox.setVisible(True)
        self.advanced_toggle_btn.setVisible(True)
        self.advanced_panel.setVisible(False)

        # 3. Hide standalone panel
        self.standalone_panel.setVisible(False)
        
        # 4. Reset process button state
        self._set_process_btn_state("idle")
        self.console_text.append("[STANDALONE] UI exited Standalone Reconstruction mode.")

    def _on_standalone_poisson_depth_changed(self, value):
        self.standalone_poisson_label.setText(f"Poisson Depth: {value}")

    def _cancel_video_extraction(self):
        self.console_text.append("[VIDEO] Cancelling video extraction...")
        if hasattr(self, 'extraction_worker') and self.extraction_worker:
            self.extraction_worker.cancel()
        self._cleanup_extraction_ui(cancelled=True)

    def _cleanup_extraction_ui(self, cancelled=False):
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        self.bg_remove_btn.setEnabled(len(self.image_list) > 0)
        
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
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        self.bg_remove_btn.setEnabled(True)
        
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
                
        self._handle_dropped_images(self.image_list)
        self.status_label.setText("Extraction complete!")
        self.console_text.append(f"[VIDEO] Complete. Added {added_count} frames from video sources.")

    def _remove_backgrounds_clicked(self):
        if not self.image_list:
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Background Removal")
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("Do you want to remove the background of all loaded images?")
        msg_box.setInformativeText(
            "This will create preprocessed working copies of the images in the project's temporary reconstruction folder and remove their backgrounds offline.\n\n"
            "Your original camera files will NOT be modified.\n\n"
            "Do you want to proceed?"
        )
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        
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
        self.browse_btn.setEnabled(False)
        self.mobile_import_btn.setEnabled(False)
        self.bg_remove_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self._set_export_actions_enabled(False)
        self.photos_tab.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.status_label.setText("Preparing working copies...")
        
        # Create reconstruction out workspace folder for preprocessed images
        import shutil
        preprocessed_dir = os.path.join(get_reconstruction_out_dir(), "preprocessed_images")
        
        # Clean up any existing preprocessed folder to avoid mix-up
        if os.path.exists(preprocessed_dir):
            try:
                shutil.rmtree(preprocessed_dir)
            except Exception:
                pass
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        self.console_text.append(f"[PREP] Copying {len(self.image_list)} images to workspace: {preprocessed_dir}")
        
        copied_list = []
        for path in self.image_list:
            filename = os.path.basename(path)
            dest_path = os.path.join(preprocessed_dir, filename)
            try:
                shutil.copy2(path, dest_path)
                copied_list.append(os.path.normpath(dest_path))
            except Exception as e:
                self.console_text.append(f"[ERROR] Failed to copy {filename} to workspace: {e}")
                
        if not copied_list:
            self.console_text.append("[ERROR] No images could be prepared in the workspace directory.")
            self._on_bg_removal_finished(False, self.image_list, "Failed to copy images to workspace.")
            return

        from pipeline_manager import BackgroundRemovalWorker
        self.worker = BackgroundRemovalWorker(copied_list, self)
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.log_message.connect(self._append_log)
        self.worker.finished.connect(self._on_bg_removal_finished)
        
        self.console_text.append("[START] Initializing background removal worker thread...")
        self.worker.start()

    def _on_bg_removal_finished(self, success: bool, updated_list: list, message: str):
        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
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

    def _stage_images_for_reconstruction(self) -> str | None:
        """
        Copy every path in self.image_list into a single flat staging directory
        so PipelineWorker can find all images (originals + extracted frames) in one place.

        Returns the staging directory path, or None if no images could be staged.
        """
        import shutil
        staging_dir = os.path.join(get_reconstruction_out_dir(), "input_images")
        
        # Always start fresh so stale files from a previous run don't contaminate this one
        if os.path.exists(staging_dir):
            try:
                shutil.rmtree(staging_dir)
            except Exception as e:
                self.console_text.append(f"[WARNING] Could not clear staging dir: {e}")
        os.makedirs(staging_dir, exist_ok=True)

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
                shutil.copy2(path, dest)
                staged.append(dest)
            except Exception as e:
                self.console_text.append(f"[WARNING] Could not stage {os.path.basename(path)}: {e}")

        if not staged:
            return None

        self.console_text.append(f"[PREP] Staged {len(staged)} images for reconstruction → {staging_dir}")
        return staging_dir

    def _toggle_advanced_options(self):
        is_visible = not self.advanced_panel.isVisible()
        self.advanced_panel.setVisible(is_visible)
        self.advanced_toggle_btn.setText("▾  Advanced Options" if is_visible else "▸  Advanced Options")

    def _on_mesh_mode_changed(self, index):
        self.poisson_widget.setVisible(index == 1)

    def _on_poisson_depth_changed(self, value):
        self.poisson_depth_label.setText(f"Poisson Depth: {value}")

    def _on_custom_settings_toggled(self, checked):
        self.mapper_label.setEnabled(checked)
        self.mapper_combo.setEnabled(checked)
        self.mesh_mode_label.setEnabled(checked)
        self.mesh_mode_combo.setEnabled(checked)
        self.poisson_widget.setEnabled(checked)
        self.custom_settings_container.setEnabled(checked)
        self.quality_label.setEnabled(not checked)
        self.quality_combo.setEnabled(not checked)
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
            self.custom_guided_check.setChecked(self.plain_surfaces_checkbox.isChecked())
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
            self.custom_ba_check.setChecked(True)
            self.custom_densify_res_combo.setCurrentIndex(0)
            self.custom_densify_views_spin.setValue(8)
            self.custom_refine_scales_spin.setValue(3)
            self.custom_texture_res_combo.setCurrentIndex(0)

    def _start_processing(self):
        if not self.standalone_cloud_path and not self.image_list:
            return
            
        # Terminate any active viewer to prevent lock conflict on MVS files during reconstruction
        self._terminate_viewer()
        
        self._set_process_btn_state("progress")
        self.browse_btn.setEnabled(False)
        self.mobile_import_btn.setEnabled(False)
        self._set_export_actions_enabled(False)
        self.bg_remove_btn.setEnabled(False)
        # self.ref_cloud_btn.setEnabled(False)
        # self.ref_cloud_clear_btn.setEnabled(False)
        self.quality_combo.setEnabled(False)
        self.gpu_combo.setEnabled(False)
        self.plain_surfaces_checkbox.setEnabled(False)
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
            self.standalone_cloud_btn.setEnabled(False)
            self.standalone_cloud_clear_btn.setEnabled(False)
            self.standalone_poisson_slider.setEnabled(False)
            self.vertex_color_toggle.setEnabled(False)
            
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
            self.worker.start()
            self._update_file_menu_states()
            return
        
        # stage all images into one flat directory
        image_dir = self._stage_images_for_reconstruction()
        if image_dir is None:
            self.console_text.append("[ERROR] Could not stage images. Aborting reconstruction.")
            self._set_process_btn_state("ready")
            self.browse_btn.setEnabled(True)
            self.mobile_import_btn.setEnabled(True)
            self.bg_remove_btn.setEnabled(True)
            # self.ref_cloud_btn.setEnabled(True)
            # self.ref_cloud_clear_btn.setEnabled(True)
            self.gpu_combo.setEnabled(True)
            self.plain_surfaces_checkbox.setEnabled(True)
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
        has_plain = self.plain_surfaces_checkbox.isChecked()
        mesh_mode = "poisson" if self.mesh_mode_combo.currentIndex() == 1 else "default"
        poisson_depth = self.poisson_depth_slider.value()

        custom_params = None
        if self.custom_settings_toggle.isChecked():
            custom_params = {
                "colmap_max_num_features": self.custom_features_spin.value(),
                "colmap_max_num_matches": self.custom_matches_spin.value(),
                "guided_matching": "1" if self.custom_guided_check.isChecked() else "0",
                "run_bundle_adjuster": self.custom_ba_check.isChecked(),
                "densify_res": str(self.custom_densify_res_combo.currentIndex()),
                "densify_views": str(self.custom_densify_views_spin.value()),
                "refine_scales": str(self.custom_refine_scales_spin.value()),
                "texture_res": str(self.custom_texture_res_combo.currentIndex())
            }

        from pipeline_manager import PipelineWorker
        self.worker = PipelineWorker(
            image_dir, 
            output_dir, 
            quality_preset=quality_preset, 
            gpu_mode=gpu_mode, 
            has_plain_surfaces=has_plain,
            mapper_mode=mapper_mode,
            ref_cloud_path=None,
            mesh_mode=mesh_mode,
            poisson_depth=poisson_depth,
            custom_params=custom_params,
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
                self.viewer_widget.mode_select.blockSignals(True)
                self.viewer_widget.mode_select.setCurrentIndex(0)
                self.viewer_widget.mode_select.blockSignals(False)
                self._reload_viewer(scene_mvs)
                self.view_scene_btn.setEnabled(True)

    def _on_pipeline_finished(self, success: bool, msg: str):
        if hasattr(self, 'standalone_cloud_path') and self.standalone_cloud_path:
            # Standalone reconstruction mode finished cleanup
            self.view_scene_btn.setEnabled(True)
            self.standalone_cloud_btn.setEnabled(True)
            self.standalone_cloud_clear_btn.setEnabled(True)
            self.standalone_poisson_slider.setEnabled(True)
            self.vertex_color_toggle.setEnabled(True)
            
            if success:
                self._set_process_btn_state("ready")
                self.console_text.append(f"[FINISHED] {msg}")
                self._update_upload_button_state()
                
                mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
                self.viewer_widget.set_mvs_directory(mvs_dir)
                self.viewer_widget.mode_select.blockSignals(True)
                self.viewer_widget.mode_select.setCurrentIndex(2) # Default to mesh mode (index 2)
                self.viewer_widget.mode_select.blockSignals(False)
                
                mesh_path = self.viewer_widget.get_selected_file_path()
                if mesh_path:
                    self._reload_viewer(mesh_path)
            else:
                self._set_process_btn_state("failed")
                self.console_text.append(f"[FAILED] Reconstruction failed: {msg}")
            self._update_file_menu_states()
            return

        self.browse_btn.setEnabled(True)
        self.mobile_import_btn.setEnabled(True)
        self.view_scene_btn.setEnabled(True)
        self.gpu_combo.setEnabled(True)
        self.plain_surfaces_checkbox.setEnabled(True)
        self.custom_settings_toggle.setEnabled(True)
        self._on_custom_settings_toggled(self.custom_settings_toggle.isChecked())
        self.advanced_toggle_btn.setEnabled(True)
        # self.ref_cloud_btn.setEnabled(True)
        # self.ref_cloud_clear_btn.setEnabled(True)
        self.bg_remove_btn.setEnabled(len(self.image_list) > 0)
        
        if success:
            self._set_process_btn_state("ready")
            self.console_text.append(f"[FINISHED] {msg}")
            self._update_upload_button_state()
            
            mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
            self.viewer_widget.set_mvs_directory(mvs_dir)
            self.viewer_widget.mode_select.blockSignals(True)
            
            # Pick best available viewer mode
            mesh_exists = False
            for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_texture.obj", "scene_dense_mesh_refine.ply", "scene_dense_mesh_refcloud.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
                if os.path.exists(os.path.join(mvs_dir, candidate)):
                    mesh_exists = True
                    break
            
            dense_exists = os.path.exists(os.path.join(mvs_dir, "scene_dense.mvs"))
            
            if mesh_exists:
                self.viewer_widget.mode_select.setCurrentIndex(2)
            elif dense_exists:
                self.viewer_widget.mode_select.setCurrentIndex(1)
            else:
                self.viewer_widget.mode_select.setCurrentIndex(0)
                
            self.viewer_widget.mode_select.blockSignals(False)
            
            mesh_path = self.viewer_widget.get_selected_file_path()
            if mesh_path:
                self._reload_viewer(mesh_path)
        else:
            self._set_process_btn_state("failed")
            self.console_text.append(f"[FAILED] Reconstruction failed: {msg}")
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
                    self.console_text.append("[INFO] Converting OBJ to GLB using obj2gltf...")
                    try:
                        import subprocess
                        import sys
                        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                        # Run obj2gltf to convert obj to glb with embedded textures (-b)
                        subprocess.run(["obj2gltf", "-i", src_obj, "-o", file_path, "-b"], capture_output=True, text=True, check=True, shell=True, creationflags=creationflags)
                        self.console_text.append(f"[EXPORT] GLB mesh successfully written to {file_path}")
                    except Exception as e:
                        self.console_text.append(f"[ERROR] Failed to convert to GLB. Ensure Node.js and obj2gltf are installed: {e}")
                else:
                    self.console_text.append(f"[ERROR] Could not find reconstructed GLB or OBJ file at {mvs_out}")
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
        """Checks if a previous reconstruction scene exists and updates recover action state."""
        output_dir = get_reconstruction_out_dir()
        mvs_dir = os.path.join(output_dir, "mvs")
        scene_mvs = os.path.join(mvs_dir, "scene.mvs")
        if os.path.exists(scene_mvs):
            self.viewer_widget.set_mvs_directory(mvs_dir)
            self.console_text.append("[INFO] Detected previous reconstruction. Go to File Menu -> Recover Last Session to load it.")
        self._update_file_menu_states()

    def _retrieve_last_session(self):
        """Retrieves and displays the last session, and enables export/upload buttons."""
        self.view_scene_btn.setEnabled(True)
        self.console_text.append("[INFO] Retrieved last session. 3D Viewer is ready to display.")
        self._update_upload_button_state()
        
        # Determine the best view mode and load it immediately
        output_dir = get_reconstruction_out_dir()
        mvs_dir = os.path.join(output_dir, "mvs")
        
        self.viewer_widget.mode_select.blockSignals(True)
        mesh_exists = False
        for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_texture.obj", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
            if os.path.exists(os.path.join(mvs_dir, candidate)):
                mesh_exists = True
                break
        
        dense_exists = os.path.exists(os.path.join(mvs_dir, "scene_dense.mvs"))
        
        if mesh_exists:
            self.viewer_widget.mode_select.setCurrentIndex(2)
        elif dense_exists:
            self.viewer_widget.mode_select.setCurrentIndex(1)
        else:
            self.viewer_widget.mode_select.setCurrentIndex(0)
            
        self.viewer_widget.mode_select.blockSignals(False)
        
        path = self.viewer_widget.get_selected_file_path()
        if path:
            self._reload_viewer(path)
            
        self._update_file_menu_states()

    def _save_project(self):
        mvs_dir = self.viewer_widget.current_mvs_dir
        if not mvs_dir or not os.path.exists(mvs_dir):
            self.console_text.append("[ERROR] No active 3D reconstruction session to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project File",
            self.last_accessed_dir,
            "Proximap Project (*.pxm)"
        )
        if not file_path:
            return
            
        if not file_path.lower().endswith(".pxm"):
            file_path += ".pxm"
            
        self.last_accessed_dir = os.path.dirname(file_path)
        self.console_text.append(f"[SAVE] Packing reconstruction assets from {mvs_dir} to {file_path}...")
        
        self.save_dialog = ProjectProgressDialog("Saving Project", "Compressing assets and saving project file...", self)
        self.save_dialog.show()
        
        self.save_worker = SaveWorker(mvs_dir, file_path)
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
        self.view_scene_btn.setEnabled(True)
        self._update_upload_button_state()
        
        # Determine the best view mode and load it immediately
        self.viewer_widget.mode_select.blockSignals(True)
        mesh_exists = False
        for candidate in ["scene_dense_mesh_texture.ply", "scene_dense_mesh_texture.obj", "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply", "scene_mesh.ply"]:
            if os.path.exists(os.path.join(mvs_dir, candidate)):
                mesh_exists = True
                break
        
        dense_exists = os.path.exists(os.path.join(mvs_dir, "scene_dense.mvs"))
        
        if mesh_exists:
            self.viewer_widget.mode_select.setCurrentIndex(2)
        elif dense_exists:
            self.viewer_widget.mode_select.setCurrentIndex(1)
        else:
            self.viewer_widget.mode_select.setCurrentIndex(0)
            
        self.viewer_widget.mode_select.blockSignals(False)
        
        path = self.viewer_widget.get_selected_file_path()
        if path:
            self._reload_viewer(path)
            
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
        self.view_scene_btn.setEnabled(False)

        # 4. Re-enable Step 1 buttons and restore photogrammetry settings panel
        self._exit_standalone_mode()
        self.bg_remove_btn.setEnabled(False)

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
            return

        self.viewer_widget.action_new.setEnabled(True)
            
        # We can save if we have a valid MVS directory containing files
        mvs_dir = self.viewer_widget.current_mvs_dir
        has_assets = False
        if mvs_dir and os.path.exists(mvs_dir):
            if os.path.exists(os.path.join(mvs_dir, "scene.mvs")):
                has_assets = True
        self.viewer_widget.action_save.setEnabled(has_assets)
        
        # We can load at any time when not running
        self.viewer_widget.action_load.setEnabled(True)
        
        # We can recover if there's an existing scene in the base reconstruction directory
        output_dir = get_reconstruction_out_dir()
        scene_mvs = os.path.join(output_dir, "mvs", "scene.mvs")
        self.viewer_widget.action_recover.setEnabled(os.path.exists(scene_mvs))

    def _toggle_viewer_mode(self):
        """Reloads the embedded 3D viewer."""
        path = self.viewer_widget.get_selected_file_path()
        if path:
            self._reload_viewer(path)

    def _on_camera_changed(self, index):
        if self.view is None:
            return
            
        # Switch camera mode (0: Arcball, 1: Turntable, 2: Fly, 3: PanZoom, 4: Magnify)
        if index == 0:
            self.view.camera = 'arcball'
        elif index == 1:
            self.view.camera = 'turntable'
        elif index == 2:
            self.view.camera = 'fly'
            try:
                from vispy.util.keys import SPACE
                self.view.camera._keymap[SPACE] = (1, 3)
                self.view.camera.auto_roll = False
            except Exception:
                pass
        elif index == 3:
            self.view.camera = 'panzoom'
        elif index == 4:
            import vispy.scene.cameras as cams
            self.view.camera = cams.MagnifyCamera()
            
        # Re-center and re-scale if we have loaded points
        if self._last_points is not None and len(self._last_points) > 0:
            import numpy as np
            bbox_min = np.min(self._last_points, axis=0)
            bbox_max = np.max(self._last_points, axis=0)
            center = (bbox_min + bbox_max) / 2.0
            scale = np.max(bbox_max - bbox_min)
            
            if hasattr(self.view.camera, 'rect'):
                self.view.camera.rect = (bbox_min[0], bbox_min[1], scale, scale)
            else:
                if hasattr(self.view.camera, 'center'):
                    self.view.camera.center = center
                if hasattr(self.view.camera, 'distance'):
                    self.view.camera.distance = max(0.1, scale * 1.5)
                elif hasattr(self.view.camera, 'scale_factor'):
                    self.view.camera.scale_factor = scale
                    
                if index == 1:
                    self.view.camera.elevation = 30
                    self.view.camera.azimuth = 45
                    
        # Update overlay content if visible
        if hasattr(self, 'overlay_label') and self.overlay_label.isVisible():
            self._update_overlay_content()
            self._position_overlay()

    def _clear_visuals(self):
        if hasattr(self, 'markers_visual') and self.markers_visual is not None:
            try:
                self.markers_visual.unparent()
            except AttributeError:
                self.markers_visual.parent = None
            self.markers_visual = None
        if hasattr(self, 'mesh_visual') and self.mesh_visual is not None:
            try:
                self.mesh_visual.unparent()
            except AttributeError:
                self.mesh_visual.parent = None
            self.mesh_visual = None
        if hasattr(self, 'cameras_visual') and self.cameras_visual is not None:
            try:
                self.cameras_visual.unparent()
            except AttributeError:
                self.cameras_visual.parent = None
            self.cameras_visual = None
        self._last_points = None

    def _read_points3d_binary(self, path_to_model_file):
        import struct
        import numpy as np
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
        return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)

    def _read_images_binary(self, path_to_model_file):
        import struct
        import numpy as np
        images_data = []
        if not os.path.exists(path_to_model_file):
            return images_data
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
                    camera_center = -R.T @ tvec
                    images_data.append({
                        "center": camera_center,
                        "R": R
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
                         
        return vertices, texcoords, faces, texture_path

    def _draw_cameras(self, cameras_data):
        import numpy as np
        if not cameras_data:
            return
            
        d = 0.15 # depth of frustum
        w = 0.12 # half-width of image plane
        h = 0.08 # half-height of image plane
        
        local_corners = np.array([
            [-w, -h, d],
            [ w, -h, d],
            [ w,  h, d],
            [-w,  h, d]
        ])
        
        line_vertices = []
        for cam in cameras_data:
            C = cam["center"]
            R = cam["R"]
            R_cw = R.T
            
            world_corners = []
            for corner in local_corners:
                world_corners.append(R_cw @ corner + C)
                
            # Line segments connections
            for wc in world_corners:
                line_vertices.append(C)
                line_vertices.append(wc)
                
            line_vertices.append(world_corners[0])
            line_vertices.append(world_corners[1])
            
            line_vertices.append(world_corners[1])
            line_vertices.append(world_corners[2])
            
            line_vertices.append(world_corners[2])
            line_vertices.append(world_corners[3])
            
            line_vertices.append(world_corners[3])
            line_vertices.append(world_corners[0])
            
        if line_vertices:
            pos = np.array(line_vertices, dtype=np.float32)
            self.cameras_visual = scene.visuals.Line(
                pos=pos,
                color='#00E676',
                width=1.5,
                connect='segments'
            )
            self.cameras_visual.parent = self.view.scene

    def _render_in_vispy_from_data(self, points, colors, faces, texcoords, texture_path, mode):
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
        else:
            marker_colors = 'white'
            if colors is not None and len(colors) > 0:
                mc = colors.astype(np.float32) / 255.0
                if mc.shape[1] == 3:
                    mc = np.hstack([mc, np.ones((mc.shape[0], 1), dtype=np.float32)])
                marker_colors = mc
            self.markers_visual = scene.visuals.Markers(parent=self.view.scene)
            self.markers_visual.set_data(pos=points, face_color=marker_colors, size=2, edge_width=0)

        self._last_points = points
        self.canvas.native.show()
        self.viewer_widget.fallback_label.hide()

        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        scale  = np.max(bbox_max - bbox_min)
        self.view.camera.center   = center
        self.view.camera.distance = max(0.1, scale * 1.5)
        if self.viewer_widget.cam_select.currentIndex() == 1:
            self.view.camera.elevation = 30
            self.view.camera.azimuth   = 45
        self.canvas.update()

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
            output_dir = get_reconstruction_out_dir()
            points_bin = os.path.join(output_dir, "colmap", "sparse", "points3D.bin")
            if not os.path.exists(points_bin):
                points_bin = os.path.join(output_dir, "colmap", "sparse", "0", "points3D.bin")
                
            if os.path.exists(points_bin):
                points, colors = self._read_points3d_binary(points_bin)
            else:
                scene_ply = os.path.join(output_dir, "mvs", "scene.ply")
                if os.path.exists(scene_ply):
                    points, colors, _ = self._read_ply(scene_ply)
                    
            images_bin = os.path.join(output_dir, "colmap", "sparse", "images.bin")
            if not os.path.exists(images_bin):
                images_bin = os.path.join(output_dir, "colmap", "sparse", "0", "images.bin")
                
            if os.path.exists(images_bin):
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
                    try:
                        import point_cloud_io
                        res = point_cloud_io.load_point_cloud(file_path)
                        if res.success and res.cloud is not None:
                            points = np.asarray(res.cloud.points, dtype=np.float32)
                            if res.has_colors:
                                colors = (np.asarray(res.cloud.colors) * 255.0).astype(np.uint8)
                            else:
                                colors = np.ones((len(points), 3), dtype=np.uint8) * 180
                    except Exception as e:
                        self.console_text.append(f"[WARNING] Vispy fallback loader failed: {e}")
                else:
                    dirname = os.path.dirname(file_path)
                    cand_obj = os.path.join(dirname, "scene_dense_mesh_texture.obj")
                    obj_cand = file_path.replace(".ply", ".obj").replace(".mvs", ".obj")
                    if os.path.exists(cand_obj):
                        vertices, texcoords, faces, texture_path = self._read_obj(cand_obj)
                        points = vertices
                    elif os.path.exists(obj_cand):
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
            self.markers_visual.set_data(
                pos=points,
                face_color=marker_colors,
                size=2,
                edge_width=0
            )
            
        else:
            self.canvas.native.hide()
            self.viewer_widget.fallback_label.setText("No valid 3D points or faces could be parsed.")
            self.viewer_widget.fallback_label.show()
            return
            
        # Store points reference for camera switches
        self._last_points = points
        
        self.canvas.native.show()
        self.viewer_widget.fallback_label.hide()
        
        # Center and zoom camera
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        scale = np.max(bbox_max - bbox_min)
        
        self.view.camera.center = center
        self.view.camera.distance = max(0.1, scale * 1.5)
        # Apply turntable elevation/azimuth if selected
        if self.viewer_widget.cam_select.currentIndex() == 1:
            self.view.camera.elevation = 30
            self.view.camera.azimuth = 45
            
        self.canvas.update()

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
        if not os.path.exists(file_path):
            self.viewer_widget.fallback_label.setText(
                f"File not found: {os.path.basename(file_path)}\nRun reconstruction to generate this file first."
            )
            self.viewer_widget.fallback_label.show()
            self.console_text.append(f"[WARNING] 3D file not found: {file_path}")
            return

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlay()

    def _position_overlay(self):
        if hasattr(self, 'overlay_label') and self.overlay_label.isVisible():
            container_w = self.viewer_widget.container_area.width()
            container_h = self.viewer_widget.container_area.height()
            
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

    def _on_show_controls_changed(self, state):
        visible = (state == Qt.Checked.value or state == 2)
        self.overlay_label.setVisible(visible)
        if visible:
            self._update_overlay_content()
            self._position_overlay()

    def _update_overlay_content(self):
        index = self.viewer_widget.cam_select.currentIndex()
        controls_text = CAMERA_CONTROLS.get(index, "")
        self.overlay_label.setText(controls_text)
        self.overlay_label.adjustSize()

    def _upload_to_proximap(self):
        mvs_out = self._get_active_mvs_dir()
        
        src_glb = os.path.join(mvs_out, "scene_dense_mesh_texture.glb")
        src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
        
        if not os.path.exists(src_glb):
            if os.path.exists(src_obj):
                self.console_text.append("[BRIDGE] Pre-converting reconstructed OBJ to GLB for upload...")
                try:
                    import subprocess
                    import sys
                    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    subprocess.run(["obj2gltf", "-i", src_obj, "-o", src_glb, "-b"], capture_output=True, text=True, check=True, shell=True, creationflags=creationflags)
                except Exception as e:
                    self.console_text.append(f"[BRIDGE ERROR] Could not convert model to GLB. Ensure obj2gltf is installed: {e}")
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
            for f in files:
                normalized = os.path.normpath(f)
                ext = os.path.splitext(normalized)[1].lower()
                if ext in IMAGE_EXTS:
                    images.append(normalized)
                elif ext in VIDEO_EXTS:
                    videos.append(normalized)
            
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
                target_file = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
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
        self.setWindowTitle("Mobile Device Bridge — " + ("Import Media" if mode == "import" else "Download 3D Model"))
        self.setFixedSize(420, 520)
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
        layout.setSpacing(12)

        title = QLabel("📱 Scan with your Mobile Phone", self)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00E676;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Get active network Wi-Fi SSID if possible
        wifi_ssid = None
        if os.name == "nt":
            try:
                import subprocess
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "(Get-NetConnectionProfile).Name"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if res.returncode == 0:
                    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                    if lines:
                        wifi_ssid = lines[0]
            except Exception:
                pass

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

        # QR Code Image Label
        self.qr_label = QLabel(self)
        self.qr_label.setFixedSize(240, 240)
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
        lbl_url = QLabel(primary_url, url_box)
        lbl_url.setStyleSheet("font-size: 13px; font-weight: bold; color: #00E676;")
        lbl_url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        url_layout.addWidget(lbl_type)
        url_layout.addWidget(lbl_url)
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

        lbl = QLabel("Choose where to save imported images/videos on your PC:", self)
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
        import subprocess
        
        # 1. Get active network connections from psutil
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        active_nets = []
        
        for name, stat in stats.items():
            if stat.isup and name in addrs:
                for addr in addrs[name]:
                    if addr.family == socket.AF_INET:
                        ip = addr.address
                        if ip and ip != "127.0.0.1" and not ip.startswith("169.254"):
                            name_lower = name.lower()
                            if any(x in name_lower for x in ["wi-fi", "wifi", "wlan", "wireless", "802.11"]):
                                conn_type = "Wi-Fi"
                            elif any(x in name_lower for x in ["local area connection*", "direct"]):
                                conn_type = "Wi-Fi Hotspot"
                            elif any(x in name_lower for x in ["ethernet", "lan"]):
                                conn_type = "Ethernet"
                            else:
                                conn_type = "LAN"
                            active_nets.append((name, ip, conn_type))
                            
        # 2. Try to get Wi-Fi network name / SSID (Windows)
        wifi_ssid = None
        if os.name == "nt":
            try:
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "(Get-NetConnectionProfile).Name"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if res.returncode == 0:
                    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                    if lines:
                        wifi_ssid = lines[0]
            except Exception:
                pass
                
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
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    
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
