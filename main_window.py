import os
import sys
import subprocess
import ctypes
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QProgressBar, QRadioButton, QButtonGroup,
    QFrame, QFileDialog, QTextEdit, QStackedWidget, QComboBox,
    QScrollArea, QTabWidget, QGridLayout, QCheckBox, QSlider
)

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

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


from PySide6.QtCore import Qt, QSize, Signal, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QFont, QWindow, QPixmap

import hardware_profiler
from pipeline_manager import PipelineWorker

import http.server
import socketserver
import threading
import webbrowser

class ModelServerHandler(http.server.BaseHTTPRequestHandler):
    model_path = ""
    
    def log_message(self, format, *args):
        # Suppress standard logging to console for clean output
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()
        
    def do_OPTIONS(self):
        self.send_response(200)
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
        
        self.instruction_label = QLabel("Drag images or folder here to start", self)
        self.instruction_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #b3b3b3;")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        
        self.sub_label = QLabel("Supports JPG, PNG, TIFF", self)
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
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if os.path.isdir(local_path):
                # Scan folder for images
                for root, _, filenames in os.walk(local_path):
                    for filename in filenames:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                            files.append(os.path.join(root, filename))
            elif os.path.isfile(local_path):
                if local_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    files.append(local_path)
                    
        if files:
            self.images_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()


class ViewerWrapperWidget(QFrame):
    """
    Widget wrapper that hosts the embedded OpenMVS Viewer.exe native window
    and provides a control bar to reload or change MVS scene modes.
    Now also acts as the main drag-and-drop landing area!
    """
    images_dropped = Signal(list)
    reload_requested = Signal(str)  # Emits target file path to reload
    external_launch_requested = Signal(str)  # Emits target file path to launch externally
    back_requested = Signal()  # Emits when the user wants to go back to import view

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
        
        self.back_btn = QPushButton("⬅ Back to Import", self.control_bar)
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                font-size: 11px;
                padding: 4px 8px;
                font-weight: normal;
                border-color: #444444;
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00E676;
            }
        """)
        
        title_label = QLabel("3D Spatial Visualization", self.control_bar)
        title_label.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 13px; margin-left: 5px;")
        
        # Dropdown to choose MVS scene mode
        self.mode_select = QComboBox(self.control_bar)
        self.mode_select.setMinimumWidth(200)
        self.mode_select.addItems([
            "Sparse Point Cloud & Cameras",
            "Dense Point Cloud",
            "Textured Mesh"
        ])
        
        # Action buttons
        self.reload_btn = QPushButton("🔄 Reload", self.control_bar)
        self.reload_btn.setStyleSheet("font-size: 11px; padding: 4px 8px; font-weight: normal;")
        self.external_btn = QPushButton("↗ Open Externally", self.control_bar)
        self.external_btn.setStyleSheet("font-size: 11px; padding: 4px 8px; font-weight: normal;")
        
        control_layout.addWidget(self.back_btn)
        control_layout.addWidget(title_label)
        control_layout.addStretch()
        control_layout.addWidget(self.mode_select)
        control_layout.addWidget(self.reload_btn)
        control_layout.addWidget(self.external_btn)
        
        layout.addWidget(self.control_bar)
        
        # Container for the embedded window
        self.container_area = QWidget(self)
        self.container_area_layout = QVBoxLayout(self.container_area)
        self.container_area_layout.setContentsMargins(0, 0, 0, 0)
        self.container_area_layout.setSpacing(0)
        
        # A simple fallback label when no viewer is running
        self.fallback_label = QLabel("Drag Images Here or Process to View 3D Scene", self.container_area)
        self.fallback_label.setAlignment(Qt.AlignCenter)
        self.fallback_label.setStyleSheet("color: #737373; font-size: 14px;")
        self.container_area_layout.addWidget(self.fallback_label)
        
        layout.addWidget(self.container_area)
        
        # Setup actions
        self.back_btn.clicked.connect(self._on_back_clicked)
        self.reload_btn.clicked.connect(self._on_reload_clicked)
        self.external_btn.clicked.connect(self._on_external_clicked)
        self.mode_select.currentIndexChanged.connect(self._on_mode_changed)
        
        self.current_mvs_dir = None

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
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if os.path.isdir(local_path):
                # Scan folder for images
                for root, _, filenames in os.walk(local_path):
                    for filename in filenames:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                            files.append(os.path.join(root, filename))
            elif os.path.isfile(local_path):
                if local_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    files.append(local_path)
                    
        if files:
            self.images_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()

    def set_mvs_directory(self, mvs_dir: str):
        self.current_mvs_dir = mvs_dir

    def get_selected_file_path(self) -> str:
        if not self.current_mvs_dir:
            return None
            
        index = self.mode_select.currentIndex()
        if index == 0:
            return os.path.join(self.current_mvs_dir, "scene.mvs")
        elif index == 1:
            return os.path.join(self.current_mvs_dir, "scene_dense.mvs")
        elif index == 2:
            # We want to load the textured mesh.
            # OpenMVS Viewer.exe can load .ply, .obj, or .glb directly with textures.
            # We check files in priority order (textured PLY first for Viewer.exe compatibility, then fallback candidates):
            for candidate in [
                "scene_dense_mesh_texture.ply",
                "scene_dense_mesh_texture.obj",
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
            return os.path.join(self.current_mvs_dir, "scene_dense_mesh_texture.ply")
        return None

    def _on_back_clicked(self):
        self.back_requested.emit()

    def _on_reload_clicked(self):
        path = self.get_selected_file_path()
        if path:
            self.reload_requested.emit(path)

    def _on_external_clicked(self):
        path = self.get_selected_file_path()
        if path:
            self.external_launch_requested.emit(path)

    def _on_mode_changed(self, index):
        path = self.get_selected_file_path()
        if path:
            self.reload_requested.emit(path)


class PhotoItemWidget(QWidget):
    """
    Individual photo thumbnail card display with a selection checkbox.
    """
    def __init__(self, file_path, size, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.size = size
        self.selected = False
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
        
        # Load and scale image to thumbnail
        pixmap = QPixmap(self.file_path)
        if not pixmap.isNull():
            scaled_pix = pixmap.scaled(self.size, self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pix)
        else:
            self.image_label.setText("⚠️")
            self.image_label.setStyleSheet("font-size: 24px;")
            
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
        
    def set_images(self, image_paths):
        self.image_paths = image_paths
        self.rebuild_grid()

    def clear_grid(self):
        while self.layout.count() > 0:
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.image_items.clear()
        
    def rebuild_grid(self):
        self.clear_grid()
        if not self.image_paths:
            return
            
        width = self.width()
        if width < 100:
            width = 400  # Fallback minimum width estimation
            
        col_width = self.thumbnail_size + 20
        cols = max(1, width // col_width)
        
        for idx, path in enumerate(self.image_paths):
            item_widget = PhotoItemWidget(path, self.thumbnail_size, self)
            self.image_items.append(item_widget)
            row = idx // cols
            col = idx % cols
            self.layout.addWidget(item_widget, row, col)
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rebuild_grid()


class PhotosTabWidget(QWidget):
    """
    Tab widget containing the Photos toolbar and dynamic photo grid area.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_list = []
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
        self.btn_select_all = QPushButton("✔️", self.toolbar)
        self.btn_select_all.setToolTip("Select All")
        self.btn_select_all.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
        self.btn_deselect_all = QPushButton("➖", self.toolbar)
        self.btn_deselect_all.setToolTip("Deselect All")
        self.btn_deselect_all.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
        self.btn_remove_selected = QPushButton("❌", self.toolbar)
        self.btn_remove_selected.setToolTip("Remove Selected")
        self.btn_remove_selected.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 12px; background-color: transparent; border: none; } QPushButton:hover { background-color: #333333; border-radius: 4px; }")
        
        self.btn_add_photos = QPushButton("📂", self.toolbar)
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
        self.scroll_area.setWidget(self.grid_widget)
        
        layout.addWidget(self.scroll_area)
        
        # Connections
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_deselect_all.clicked.connect(self.deselect_all)
        self.size_slider.valueChanged.connect(self.change_thumbnail_size)
        
    def set_images(self, image_paths):
        self.image_list = image_paths
        self.grid_widget.set_images(image_paths)
        
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
        self.grid_widget.rebuild_grid()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proximap - Photogrammetry Dashboard")
        self.setMinimumSize(1100, 750)
        self.image_list = []
        self.worker = None
        
        # OpenMVS Viewer Subprocess States
        self.viewer_process = None
        self.viewer_hwnd = None
        self.viewer_timer = QTimer(self)
        self.viewer_timer.timeout.connect(self._check_for_viewer_window)
        self.viewer_timer.setInterval(150)
        
        # Load hardware properties
        self.total_ram_gb = hardware_profiler.get_total_memory() / (1024**3)
        self.available_ram_gb = hardware_profiler.get_available_memory() / (1024**3)
        self.dgpu_detected = not hardware_profiler.use_low_hardware_fallback
        
        self._init_ui()
        self._apply_styling()
        self._check_existing_scene()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
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
        
        # Hardware Status Badge
        self.badge = QLabel("Memory Check...", step1_box)
        self.badge.setObjectName("Badge")
        self.badge.setAlignment(Qt.AlignCenter)
        self._update_system_badge()
        
        self.browse_btn = QPushButton("Select Images Directory", step1_box)
        self.browse_btn.clicked.connect(self._open_dir_dialog)
        
        step1_layout.addWidget(s1_title)
        step1_layout.addWidget(self.img_count_label)
        step1_layout.addWidget(self.camera_label)
        step1_layout.addWidget(self.badge)
        step1_layout.addWidget(self.browse_btn)
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
        step2_layout.addWidget(self.process_btn)
        step2_layout.addWidget(self.progress_bar)
        step2_layout.addWidget(self.status_label)
        scroll_content_layout.addWidget(step2_box)
        
        # STEP 3: Export Mesh
        self.step3_box = QFrame(scroll_content)
        self.step3_box.setObjectName("StepBox")
        step3_layout = QVBoxLayout(self.step3_box)
        
        s3_title = QLabel("Step 3: Export Mesh", self.step3_box)
        s3_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #00E676;")
        
        self.radio_glb = QRadioButton("Export as .glb (Textured)", self.step3_box)
        self.radio_obj = QRadioButton("Export as .obj (Separated)", self.step3_box)
        self.radio_ply = QRadioButton("Export as .ply (Point Cloud)", self.step3_box)
        self.radio_obj.setChecked(True)
        
        self.radio_group = QButtonGroup(self.step3_box)
        self.radio_group.addButton(self.radio_glb)
        self.radio_group.addButton(self.radio_obj)
        self.radio_group.addButton(self.radio_ply)
        
        self.export_btn = QPushButton("Export...", self.step3_box)
        self.export_btn.clicked.connect(self._export_mesh)
        
        self.upload_portal_btn = QPushButton("☁  Upload to ProximaXR", self.step3_box)
        self.upload_portal_btn.clicked.connect(self._upload_to_proximaxr)
        self.upload_portal_btn.setStyleSheet("""
            QPushButton {
                background-color: #1A1A1A;
                color: #00E676;
                border: 1px solid #00E676;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #00E676;
                color: #121212;
            }
        """)
        
        step3_layout.addWidget(s3_title)
        step3_layout.addWidget(self.radio_glb)
        step3_layout.addWidget(self.radio_obj)
        step3_layout.addWidget(self.radio_ply)
        step3_layout.addWidget(self.export_btn)
        step3_layout.addWidget(self.upload_portal_btn)
        scroll_content_layout.addWidget(self.step3_box)
        
        # Disable Step 3 until processing finishes
        self.step3_box.setEnabled(False)
        
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
        self.viewer_widget.images_dropped.connect(self._handle_dropped_images)
        self.viewer_widget.reload_requested.connect(self._reload_viewer)
        self.viewer_widget.external_launch_requested.connect(self._launch_external_viewer)
        
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
        
        console_title = QLabel("System Output Log", self.console_frame)
        console_title.setStyleSheet("font-weight: bold; color: #888888; font-size: 11px; text-transform: uppercase;")
        self.console_text = QTextEdit(self.console_frame)
        self.console_text.setReadOnly(True)
        self.console_text.setObjectName("Console")
        
        console_layout.addWidget(console_title)
        console_layout.addWidget(self.console_text)
        
        # Add tabs
        self.bottom_tabs.addTab(self.photos_tab, "Photos")
        self.bottom_tabs.addTab(self.console_frame, "Console")
        
        right_layout.addWidget(self.bottom_tabs, stretch=2)
        
        main_layout.addWidget(right_panel, stretch=1)
        self._set_process_btn_state("idle")

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
            QTabBar::tab {
                background-color: #242424;
                color: #aaaaaa;
                border: 1px solid #2B2B2B;
                padding: 6px 16px;
                font-weight: bold;
                font-size: 11px;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #e0e0e0;
                color: #121212;
                border-top: none;
            }
            QTabBar::tab:hover:!selected {
                background-color: #333333;
                color: #ffffff;
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
                
        self.camera_label.setText(f"Camera: {camera_name}")
        if files:
            self.console_text.append(f"[INFO] Successfully imported {len(files)} files. Camera identified: {camera_name}")
            self._set_process_btn_state("ready")
        else:
            self.console_text.append("[INFO] Image list cleared.")
            self._set_process_btn_state("idle")

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
            self, "Select Images to Add", "", "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if files:
            current_set = set(self.image_list)
            added_count = 0
            for f in files:
                normalized = os.path.normpath(f)
                if normalized not in current_set:
                    self.image_list.append(normalized)
                    added_count += 1
            if added_count > 0:
                self._handle_dropped_images(self.image_list)
                self.console_text.append(f"[INFO] Added {added_count} new images.")

    def _open_dir_dialog(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if dir_path:
            files = []
            for root, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        files.append(os.path.join(root, filename))
            if files:
                self._handle_dropped_images(files)
            else:
                self.console_text.append("[WARNING] No valid images found in selected folder.")

    def _start_processing(self):
        if not self.image_list:
            return
            
        # Terminate any active viewer to prevent lock conflict on MVS files during reconstruction
        self._terminate_viewer()
        
        self._set_process_btn_state("progress")
        self.browse_btn.setEnabled(False)
        self.step3_box.setEnabled(False)
        
        # Temp output dir inside the workspace or local appdata if not writable
        output_dir = get_reconstruction_out_dir()
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract quality and gpu mode
        quality_presets = ["preview", "medium", "high", "ultra"]
        gpu_modes = ["auto", "force_gpu", "force_cpu"]
        quality_preset = quality_presets[self.quality_combo.currentIndex()]
        gpu_mode = gpu_modes[self.gpu_combo.currentIndex()]

        self.worker = PipelineWorker(
            os.path.dirname(self.image_list[0]), 
            output_dir, 
            quality_preset=quality_preset, 
            gpu_mode=gpu_mode, 
            parent=self
        )
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.log_message.connect(self._append_log)
        self.worker.finished.connect(self._on_pipeline_finished)
        
        self.console_text.append("[START] Initializing asynchronous reconstruction task thread...")
        self.worker.start()

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
        self.browse_btn.setEnabled(True)
        self.view_scene_btn.setEnabled(True)
        
        if success:
            self._set_process_btn_state("ready")
            self.console_text.append(f"[FINISHED] {msg}")
            self.step3_box.setEnabled(True)
            
            mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
            self.viewer_widget.set_mvs_directory(mvs_dir)
            self.viewer_widget.mode_select.blockSignals(True)
            
            # Pick best available viewer mode
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
            
            mesh_path = self.viewer_widget.get_selected_file_path()
            if mesh_path:
                self._reload_viewer(mesh_path)
        else:
            self._set_process_btn_state("failed")
            self.console_text.append(f"[FAILED] Reconstruction failed: {msg}")

    def _export_mesh(self):
        # Determine format selection
        fmt = ".obj"
        if self.radio_ply.isChecked():
            fmt = ".ply"
        elif self.radio_glb.isChecked():
            fmt = ".glb"
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Final Reconstruction Mesh", f"reconstructed_mesh{fmt}", f"Mesh Files (*{fmt})"
        )
        if not file_path:
            return
            
        output_dir = get_reconstruction_out_dir()
        mvs_out = os.path.join(output_dir, "mvs")
        
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
                        # The MTL file might reference a different filename, so we keep the original name
                        shutil.copy2(src_mtl, os.path.join(dest_dir, "scene_dense_mesh_texture.mtl"))
                        
                        # Parse the MTL to find the texture image(s) and copy them
                        with open(src_mtl, 'r') as f:
                            for line in f:
                                if line.strip().startswith("map_Kd "):
                                    tex_filename = line.strip().split(" ", 1)[1]
                                    src_tex = os.path.join(mvs_out, tex_filename)
                                    if os.path.exists(src_tex):
                                        shutil.copy2(src_tex, os.path.join(dest_dir, tex_filename))
                        
                    self.console_text.append(f"[EXPORT] OBJ mesh and textures successfully written to {dest_dir}")
                else:
                    self.console_text.append(f"[ERROR] Could not find reconstructed OBJ file at {src_obj}")
            elif fmt == ".glb":
                src_obj = os.path.join(mvs_out, "scene_dense_mesh_texture.obj")
                if os.path.exists(src_obj):
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
                    self.console_text.append(f"[ERROR] Could not find reconstructed OBJ file at {src_obj}")
        except Exception as e:
            self.console_text.append(f"[ERROR] Failed to export mesh: {e}")

    def _check_existing_scene(self):
        """Checks if a previous reconstruction scene exists to enable the viewer button."""
        output_dir = get_reconstruction_out_dir()
        mvs_dir = os.path.join(output_dir, "mvs")
        scene_mvs = os.path.join(mvs_dir, "scene.mvs")
        if os.path.exists(scene_mvs):
            self.viewer_widget.set_mvs_directory(mvs_dir)
            self.view_scene_btn.setEnabled(True)
            self.step3_box.setEnabled(True)
            self.console_text.append("[INFO] Detected previous reconstruction. 3D Viewer is ready to display.")

    def _toggle_viewer_mode(self):
        """Reloads the embedded 3D viewer."""
        path = self.viewer_widget.get_selected_file_path()
        if path:
            self._reload_viewer(path)

    def _check_for_viewer_window(self):
        if not self.viewer_process or self.viewer_process.poll() is not None:
            self.viewer_timer.stop()
            return

        target_pid = self.viewer_process.pid
        hwnd_found = None

        def enum_callback(hwnd, extra):
            nonlocal hwnd_found
            pid = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == target_pid:
                if ctypes.windll.user32.IsWindowVisible(hwnd):
                    hwnd_found = hwnd
                    return False  # Stop enumeration
            return True

        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        ctypes.windll.user32.EnumWindows(EnumWindowsProc(enum_callback), 0)

        if hwnd_found:
            self.viewer_timer.stop()
            self._embed_hwnd(hwnd_found)

    def _embed_hwnd(self, hwnd):
        self.viewer_hwnd = hwnd
        
        # Win32 Style configuration for borderless child embedding
        GWL_STYLE = -16
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
        style &= ~0x00C00000  # WS_CAPTION
        style &= ~0x00040000  # WS_THICKFRAME
        style &= ~0x00080000  # WS_SYSMENU
        style |= 0x40000000   # WS_CHILD
        
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)
        ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0027) # SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER

        window = QWindow.fromWinId(hwnd)
        self._clear_viewer_container()
        
        self.qt_viewer_container = QWidget.createWindowContainer(window, self.viewer_widget.container_area)
        self.viewer_widget.container_area_layout.addWidget(self.qt_viewer_container)
        self.viewer_widget.fallback_label.hide()

    def _clear_viewer_container(self):
        layout = self.viewer_widget.container_area_layout
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                if widget == self.viewer_widget.fallback_label:
                    widget.hide()
                else:
                    widget.deleteLater()

    def _reload_viewer(self, file_path):
        self._terminate_viewer()
        
        if not os.path.exists(file_path):
            self.viewer_widget.fallback_label.setText(f"File not found: {os.path.basename(file_path)}\nRun reconstruction to generate this file first.")
            self.viewer_widget.fallback_label.show()
            self.console_text.append(f"[WARNING] 3D file not found: {file_path}")
            return
            
        viewer_exe = os.path.join(get_base_dir(), "backend_bin", "openMVS", "Viewer.exe")
        if not os.path.exists(viewer_exe):
            self.console_text.append(f"[ERROR] Viewer.exe not found at {viewer_exe}")
            return
            
        self.viewer_widget.fallback_label.setText("Loading 3D scene in embedded view...")
        self.viewer_widget.fallback_label.show()
        
        try:
            import sys
            import subprocess
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            self.viewer_process = subprocess.Popen(
                [viewer_exe, file_path],
                cwd=os.path.dirname(file_path),
                creationflags=creationflags
            )
            self.viewer_timer.start()
            self.console_text.append(f"[INFO] Launched Viewer.exe on {os.path.basename(file_path)}")
        except Exception as e:
            self.console_text.append(f"[ERROR] Failed to start Viewer.exe: {e}")
 
    def _launch_external_viewer(self, file_path):
        if not os.path.exists(file_path):
            self.console_text.append(f"[WARNING] File not found for external viewer: {file_path}")
            return
            
        viewer_exe = os.path.join(get_base_dir(), "backend_bin", "openMVS", "Viewer.exe")
        import sys
        import subprocess
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        try:
            subprocess.Popen(
                [viewer_exe, file_path],
                cwd=os.path.dirname(file_path),
                creationflags=creationflags
            )
            self.console_text.append(f"[INFO] Launched external Viewer.exe window on {os.path.basename(file_path)}")
        except Exception as e:
            self.console_text.append(f"[ERROR] Failed to launch external viewer: {e}")

    def _terminate_viewer(self):
        self.viewer_timer.stop()
        if self.viewer_process:
            try:
                self.viewer_process.terminate()
                self.viewer_process.wait(timeout=1.0)
            except Exception:
                try:
                    self.viewer_process.kill()
                except Exception:
                    pass
            self.viewer_process = None
        self.viewer_hwnd = None
        self.viewer_widget.fallback_label.setText("3D Viewer Idle")
        self.viewer_widget.fallback_label.show()

    def closeEvent(self, event):
        self._terminate_viewer()
        if hasattr(self, 'loopback_server') and self.loopback_server:
            try:
                self.loopback_server.stop()
            except Exception:
                pass
        super().closeEvent(event)

    def _upload_to_proximaxr(self):
        output_dir = get_reconstruction_out_dir()
        mvs_out = os.path.join(output_dir, "mvs")
        
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
        
        bridge_url = f"http://localhost:3000/upload-bridge?local_url={local_url}&name={model_name}"
        
        self.console_text.append(f"[BRIDGE] Directing system browser to: {bridge_url}")
        import webbrowser
        webbrowser.open(bridge_url)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
