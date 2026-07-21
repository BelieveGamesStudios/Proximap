import numpy as np
import os
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QScrollArea,
    QListWidget, QGridLayout, QDoubleSpinBox, QCheckBox, QPushButton,
    QColorDialog, QComboBox, QFileDialog, QMessageBox, QDialog, QProgressBar,
    QMenuBar, QMenu, QWidgetAction
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from mesh_editor.viewport import MeshEditorViewport
from mesh_editor.scene import Object, load_mesh_file, apply_texture_to_meshes
from mesh_editor.undo import (
    UndoStack, BaseCommand, TransformCommand, SpinboxTransformCommand,
    DeleteCommand, AddObjectCommand
)

class MeshImportWorker(QThread):
    # Signals: finished (success, meshes, error_message)
    finished = Signal(bool, list, str)
    
    def __init__(self, file_path: str, texture_path: str = None):
        super().__init__()
        self.file_path = file_path
        self.texture_path = texture_path
        
    def run(self):
        try:
            # Load mesh in the worker thread (heavy CPU task)
            meshes = load_mesh_file(self.file_path)
            if self.texture_path:
                apply_texture_to_meshes(meshes, self.texture_path)
            self.finished.emit(True, meshes, "")
        except Exception as e:
            import traceback
            err_details = traceback.format_exc()
            self.finished.emit(False, None, f"{str(e)}\n\n{err_details}")

class MeshExportWorker(QThread):
    # Signals: finished (success, error_message)
    finished = Signal(bool, str)
    
    def __init__(self, scene, file_path: str):
        super().__init__()
        self.scene = scene
        self.file_path = file_path
        
    def run(self):
        try:
            from mesh_editor.scene import export_scene_to_file
            export_scene_to_file(self.scene, self.file_path)
            self.finished.emit(True, "")
        except Exception as e:
            import traceback
            err_details = traceback.format_exc()
            self.finished.emit(False, f"{str(e)}\n\n{err_details}")

class MeshImportProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.setWindowTitle("Importing Model")
        self.setFixedSize(320, 110)
        self.setStyleSheet("""
            QDialog {
                background-color: #1A1A1A;
                border: 1px solid #333333;
                border-radius: 8px;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)
        
        self.label = QLabel("Parsing geometry & processing mesh...", self)
        self.label.setStyleSheet("color: #00E676; font-weight: bold;")
        
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0)  # Indeterminate animation
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                background-color: #222222;
                height: 12px;
            }
            QProgressBar::chunk {
                background-color: #00E676;
                border-radius: 4px;
            }
        """)
        
        layout.addWidget(self.label)
        layout.addWidget(self.progress)

class MeshEditorWidget(QWidget):
    """The main wrapper widget for the Mesh Editor tab."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MeshEditorWidget")
        self.setAcceptDrops(True)
        
        # Initialize Undo / Redo Stack Engine (Limit 30 items)
        self.undo_stack = UndoStack(max_size=30)
        self._last_spinbox_pos = None
        self._last_spinbox_rot = None
        self._last_spinbox_scale = None
        
        # Build UI layout
        self._init_ui()
        self._wire_events()
        self._populate_outliner()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # --- LEFT PANEL: Sidebar Controls ---
        self.sidebar = QFrame(self)
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(360)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # Header/Title of Mesh Editor sidebar
        title_container = QWidget(self.sidebar)
        title_container.setStyleSheet("background-color: #1A1A1A; border-top-left-radius: 8px; border-top-right-radius: 8px;")
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(20, 20, 20, 10)
        
        title_label = QLabel("Mesh Editor controls", title_container)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff; padding-bottom: 10px; border-bottom: 1px solid #3d3d3d;")
        title_layout.addWidget(title_label)
        sidebar_layout.addWidget(title_container)
        
        # Scroll area for controls
        scroll_area = QScrollArea(self.sidebar)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("QScrollArea { background-color: #1A1A1A; border: none; }")
        
        scroll_content = QWidget()
        scroll_content.setObjectName("ScrollContent")
        scroll_content.setStyleSheet("QWidget#ScrollContent { background-color: #1A1A1A; }")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 10, 20, 20)
        scroll_layout.setSpacing(20)
        
        # SECTION 1: Outliner
        outliner_box = QFrame(scroll_content)
        outliner_box.setObjectName("StepBox")
        outliner_vlayout = QVBoxLayout(outliner_box)
        
        outliner_title = QLabel("Scene Outliner", outliner_box)
        outliner_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #00E676;")
        outliner_vlayout.addWidget(outliner_title)
        
        self.outliner_list = QListWidget(outliner_box)
        self.outliner_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.outliner_list.setMinimumHeight(120)
        self.outliner_list.setStyleSheet("""
            QListWidget {
                background-color: #121212;
                border: 1px solid #333333;
                border-radius: 4px;
                color: #e0e0e0;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #00E676;
                color: #121212;
            }
        """)
        outliner_vlayout.addWidget(self.outliner_list)
        
        # Actions for Scene Outliner (only Delete Selected button)
        self.btn_delete_mesh = QPushButton("Delete Selected", outliner_box)
        self.btn_delete_mesh.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                padding: 6px;
                background-color: #331A1A;
                color: #FF5252;
                border: 1px solid #FF5252;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #FF5252;
                color: #ffffff;
            }
            QPushButton:disabled {
                background-color: #222222;
                color: #666666;
                border: 1px solid #333333;
            }
        """)
        self.btn_delete_mesh.setEnabled(False)
        outliner_vlayout.addWidget(self.btn_delete_mesh)
        
        scroll_layout.addWidget(outliner_box, stretch=1)
        
        # SECTION 2: Transform Properties
        self.properties_box = QFrame(scroll_content)
        self.properties_box.setObjectName("StepBox")
        properties_vlayout = QVBoxLayout(self.properties_box)
        
        properties_title = QLabel("Transform Properties", self.properties_box)
        properties_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #00E676;")
        properties_vlayout.addWidget(properties_title)
        
        grid_layout = QGridLayout()
        grid_layout.setSpacing(6)
        
        # Helper function to create standard spinboxes
        def create_spinbox(min_val=-1000.0, max_val=1000.0, step=0.1):
            sb = QDoubleSpinBox()
            sb.setRange(min_val, max_val)
            sb.setSingleStep(step)
            sb.setDecimals(3)
            sb.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #333333;
                    color: #ffffff;
                    border: 1px solid #444444;
                    border-radius: 4px;
                    padding: 4px;
                    padding-right: 20px; /* Make room for standard arrow buttons */
                }
                QDoubleSpinBox::up-button {
                    subcontrol-origin: border;
                    subcontrol-position: top right;
                    width: 18px;
                    border-left: 1px solid #444444;
                    border-bottom: 1px solid #222222;
                    border-top-right-radius: 4px;
                    background-color: #2D2D2D;
                }
                QDoubleSpinBox::up-button:hover {
                    background-color: #3D3D3D;
                }
                QDoubleSpinBox::up-button:pressed {
                    background-color: #00E676;
                }
                QDoubleSpinBox::up-arrow {
                    image: url(no-image);
                    width: 0;
                    height: 0;
                    border-left: 3px solid transparent;
                    border-right: 3px solid transparent;
                    border-bottom: 5px solid #ffffff;
                }
                QDoubleSpinBox::up-arrow:pressed {
                    border-bottom-color: #121212;
                }
                QDoubleSpinBox::down-button {
                    subcontrol-origin: border;
                    subcontrol-position: bottom right;
                    width: 18px;
                    border-left: 1px solid #444444;
                    border-bottom-right-radius: 4px;
                    background-color: #2D2D2D;
                }
                QDoubleSpinBox::down-button:hover {
                    background-color: #3D3D3D;
                }
                QDoubleSpinBox::down-button:pressed {
                    background-color: #00E676;
                }
                QDoubleSpinBox::down-arrow {
                    image: url(no-image);
                    width: 0;
                    height: 0;
                    border-left: 3px solid transparent;
                    border-right: 3px solid transparent;
                    border-top: 5px solid #ffffff;
                }
                QDoubleSpinBox::down-arrow:pressed {
                    border-top-color: #121212;
                }
            """)
            return sb
            
        lbl_style = "color: #aaaaaa; font-weight: bold;"
        hdr_style = "color: #00E676; font-weight: bold; font-size: 11px; margin-top: 5px;"
        
        # Location Group
        lbl_loc_hdr = QLabel("Location", self.properties_box)
        lbl_loc_hdr.setStyleSheet(hdr_style)
        grid_layout.addWidget(lbl_loc_hdr, 0, 0, 1, 2)
        
        lbl_x = QLabel("X", self.properties_box)
        lbl_x.setStyleSheet(lbl_style)
        self.sp_pos_x = create_spinbox()
        grid_layout.addWidget(lbl_x, 1, 0)
        grid_layout.addWidget(self.sp_pos_x, 1, 1)
        
        lbl_y = QLabel("Y", self.properties_box)
        lbl_y.setStyleSheet(lbl_style)
        self.sp_pos_y = create_spinbox()
        grid_layout.addWidget(lbl_y, 2, 0)
        grid_layout.addWidget(self.sp_pos_y, 2, 1)
        
        lbl_z = QLabel("Z", self.properties_box)
        lbl_z.setStyleSheet(lbl_style)
        self.sp_pos_z = create_spinbox()
        grid_layout.addWidget(lbl_z, 3, 0)
        grid_layout.addWidget(self.sp_pos_z, 3, 1)
        
        # Rotation Group
        lbl_rot_hdr = QLabel("Rotation (Deg)", self.properties_box)
        lbl_rot_hdr.setStyleSheet(hdr_style)
        grid_layout.addWidget(lbl_rot_hdr, 4, 0, 1, 2)
        
        lbl_rx = QLabel("X", self.properties_box)
        lbl_rx.setStyleSheet(lbl_style)
        self.sp_rot_x = create_spinbox(-360.0, 360.0, 5.0)
        grid_layout.addWidget(lbl_rx, 5, 0)
        grid_layout.addWidget(self.sp_rot_x, 5, 1)
        
        lbl_ry = QLabel("Y", self.properties_box)
        lbl_ry.setStyleSheet(lbl_style)
        self.sp_rot_y = create_spinbox(-360.0, 360.0, 5.0)
        grid_layout.addWidget(lbl_ry, 6, 0)
        grid_layout.addWidget(self.sp_rot_y, 6, 1)
        
        lbl_rz = QLabel("Z", self.properties_box)
        lbl_rz.setStyleSheet(lbl_style)
        self.sp_rot_z = create_spinbox(-360.0, 360.0, 5.0)
        grid_layout.addWidget(lbl_rz, 7, 0)
        grid_layout.addWidget(self.sp_rot_z, 7, 1)
        
        # Scale Group
        lbl_scl_hdr = QLabel("Scale", self.properties_box)
        lbl_scl_hdr.setStyleSheet(hdr_style)
        grid_layout.addWidget(lbl_scl_hdr, 8, 0, 1, 2)
        
        lbl_sx = QLabel("X", self.properties_box)
        lbl_sx.setStyleSheet(lbl_style)
        self.sp_scale_x = create_spinbox(0.01, 100.0, 0.1)
        grid_layout.addWidget(lbl_sx, 9, 0)
        grid_layout.addWidget(self.sp_scale_x, 9, 1)
        
        lbl_sy = QLabel("Y", self.properties_box)
        lbl_sy.setStyleSheet(lbl_style)
        self.sp_scale_y = create_spinbox(0.01, 100.0, 0.1)
        grid_layout.addWidget(lbl_sy, 10, 0)
        grid_layout.addWidget(self.sp_scale_y, 10, 1)
        
        lbl_sz = QLabel("Z", self.properties_box)
        lbl_sz.setStyleSheet(lbl_style)
        self.sp_scale_z = create_spinbox(0.01, 100.0, 0.1)
        grid_layout.addWidget(lbl_sz, 11, 0)
        grid_layout.addWidget(self.sp_scale_z, 11, 1)
        
        properties_vlayout.addLayout(grid_layout)
        scroll_layout.addWidget(self.properties_box, stretch=0)
        
        scroll_area.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll_area)
        layout.addWidget(self.sidebar)
        
        # Create Settings widgets for the View dropdown menu
        self.btn_projection = QPushButton("Camera: Perspective")
        self.btn_projection.setStyleSheet("QPushButton { font-size: 11px; padding: 6px 12px; }")
        
        self.cb_snap = QCheckBox("Enable Snapping")
        self.cb_snap.setStyleSheet("color: #ffffff; font-size: 11px;")
        
        self.cb_invert_y = QCheckBox("Invert Y Mouse")
        self.cb_invert_y.setStyleSheet("color: #ffffff; font-size: 11px;")
        
        # Snapping options
        self.sp_snap_t = create_spinbox(0.01, 10.0, 0.1)
        self.sp_snap_t.setValue(0.5)
        
        self.sp_snap_r = create_spinbox(1.0, 90.0, 5.0)
        self.sp_snap_r.setValue(15.0)
        
        # Grid line controls
        self.sp_subdivs = create_spinbox(1.0, 100.0, 1.0)
        self.sp_subdivs.setValue(10.0)
        
        # Color pickers
        self.btn_color_bg = QPushButton("Background Color")
        self.btn_color_bg.setStyleSheet("QPushButton { font-size: 11px; padding: 6px 12px; }")
        self.btn_color_grid = QPushButton("Grid Major Color")
        self.btn_color_grid.setStyleSheet("QPushButton { font-size: 11px; padding: 6px 12px; }")
        
        # --- RIGHT PANEL: Viewport Container with Top Toolbar ---
        self.viewport_container = QWidget(self)
        self.viewport_container.setObjectName("ViewportContainer")
        viewport_layout = QVBoxLayout(self.viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        
        # Create Menu Bar / Toolbar
        self.menu_bar = QMenuBar(self.viewport_container)
        self.menu_bar.setObjectName("ViewportMenuBar")
        
        # 1. File Menu
        file_menu = QMenu("File", self.menu_bar)
        
        action_import = file_menu.addAction("Import Mesh (.obj, .glb)")
        action_import.triggered.connect(self._on_import_model_clicked)
        
        action_load_recon = file_menu.addAction("Load Reconstructed Model")
        action_load_recon.triggered.connect(self._on_load_reconstructed_model_clicked)
        
        action_import_pxm = file_menu.addAction("Import .PXM")
        action_import_pxm.triggered.connect(self._on_import_pxm_clicked)
        
        file_menu.addSeparator()
        
        action_export = file_menu.addAction("Export Scene (.obj, .glb)")
        action_export.triggered.connect(self._on_export_scene_clicked)
        
        file_menu.addSeparator()
        self.action_upload_proximap = file_menu.addAction("Upload to Proximap")
        
        self.menu_bar.addMenu(file_menu)
        
        # 2. Edit Menu
        edit_menu = QMenu("Edit", self.menu_bar)
        
        self.action_undo = edit_menu.addAction("Undo")
        self.action_undo.setShortcut("Ctrl+Z")
        self.action_undo.setEnabled(False)
        self.action_undo.triggered.connect(self._perform_undo)
        
        self.action_redo = edit_menu.addAction("Redo")
        self.action_redo.setShortcut("Ctrl+Y")
        self.action_redo.setEnabled(False)
        self.action_redo.triggered.connect(self._perform_redo)
        
        edit_menu.addSeparator()
        
        action_delete = edit_menu.addAction("Delete Selected")
        action_delete.triggered.connect(self._delete_selected_object)
        self.menu_bar.addMenu(edit_menu)
        
        # 3. View Menu with embedded Settings panel (dropdown style)
        view_menu = QMenu("View", self.menu_bar)
        
        settings_widget = QWidget(view_menu)
        settings_widget.setStyleSheet("background-color: #1A1A1A; color: #ffffff;")
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(15, 12, 15, 12)
        settings_layout.setSpacing(10)
        
        lbl_title = QLabel("Viewport Settings", settings_widget)
        lbl_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #00E676; padding-bottom: 4px; border-bottom: 1px solid #333333;")
        settings_layout.addWidget(lbl_title)
        
        settings_layout.addWidget(self.btn_projection)
        settings_layout.addWidget(self.cb_snap)
        settings_layout.addWidget(self.cb_invert_y)
        
        # Grid for Snap Inputs
        snap_grid = QGridLayout()
        snap_grid.setSpacing(6)
        
        lbl_tsnap = QLabel("T-Snap (m):", settings_widget)
        lbl_tsnap.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        snap_grid.addWidget(lbl_tsnap, 0, 0)
        snap_grid.addWidget(self.sp_snap_t, 0, 1)
        
        lbl_rsnap = QLabel("R-Snap (deg):", settings_widget)
        lbl_rsnap.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        snap_grid.addWidget(lbl_rsnap, 1, 0)
        snap_grid.addWidget(self.sp_snap_r, 1, 1)
        
        lbl_subdiv = QLabel("Grid Subdivs:", settings_widget)
        lbl_subdiv.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        snap_grid.addWidget(lbl_subdiv, 2, 0)
        snap_grid.addWidget(self.sp_subdivs, 2, 1)
        
        settings_layout.addLayout(snap_grid)
        
        # Color pickers
        settings_layout.addWidget(self.btn_color_bg)
        settings_layout.addWidget(self.btn_color_grid)
        
        # Wrap the settings widget in a QWidgetAction
        view_settings_action = QWidgetAction(view_menu)
        view_settings_action.setDefaultWidget(settings_widget)
        view_menu.addAction(view_settings_action)
        
        self.menu_bar.addMenu(view_menu)
        
        viewport_layout.addWidget(self.menu_bar)
        
        # Horizontal layout for the toolbar and the viewport
        viewport_hbox = QHBoxLayout()
        viewport_hbox.setContentsMargins(0, 0, 0, 0)
        viewport_hbox.setSpacing(0)
        
        # Create Vertical Toolbar Frame
        self.trans_toolbar = QFrame(self.viewport_container)
        self.trans_toolbar.setObjectName("ViewportToolbar")
        self.trans_toolbar.setFixedWidth(42)
        self.trans_toolbar.setStyleSheet("""
            QFrame#ViewportToolbar {
                background-color: #1A1A1A;
                border-right: 1px solid #2D2D2D;
            }
            QPushButton {
                background-color: transparent;
                color: #aaaaaa;
                border: 1px solid transparent;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
                margin: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #2D2D2D;
                color: #ffffff;
                border: 1px solid #444444;
            }
            QPushButton:checked {
                background-color: #00E676;
                color: #121212;
                border: 1px solid #00E676;
            }
        """)
        
        toolbar_layout = QVBoxLayout(self.trans_toolbar)
        toolbar_layout.setContentsMargins(0, 4, 0, 4)
        toolbar_layout.setSpacing(4)
        
        from PySide6.QtWidgets import QButtonGroup
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        
        # Translate Tool Button
        self.btn_tool_translate = QPushButton("✥", self.trans_toolbar)
        self.btn_tool_translate.setCheckable(True)
        self.btn_tool_translate.setChecked(True)
        self.btn_tool_translate.setToolTip("Translate (G)")
        self.tool_group.addButton(self.btn_tool_translate)
        toolbar_layout.addWidget(self.btn_tool_translate)
        
        # Rotate Tool Button
        self.btn_tool_rotate = QPushButton("⟳", self.trans_toolbar)
        self.btn_tool_rotate.setCheckable(True)
        self.btn_tool_rotate.setToolTip("Rotate (R)")
        self.tool_group.addButton(self.btn_tool_rotate)
        toolbar_layout.addWidget(self.btn_tool_rotate)
        
        # Scale Tool Button
        self.btn_tool_scale = QPushButton("⤢", self.trans_toolbar)
        self.btn_tool_scale.setCheckable(True)
        self.btn_tool_scale.setToolTip("Scale (S)")
        self.tool_group.addButton(self.btn_tool_scale)
        toolbar_layout.addWidget(self.btn_tool_scale)
        
        # Separator line
        line = QFrame(self.trans_toolbar)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #333333; margin: 4px;")
        toolbar_layout.addWidget(line)
        
        # Space Toggle Button (Defaults to Local)
        self.btn_tool_space = QPushButton("Local", self.trans_toolbar)
        self.btn_tool_space.setToolTip("Transform Space: Local (Click to toggle)")
        self.btn_tool_space.setStyleSheet("""
            QPushButton {
                font-size: 10px;
                font-weight: bold;
                padding: 6px 2px;
                background-color: #2D2D2D;
                color: #00E676;
                border: 1px solid #444444;
                border-radius: 4px;
                margin: 4px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
                border-color: #00E676;
            }
        """)
        toolbar_layout.addWidget(self.btn_tool_space)
        
        # Separator line
        line2 = QFrame(self.trans_toolbar)
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        line2.setStyleSheet("background-color: #333333; margin: 4px;")
        toolbar_layout.addWidget(line2)

        # Projection button
        self.btn_projection_toolbar = QPushButton("Persp", self.trans_toolbar)
        self.btn_projection_toolbar.setCheckable(True)
        self.btn_projection_toolbar.setChecked(True)
        self.btn_projection_toolbar.setToolTip("Toggle Perspective/Orthographic (Numpad 5)")
        self.btn_projection_toolbar.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                font-weight: bold;
                padding: 6px 2px;
                background-color: #2D2D2D;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                margin: 4px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
            QPushButton:checked {
                background-color: #00E676;
                color: #121212;
                border-color: #00E676;
            }
        """)
        toolbar_layout.addWidget(self.btn_projection_toolbar)

        toolbar_layout.addStretch()
        
        viewport_hbox.addWidget(self.trans_toolbar)
        
        self.viewport = MeshEditorViewport(self.viewport_container)
        viewport_hbox.addWidget(self.viewport, stretch=1)
        
        viewport_layout.addLayout(viewport_hbox, stretch=1)
        
        layout.addWidget(self.viewport_container, stretch=4)
        
        # Disable properties inputs initially until an object is selected
        self._set_properties_enabled(False)
        self.properties_box.setVisible(False)

    def _wire_events(self):
        # 1. Viewport Selection -> Sync Sidebar
        self.viewport.selection_changed.connect(self._on_viewport_selection_changed)
        # 2. Viewport Gizmo Edit -> Sync Spinboxes & Commit Undo
        self.viewport.transform_changed.connect(self._on_viewport_transform_changed)
        self.viewport.transform_committed.connect(self._on_viewport_transform_committed)
        self.viewport.undo_requested.connect(self._perform_undo)
        self.viewport.redo_requested.connect(self._perform_redo)
        
        # 3. Sidebar Selection -> Sync Viewport Selection
        self.outliner_list.itemSelectionChanged.connect(self._on_outliner_selection_changed)
        
        # 4. Spinbox value changes -> Sync Viewport Object Transform & Commit Undo
        for sb in [self.sp_pos_x, self.sp_pos_y, self.sp_pos_z]:
            sb.valueChanged.connect(self._on_translation_spinbox_changed)
            sb.editingFinished.connect(self._on_translation_spinbox_committed)
        for sb in [self.sp_rot_x, self.sp_rot_y, self.sp_rot_z]:
            sb.valueChanged.connect(self._on_rotation_spinbox_changed)
            sb.editingFinished.connect(self._on_rotation_spinbox_committed)
        for sb in [self.sp_scale_x, self.sp_scale_y, self.sp_scale_z]:
            sb.valueChanged.connect(self._on_scale_spinbox_changed)
            sb.editingFinished.connect(self._on_scale_spinbox_committed)
            
        # 5. Projection Toggles
        self.btn_projection.clicked.connect(self._on_projection_clicked)
        self.btn_projection_toolbar.clicked.connect(self._on_projection_clicked)
        self.viewport.camera_changed.connect(self._on_camera_changed)
        
        # 6. Snapping Toggles and values
        self.cb_snap.stateChanged.connect(self._on_snap_toggled)
        self.cb_invert_y.stateChanged.connect(self._on_invert_y_toggled)
        self.sp_snap_t.valueChanged.connect(self._on_snap_values_changed)
        self.sp_snap_r.valueChanged.connect(self._on_snap_values_changed)
        
        # 7. Subdivisions changed
        self.sp_subdivs.valueChanged.connect(self._on_subdivs_changed)
        
        # 8. Colors changed
        self.btn_color_bg.clicked.connect(self._on_color_bg_clicked)
        self.btn_color_grid.clicked.connect(self._on_color_grid_clicked)
        
        # 9. Delete model clicked
        self.btn_delete_mesh.clicked.connect(self._delete_selected_object)
        
        # 10. Viewport key shortcuts
        self.viewport.delete_pressed.connect(self._delete_selected_object)
        
        # 11. Toolbar events
        self.btn_tool_translate.clicked.connect(self._change_tool_to_translate)
        self.btn_tool_rotate.clicked.connect(self._change_tool_to_rotate)
        self.btn_tool_scale.clicked.connect(self._change_tool_to_scale)
        self.btn_tool_space.clicked.connect(self._toggle_transform_space)
        self.viewport.tool_changed.connect(self._on_viewport_tool_changed)

    def _populate_outliner(self):
        self.outliner_list.blockSignals(True)
        self.outliner_list.clear()
        for obj in self.viewport.scene.objects:
            self.outliner_list.addItem(obj.name)
        self.outliner_list.blockSignals(False)

    def _set_properties_enabled(self, enabled: bool):
        for sb in [
            self.sp_pos_x, self.sp_pos_y, self.sp_pos_z,
            self.sp_rot_x, self.sp_rot_y, self.sp_rot_z,
            self.sp_scale_x, self.sp_scale_y, self.sp_scale_z
        ]:
            sb.setEnabled(enabled)

    def _block_properties_signals(self, block: bool):
        for sb in [
            self.sp_pos_x, self.sp_pos_y, self.sp_pos_z,
            self.sp_rot_x, self.sp_rot_y, self.sp_rot_z,
            self.sp_scale_x, self.sp_scale_y, self.sp_scale_z
        ]:
            sb.blockSignals(block)

    # --- SLOT HANDLERS ---
    def _update_outliner_highlights(self):
        self.outliner_list.blockSignals(True)
        active_obj = self.viewport.scene.active_object
        selected_objs = self.viewport.scene.selected_objects
        
        for i in range(self.outliner_list.count()):
            item = self.outliner_list.item(i)
            item_name = item.text()
            
            obj = None
            for o in self.viewport.scene.objects:
                if o.name == item_name:
                    obj = o
                    break
                    
            if obj in selected_objs:
                item.setSelected(True)
                if obj == active_obj:
                    # Active object (last clicked): Bright Green text, darker green/black background
                    item.setForeground(QColor("#00FF9C"))
                    item.setBackground(QColor("#1e2d27"))
                else:
                    # Secondary selected: Muted green text, subtle background
                    item.setForeground(QColor("#1A7A4A"))
                    item.setBackground(QColor("#121b16"))
            else:
                item.setSelected(False)
                item.setForeground(QColor("#e0e0e0"))
                item.setBackground(QColor("#121212"))
                
        self.outliner_list.blockSignals(False)

    # --- SLOT HANDLERS ---
    def _on_viewport_selection_changed(self, selected_obj: Object):
        selected_objs = self.viewport.scene.selected_objects
        self.btn_delete_mesh.setEnabled(len(selected_objs) > 0)
        
        if not selected_objs:
            self.outliner_list.blockSignals(True)
            self.outliner_list.clearSelection()
            self.outliner_list.blockSignals(False)
            self._set_properties_enabled(False)
            self.properties_box.setVisible(False)
            self._block_properties_signals(True)
            for sb in [self.sp_pos_x, self.sp_pos_y, self.sp_pos_z, self.sp_rot_x, self.sp_rot_y, self.sp_rot_z]:
                sb.setValue(0.0)
            for sb in [self.sp_scale_x, self.sp_scale_y, self.sp_scale_z]:
                sb.setValue(1.0)
            self._block_properties_signals(False)
        else:
            self._update_outliner_highlights()
            
            # Populate fields based on active selection (selected_obj)
            if selected_obj is not None:
                self._set_properties_enabled(True)
                self.properties_box.setVisible(True)
                self._sync_properties_from_object(selected_obj)
            else:
                self._set_properties_enabled(False)

    def _on_viewport_transform_changed(self, obj: Object):
        # Update inputs dynamically from gizmo dragging without loops
        if obj:
            self._sync_properties_from_object(obj)

    def _on_outliner_selection_changed(self):
        selected_items = self.outliner_list.selectedItems()
        if not selected_items:
            self.viewport.scene.selected_objects = []
            self.viewport.scene.active_object = None
            active_obj = None
        else:
            new_selected = []
            for item in selected_items:
                for obj in self.viewport.scene.objects:
                    if obj.name == item.text():
                        new_selected.append(obj)
                        break
            self.viewport.scene.selected_objects = new_selected
            
            current_item = self.outliner_list.currentItem()
            active_obj = None
            if current_item and current_item.isSelected():
                for obj in self.viewport.scene.objects:
                    if obj.name == current_item.text():
                        active_obj = obj
                        break
            if active_obj is None and new_selected:
                active_obj = new_selected[-1]
            self.viewport.scene.active_object = active_obj
            
        selected_objs = self.viewport.scene.selected_objects
        self.btn_delete_mesh.setEnabled(len(selected_objs) > 0)
        
        if active_obj is None:
            self._set_properties_enabled(False)
            self.properties_box.setVisible(False)
        else:
            self._set_properties_enabled(True)
            self.properties_box.setVisible(True)
            self._sync_properties_from_object(active_obj)
            
        self._update_outliner_highlights()
        self.viewport.update()

    def push_command(self, cmd: BaseCommand):
        """Public API for registering custom commands into the Undo/Redo stack (supports addons)."""
        self.undo_stack.push(cmd)
        self._update_undo_redo_actions()

    def _perform_undo(self):
        label = self.undo_stack.undo()
        if label:
            self._populate_outliner()
            active = self.viewport.scene.active_object
            self._on_viewport_selection_changed(active)
            self._update_undo_redo_actions()
            self.viewport.update()

    def _perform_redo(self):
        label = self.undo_stack.redo()
        if label:
            self._populate_outliner()
            active = self.viewport.scene.active_object
            self._on_viewport_selection_changed(active)
            self._update_undo_redo_actions()
            self.viewport.update()

    def _update_undo_redo_actions(self):
        if hasattr(self, 'action_undo') and hasattr(self, 'action_redo'):
            self.action_undo.setEnabled(self.undo_stack.can_undo())
            self.action_redo.setEnabled(self.undo_stack.can_redo())
            self.action_undo.setText(f"Undo {self.undo_stack.undo_label()}" if self.undo_stack.can_undo() else "Undo")
            self.action_redo.setText(f"Redo {self.undo_stack.redo_label()}" if self.undo_stack.can_redo() else "Redo")

    def _on_viewport_transform_committed(self, payload):
        states, operation = payload
        cmd = TransformCommand(states, viewport=self.viewport, operation=operation)
        self.push_command(cmd)

    def _sync_properties_from_object(self, obj: Object):
        self._block_properties_signals(True)
        self.sp_pos_x.setValue(obj.position[0])
        self.sp_pos_y.setValue(obj.position[1])
        self.sp_pos_z.setValue(obj.position[2])
        
        self.sp_rot_x.setValue(obj.rotation[0])
        self.sp_rot_y.setValue(obj.rotation[1])
        self.sp_rot_z.setValue(obj.rotation[2])
        
        self.sp_scale_x.setValue(obj.scale[0])
        self.sp_scale_y.setValue(obj.scale[1])
        self.sp_scale_z.setValue(obj.scale[2])
        self._block_properties_signals(False)
        self._last_spinbox_pos = np.copy(obj.position)
        self._last_spinbox_rot = np.copy(obj.rotation)
        self._last_spinbox_scale = np.copy(obj.scale)

    def _on_translation_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.position = np.array([
                self.sp_pos_x.value(),
                self.sp_pos_y.value(),
                self.sp_pos_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_translation_spinbox_committed(self):
        obj = self.viewport.scene.selected_object
        if obj and self._last_spinbox_pos is not None:
            new_pos = np.array([self.sp_pos_x.value(), self.sp_pos_y.value(), self.sp_pos_z.value()], dtype=np.float32)
            if not np.array_equal(self._last_spinbox_pos, new_pos):
                cmd = SpinboxTransformCommand(obj, 'position', self._last_spinbox_pos, new_pos, self.viewport)
                self.push_command(cmd)
                self._last_spinbox_pos = np.copy(new_pos)

    def _on_rotation_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.rotation = np.array([
                self.sp_rot_x.value(),
                self.sp_rot_y.value(),
                self.sp_rot_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_rotation_spinbox_committed(self):
        obj = self.viewport.scene.selected_object
        if obj and self._last_spinbox_rot is not None:
            new_rot = np.array([self.sp_rot_x.value(), self.sp_rot_y.value(), self.sp_rot_z.value()], dtype=np.float32)
            if not np.array_equal(self._last_spinbox_rot, new_rot):
                cmd = SpinboxTransformCommand(obj, 'rotation', self._last_spinbox_rot, new_rot, self.viewport)
                self.push_command(cmd)
                self._last_spinbox_rot = np.copy(new_rot)

    def _on_scale_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.scale = np.array([
                self.sp_scale_x.value(),
                self.sp_scale_y.value(),
                self.sp_scale_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_scale_spinbox_committed(self):
        obj = self.viewport.scene.selected_object
        if obj and self._last_spinbox_scale is not None:
            new_scale = np.array([self.sp_scale_x.value(), self.sp_scale_y.value(), self.sp_scale_z.value()], dtype=np.float32)
            if not np.array_equal(self._last_spinbox_scale, new_scale):
                cmd = SpinboxTransformCommand(obj, 'scale', self._last_spinbox_scale, new_scale, self.viewport)
                self.push_command(cmd)
                self._last_spinbox_scale = np.copy(new_scale)

    def _on_projection_clicked(self):
        cam = self.viewport.camera
        cam.is_perspective = not cam.is_perspective
        self.viewport.notify_camera_changed()
        self.viewport.update()

    def _on_camera_changed(self, camera):
        # Update sidebar projection setting toggle button
        if camera.is_perspective:
            self.btn_projection.setText("Camera: Perspective")
            self.btn_projection_toolbar.setText("Persp")
            self.btn_projection_toolbar.setChecked(True)
        else:
            self.btn_projection.setText("Camera: Orthographic")
            self.btn_projection_toolbar.setText("Ortho")
            self.btn_projection_toolbar.setChecked(False)

    def _on_snap_toggled(self, state):
        self.viewport.gizmo.use_snapping = (state == Qt.CheckState.Checked.value)
        self.viewport.update()
        
    def _on_invert_y_toggled(self, state):
        self.viewport.invert_y = (state == Qt.CheckState.Checked.value)
        self.viewport.update()

    def _on_snap_values_changed(self):
        gizmo = self.viewport.gizmo
        gizmo.snap_translation = self.sp_snap_t.value()
        gizmo.snap_rotation = self.sp_snap_r.value()
        # Keep scale snapping proportional to 10% of value
        gizmo.snap_scale = 0.1
        self.viewport.update()

    def _on_subdivs_changed(self):
        self.viewport.grid_subdivisions = self.sp_subdivs.value()
        self.viewport.update()

    def _on_color_bg_clicked(self):
        from PySide6.QtGui import QColor
        c = self.viewport.bg_color
        initial = QColor.fromRgbF(c[0], c[1], c[2])
        cur_color = QColorDialog.getColor(
            initial, self, "Select Viewport Background Color"
        )
        if cur_color.isValid():
            r, g, b, _ = cur_color.getRgbF()
            self.viewport.bg_color = np.array([r, g, b], dtype=np.float32)
            self.viewport.update()

    def _on_color_grid_clicked(self):
        from PySide6.QtGui import QColor
        c = self.viewport.grid_color_1
        initial = QColor.fromRgbF(c[0], c[1], c[2], c[3])
        cur_color = QColorDialog.getColor(
            initial, self, "Select Major Grid Line Color"
        )
        if cur_color.isValid():
            r, g, b, a = cur_color.getRgbF()
            self.viewport.grid_color_1 = np.array([r, g, b, a], dtype=np.float32)
            self.viewport.update()

    def _import_mesh_from_path(self, file_path: str, texture_path: str = None):
        if not file_path or not os.path.exists(file_path):
            QMessageBox.critical(
                self, "Import Error", f"Model file not found:\n{file_path}"
            )
            return
            
        # Display modal dialog
        self.import_dialog = MeshImportProgressDialog(self)
        
        # Setup worker thread
        self.import_worker = MeshImportWorker(file_path, texture_path)
        self.import_worker.finished.connect(self._on_import_finished)
        
        # Start worker thread and dialog
        self.import_worker.start()
        self.import_dialog.exec()

    def _on_import_model_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import 3D Model", "", "3D Mesh Files (*.obj *.glb *.gltf *.ply)"
        )
        if not file_path:
            return
            
        # Check if the model has an auto-detectable texture
        has_texture = False
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.glb', '.gltf'):
            has_texture = True
        elif ext == '.obj':
            mtl_path = os.path.splitext(file_path)[0] + '.mtl'
            if os.path.exists(mtl_path):
                try:
                    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            if line.strip().startswith('map_Kd'):
                                has_texture = True
                                break
                except Exception:
                    pass
                    
        if not has_texture:
            reply = QMessageBox.question(
                self,
                "Select Texture File",
                "This model does not appear to have an embedded or associated texture map.\n\n"
                "Would you like to select a companion texture file (e.g., .png, .jpg, .mtl)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                selected_paths, _ = QFileDialog.getOpenFileNames(
                    self,
                    "Select Texture / Material File(s)",
                    os.path.dirname(file_path),
                    "Texture & Material Files (*.png *.jpg *.jpeg *.bmp *.tga *.mtl)"
                )
                if selected_paths:
                    mtl_files = [p for p in selected_paths if p.lower().endswith('.mtl')]
                    img_files = [p for p in selected_paths if not p.lower().endswith('.mtl')]
                    texture_path = mtl_files[0] if mtl_files else img_files[0]
                    self._import_mesh_from_path(file_path, texture_path)
                    return
                    
        self._import_mesh_from_path(file_path)

    def _on_load_reconstructed_model_clicked(self):
        # Traverse up parent to find active mvs_dir
        mvs_dir = None
        curr = self.parent()
        while curr is not None:
            if hasattr(curr, '_get_active_mvs_dir'):
                mvs_dir = curr._get_active_mvs_dir()
                break
            curr = curr.parent()
            
        if not mvs_dir:
            try:
                from main_window import get_reconstruction_out_dir
                mvs_dir = os.path.join(get_reconstruction_out_dir(), "mvs")
            except Exception:
                local_appdata = os.environ.get("LOCALAPPDATA")
                if local_appdata:
                    mvs_dir = os.path.join(local_appdata, "Proximap", "reconstruction_out", "mvs")
                else:
                    mvs_dir = os.path.join(os.path.expanduser("~"), ".proximap", "reconstruction_out", "mvs")
                    
        # Check candidates
        found_path = None
        for candidate in [
            "scene_dense_mesh_texture.obj",
            "scene_dense_mesh_texture.glb",
            "scene_dense_mesh_texture.ply",
            "scene_dense_mesh_refine.ply",
            "scene_dense_mesh.ply",
            "scene_mesh.ply"
        ]:
            path = os.path.join(mvs_dir, candidate)
            if os.path.exists(path):
                found_path = path
                break
                
        if found_path:
            self._import_mesh_from_path(found_path)
        else:
            QMessageBox.warning(
                self, 
                "Reconstruction Model Not Found", 
                f"No reconstructed model could be found in the current session folder:\n{mvs_dir}\n\nPlease run the dense reconstruction/texturing step first."
            )

    def _on_import_pxm_clicked(self, file_path: str = None):
        if not isinstance(file_path, str):
            file_path = None
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Import Proximap Project", "", "Proximap Project Files (*.pxm)"
            )
            if not file_path:
                return
            
        import zipfile
        import tempfile
        import uuid
        
        try:
            # Create a temporary extraction directory inside temp root
            temp_dir = os.path.join(tempfile.gettempdir(), f"proximap_pxm_import_{uuid.uuid4().hex}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract zip contents
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
                
            # Scan for candidate 3D mesh files
            found_mesh_path = None
            standard_obj = os.path.join(temp_dir, "scene_dense_mesh_texture.obj")
            if os.path.exists(standard_obj):
                found_mesh_path = standard_obj
            else:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.obj', '.glb', '.gltf', '.ply')):
                            found_mesh_path = os.path.join(root, file)
                            break
                    if found_mesh_path:
                        break
                        
            if found_mesh_path:
                self._import_mesh_from_path(found_mesh_path)
            else:
                QMessageBox.warning(
                    self,
                    "Import PXM Error",
                    "The selected .pxm file was successfully unpacked, but no 3D model files (.obj, .glb, etc.) were found inside."
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Import PXM Error",
                f"Failed to extract and import .pxm file:\n{str(e)}"
            )

    def _on_import_finished(self, success: bool, meshes: list, error_msg: str):
        # 1. Close modal dialog
        if hasattr(self, 'import_dialog') and self.import_dialog:
            self.import_dialog.accept()
            self.import_dialog = None
            
        if not success:
            QMessageBox.critical(
                self, "Import Error", f"Failed to load 3D model:\n{error_msg}"
            )
            self.import_worker = None
            return
            
        try:
            # 2. Setup OpenGL buffers on the main thread with context active
            self.viewport.makeCurrent()
            file_path = self.import_worker.file_path
            base_file_name = os.path.basename(file_path)
            existing_names = [o.name for o in self.viewport.scene.objects]
            
            created_objects = []
            for node_name, mesh in meshes:
                mesh.setup_buffers()
                
                # Prefer geometry's own node name, fallback to base file name
                name = node_name if node_name else base_file_name
                unique_name = name
                idx = 1
                while unique_name in existing_names:
                    unique_name = f"{name} ({idx})"
                    idx += 1
                existing_names.append(unique_name)
                
                obj = Object(unique_name, mesh)
                self.viewport.scene.add_object(obj)
                created_objects.append(obj)
                
            self.viewport.doneCurrent()
            
            # 3. Refresh outliner list
            self._populate_outliner()
            
            # 4. Select the last created object (or first) and push Undo command
            if created_objects:
                self.viewport.scene.selected_object = created_objects[-1]
                self._on_viewport_selection_changed(created_objects[-1])
                cmd = AddObjectCommand(created_objects, self.viewport, on_scene_changed_cb=self._populate_outliner)
                self.push_command(cmd)
                
            # 5. Update rendering
            self.viewport.update()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Import Error", f"Failed to configure GL buffers:\n{str(e)}"
            )
            
        self.import_worker = None

    def _delete_selected_object(self):
        selected_objs = list(self.viewport.scene.selected_objects)
        if selected_objs:
            cmd = DeleteCommand(selected_objs, self.viewport, on_scene_changed_cb=self._populate_outliner)
            cmd.execute()
            self.push_command(cmd)

    def _on_export_scene_clicked(self):
        if not self.viewport.scene.objects:
            QMessageBox.warning(
                self, "Export Warning", "There are no objects in the scene to export."
            )
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Scene", "", "3D Mesh Files (*.obj *.glb)"
        )
        if not file_path:
            return
            
        # Create and configure the waiting dialog
        self.export_dialog = QDialog(self, Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.export_dialog.setWindowTitle("Exporting Scene")
        self.export_dialog.setMinimumSize(280, 100)
        self.export_dialog.setModal(True)
        
        dialog_layout = QVBoxLayout(self.export_dialog)
        dialog_layout.setContentsMargins(20, 20, 20, 20)
        dialog_layout.setSpacing(10)
        
        lbl_msg = QLabel("Merging and exporting scene objects...\nPlease wait.", self.export_dialog)
        lbl_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_msg.setStyleSheet("color: #ffffff; font-size: 11px;")
        dialog_layout.addWidget(lbl_msg)
        
        progress_bar = QProgressBar(self.export_dialog)
        progress_bar.setRange(0, 0)  # Infinite progress bar
        dialog_layout.addWidget(progress_bar)
        
        # Start export thread
        self.export_worker = MeshExportWorker(self.viewport.scene, file_path)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.start()
        
        self.export_dialog.exec()

    def _on_export_finished(self, success: bool, error_msg: str):
        if hasattr(self, "export_dialog") and self.export_dialog:
            self.export_dialog.accept()
            self.export_dialog = None
            
        if success:
            QMessageBox.information(
                self, "Export Success", "Successfully consolidated and exported the scene!"
            )
        else:
            QMessageBox.critical(
                self, "Export Error", f"Consolidation/export failed:\n{error_msg}"
            )
            
        self.export_worker = None

    def _change_tool_to_translate(self):
        self.viewport.gizmo.operation = "translate"
        self.btn_tool_translate.setChecked(True)
        self.viewport.update()

    def _change_tool_to_rotate(self):
        self.viewport.gizmo.operation = "rotate"
        self.btn_tool_rotate.setChecked(True)
        self.viewport.update()

    def _change_tool_to_scale(self):
        self.viewport.gizmo.operation = "scale"
        self.btn_tool_scale.setChecked(True)
        self.viewport.update()

    def _on_viewport_tool_changed(self, operation):
        if operation == "translate":
            self.btn_tool_translate.setChecked(True)
        elif operation == "rotate":
            self.btn_tool_rotate.setChecked(True)
        elif operation == "scale":
            self.btn_tool_scale.setChecked(True)

    def _toggle_transform_space(self):
        gizmo = self.viewport.gizmo
        if gizmo.space == "local":
            gizmo.space = "global"
            self.btn_tool_space.setText("Global")
            self.btn_tool_space.setToolTip("Transform Space: Global (Click to toggle)")
            self.btn_tool_space.setStyleSheet("""
                QPushButton {
                    font-size: 10px;
                    font-weight: bold;
                    padding: 6px 2px;
                    background-color: #2D2D2D;
                    color: #ffffff;
                    border: 1px solid #444444;
                    border-radius: 4px;
                    margin: 4px;
                }
                QPushButton:hover {
                    background-color: #3D3D3D;
                    border-color: #ffffff;
                }
            """)
        else:
            gizmo.space = "local"
            self.btn_tool_space.setText("Local")
            self.btn_tool_space.setToolTip("Transform Space: Local (Click to toggle)")
            self.btn_tool_space.setStyleSheet("""
                QPushButton {
                    font-size: 10px;
                    font-weight: bold;
                    padding: 6px 2px;
                    background-color: #2D2D2D;
                    color: #00E676;
                    border: 1px solid #444444;
                    border-radius: 4px;
                    margin: 4px;
                }
                QPushButton:hover {
                    background-color: #3D3D3D;
                    border-color: #00E676;
                }
            """)
        self.viewport.update()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            # Check if there's at least one supported file
            has_supported = False
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.obj', '.glb', '.gltf', '.ply', '.pxm'):
                    has_supported = True
                    break
            if has_supported:
                self.viewport_container.setStyleSheet(
                    "QWidget#ViewportContainer { border: 2px dashed #00E676; background-color: #1A2820; }"
                )
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.viewport_container.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self.viewport_container.setStyleSheet("")
        
        mesh_files = []
        pxm_files = []
        texture_files = []
        
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path or not os.path.exists(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.obj', '.glb', '.gltf', '.ply'):
                mesh_files.append(os.path.normpath(path))
            elif ext == '.pxm':
                pxm_files.append(os.path.normpath(path))
            elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tga', '.mtl'):
                texture_files.append(os.path.normpath(path))
                
        # Handle project file drop
        if pxm_files:
            # We process the first pxm file dropped
            self._on_import_pxm_clicked(pxm_files[0])
            event.acceptProposedAction()
            return
            
        # Handle mesh files drop
        if mesh_files:
            for mesh_file in mesh_files:
                # Find matching companion texture if dropped together
                companion_texture = None
                mesh_name = os.path.splitext(os.path.basename(mesh_file))[0].lower()
                
                # Check for exact name match (ignoring extensions)
                for tex in texture_files:
                    tex_name = os.path.splitext(os.path.basename(tex))[0].lower()
                    if tex_name == mesh_name:
                        companion_texture = tex
                        break
                
                # If no exact name match, but exactly one texture file was dropped, use it
                if companion_texture is None and len(texture_files) == 1:
                    companion_texture = texture_files[0]
                    
                if companion_texture:
                    self._import_mesh_from_path(mesh_file, companion_texture)
                else:
                    # No companion texture found, use standard prompt (which does auto-detect check internally)
                    has_texture = False
                    ext = os.path.splitext(mesh_file)[1].lower()
                    if ext in ('.glb', '.gltf'):
                        has_texture = True
                    elif ext == '.obj':
                        mtl_path = os.path.splitext(mesh_file)[0] + '.mtl'
                        if os.path.exists(mtl_path):
                            try:
                                with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    for line in f:
                                        if line.strip().startswith('map_Kd'):
                                            has_texture = True
                                            break
                            except Exception:
                                pass
                                
                    if not has_texture:
                        reply = QMessageBox.question(
                            self,
                            "Select Texture File",
                            f"Model '{os.path.basename(mesh_file)}' does not appear to have an embedded or associated texture map.\n\n"
                            "Would you like to select a companion texture file (e.g., .png, .jpg, .mtl)?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No
                        )
                        if reply == QMessageBox.Yes:
                            selected_paths, _ = QFileDialog.getOpenFileNames(
                                self,
                                "Select Texture / Material File(s)",
                                os.path.dirname(mesh_file),
                                "Texture & Material Files (*.png *.jpg *.jpeg *.bmp *.tga *.mtl)"
                            )
                            if selected_paths:
                                mtl_files = [p for p in selected_paths if p.lower().endswith('.mtl')]
                                img_files = [p for p in selected_paths if not p.lower().endswith('.mtl')]
                                texture_path = mtl_files[0] if mtl_files else img_files[0]
                                self._import_mesh_from_path(mesh_file, texture_path)
                                continue
                    self._import_mesh_from_path(mesh_file)
            event.acceptProposedAction()
        else:
            event.ignore()
