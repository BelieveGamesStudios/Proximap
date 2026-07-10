import numpy as np
import os
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QScrollArea,
    QListWidget, QGridLayout, QDoubleSpinBox, QCheckBox, QPushButton,
    QColorDialog, QComboBox, QFileDialog, QMessageBox, QDialog, QProgressBar,
    QMenuBar, QMenu, QWidgetAction
)
from PySide6.QtCore import Qt, QThread, Signal
from mesh_editor.viewport import MeshEditorViewport
from mesh_editor.scene import Object, load_mesh_file

class MeshImportWorker(QThread):
    # Signals: finished (success, mesh, error_message)
    finished = Signal(bool, object, str)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        try:
            # Load mesh in the worker thread (heavy CPU task)
            mesh = load_mesh_file(self.file_path)
            self.finished.emit(True, mesh, "")
        except Exception as e:
            import traceback
            err_details = traceback.format_exc()
            self.finished.emit(False, None, f"{str(e)}\n\n{err_details}")

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
        self.outliner_list.setFixedHeight(120)
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
        
        scroll_layout.addWidget(outliner_box)
        
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
        scroll_layout.addWidget(self.properties_box)
        
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll_area)
        layout.addWidget(self.sidebar)
        
        # Create Settings widgets for the View dropdown menu
        self.btn_projection = QPushButton("Camera: Perspective")
        self.btn_projection.setStyleSheet("QPushButton { font-size: 11px; padding: 6px 12px; }")
        
        self.cb_snap = QCheckBox("Enable Snapping")
        self.cb_snap.setStyleSheet("color: #ffffff; font-size: 11px;")
        
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
        viewport_container = QWidget(self)
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        
        # Create Menu Bar / Toolbar
        self.menu_bar = QMenuBar(viewport_container)
        self.menu_bar.setObjectName("ViewportMenuBar")
        
        # 1. File Menu
        file_menu = QMenu("File", self.menu_bar)
        action_import = file_menu.addAction("Import Mesh (.obj, .glb)")
        action_import.triggered.connect(self._on_import_model_clicked)
        action_export = file_menu.addAction("Export Scene (.obj, .glb)")
        action_export.triggered.connect(self._on_export_scene_clicked)
        self.menu_bar.addMenu(file_menu)
        
        # 2. Edit Menu
        edit_menu = QMenu("Edit", self.menu_bar)
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
        
        self.viewport = MeshEditorViewport(viewport_container)
        viewport_layout.addWidget(self.viewport, stretch=1)
        
        layout.addWidget(viewport_container, stretch=4)
        
        # Disable properties inputs initially until an object is selected
        self._set_properties_enabled(False)

    def _wire_events(self):
        # 1. Viewport Selection -> Sync Sidebar
        self.viewport.selection_changed.connect(self._on_viewport_selection_changed)
        # 2. Viewport Gizmo Edit -> Sync Spinboxes
        self.viewport.transform_changed.connect(self._on_viewport_transform_changed)
        
        # 3. Sidebar Selection -> Sync Viewport Selection
        self.outliner_list.currentRowChanged.connect(self._on_outliner_row_changed)
        
        # 4. Spinbox value changes -> Sync Viewport Object Transform
        for sb in [self.sp_pos_x, self.sp_pos_y, self.sp_pos_z]:
            sb.valueChanged.connect(self._on_translation_spinbox_changed)
        for sb in [self.sp_rot_x, self.sp_rot_y, self.sp_rot_z]:
            sb.valueChanged.connect(self._on_rotation_spinbox_changed)
        for sb in [self.sp_scale_x, self.sp_scale_y, self.sp_scale_z]:
            sb.valueChanged.connect(self._on_scale_spinbox_changed)
            
        # 5. Projection Toggles
        self.btn_projection.clicked.connect(self._on_projection_clicked)
        
        # 6. Snapping Toggles and values
        self.cb_snap.stateChanged.connect(self._on_snap_toggled)
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
    def _on_viewport_selection_changed(self, selected_obj: Object):
        self.btn_delete_mesh.setEnabled(selected_obj is not None)
        if selected_obj is None:
            self.outliner_list.blockSignals(True)
            self.outliner_list.clearSelection()
            self.outliner_list.blockSignals(False)
            self._set_properties_enabled(False)
            self._block_properties_signals(True)
            for sb in [self.sp_pos_x, self.sp_pos_y, self.sp_pos_z, self.sp_rot_x, self.sp_rot_y, self.sp_rot_z]:
                sb.setValue(0.0)
            for sb in [self.sp_scale_x, self.sp_scale_y, self.sp_scale_z]:
                sb.setValue(1.0)
            self._block_properties_signals(False)
        else:
            # Sync outliner selection
            self.outliner_list.blockSignals(True)
            for i in range(self.outliner_list.count()):
                item_name = self.outliner_list.item(i).text()
                if item_name == selected_obj.name:
                    self.outliner_list.setCurrentRow(i)
                    break
            self.outliner_list.blockSignals(False)
            
            # Populate fields
            self._set_properties_enabled(True)
            self._sync_properties_from_object(selected_obj)

    def _on_viewport_transform_changed(self, obj: Object):
        # Update inputs dynamically from gizmo dragging without loops
        self._sync_properties_from_object(obj)

    def _on_outliner_row_changed(self, row: int):
        if row < 0 or row >= len(self.viewport.scene.objects):
            self.viewport.scene.selected_object = None
        else:
            selected_obj = self.viewport.scene.objects[row]
            self.viewport.scene.selected_object = selected_obj
            self._set_properties_enabled(True)
            self._sync_properties_from_object(selected_obj)
        self.viewport.update()

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

    def _on_translation_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.position = np.array([
                self.sp_pos_x.value(),
                self.sp_pos_y.value(),
                self.sp_pos_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_rotation_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.rotation = np.array([
                self.sp_rot_x.value(),
                self.sp_rot_y.value(),
                self.sp_rot_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_scale_spinbox_changed(self):
        obj = self.viewport.scene.selected_object
        if obj:
            obj.scale = np.array([
                self.sp_scale_x.value(),
                self.sp_scale_y.value(),
                self.sp_scale_z.value()
            ], dtype=np.float32)
            self.viewport.update()

    def _on_projection_clicked(self):
        cam = self.viewport.camera
        cam.is_perspective = not cam.is_perspective
        if cam.is_perspective:
            self.btn_projection.setText("Camera: Perspective")
        else:
            self.btn_projection.setText("Camera: Orthographic")
        self.viewport.update()

    def _on_snap_toggled(self, state):
        self.viewport.imgui_bridge.use_snapping = (state == Qt.CheckState.Checked.value)
        self.viewport.update()

    def _on_snap_values_changed(self):
        bridge = self.viewport.imgui_bridge
        bridge.snap_translation = self.sp_snap_t.value()
        bridge.snap_rotation = self.sp_snap_r.value()
        # Keep scale snapping proportional to 10% of value
        bridge.snap_scale = 0.1
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

    def _on_import_model_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import 3D Model", "", "3D Mesh Files (*.obj *.glb *.gltf *.ply)"
        )
        if not file_path:
            return
            
        # Display modal dialog
        self.import_dialog = MeshImportProgressDialog(self)
        
        # Setup worker thread
        self.import_worker = MeshImportWorker(file_path)
        self.import_worker.finished.connect(self._on_import_finished)
        
        # Start worker thread and dialog
        self.import_worker.start()
        self.import_dialog.exec()

    def _on_import_finished(self, success: bool, mesh, error_msg: str):
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
            mesh.setup_buffers()
            self.viewport.doneCurrent()
            
            # 3. Create the scene object (using filename as object name)
            file_path = self.import_worker.file_path
            name = os.path.basename(file_path)
            # Ensure unique name in outliner
            base_name = name
            idx = 1
            existing_names = [o.name for o in self.viewport.scene.objects]
            while name in existing_names:
                name = f"{base_name} ({idx})"
                idx += 1
                
            obj = Object(name, mesh)
            
            # Add to scene
            self.viewport.scene.add_object(obj)
            
            # Refresh outliner list and select it
            self._populate_outliner()
            
            # Select the newly imported object
            self.viewport.scene.selected_object = obj
            self._on_viewport_selection_changed(obj)
            
            # Update rendering
            self.viewport.update()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Import Error", f"Failed to configure GL buffers:\n{str(e)}"
            )
            
        self.import_worker = None

    def _delete_selected_object(self):
        selected_obj = self.viewport.scene.selected_object
        if selected_obj is not None:
            # 1. Clean up OpenGL resources for this mesh
            self.viewport.makeCurrent()
            selected_obj.mesh.cleanup()
            self.viewport.doneCurrent()
            
            # 2. Remove from scene list
            self.viewport.scene.remove_object(selected_obj)
            
            # 3. Clear selected state
            self.viewport.scene.selected_object = None
            self._on_viewport_selection_changed(None)
            
            # 4. Refresh outliner and redraw
            self._populate_outliner()
            self.viewport.update()

    def _on_export_scene_clicked(self):
        if not self.viewport.scene.objects:
            QMessageBox.warning(
                self, "Export Warning", "There are no objects in the scene to export."
            )
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Scene", "", "3D Mesh Files (*.obj *.glb)"
        )
        if file_path:
            try:
                # Perform the export
                from mesh_editor.scene import export_scene_to_file
                export_scene_to_file(self.viewport.scene, file_path)
                QMessageBox.information(
                    self, "Export Success", f"Successfully exported scene to:\n{file_path}"
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export scene:\n{str(e)}"
                )
