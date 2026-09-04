"""
Point Cloud Transform Card Overlay
Provides interactive translation, rotation, PCA auto-leveling, centering, and ground snapping
for point clouds in the 3D Reconstruction Viewport.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSlider, QGridLayout, QGroupBox, QSizePolicy
)


class PointCloudTransformCard(QFrame):
    """Draggable floating transform card overlay for orienting point clouds."""
    
    transform_changed = Signal(object)      # Emits 4x4 affine transform matrix (np.ndarray) for live preview
    transform_applied = Signal(object)      # Emits 4x4 matrix when user clicks 'Apply'
    transform_reset = Signal()              # Emits when user clicks 'Reset'
    transform_closed = Signal()             # Emits when card is closed

    def __init__(self, parent=None, points_provider=None):
        super().__init__(parent)
        self.points_provider = points_provider  # Callable returning current np.ndarray of points
        self._drag_pos = QPoint()
        self._is_dragging = False
        self._updating_ui = False

        self.setObjectName("PointCloudTransformCard")
        self.setFixedWidth(290)
        self.setStyleSheet("""
            QFrame#PointCloudTransformCard {
                background-color: rgba(24, 24, 24, 0.95);
                border: 1px solid #3A3A3A;
                border-radius: 8px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 11px;
            }
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 11px;
                font-weight: bold;
                color: #00E676;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
            QDoubleSpinBox {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
            }
            QDoubleSpinBox:focus {
                border-color: #00E676;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #333333;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #00E676;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #777777;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #2e2e2e;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
                border-color: #00E676;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
            QPushButton#ApplyBtn {
                background-color: #00E676;
                color: #121212;
                font-weight: bold;
                border: 1px solid #00E676;
            }
            QPushButton#ApplyBtn:hover {
                background-color: #00C853;
            }
            QPushButton#CloseBtn {
                background: transparent;
                border: none;
                color: #888888;
                font-size: 13px;
                font-weight: bold;
                padding: 0px 4px;
            }
            QPushButton#CloseBtn:hover {
                color: #ff5252;
            }
        """)

        self._build_ui()
        self.adjustSize()

    def showEvent(self, event):
        super().showEvent(event)
        self.adjustSize()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 10)
        main_layout.setSpacing(8)

        # 1. Header (Draggable Title Bar + Close Button)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = QLabel("Point Cloud Transform")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #ffffff;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("CloseBtn")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.clicked.connect(self._on_close_clicked)
        header_layout.addWidget(self.close_btn)

        main_layout.addLayout(header_layout)

        # 2. Translation Group
        trans_group = QGroupBox("Translation (m)")
        trans_layout = QVBoxLayout(trans_group)
        trans_layout.setContentsMargins(6, 8, 6, 6)
        trans_layout.setSpacing(4)

        self.spin_tx, self.slider_tx = self._create_axis_row("X:", -50.0, 50.0, 0.0, 0.05, trans_layout)
        self.spin_ty, self.slider_ty = self._create_axis_row("Y:", -50.0, 50.0, 0.0, 0.05, trans_layout)
        self.spin_tz, self.slider_tz = self._create_axis_row("Z:", -50.0, 50.0, 0.0, 0.05, trans_layout)

        # Quick Translation Actions
        quick_trans_layout = QHBoxLayout()
        quick_trans_layout.setSpacing(4)
        
        self.btn_center = QPushButton("Center to Origin")
        self.btn_center.setToolTip("Translates point cloud so its bounding center is at (0,0,0)")
        self.btn_center.clicked.connect(self._center_to_origin)
        quick_trans_layout.addWidget(self.btn_center)

        self.btn_snap_ground = QPushButton("Snap to Ground")
        self.btn_snap_ground.setToolTip("Aligns bottom of point cloud to Y=0 ground grid")
        self.btn_snap_ground.clicked.connect(self._snap_to_ground)
        quick_trans_layout.addWidget(self.btn_snap_ground)

        trans_layout.addLayout(quick_trans_layout)
        main_layout.addWidget(trans_group)

        # 3. Rotation Group
        rot_group = QGroupBox("Rotation (°)")
        rot_layout = QVBoxLayout(rot_group)
        rot_layout.setContentsMargins(6, 8, 6, 6)
        rot_layout.setSpacing(4)

        self.spin_rx, self.slider_rx = self._create_axis_row("X:", -180.0, 180.0, 0.0, 1.0, rot_layout, is_angle=True)
        self.spin_ry, self.slider_ry = self._create_axis_row("Y:", -180.0, 180.0, 0.0, 1.0, rot_layout, is_angle=True)
        self.spin_rz, self.slider_rz = self._create_axis_row("Z:", -180.0, 180.0, 0.0, 1.0, rot_layout, is_angle=True)

        # Quick 90 deg rotation buttons
        rot_btn_grid = QGridLayout()
        rot_btn_grid.setSpacing(4)

        btn_x_plus = QPushButton("+90° X")
        btn_x_plus.clicked.connect(lambda: self._add_rotation(90, 0, 0))
        btn_x_minus = QPushButton("-90° X")
        btn_x_minus.clicked.connect(lambda: self._add_rotation(-90, 0, 0))
        rot_btn_grid.addWidget(btn_x_plus, 0, 0)
        rot_btn_grid.addWidget(btn_x_minus, 0, 1)

        btn_y_plus = QPushButton("+90° Y")
        btn_y_plus.clicked.connect(lambda: self._add_rotation(0, 90, 0))
        btn_y_minus = QPushButton("-90° Y")
        btn_y_minus.clicked.connect(lambda: self._add_rotation(0, -90, 0))
        rot_btn_grid.addWidget(btn_y_plus, 1, 0)
        rot_btn_grid.addWidget(btn_y_minus, 1, 1)

        btn_z_plus = QPushButton("+90° Z")
        btn_z_plus.clicked.connect(lambda: self._add_rotation(0, 0, 90))
        btn_z_minus = QPushButton("-90° Z")
        btn_z_minus.clicked.connect(lambda: self._add_rotation(0, 0, -90))
        rot_btn_grid.addWidget(btn_z_plus, 2, 0)
        rot_btn_grid.addWidget(btn_z_minus, 2, 1)

        btn_flip_x = QPushButton("180° Flip X")
        btn_flip_x.clicked.connect(lambda: self._add_rotation(180, 0, 0))
        btn_flip_z = QPushButton("180° Flip Z")
        btn_flip_z.clicked.connect(lambda: self._add_rotation(0, 0, 180))
        rot_btn_grid.addWidget(btn_flip_x, 3, 0)
        rot_btn_grid.addWidget(btn_flip_z, 3, 1)

        rot_layout.addLayout(rot_btn_grid)
        main_layout.addWidget(rot_group)

        # 4. Auto-Alignment (PCA Leveling)
        align_layout = QHBoxLayout()
        self.btn_auto_level = QPushButton("⚡ Auto-Level (PCA)")
        self.btn_auto_level.setToolTip("Automatically levels point cloud using Principal Component Analysis")
        self.btn_auto_level.setStyleSheet("font-weight: bold; color: #00E676; border-color: #00E676;")
        self.btn_auto_level.clicked.connect(self._auto_level_pca)
        align_layout.addWidget(self.btn_auto_level)
        main_layout.addLayout(align_layout)

        # 5. Action Footer Buttons (Apply, Reset)
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(6)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_transform)
        footer_layout.addWidget(self.btn_reset, stretch=1)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("ApplyBtn")
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        footer_layout.addWidget(self.btn_apply, stretch=2)

        main_layout.addLayout(footer_layout)

    def _create_axis_row(self, label_text, min_val, max_val, default_val, step, parent_layout, is_angle=False):
        row = QHBoxLayout()
        row.setSpacing(4)

        lbl = QLabel(label_text)
        lbl.setFixedWidth(16)
        row.addWidget(lbl)

        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setSingleStep(step)
        spin.setDecimals(2 if not is_angle else 1)
        spin.setFixedWidth(65)
        row.addWidget(spin)

        slider = QSlider(Qt.Horizontal)
        slider_scale = 10.0 if not is_angle else 1.0
        slider.setRange(int(min_val * slider_scale), int(max_val * slider_scale))
        slider.setValue(int(default_val * slider_scale))
        row.addWidget(slider)

        # Sync spinbox and slider
        def on_spin_changed(val):
            if not self._updating_ui:
                self._updating_ui = True
                slider.setValue(int(val * slider_scale))
                self._updating_ui = False
                self._emit_transform_preview()

        def on_slider_changed(val):
            if not self._updating_ui:
                self._updating_ui = True
                spin.setValue(val / slider_scale)
                self._updating_ui = False
                self._emit_transform_preview()

        spin.valueChanged.connect(on_spin_changed)
        slider.valueChanged.connect(on_slider_changed)

        parent_layout.addLayout(row)
        return spin, slider

    def get_transform_matrix(self) -> np.ndarray:
        """Computes 4x4 affine transformation matrix from current spinbox values."""
        tx = self.spin_tx.value()
        ty = self.spin_ty.value()
        tz = self.spin_tz.value()

        rx = np.radians(self.spin_rx.value())
        ry = np.radians(self.spin_ry.value())
        rz = np.radians(self.spin_rz.value())

        # Rotation matrices (XYZ Euler order)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ], dtype=np.float32)

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ], dtype=np.float32)

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        R = Rz @ Ry @ Rx

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def _emit_transform_preview(self):
        mat = self.get_transform_matrix()
        self.transform_changed.emit(mat)

    def _add_rotation(self, dx, dy, dz):
        self._updating_ui = True
        
        def wrap_angle(angle):
            while angle > 180.0: angle -= 360.0
            while angle < -180.0: angle += 360.0
            return angle

        new_rx = wrap_angle(self.spin_rx.value() + dx)
        new_ry = wrap_angle(self.spin_ry.value() + dy)
        new_rz = wrap_angle(self.spin_rz.value() + dz)

        self.spin_rx.setValue(new_rx)
        self.slider_rx.setValue(int(new_rx))
        self.spin_ry.setValue(new_ry)
        self.slider_ry.setValue(int(new_ry))
        self.spin_rz.setValue(new_rz)
        self.slider_rz.setValue(int(new_rz))

        self._updating_ui = False
        self._emit_transform_preview()

    def _get_active_points(self):
        if callable(self.points_provider):
            return self.points_provider()
        return None

    def _center_to_origin(self):
        pts = self._get_active_points()
        if pts is None or len(pts) == 0:
            return
        
        # Apply current rotation to points first to find oriented centroid
        R = self.get_transform_matrix()[:3, :3]
        rotated_pts = pts @ R.T
        center = np.mean(rotated_pts, axis=0)

        self._updating_ui = True
        self.spin_tx.setValue(-float(center[0]))
        self.slider_tx.setValue(int(-center[0] * 10.0))
        self.spin_ty.setValue(-float(center[1]))
        self.slider_ty.setValue(int(-center[1] * 10.0))
        self.spin_tz.setValue(-float(center[2]))
        self.slider_tz.setValue(int(-center[2] * 10.0))
        self._updating_ui = False
        self._emit_transform_preview()

    def _snap_to_ground(self):
        pts = self._get_active_points()
        if pts is None or len(pts) == 0:
            return
        
        R = self.get_transform_matrix()[:3, :3]
        rotated_pts = pts @ R.T
        min_y = np.min(rotated_pts[:, 1])

        self._updating_ui = True
        self.spin_ty.setValue(-float(min_y))
        self.slider_ty.setValue(int(-min_y * 10.0))
        self._updating_ui = False
        self._emit_transform_preview()

    def _auto_level_pca(self):
        """Computes PCA orientation to level the point cloud to the ground plane."""
        pts = self._get_active_points()
        if pts is None or len(pts) < 10:
            return
        
        # Subsample for fast PCA calculation if large
        sample = pts if len(pts) <= 10000 else pts[np.random.choice(len(pts), 10000, replace=False)]
        centroid = np.mean(sample, axis=0)
        centered = sample - centroid
        
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by decreasing eigenvalue
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_idx]

        # In PCA:
        # e0, e1 span the main dominant planes (ground / surface)
        # e2 (smallest variance) is the surface normal (up/down axis)
        v_up = eigenvectors[:, 2]
        if v_up[1] < 0:  # Orient positive Y upwards
            v_up = -v_up
            
        # Target up vector is Y-up [0, 1, 0]
        target_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v_cross = np.cross(v_up, target_up)
        norm_cross = np.linalg.norm(v_cross)
        v_dot = np.dot(v_up, target_up)

        if norm_cross > 1e-4:
            # Rodrigues rotation axis-angle
            axis = v_cross / norm_cross
            angle = np.arccos(np.clip(v_dot, -1.0, 1.0))
            
            # Convert axis-angle to rotation matrix
            s = np.sin(angle)
            c = np.cos(angle)
            t = 1.0 - c
            x, y, z = axis
            
            R_align = np.array([
                [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
            ], dtype=np.float32)
            
            # Extract Euler angles
            import trimesh
            rx, ry, rz = np.degrees(trimesh.transformations.euler_from_matrix(R_align, 'sxyz'))
            
            self._updating_ui = True
            self.spin_rx.setValue(float(rx))
            self.slider_rx.setValue(int(rx))
            self.spin_ry.setValue(float(ry))
            self.slider_ry.setValue(int(ry))
            self.spin_rz.setValue(float(rz))
            self.slider_rz.setValue(int(rz))
            self._updating_ui = False
            
            self._center_to_origin()
            self._snap_to_ground()

    def reset_transform(self):
        self._updating_ui = True
        for spin, slider in [
            (self.spin_tx, self.slider_tx), (self.spin_ty, self.slider_ty), (self.spin_tz, self.slider_tz),
            (self.spin_rx, self.slider_rx), (self.spin_ry, self.slider_ry), (self.spin_rz, self.slider_rz)
        ]:
            spin.setValue(0.0)
            slider.setValue(0)
        self._updating_ui = False
        self.transform_reset.emit()

    def _on_apply_clicked(self):
        mat = self.get_transform_matrix()
        self.transform_applied.emit(mat)
        # Reset internal sliders since transform is now baked into points
        self._updating_ui = True
        for spin, slider in [
            (self.spin_tx, self.slider_tx), (self.spin_ty, self.slider_ty), (self.spin_tz, self.slider_tz),
            (self.spin_rx, self.slider_rx), (self.spin_ry, self.slider_ry), (self.spin_rz, self.slider_rz)
        ]:
            spin.setValue(0.0)
            slider.setValue(0)
        self._updating_ui = False

    def _on_close_clicked(self):
        self.reset_transform()
        self.hide()
        self.transform_closed.emit()

    # Draggable widget handling
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() & Qt.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_pos
            parent = self.parentWidget()
            if parent:
                # Clamp within parent bounds
                max_x = parent.width() - self.width() - 5
                max_y = parent.height() - self.height() - 5
                clamped_x = max(5, min(new_pos.x(), max_x))
                clamped_y = max(5, min(new_pos.y(), max_y))
                self.move(clamped_x, clamped_y)
            else:
                self.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._is_dragging = False
        super().mouseReleaseEvent(event)
