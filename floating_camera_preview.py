"""
floating_camera_preview.py
==========================
Floating widgets that appear over the VisPy 3D viewport:

  FloatingCameraPreviewWidget
      Bottom-right PiP inspector card showing the clicked camera's photo,
      filename, camera ID, and action buttons ("Look Through Camera", "Close").

  CameraSettingsPopover
      Compact popover card (anchored near the "Camera Settings" toolbar button)
      containing Camera Frustum Scale and Photo Opacity sliders.
"""

import os

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSlider, QSizePolicy,
)


# ---------------------------------------------------------------------------
# Common styling helpers
# ---------------------------------------------------------------------------

_CARD_STYLE = """
    QWidget#CameraPreviewCard, QWidget#CameraSettingsCard {
        background-color: rgba(22, 22, 22, 230);
        border: 1px solid #3A3A3A;
        border-radius: 8px;
    }
"""

_BTN_STYLE = """
    QPushButton {
        font-size: 11px;
        padding: 5px 10px;
        background-color: #2A2A2A;
        color: #e0e0e0;
        border: 1px solid #444444;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #333333;
        border-color: #00E5FF;
        color: #00E5FF;
    }
"""

_PRIMARY_BTN_STYLE = """
    QPushButton {
        font-size: 11px;
        padding: 5px 10px;
        background-color: #1A3A3A;
        color: #00E5FF;
        border: 1px solid #00E5FF;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #1E4A4A;
    }
"""

_LABEL_STYLE = "color: #c0c0c0; font-size: 11px;"
_TITLE_STYLE = "color: #e8e8e8; font-size: 12px; font-weight: bold;"
_SLIDER_STYLE = """
    QSlider::groove:horizontal {
        background: #333333;
        height: 4px;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #00E5FF;
        width: 12px;
        height: 12px;
        margin: -4px 0;
        border-radius: 6px;
    }
    QSlider::sub-page:horizontal {
        background: #00E5FF;
        border-radius: 2px;
    }
"""


# ---------------------------------------------------------------------------
# FloatingCameraPreviewWidget
# ---------------------------------------------------------------------------

class FloatingCameraPreviewWidget(QWidget):
    """PiP inspector card pinned to the bottom-right of the viewport.

    Signals:
        look_through_requested(int)   – user clicked "Look Through Camera"
        closed()                      – user clicked "Close"
    """

    look_through_requested = Signal(int)
    closed = Signal()

    _PREVIEW_SIZE = 220  # px – square preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CameraPreviewCard")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(_CARD_STYLE)

        self._camera_index = -1
        self._full_res_path = ""

        self._build_ui()
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_camera(self, index: int, cam: dict):
        """Populate the card with *cam* data and make it visible."""
        self._camera_index = index
        image_path = cam.get("image_path", "")
        image_name = cam.get("image_name", os.path.basename(image_path) if image_path else "Unknown")
        camera_id = cam.get("image_id", index + 1)

        self._lbl_name.setText(image_name)
        self._lbl_id.setText(f"Camera {camera_id}")
        self._full_res_path = image_path

        # Load a preview-sized version (up to 220 px, keeps aspect)
        self._load_preview(image_path)
        self.adjustSize()

        # Reposition bottom-right (parent must be set correctly)
        self._reposition()
        self.show()
        self.raise_()

    def hide_card(self):
        self._camera_index = -1
        self.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Title row
        title_row = QHBoxLayout()
        self._lbl_id = QLabel("Camera 0")
        self._lbl_id.setStyleSheet(_TITLE_STYLE)
        title_row.addWidget(self._lbl_id)
        title_row.addStretch()

        self._btn_close = QPushButton("Close")
        self._btn_close.setStyleSheet(_BTN_STYLE)
        self._btn_close.setFixedHeight(24)
        self._btn_close.clicked.connect(self._on_close)
        title_row.addWidget(self._btn_close)
        layout.addLayout(title_row)

        # Image preview
        self._img_label = QLabel()
        self._img_label.setFixedSize(self._PREVIEW_SIZE, self._PREVIEW_SIZE)
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet(
            "background-color: #111111; border: 1px solid #2B2B2B; border-radius: 4px;"
        )
        self._img_label.setText("Loading…")
        layout.addWidget(self._img_label)

        # Filename
        self._lbl_name = QLabel("–")
        self._lbl_name.setStyleSheet(_LABEL_STYLE)
        self._lbl_name.setWordWrap(True)
        self._lbl_name.setMaximumWidth(self._PREVIEW_SIZE)
        layout.addWidget(self._lbl_name)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_look = QPushButton("Look Through Camera")
        self._btn_look.setStyleSheet(_PRIMARY_BTN_STYLE)
        self._btn_look.clicked.connect(self._on_look_through)
        btn_row.addWidget(self._btn_look)
        layout.addLayout(btn_row)

    def _load_preview(self, path: str):
        if not path or not os.path.isfile(path):
            self._img_label.setPixmap(QPixmap())
            self._img_label.setText("No image")
            return
        try:
            pix = QPixmap(path)
            if pix.isNull():
                raise RuntimeError("null pixmap")
            pix = pix.scaled(
                self._PREVIEW_SIZE, self._PREVIEW_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._img_label.setText("")
            self._img_label.setPixmap(pix)
        except Exception:
            self._img_label.setPixmap(QPixmap())
            self._img_label.setText("Preview unavailable")

    def _reposition(self):
        p = self.parent()
        if p is None:
            return
        margin = 12
        x = p.width() - self.width() - margin
        y = p.height() - self.height() - margin
        self.move(max(0, x), max(0, y))

    def _on_close(self):
        self.hide_card()
        self.closed.emit()

    def _on_look_through(self):
        self.look_through_requested.emit(self._camera_index)


# ---------------------------------------------------------------------------
# CameraSettingsPopover
# ---------------------------------------------------------------------------

class CameraSettingsPopover(QWidget):
    """Compact popover with Camera Frustum Scale and Photo Opacity sliders.

    Signals:
        scale_changed(float)    – 0.2 to 3.0
        opacity_changed(float)  – 0.0 to 1.0
    """

    scale_changed = Signal(float)
    opacity_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CameraSettingsCard")
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(_CARD_STYLE)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Title
        title = QLabel("Camera Settings")
        title.setStyleSheet(_TITLE_STYLE)
        layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333333;")
        layout.addWidget(sep)

        # --- Scale ---
        lbl_scale = QLabel("Frustum Scale")
        lbl_scale.setStyleSheet(_LABEL_STYLE)
        layout.addWidget(lbl_scale)

        scale_row = QHBoxLayout()
        self._scale_slider = QSlider(Qt.Horizontal)
        self._scale_slider.setRange(20, 300)   # represents 0.2 to 3.0
        self._scale_slider.setValue(100)        # default 1.0
        self._scale_slider.setStyleSheet(_SLIDER_STYLE)
        self._scale_slider.valueChanged.connect(self._on_scale_changed)
        scale_row.addWidget(self._scale_slider)

        self._scale_val_lbl = QLabel("1.0x")
        self._scale_val_lbl.setStyleSheet(_LABEL_STYLE)
        self._scale_val_lbl.setFixedWidth(36)
        scale_row.addWidget(self._scale_val_lbl)
        layout.addLayout(scale_row)

        # --- Opacity ---
        lbl_opacity = QLabel("Photo Opacity")
        lbl_opacity.setStyleSheet(_LABEL_STYLE)
        layout.addWidget(lbl_opacity)

        opacity_row = QHBoxLayout()
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(10, 100)  # 10% to 100%
        self._opacity_slider.setValue(85)        # default 85%
        self._opacity_slider.setStyleSheet(_SLIDER_STYLE)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)

        self._opacity_val_lbl = QLabel("85%")
        self._opacity_val_lbl.setStyleSheet(_LABEL_STYLE)
        self._opacity_val_lbl.setFixedWidth(36)
        opacity_row.addWidget(self._opacity_val_lbl)
        layout.addLayout(opacity_row)

        self.setFixedWidth(230)

    def _on_scale_changed(self, value: int):
        scale = value / 100.0
        self._scale_val_lbl.setText(f"{scale:.1f}x")
        self.scale_changed.emit(scale)

    def _on_opacity_changed(self, value: int):
        opacity = value / 100.0
        self._opacity_val_lbl.setText(f"{value}%")
        self.opacity_changed.emit(opacity)

    def show_near(self, global_pos):
        """Show the popover below *global_pos* (e.g. a toolbar button's bottom-left)."""
        self.adjustSize()
        self.move(global_pos)
        self.show()
        self.raise_()
