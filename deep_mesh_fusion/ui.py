from __future__ import annotations

import json
import shutil
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QMimeData, QPointF, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QMouseEvent, QPainter, QPen, QPolygonF, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QTextEdit, QToolButton, QVBoxLayout,
    QWidget, QInputDialog,
)

from point_cloud_io import load_point_cloud

from .models import DeepMeshFusionConfig
from .photogrammetry import ColmapTextModelLoader
from .viewport import DeepMeshFusionViewport as FusionViewport
from .workspace import DeepMeshFusionWorkspace


PIPELINE_STAGES = (
    "Scan preparation", "Scan alignment", "Surface fusion",
    "Surface validation", "Cleanup", "Texture", "Final quality",
)


@dataclass
class ScanPreview:
    scan_id: str
    name: str
    source_path: str
    points: np.ndarray
    color: QColor
    visible: bool = True


class LegacyFusionViewport(QWidget):
    """Responsive orbitable point-cloud preview; processing still uses full sources."""

    ply_files_dropped = Signal(list)
    MAX_POINTS_PER_LAYER = 18_000
    MAX_PREVIEW_FACES = 20_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("deepMeshFusionViewport")
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.layers: Dict[str, Dict] = {}
        self.mesh_layer = None
        self._center = np.zeros(3, dtype=float)
        self._span = 1.0
        self._yaw, self._pitch, self._zoom = -35.0, 24.0, 1.0
        self._pan = np.zeros(2, dtype=float)
        self._last_mouse = None
        self._drop_active = False

    def set_layer(self, layer_id: str, points, color: QColor, visible=True, selected=False):
        points = np.asarray(points, dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1)] if len(points) else points.reshape(0, 3)
        if len(points) > self.MAX_POINTS_PER_LAYER:
            points = points[np.linspace(0, len(points) - 1, self.MAX_POINTS_PER_LAYER, dtype=int)]
        self.layers[layer_id] = {"points": points, "color": QColor(color), "visible": bool(visible), "selected": bool(selected)}
        self.update()

    def remove_layer(self, layer_id):
        self.layers.pop(layer_id, None); self.fit_to_layers()

    def set_mesh(self, vertices, faces, source_face_count=None):
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("A viewport mesh requires vertices (N, 3) and triangular faces (M, 3)")
        valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
        self.mesh_layer = {
            "vertices": vertices,
            "faces": faces[valid],
            "source_face_count": int(source_face_count if source_face_count is not None else len(faces)),
        }
        self.fit_to_layers()

    def clear_mesh(self):
        self.mesh_layer = None; self.fit_to_layers()

    def set_layer_visible(self, layer_id, visible):
        if layer_id in self.layers:
            self.layers[layer_id]["visible"] = bool(visible); self.update()

    def select_layer(self, layer_id):
        for key, layer in self.layers.items(): layer["selected"] = key == layer_id
        self.update()

    def fit_to_layers(self):
        visible = [layer["points"] for layer in self.layers.values() if layer["visible"] and len(layer["points"])]
        if self.mesh_layer is not None and len(self.mesh_layer["vertices"]): visible.append(self.mesh_layer["vertices"])
        if visible:
            points = np.vstack(visible); lower, upper = np.min(points, axis=0), np.max(points, axis=0)
            self._center = (lower + upper) * .5; self._span = max(float(np.max(upper - lower)), 1e-4)
        else:
            self._center = np.zeros(3); self._span = 1.0
        self._zoom = 1.0; self._pan[:] = 0.0; self.update()

    def paintEvent(self, _event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#0C0C0C")); self._draw_grid(painter)
        layers = [layer for layer in self.layers.values() if layer["visible"] and len(layer["points"])]
        if self.mesh_layer is None and not layers:
            painter.setPen(QColor("#777777")); painter.drawText(self.rect(), Qt.AlignCenter, "3D VIEWPORT\n\nDrag and drop .PLY scans here\nor use Add PLY Passes")
            self._draw_drop_overlay(painter)
            return
        if self.mesh_layer is not None:
            self._draw_mesh(painter)
        for layer in layers:
            projected, depth = self._project(layer["points"]); projected = projected[np.argsort(depth)]
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in projected])
            color = QColor(layer["color"]); color.setAlpha(235 if layer["selected"] else 175)
            painter.setPen(QPen(color, 2.2 if layer["selected"] else 1.25)); painter.drawPoints(polygon)
        painter.setPen(QColor("#B0B0B0")); painter.drawText(14, 22, "Orbit: drag   Pan: right-drag   Zoom: wheel")
        self._draw_drop_overlay(painter)

    def _draw_mesh(self, painter):
        vertices = self.mesh_layer["vertices"]; faces = self.mesh_layer["faces"]
        if not len(vertices) or not len(faces): return
        projected, depth, rotated = self._project(vertices, include_rotated=True)
        triangles = rotated[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        light = np.full(len(faces), .45, dtype=float)
        light[valid] = .35 + .65 * np.abs(normals[valid, 2] / lengths[valid])
        bins = np.clip((light * 7).astype(int), 0, 7)
        colors = [QColor(int(78 + i * 19), int(83 + i * 19), int(81 + i * 19)) for i in range(8)]
        painter.setPen(Qt.NoPen)
        for face_index in np.argsort(np.mean(depth[faces], axis=1)):
            polygon = QPolygonF([QPointF(float(projected[index, 0]), float(projected[index, 1])) for index in faces[face_index]])
            painter.setBrush(QBrush(colors[int(bins[face_index])]))
            painter.drawPolygon(polygon)
        painter.setBrush(Qt.NoBrush); painter.setPen(QColor("#00E676"))
        painter.drawText(14, self.height() - 16, f"VALIDATED FUSED SURFACE  ·  {self.mesh_layer['source_face_count']:,} faces")

    def _draw_drop_overlay(self, painter):
        if not self._drop_active: return
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(0, 230, 118, 28))
        painter.setPen(QPen(QColor("#00E676"), 3, Qt.DashLine))
        painter.drawRoundedRect(self.rect().adjusted(10, 10, -10, -10), 8, 8)
        painter.setPen(QColor("#00E676"))
        painter.drawText(self.rect(), Qt.AlignCenter, "DROP PLY SCANS TO IMPORT")

    def _draw_grid(self, painter):
        width, height = self.width(), self.height(); spacing = max(32, min(width, height) // 10)
        painter.setPen(QPen(QColor("#2B2B2B"), 1))
        for x in range(width // 2 % spacing, width, spacing): painter.drawLine(x, 0, x, height)
        for y in range(height // 2 % spacing, height, spacing): painter.drawLine(0, y, width, y)
        painter.setPen(QPen(QColor("#444444"), 1)); painter.drawLine(width // 2, 0, width // 2, height); painter.drawLine(0, height // 2, width, height // 2)

    def _project(self, points, include_rotated=False):
        yaw, pitch = np.radians(self._yaw), np.radians(self._pitch)
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))); rx = np.array(((1, 0, 0), (0, cp, -sp), (0, sp, cp)))
        rotated = (np.asarray(points) - self._center) @ (ry @ rx).T
        scale = .78 * min(self.width(), self.height()) * self._zoom / self._span
        projected = np.column_stack((self.width() * .5 + self._pan[0] + rotated[:, 0] * scale, self.height() * .5 + self._pan[1] - rotated[:, 1] * scale))
        return (projected, rotated[:, 2], rotated) if include_rotated else (projected, rotated[:, 2])

    def mousePressEvent(self, event: QMouseEvent): self._last_mouse = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse is None: return
        delta = event.position() - self._last_mouse
        if event.buttons() & Qt.LeftButton:
            self._yaw += delta.x() * .45; self._pitch = float(np.clip(self._pitch + delta.y() * .35, -89, 89))
        elif event.buttons() & (Qt.RightButton | Qt.MiddleButton): self._pan += (delta.x(), delta.y())
        self._last_mouse = event.position(); self.update()

    def mouseReleaseEvent(self, _event): self._last_mouse = None

    def wheelEvent(self, event: QWheelEvent):
        self._zoom = float(np.clip(self._zoom * (1.12 ** (event.angleDelta().y() / 120.0)), .08, 30)); self.update()

    @staticmethod
    def _ply_paths_from_mime(mime: QMimeData):
        if not mime or not mime.hasUrls(): return []
        return [
            str(Path(url.toLocalFile()).resolve())
            for url in mime.urls()
            if url.isLocalFile() and Path(url.toLocalFile()).is_file()
            and Path(url.toLocalFile()).suffix.lower() == ".ply"
        ]

    def dragEnterEvent(self, event):
        if self._ply_paths_from_mime(event.mimeData()):
            self._drop_active = True; event.acceptProposedAction(); self.update()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._ply_paths_from_mime(event.mimeData()): event.acceptProposedAction()
        else: event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_active = False; event.accept(); self.update()

    def dropEvent(self, event):
        paths = self._ply_paths_from_mime(event.mimeData()); self._drop_active = False; self.update()
        if paths:
            event.acceptProposedAction(); self.ply_files_dropped.emit(paths)
        else:
            event.ignore()


class ScanPassRow(QFrame):
    visibility_changed = Signal(str, bool); selected = Signal(str)
    rename_requested = Signal(str); remove_requested = Signal(str)

    def __init__(self, preview: ScanPreview, parent=None):
        super().__init__(parent); self.scan_id = preview.scan_id; self.setObjectName("scanPassRow")
        layout = QHBoxLayout(self); layout.setContentsMargins(6, 6, 4, 6)
        self.visibility = QCheckBox(); self.visibility.setChecked(preview.visible); self.visibility.setToolTip("Show or hide this pass")
        swatch = QLabel("●"); swatch.setStyleSheet(f"color:{preview.color.name()};font-size:16px")
        text = QVBoxLayout(); self.name_label = QLabel(preview.name); self.name_label.setStyleSheet("font-weight:600;color:#FFFFFF")
        self.file_label = QLabel(Path(preview.source_path).name); self.file_label.setStyleSheet("color:#A0A0A0;font-size:10px"); self.file_label.setToolTip(preview.source_path)
        text.addWidget(self.name_label); text.addWidget(self.file_label)
        rename = QToolButton(); rename.setText("✎"); rename.setToolTip("Rename scan")
        remove = QToolButton(); remove.setText("×"); remove.setToolTip("Remove scan")
        layout.addWidget(self.visibility); layout.addWidget(swatch); layout.addLayout(text, 1); layout.addWidget(rename); layout.addWidget(remove)
        self.visibility.toggled.connect(lambda checked: self.visibility_changed.emit(self.scan_id, checked))
        rename.clicked.connect(lambda: self.rename_requested.emit(self.scan_id)); remove.clicked.connect(lambda: self.remove_requested.emit(self.scan_id))

    def mousePressEvent(self, event): self.selected.emit(self.scan_id); super().mousePressEvent(event)

    def set_selected(self, selected):
        border = "#00E676" if selected else "#333333"; background = "#213328" if selected else "#1A1A1A"
        self.setStyleSheet(f"QFrame#scanPassRow{{background:{background};border:1px solid {border};border-radius:6px;}}")


class DeepMeshFusionTaskWorker(QThread):
    log_message = Signal(str); completed = Signal(str, object); failed = Signal(str, str)

    def __init__(self, task, *, workspace=None, workspace_path=None, scan_paths=None, scan_names=None, config=None, photogrammetry=None, allow_review=False, reference_pass_id=None, mesh_path=None, parent=None):
        super().__init__(parent); self.task = task; self.workspace = workspace; self.workspace_path = workspace_path
        self.scan_paths = list(scan_paths or []); self.scan_names = list(scan_names or []); self.config = config
        self.photogrammetry = photogrammetry; self.allow_review = allow_review; self.reference_pass_id = reference_pass_id; self.mesh_path = mesh_path

    def run(self):
        try:
            if self.task == "diagnostics":
                workspace = DeepMeshFusionWorkspace(self.workspace_path, self.config, self.log_message.emit)
                for index, path in enumerate(self.scan_paths): self._check(); workspace.add_pass(path, self.scan_names[index])
                self.completed.emit(self.task, {"workspace": workspace, "passes": workspace.analyze_passes()})
            elif self.task == "alignment":
                self._check(); self.completed.emit(self.task, self.workspace.register_passes(self.reference_pass_id))
            elif self.task == "point_fusion":
                self._check(); self.completed.emit(self.task, self.workspace.fuse_points_registered())
            elif self.task == "surface":
                self._check(); self.completed.emit(self.task, self.workspace.reconstruct_fused_surface())
            elif self.task == "cleanup_validation":
                self._check(); self.completed.emit(self.task, self.workspace.validate_cleaned_mesh(self.mesh_path))
            elif self.task == "photo_registration":
                model, images, dense = self.photogrammetry
                photo = self.workspace.prepare_photogrammetry(model, images, dense, allow_geometry_review=self.allow_review)
                self.completed.emit(self.task, photo)
            elif self.task == "texture":
                texture = self.workspace.bake_textures(allow_texture_review=self.allow_review)
                self._check(); final = self.workspace.finalize_asset(allow_texture_review=self.allow_review)
                self._check(); quality = self.workspace.evaluate_tour_readiness()
                self.completed.emit(self.task, {"texture": texture, "final": final, "quality": quality})
            else: raise ValueError(f"Unknown Deep Mesh Fusion task: {self.task}")
        except InterruptedError: self.failed.emit(self.task, "Processing cancelled after the current safe stage.")
        except Exception as error: self.failed.emit(self.task, f"{type(error).__name__}: {error}")

    def _check(self):
        if self.isInterruptionRequested(): raise InterruptedError


class DeepMeshFusionPanel(QWidget):
    """Viewport-first professional workflow backed by Milestones 1–11."""

    request_reconstruction = Signal()
    COLORS = ("#00E676", "#29B6F6", "#FFB74D", "#AB47BC", "#FF5252", "#26C6DA", "#EC407A")

    def __init__(self, parent=None, reconstruction_root: Optional[str] = None):
        super().__init__(parent)
        self.setObjectName("deepMeshFusionPanel")
        self.reconstruction_root = Path(reconstruction_root or "reconstruction_out").resolve()
        self.worker = None; self.workspace = None; self.scan_previews = {}; self.scan_rows = {}; self.selected_scan_id = None
        self.photogrammetry_paths = None; self.photogrammetry_loaded = False; self._loaded_photogrammetry_paths = None
        self.last_result = {}; self.current_stage = 0; self.unlocked_stage = 0; self.completed_stages = set()
        self.processing = False; self.analysis_requires_review = False; self.mesh_op_worker = None; self.cleanup_history = []
        self._setup_ui(); self._restore_workspace_state(); self.refresh_reconstruction_state(); self._update_state()

    def _setup_ui(self):
        self.setStyleSheet("""
            QWidget#deepMeshFusionPanel { background:#121212; color:#FFFFFF; }
            QScrollArea { background:#1A1A1A; border:1px solid #2B2B2B; }
            QScrollArea > QWidget > QWidget { background:#1A1A1A; }
            QFrame#fusionSidebar { background:#1A1A1A; border:1px solid #2B2B2B; }
            QFrame#fusionDiagnostics { background:#1A1A1A; border:1px solid #2B2B2B; }
            QSplitter::handle { background:#2B2B2B; }
            QSplitter::handle:hover { background:#00E676; }
            QPushButton, QToolButton { background:#2A2A2A; color:#FFFFFF; border:1px solid #3A3A3A; border-radius:4px; padding:5px; }
            QPushButton:hover, QToolButton:hover { background:#333333; color:#00E676; border-color:#00E676; }
            QPushButton:pressed, QToolButton:pressed { background:#213328; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background:#1E1E1E; color:#FFFFFF; border:1px solid #333333; border-radius:4px; }
            QCheckBox { color:#FFFFFF; }
        """)
        root = QVBoxLayout(self); root.setContentsMargins(12, 10, 12, 10); root.setSpacing(8)
        header = QHBoxLayout(); title_box = QVBoxLayout(); title = QLabel("Deep Mesh Fusion")
        title.setStyleSheet("font-size:21px;font-weight:700;color:#FFFFFF")
        subtitle = QLabel("Combine multiple LiDAR scans into a reliable, tour-ready environment"); subtitle.setStyleSheet("color:#B0B0B0")
        title_box.addWidget(title); title_box.addWidget(subtitle); header.addLayout(title_box); header.addStretch()
        self.stage_counter = QLabel(f"Stage 1 of {len(PIPELINE_STAGES)}"); self.stage_counter.setStyleSheet("color:#00E676;font-weight:600")
        header.addWidget(self.stage_counter); root.addLayout(header)
        self.stage_labels = []; stage_bar = QHBoxLayout(); stage_bar.setSpacing(5)
        for index, stage in enumerate(PIPELINE_STAGES):
            label = QToolButton(); label.setText(stage); label.setToolButtonStyle(Qt.ToolButtonTextOnly)
            label.clicked.connect(lambda _checked=False, stage_index=index: self._select_stage(stage_index))
            stage_bar.addWidget(label, 1); self.stage_labels.append(label)
        root.addLayout(stage_bar)
        self.main_splitter = QSplitter(Qt.Horizontal); self.main_splitter.setChildrenCollapsible(False); self.main_splitter.setHandleWidth(5)
        self.main_splitter.addWidget(self._sidebar())
        self.detail_splitter = QSplitter(Qt.Vertical); self.detail_splitter.setChildrenCollapsible(False); self.detail_splitter.setHandleWidth(5)
        self.viewport = FusionViewport(); self.viewport.ply_files_dropped.connect(self._add_scan_paths)
        self.detail_splitter.addWidget(self.viewport)
        self.detail_splitter.addWidget(self._diagnostics_pane())
        self.detail_splitter.setStretchFactor(0, 1); self.detail_splitter.setStretchFactor(1, 0)
        self.detail_splitter.setSizes((900, 44)); self.detail_splitter.splitterMoved.connect(self._diagnostics_splitter_moved)
        self.main_splitter.addWidget(self.detail_splitter); self.main_splitter.setSizes((280, 900)); self.main_splitter.setStretchFactor(1, 1)
        root.addWidget(self.main_splitter, 1)

    def _diagnostics_pane(self):
        pane = QFrame(); pane.setObjectName("fusionDiagnostics"); pane.setMinimumHeight(44)
        pane_layout = QVBoxLayout(pane); pane_layout.setContentsMargins(8, 5, 8, 7); pane_layout.setSpacing(5)
        header = QHBoxLayout(); header.setSpacing(7)
        self.status_dot = QLabel("●"); self.status_dot.setStyleSheet("color:#00E676")
        self.status_text = QLabel("Ready to import scan passes"); self.status_text.setStyleSheet("color:#D0D0D0")
        self.diagnostics_toggle = QToolButton(); self.diagnostics_toggle.setText("Console  ▸"); self.diagnostics_toggle.setCheckable(True); self.diagnostics_toggle.toggled.connect(self._toggle_diagnostics)
        self.clear_console_button = QToolButton(); self.clear_console_button.setText("Clear"); self.clear_console_button.clicked.connect(self._clear_diagnostics)
        header.addWidget(self.status_dot); header.addWidget(self.status_text); header.addStretch(); header.addWidget(self.diagnostics_toggle); header.addWidget(self.clear_console_button)
        pane_layout.addLayout(header)
        self.console = QTextEdit(); self.console.setObjectName("deepMeshFusionConsole"); self.console.setReadOnly(True); self.console.setVisible(False); self.console.setMinimumHeight(0)
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.console.setStyleSheet("font-family:monospace;background:#101010;color:#D8D8D8;border:1px solid #333333")
        pane_layout.addWidget(self.console, 1)
        self.diagnostics_pane = pane
        return pane

    def _sidebar(self):
        sidebar = QFrame(); sidebar.setObjectName("fusionSidebar"); sidebar.setMinimumWidth(250); sidebar.setMaximumWidth(410)
        sidebar_layout = QVBoxLayout(sidebar); sidebar_layout.setContentsMargins(0, 0, 0, 7); sidebar_layout.setSpacing(7)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setMinimumWidth(250); scroll.setMaximumWidth(410)
        panel = QWidget(); layout = QVBoxLayout(panel); layout.setContentsMargins(6, 4, 8, 4); layout.setSpacing(10)
        self.stage_panels = []

        self.preparation_panel = QFrame(); preparation = QVBoxLayout(self.preparation_panel); preparation.setContentsMargins(0, 0, 0, 0); preparation.setSpacing(8)
        scans_header = QHBoxLayout(); scans_header.setContentsMargins(0, 0, 0, 0)
        scans_header.addWidget(self._section_label("SCANS")); scans_header.addStretch()
        self.scan_count_label = QLabel("0 scans"); self.scan_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter); self.scan_count_label.setStyleSheet("color:#B0B0B0")
        scans_header.addWidget(self.scan_count_label); preparation.addLayout(scans_header)
        add_button = QPushButton("+ Add PLY Passes"); add_button.clicked.connect(self._add_scans); preparation.addWidget(add_button)
        self.scan_container = QWidget(); self.scan_layout = QVBoxLayout(self.scan_container); self.scan_layout.setContentsMargins(0, 0, 0, 0); self.scan_layout.setSpacing(6); self.scan_layout.addStretch()
        preparation.addWidget(self.scan_container); preparation.addStretch(); layout.addWidget(self.preparation_panel); self.stage_panels.append(self.preparation_panel)

        self.alignment_panel = QFrame(); alignment_layout = QVBoxLayout(self.alignment_panel); alignment_layout.setContentsMargins(0, 0, 0, 0)
        alignment_layout.addWidget(self._section_label("POINT CLOUD ALIGNMENT"))
        self.auto_reference = QCheckBox("Auto-select best reference"); self.auto_reference.setChecked(True); alignment_layout.addWidget(self.auto_reference)
        self.reference_combo = QComboBox(); self.reference_combo.setEnabled(False); alignment_layout.addWidget(self.reference_combo)
        self.auto_reference.toggled.connect(lambda checked: self.reference_combo.setEnabled(not checked))
        align_form = QFormLayout(); self.voxel = QDoubleSpinBox(); self.voxel.setRange(.005, 1); self.voxel.setDecimals(3); self.voxel.setSingleStep(.005); self.voxel.setValue(.03)
        align_form.addRow("Voxel size", self.voxel); alignment_layout.addLayout(align_form)
        self.alignment_summary = QLabel("Run alignment to compare registration fitness, RMSE, and overlap."); self.alignment_summary.setWordWrap(True); self.alignment_summary.setStyleSheet("color:#B0B0B0"); alignment_layout.addWidget(self.alignment_summary)
        self.retry_alignment = QPushButton("Retry Alignment"); self.retry_alignment.clicked.connect(self._run_alignment); self.retry_alignment.setVisible(False); alignment_layout.addWidget(self.retry_alignment)
        self.export_registered = QToolButton(); self.export_registered.setText("Export Unified Point Cloud  ▾"); self.export_registered.setPopupMode(QToolButton.InstantPopup)
        export_menu = QMenu(self.export_registered)
        lossless = QAction("Lossless registered PLY", self.export_registered); lossless.triggered.connect(lambda: self._export_registered_cloud(False))
        downsampled = QAction("Voxel-downsampled PLY", self.export_registered); downsampled.triggered.connect(lambda: self._export_registered_cloud(True))
        export_menu.addAction(lossless); export_menu.addAction(downsampled); self.export_registered.setMenu(export_menu); self.export_registered.setVisible(False); alignment_layout.addWidget(self.export_registered)
        alignment_layout.addStretch(); layout.addWidget(self.alignment_panel); self.stage_panels.append(self.alignment_panel)

        self.fusion_panel = QFrame(); fusion_layout = QVBoxLayout(self.fusion_panel); fusion_layout.setContentsMargins(0, 0, 0, 0)
        fusion_layout.addWidget(self._section_label("EVIDENCE-SUPPORTED POINT FUSION"))
        self.fusion_preset = QComboBox(); self.fusion_preset.addItems(("Balanced", "Conservative", "Aggressive")); self.fusion_preset.currentTextChanged.connect(self._apply_fusion_preset); fusion_layout.addWidget(self.fusion_preset)
        self.fusion_advanced_toggle = QToolButton(); self.fusion_advanced_toggle.setText("Advanced  ▸"); self.fusion_advanced_toggle.setCheckable(True); fusion_layout.addWidget(self.fusion_advanced_toggle)
        self.fusion_advanced = QWidget(); fusion_form = QFormLayout(self.fusion_advanced)
        self.fusion_cell = QDoubleSpinBox(); self.fusion_cell.setRange(.005, 1); self.fusion_cell.setDecimals(3); self.fusion_cell.setValue(.03)
        self.min_observations = QSpinBox(); self.min_observations.setRange(2, 12); self.min_observations.setValue(2)
        self.correspondence_distance = QDoubleSpinBox(); self.correspondence_distance.setRange(.1, 3); self.correspondence_distance.setDecimals(2); self.correspondence_distance.setValue(.85)
        self.artifact_threshold = QDoubleSpinBox(); self.artifact_threshold.setRange(.1, .95); self.artifact_threshold.setDecimals(2); self.artifact_threshold.setValue(.65)
        fusion_form.addRow("Fusion cell", self.fusion_cell); fusion_form.addRow("Min observations", self.min_observations); fusion_form.addRow("Correspondence", self.correspondence_distance); fusion_form.addRow("Artifact threshold", self.artifact_threshold)
        self.fusion_advanced.setVisible(False); self.fusion_advanced_toggle.toggled.connect(self.fusion_advanced.setVisible); fusion_layout.addWidget(self.fusion_advanced)
        self.rerun_fusion = QPushButton("Rerun Point Fusion"); self.rerun_fusion.clicked.connect(self._run_point_fusion); self.rerun_fusion.setVisible(False); fusion_layout.addWidget(self.rerun_fusion)
        self.analysis_panel = QFrame(); analysis = QFormLayout(self.analysis_panel); analysis.setContentsMargins(8, 8, 8, 8); analysis.addRow(self._section_label("FUSION ANALYSIS")); self.metric_labels = {}
        for name in ("Coverage", "Alignment", "Agreement", "Confidence"):
            value = QLabel("—"); value.setStyleSheet("font-weight:600"); analysis.addRow(name, value); self.metric_labels[name] = value
        self.analysis_panel.setVisible(False); fusion_layout.addWidget(self.analysis_panel); fusion_layout.addStretch(); layout.addWidget(self.fusion_panel); self.stage_panels.append(self.fusion_panel)

        self.validation_panel = QFrame(); validation_layout = QVBoxLayout(self.validation_panel); validation_layout.setContentsMargins(0, 0, 0, 0)
        validation_layout.addWidget(self._section_label("ARCHITECTURE-AWARE RECONSTRUCTION"))
        self.poisson_depth_label = QLabel("Poisson octree depth: 8"); validation_layout.addWidget(self.poisson_depth_label)
        self.poisson_depth = QSlider(Qt.Horizontal); self.poisson_depth.setRange(5, 12); self.poisson_depth.setValue(8); self.poisson_depth.valueChanged.connect(lambda value: self.poisson_depth_label.setText(f"Poisson octree depth: {value}")); validation_layout.addWidget(self.poisson_depth)
        axis_form = QFormLayout(); self.up_axis = QComboBox(); self.up_axis.addItems(("y", "z", "x")); axis_form.addRow("Architecture up axis", self.up_axis); validation_layout.addLayout(axis_form)
        help_label = QLabel("Planar architecture is reconstructed separately; this depth controls Screened Poisson reconstruction for complex residual geometry."); help_label.setWordWrap(True); help_label.setStyleSheet("color:#B0B0B0"); validation_layout.addWidget(help_label)
        self.rerun_surface = QPushButton("Rerun Surface Reconstruction"); self.rerun_surface.clicked.connect(self._run_surface); self.rerun_surface.setVisible(False); validation_layout.addWidget(self.rerun_surface)
        self.surface_summary = QLabel("No reconstructed surface yet."); self.surface_summary.setWordWrap(True); validation_layout.addWidget(self.surface_summary); validation_layout.addStretch(); layout.addWidget(self.validation_panel); self.stage_panels.append(self.validation_panel)

        self.cleanup_panel = QFrame(); cleanup_layout = QVBoxLayout(self.cleanup_panel); cleanup_layout.setContentsMargins(0, 0, 0, 0); cleanup_layout.setSpacing(8)
        cleanup_layout.addWidget(self._section_label("MESH CLEANUP")); cleanup_form = QFormLayout()
        self.face_reduction = QSpinBox(); self.face_reduction.setRange(10, 90); self.face_reduction.setSingleStep(5); self.face_reduction.setSuffix("%"); self.face_reduction.setValue(50); cleanup_form.addRow("Face reduction", self.face_reduction); cleanup_layout.addLayout(cleanup_form)
        self.cleanup_reduce_button = QPushButton("Cleanup & Reduce Mesh"); self.cleanup_reduce_button.clicked.connect(lambda: self._run_cleanup_operation("cleanup_reduce")); cleanup_layout.addWidget(self.cleanup_reduce_button)
        self.repair_nonmanifold_button = QPushButton("Repair Non-Manifold Edges"); self.repair_nonmanifold_button.clicked.connect(lambda: self._run_cleanup_operation("repair_nonmanifold")); cleanup_layout.addWidget(self.repair_nonmanifold_button)
        holes_form = QFormLayout(); self.max_hole_size = QSpinBox(); self.max_hole_size.setRange(5, 500); self.max_hole_size.setValue(30); holes_form.addRow("Max hole size", self.max_hole_size); cleanup_layout.addLayout(holes_form)
        self.close_holes_button = QPushButton("Close Holes"); self.close_holes_button.clicked.connect(lambda: self._run_cleanup_operation("close_holes")); cleanup_layout.addWidget(self.close_holes_button)
        merge_form = QFormLayout(); self.merge_threshold = QDoubleSpinBox(); self.merge_threshold.setRange(.001, 50); self.merge_threshold.setDecimals(3); self.merge_threshold.setSuffix("%"); self.merge_threshold.setValue(1); merge_form.addRow("Merge distance", self.merge_threshold); cleanup_layout.addLayout(merge_form)
        self.merge_vertices_button = QPushButton("Merge Close Vertices"); self.merge_vertices_button.clicked.connect(lambda: self._run_cleanup_operation("merge")); cleanup_layout.addWidget(self.merge_vertices_button)
        smooth_form = QFormLayout(); self.smooth_factor = QDoubleSpinBox(); self.smooth_factor.setRange(.01, 1); self.smooth_factor.setValue(.5); self.smooth_factor.setSingleStep(.05); self.smooth_iterations = QSpinBox(); self.smooth_iterations.setRange(1, 100); self.smooth_iterations.setValue(10); smooth_form.addRow("Smoothing factor", self.smooth_factor); smooth_form.addRow("Iterations", self.smooth_iterations); cleanup_layout.addLayout(smooth_form)
        self.smooth_button = QPushButton("Smooth Mesh"); self.smooth_button.clicked.connect(lambda: self._run_cleanup_operation("smooth")); cleanup_layout.addWidget(self.smooth_button)
        self.reset_cleanup_button = QPushButton("Reset Cleanup"); self.reset_cleanup_button.clicked.connect(self._reset_cleanup); cleanup_layout.addWidget(self.reset_cleanup_button)
        self.cleanup_allow_review = QCheckBox("Allow provisional review if validation fails"); self.cleanup_allow_review.setChecked(False); cleanup_layout.addWidget(self.cleanup_allow_review)
        self.cleanup_status = QLabel("Operations are cumulative. Reset restores the reconstructed surface."); self.cleanup_status.setWordWrap(True); self.cleanup_status.setStyleSheet("color:#B0B0B0"); cleanup_layout.addWidget(self.cleanup_status); cleanup_layout.addStretch(); layout.addWidget(self.cleanup_panel); self.stage_panels.append(self.cleanup_panel)

        self.texture_panel = QFrame(); texture_layout = QVBoxLayout(self.texture_panel); texture_layout.setContentsMargins(0, 0, 0, 0)
        self.photo_panel = QFrame(); photo = QVBoxLayout(self.photo_panel); photo.setContentsMargins(8, 8, 8, 8); photo.addWidget(self._section_label("PHOTOGRAMMETRY"))
        self.photo_status = QLabel("No reconstructed model found.\nRun a photogrammetry reconstruction first."); self.photo_status.setWordWrap(True); self.photo_status.setStyleSheet("color:#B0B0B0")
        self.photo_button = QPushButton("Run Reconstruction"); self.photo_button.clicked.connect(self._photo_action); photo.addWidget(self.photo_status); photo.addWidget(self.photo_button); texture_layout.addWidget(self.photo_panel)
        self.advanced_toggle = QToolButton(); self.advanced_toggle.setText("Advanced Settings  ▸"); self.advanced_toggle.setCheckable(True); self.advanced_toggle.toggled.connect(self._toggle_advanced); texture_layout.addWidget(self.advanced_toggle)
        self.advanced_panel = QWidget(); advanced = QFormLayout(self.advanced_panel)
        persistent_workspace = (self.reconstruction_root.parent / "deep_mesh_fusion_workspace") if self.reconstruction_root.name == "reconstruction_out" else (self.reconstruction_root / "deep_mesh_fusion_workspace")
        self.workspace_edit = QLineEdit(str(persistent_workspace.resolve()))
        self.atlas = QSpinBox(); self.atlas.setRange(64, 8192); self.atlas.setSingleStep(512); self.atlas.setValue(2048)
        self.allow_review = QCheckBox("Allow provisional review"); self.allow_review.setChecked(False)
        advanced.addRow("Texture atlas", self.atlas); advanced.addRow(self.allow_review)
        self.advanced_panel.setVisible(False); texture_layout.addWidget(self.advanced_panel); texture_layout.addStretch(); layout.addWidget(self.texture_panel); self.stage_panels.append(self.texture_panel)

        self.final_panel = QFrame(); final_layout = QVBoxLayout(self.final_panel); final_layout.setContentsMargins(0, 0, 0, 0); final_layout.addWidget(self._section_label("TOUR READINESS")); self.final_summary = QLabel("Complete texture processing to generate the final quality report."); self.final_summary.setWordWrap(True); final_layout.addWidget(self.final_summary); final_layout.addStretch(); layout.addWidget(self.final_panel); self.stage_panels.append(self.final_panel)
        layout.addStretch(); scroll.setWidget(panel); sidebar_layout.addWidget(scroll, 1)

        action_row = QHBoxLayout(); action_row.setContentsMargins(10, 0, 10, 0); action_row.addStretch()
        self.action_button = QPushButton("Continue"); self.action_button.clicked.connect(self._run_primary_action); self.action_button.setMinimumWidth(170)
        self._set_action_style("ready")
        action_row.addWidget(self.action_button); action_row.addStretch(); sidebar_layout.addLayout(action_row)
        secondary_row = QHBoxLayout(); secondary_row.setContentsMargins(10, 0, 10, 0); secondary_row.addStretch()
        self.report_button = QPushButton("Open Quality Report"); self.report_button.setVisible(False); self.report_button.clicked.connect(self._open_report)
        self.cancel_button = QPushButton("Cancel"); self.cancel_button.setVisible(False); self.cancel_button.clicked.connect(self._cancel)
        secondary_row.addWidget(self.report_button); secondary_row.addWidget(self.cancel_button); secondary_row.addStretch(); sidebar_layout.addLayout(secondary_row)
        self.sidebar = sidebar
        return sidebar

    @staticmethod
    def _section_label(text):
        label = QLabel(text); label.setStyleSheet("font-size:11px;font-weight:700;color:#00E676"); return label

    def _add_scans(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add independent LiDAR scan passes", "", "PLY scan passes (*.ply)"); self._add_scan_paths(paths)

    def _add_scan_paths(self, paths):
        existing = {item.source_path for item in self.scan_previews.values()}; failures = []
        for path_value in paths:
            path = str(Path(path_value).resolve())
            if path in existing: continue
            result = load_point_cloud(path, self._log)
            if not result.success or result.cloud is None:
                failures.append(f"{Path(path).name}: {'; '.join(result.warnings) or 'could not load'}"); continue
            points = np.asarray(result.cloud.points, dtype=np.float32)
            if len(points) > FusionViewport.MAX_POINTS_PER_LAYER:
                points = points[np.linspace(0, len(points) - 1, FusionViewport.MAX_POINTS_PER_LAYER, dtype=int)]
            scan_id = f"scan-{len(self.scan_previews) + 1}"
            while scan_id in self.scan_previews: scan_id += "-copy"
            preview = ScanPreview(scan_id, f"Scan {len(self.scan_previews) + 1:02d}", path, points, QColor(self.COLORS[len(self.scan_previews) % len(self.COLORS)]))
            self.scan_previews[scan_id] = preview; self.viewport.set_layer(scan_id, points, preview.color); self._add_scan_row(preview); existing.add(path)
        if self.scan_previews: self.viewport.fit_to_layers()
        self._invalidate_processing(); self._update_state()
        if failures: QMessageBox.warning(self, "Some scans were not imported", "\n".join(failures))

    def _add_scan_row(self, preview):
        row = ScanPassRow(preview); row.visibility_changed.connect(self._set_scan_visible); row.selected.connect(self._select_scan)
        row.rename_requested.connect(self._rename_scan); row.remove_requested.connect(self._remove_scan)
        self.scan_layout.insertWidget(self.scan_layout.count() - 1, row); self.scan_rows[preview.scan_id] = row

    def _set_scan_visible(self, scan_id, visible): self.scan_previews[scan_id].visible = visible; self.viewport.set_layer_visible(scan_id, visible)

    def _select_scan(self, scan_id):
        self.selected_scan_id = scan_id; self.viewport.select_layer(scan_id)
        for key, row in self.scan_rows.items(): row.set_selected(key == scan_id)

    def _rename_scan(self, scan_id):
        preview = self.scan_previews[scan_id]; name, accepted = QInputDialog.getText(self, "Rename scan pass", "Scan name", text=preview.name)
        if accepted and name.strip(): preview.name = name.strip(); self.scan_rows[scan_id].name_label.setText(preview.name); self._invalidate_processing()

    def _remove_scan(self, scan_id):
        row = self.scan_rows.pop(scan_id); self.scan_layout.removeWidget(row); row.deleteLater(); self.scan_previews.pop(scan_id, None); self.viewport.remove_layer(scan_id)
        if self.selected_scan_id == scan_id: self.selected_scan_id = None
        self._invalidate_processing(); self._update_state()

    def _invalidate_processing(self):
        if self.worker and self.worker.isRunning(): return
        self.workspace = None; self.last_result = {}; self.completed_stages.clear(); self.unlocked_stage = 0; self.current_stage = 0
        self.photogrammetry_loaded = False; self._loaded_photogrammetry_paths = None; self.analysis_requires_review = False
        self.analysis_panel.setVisible(False); self.report_button.setVisible(False)
        self.viewport.remove_layer("fused-geometry"); self.viewport.remove_layer("photogrammetry")
        self.viewport.clear_mesh()
        for scan_id in self.scan_previews: self.viewport.set_layer_visible(scan_id, self.scan_previews[scan_id].visible)
        self.viewport.fit_to_layers()

    def _toggle_advanced(self, visible): self.advanced_panel.setVisible(visible); self.advanced_toggle.setText(f"Advanced Settings  {'▾' if visible else '▸'}")

    def _select_stage(self, index):
        if self.processing or index > self.unlocked_stage:
            return
        self.current_stage = index
        self._refresh_viewport_for_stage(index)
        self._update_state()

    def _refresh_viewport_for_stage(self, index):
        if index == 0:
            self.viewport.clear_mesh(); self.viewport.remove_layer("fused-geometry")
            for scan_id, preview in self.scan_previews.items(): self.viewport.set_layer(scan_id, preview.points, preview.color, preview.visible, scan_id == self.selected_scan_id)
            self.viewport.fit_to_layers(); return
        if index == 1:
            self.viewport.clear_mesh(); self.viewport.remove_layer("fused-geometry")
            if self.workspace and "alignment" in self.completed_stages: self._show_registered_scans()
            return
        if index == 2 and self.workspace and "point_fusion" in self.completed_stages:
            result = self.last_result.get("point_fusion"); path = getattr(result, "fused_cloud_path", self.workspace.root / "derived" / "fused_point_cloud.ply")
            if Path(path).is_file():
                loaded = load_point_cloud(str(path), self._log)
                if loaded.success and loaded.cloud is not None:
                    for scan_id in self.scan_previews: self.viewport.set_layer_visible(scan_id, False)
                    self.viewport.clear_mesh(); self.viewport.set_layer("fused-geometry", np.asarray(loaded.cloud.points), QColor("#00E676"), True, True); self.viewport.fit_to_layers()
            return
        if index == 3 and "surface" not in self.completed_stages:
            self._refresh_viewport_for_stage(2); return
        if index >= 3 and self.workspace and "surface" in self.completed_stages:
            cleanup = self.workspace.root / "derived" / "cleanup" / "cleanup_working.ply"
            validated = self.workspace.root / "derived" / "validated_lidar_surface.ply"
            candidate = cleanup if index >= 4 and cleanup.is_file() else validated
            if candidate.is_file(): self._show_mesh_path(candidate)

    def _restore_workspace_state(self):
        root = Path(self.workspace_edit.text()).resolve(); manifest_path = root / "workspace.json"
        if not manifest_path.is_file(): return
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8")); workspace = DeepMeshFusionWorkspace.load(str(root), self._log)
            valid_sources = [item.source_path for item in workspace.passes if Path(item.source_path).is_file()]
            if len(valid_sources) != len(workspace.passes): return
            self._add_scan_paths(valid_sources); self.workspace = workspace
            self.reference_combo.clear()
            for scan_pass in workspace.passes: self.reference_combo.addItem(scan_pass.name, scan_pass.pass_id)
            names = {str(Path(item.source_path).resolve()): item.name for item in workspace.passes}
            for preview in self.scan_previews.values():
                preview.name = names.get(preview.source_path, preview.name); self.scan_rows[preview.scan_id].name_label.setText(preview.name)
            state = payload.get("workflow_state") or {}; requested = set(state.get("completed", []))
            params = state.get("parameters") or {}
            self.voxel.setValue(float(params.get("voxel_size", workspace.config.voxel_size)))
            self.fusion_preset.setCurrentText(str(params.get("fusion_preset", "Balanced")))
            self.fusion_cell.setValue(float(params.get("fusion_cell", workspace.config.effective_fusion_cell_size())))
            self.min_observations.setValue(int(params.get("min_observations", workspace.config.min_consensus_observations)))
            self.correspondence_distance.setValue(float(params.get("correspondence_distance", workspace.config.correspondence_distance_multiplier)))
            self.artifact_threshold.setValue(float(params.get("artifact_threshold", workspace.config.artifact_suppression_threshold)))
            self.poisson_depth.setValue(int(params.get("poisson_depth", workspace.config.complex_poisson_depth)))
            self.up_axis.setCurrentText(str(params.get("up_axis", workspace.config.architecture_up_axis)))
            self.face_reduction.setValue(int(params.get("face_reduction", 50))); self.max_hole_size.setValue(int(params.get("max_hole_size", 30)))
            self.merge_threshold.setValue(float(params.get("merge_threshold", 1.0))); self.smooth_factor.setValue(float(params.get("smooth_factor", .5))); self.smooth_iterations.setValue(int(params.get("smooth_iterations", 10))); self.atlas.setValue(int(params.get("texture_atlas", workspace.config.texture_atlas_size)))
            self.completed_stages.add("diagnostics")
            accepted = [item for item in workspace.passes if item.registration and item.registration.accepted]
            if len(accepted) >= 2 and "alignment" in requested: self.completed_stages.add("alignment")
            fused = payload.get("fused_artifact") or {}; fused_path = fused.get("fused_cloud_path") or fused.get("path")
            analysis_state = payload.get("cross_pass_analysis") or {}; evidence_path = analysis_state.get("evidence_map_path")
            if "point_fusion" in requested and fused_path and Path(fused_path).is_file() and evidence_path and Path(evidence_path).is_file(): self.completed_stages.add("point_fusion")
            validated = (payload.get("geometry_validation") or {}).get("validated_mesh_path")
            if "point_fusion" in self.completed_stages and "surface" in requested and validated and Path(validated).is_file(): self.completed_stages.add("surface")
            cleanup_path = root / "derived" / "cleanup" / "cleanup_working.ply"
            if "surface" in self.completed_stages and "cleanup" in requested and cleanup_path.is_file():
                self.completed_stages.add("cleanup"); self.cleanup_working_path = cleanup_path; self.cleanup_original_path = cleanup_path.with_name("reconstructed_original.ply")
            if "cleanup" in self.completed_stages and "photo_registration" in requested and payload.get("photogrammetry_preparation"): self.completed_stages.add("photo_registration")
            final_chain_ready = payload.get("texture_baking") and payload.get("final_asset") and payload.get("tour_readiness")
            if "photo_registration" in self.completed_stages and "texture" in requested and final_chain_ready: self.completed_stages.add("texture")
            if "texture" in self.completed_stages and "quality" in requested and final_chain_ready: self.completed_stages.add("quality")
            if "point_fusion" in self.completed_stages and analysis_state:
                restored_fusion = SimpleNamespace(fused_cloud_path=fused_path, mean_confidence=float(analysis_state.get("mean_confidence", 0)),
                    evidence_map_path=analysis_state.get("evidence_map_path"), region_count=int(analysis_state.get("region_count", 0)),
                    conflict_region_count=int(analysis_state.get("conflict_region_count", 0)))
                self.last_result["point_fusion"] = restored_fusion; self._show_analysis(restored_fusion)
            if "quality" in self.completed_stages:
                self.last_result["quality"] = SimpleNamespace(html_report_path=(payload.get("tour_readiness") or {}).get("html_report_path"))
            self.cleanup_history = list(state.get("cleanup_history", []))
            stage_for_key = {"diagnostics": 1, "alignment": 2, "point_fusion": 3, "surface": 4, "cleanup": 5, "texture": 6, "quality": 6}
            self.unlocked_stage = max([0] + [stage_for_key[key] for key in self.completed_stages])
            self.current_stage = min(int(state.get("current_stage", self.unlocked_stage)), self.unlocked_stage)
            self._show_registered_scans()
            self.retry_alignment.setVisible("alignment" in self.completed_stages); self.export_registered.setVisible("alignment" in self.completed_stages)
            self.rerun_fusion.setVisible("point_fusion" in self.completed_stages); self.rerun_surface.setVisible("surface" in self.completed_stages)
            if cleanup_path.is_file(): self._show_mesh_path(cleanup_path)
            self._log(f"[DEEP_FUSION] Restored workflow state from {manifest_path}")
        except Exception as error:
            self._log(f"[WARNING] Deep Mesh Fusion workspace recovery skipped: {error}")

    def _persist_workflow_state(self):
        if self.workspace is None: return
        try:
            self.workspace.save_workflow_state({"current_stage": self.current_stage, "unlocked_stage": self.unlocked_stage,
                "completed": sorted(self.completed_stages), "cleanup_history": list(self.cleanup_history),
                "parameters": {"voxel_size": self.voxel.value(), "fusion_preset": self.fusion_preset.currentText(),
                    "fusion_cell": self.fusion_cell.value(), "min_observations": self.min_observations.value(),
                    "correspondence_distance": self.correspondence_distance.value(), "artifact_threshold": self.artifact_threshold.value(),
                    "poisson_depth": self.poisson_depth.value(), "up_axis": self.up_axis.currentText(), "face_reduction": self.face_reduction.value(),
                    "max_hole_size": self.max_hole_size.value(), "merge_threshold": self.merge_threshold.value(),
                    "smooth_factor": self.smooth_factor.value(), "smooth_iterations": self.smooth_iterations.value(),
                    "texture_atlas": self.atlas.value()}})
        except Exception as error:
            self._log(f"[WARNING] Could not persist Deep Mesh Fusion UI state: {error}")

    def _apply_fusion_preset(self, name):
        presets = {
            "Conservative": (.025, 2, .70, .55),
            "Balanced": (.030, 2, .85, .65),
            "Aggressive": (.045, 2, 1.10, .78),
        }
        cell, observations, distance, artifact = presets.get(name, presets["Balanced"])
        self.fusion_cell.setValue(cell); self.min_observations.setValue(observations)
        self.correspondence_distance.setValue(distance); self.artifact_threshold.setValue(artifact)

    def _toggle_diagnostics(self, visible):
        self.console.setVisible(visible)
        self.diagnostics_toggle.setText(f"Console  {'▾' if visible else '▸'}")
        total = sum(self.detail_splitter.sizes()) or max(self.detail_splitter.height(), 500)
        bottom = max(180, int(total * .28)) if visible else 44
        self.detail_splitter.setSizes((max(total - bottom, self.viewport.minimumHeight()), bottom))

    def _diagnostics_splitter_moved(self, _position, _index):
        sizes = self.detail_splitter.sizes()
        expanded = bool(len(sizes) > 1 and sizes[1] > 70)
        if self.console.isVisible() != expanded:
            self.console.setVisible(expanded)
        if self.diagnostics_toggle.isChecked() != expanded:
            self.diagnostics_toggle.blockSignals(True); self.diagnostics_toggle.setChecked(expanded); self.diagnostics_toggle.blockSignals(False)
        self.diagnostics_toggle.setText(f"Console  {'▾' if expanded else '▸'}")

    def _clear_diagnostics(self):
        self.console.clear()

    def refresh_reconstruction_state(self):
        model = next((path for path in (self.reconstruction_root / "colmap" / "sparse" / "aligned", self.reconstruction_root / "colmap" / "sparse" / "0", self.reconstruction_root / "colmap" / "sparse") if self._is_colmap_model(path)), None)
        images = next((path for path in (
            self.reconstruction_root / "input_images",
            self.reconstruction_root / "colmap" / "images",
            self.reconstruction_root / "mvs" / "images",
        ) if self._is_image_directory(path)), None)
        dense = next((path for path in (self.reconstruction_root / "mvs" / "scene_dense.ply", self.reconstruction_root / "mvs" / "scene.ply") if path.is_file()), None)
        detected_paths = (str(model), str(images), str(dense)) if model and images and dense else None
        if self.photogrammetry_loaded and detected_paths != self._loaded_photogrammetry_paths:
            self.photogrammetry_loaded = False; self._loaded_photogrammetry_paths = None
            self.viewport.remove_layer("photogrammetry"); self.viewport.fit_to_layers()
        self.photogrammetry_paths = detected_paths
        if self.photogrammetry_paths and self.photogrammetry_loaded:
            self.photo_status.setText("✓ Reconstructed model loaded"); self.photo_status.setStyleSheet("color:#00E676;font-weight:600"); self.photo_button.setText("Reload Reconstructed Model")
        elif self.photogrammetry_paths:
            self.photo_status.setText("✓ Reconstructed model available"); self.photo_status.setStyleSheet("color:#00E676;font-weight:600"); self.photo_button.setText("Load Reconstructed Model")
        else:
            self.photo_status.setText("No reconstructed model found.\nRun a photogrammetry reconstruction first."); self.photo_status.setStyleSheet("color:#B0B0B0"); self.photo_button.setText("Run Reconstruction")
        self._update_state()

    def _photo_action(self):
        if not self.photogrammetry_paths: self.request_reconstruction.emit(); return
        try:
            model, images, dense = self.photogrammetry_paths; dataset = ColmapTextModelLoader().load(model, images, dense)
            self.viewport.set_layer("photogrammetry", dataset.points, QColor("#f9fafb"), True); self.viewport.fit_to_layers(); self.photogrammetry_loaded = True
            self._loaded_photogrammetry_paths = self.photogrammetry_paths
            self.photo_status.setText(f"✓ Reconstructed model loaded\n{len(dataset.points):,} appearance points"); self.photo_button.setText("Reload Reconstructed Model")
            self._update_state()
        except Exception as error: QMessageBox.critical(self, "Could not load reconstructed model", str(error))

    @staticmethod
    def _is_colmap_model(path):
        return path.is_dir() and (all((path / f"{name}.txt").is_file() for name in ("cameras", "images", "points3D")) or all((path / f"{name}.bin").is_file() for name in ("cameras", "images", "points3D")))

    @staticmethod
    def _is_image_directory(path):
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        return path.is_dir() and any(item.is_file() and item.suffix.lower() in extensions for item in path.rglob("*"))

    def _run_primary_action(self):
        if self.processing or (self.worker and self.worker.isRunning()): return
        stage = self.current_stage
        if stage == 0:
            if "diagnostics" in self.completed_stages: self._advance_to(1)
            else: self._run_diagnostics()
        elif stage == 1:
            if "alignment" in self.completed_stages: self._advance_to(2)
            else: self._run_alignment()
        elif stage == 2:
            if "point_fusion" in self.completed_stages: self._advance_to(3)
            else: self._run_point_fusion()
        elif stage == 3:
            if "surface" in self.completed_stages: self._advance_to(4)
            else: self._run_surface()
        elif stage == 4: self._proceed_cleanup()
        elif stage == 5:
            if "photo_registration" not in self.completed_stages: self._run_photo_registration()
            elif "texture" not in self.completed_stages: self._run_texture()
            else: self._advance_to(6)
        else: self._open_report()

    def _run_diagnostics(self):
        problems = self._input_problems()
        if problems: QMessageBox.warning(self, "Scans are not ready", "\n".join(f"• {item}" for item in problems)); return
        config = DeepMeshFusionConfig(voxel_size=self.voxel.value(), architecture_up_axis=self.up_axis.currentText(), texture_atlas_size=self.atlas.value()); previews = list(self.scan_previews.values())
        self.worker = DeepMeshFusionTaskWorker("diagnostics", workspace_path=str(self._new_workspace_path()), scan_paths=[item.source_path for item in previews], scan_names=[item.name for item in previews], config=config, parent=self); self._start_worker("Analyzing scan quality…")

    def _best_reference_pass_id(self):
        if not self.workspace or not self.workspace.passes: return None
        if not self.auto_reference.isChecked() and self.reference_combo.currentIndex() >= 0:
            return self.reference_combo.currentData()
        def score(scan_pass):
            diagnostics = scan_pass.diagnostics
            if diagnostics is None: return 0.0
            return diagnostics.point_count * max(0.0, 1.0 - diagnostics.outlier_fraction) * max(.05, diagnostics.largest_component_fraction)
        return max(self.workspace.passes, key=score).pass_id

    def _run_alignment(self):
        self._invalidate_after(1)
        self.workspace.config.voxel_size = self.voxel.value(); self.workspace.config.validate()
        self.worker = DeepMeshFusionTaskWorker("alignment", workspace=self.workspace, reference_pass_id=self._best_reference_pass_id(), parent=self)
        self._start_worker("Aligning point clouds…")

    def _run_point_fusion(self):
        self._invalidate_after(2)
        config = self.workspace.config; config.fusion_cell_size = self.fusion_cell.value(); config.min_consensus_observations = self.min_observations.value()
        config.correspondence_distance_multiplier = self.correspondence_distance.value(); config.artifact_suppression_threshold = self.artifact_threshold.value(); config.validate()
        self.worker = DeepMeshFusionTaskWorker("point_fusion", workspace=self.workspace, parent=self); self._start_worker("Fusing evidence-supported point clouds…")

    def _run_surface(self):
        self._invalidate_after(3); self.workspace.config.complex_poisson_depth = self.poisson_depth.value(); self.workspace.config.architecture_up_axis = self.up_axis.currentText(); self.workspace.config.validate()
        self.worker = DeepMeshFusionTaskWorker("surface", workspace=self.workspace, parent=self); self._start_worker("Reconstructing and validating the hybrid surface…")

    def _run_photo_registration(self):
        if not self.photogrammetry_loaded:
            if self.photogrammetry_paths: self._photo_action()
            else: self.request_reconstruction.emit()
            return
        self._sync_appearance_config(); self._invalidate_after(5)
        self.worker = DeepMeshFusionTaskWorker("photo_registration", workspace=self.workspace, photogrammetry=self.photogrammetry_paths, allow_review=self.allow_review.isChecked(), parent=self); self._start_worker("Registering photogrammetry to cleaned geometry…")

    def _run_texture(self):
        self._sync_appearance_config()
        self.worker = DeepMeshFusionTaskWorker("texture", workspace=self.workspace, allow_review=self.allow_review.isChecked(), parent=self); self._start_worker("Baking textures and evaluating final quality…")

    def _advance_to(self, stage):
        self.unlocked_stage = max(self.unlocked_stage, stage); self.current_stage = stage
        if stage == 5 and self.photogrammetry_paths and not self.photogrammetry_loaded:
            self._photo_action()
        self._refresh_viewport_for_stage(stage); self._update_state()

    def _invalidate_after(self, stage):
        key_stages = {"diagnostics": 0, "alignment": 1, "point_fusion": 2, "surface": 3, "cleanup": 4, "photo_registration": 5, "texture": 5, "quality": 6}
        for key, key_stage in list(key_stages.items()):
            if key_stage >= stage:
                self.completed_stages.discard(key); self.last_result.pop(key, None)
        self.unlocked_stage = min(self.unlocked_stage, stage)

    def _sync_appearance_config(self):
        """Apply live appearance controls to the already-created workspace."""
        if self.workspace is None:
            return
        self.workspace.update_appearance_settings(texture_atlas_size=self.atlas.value())
        self._log(f"[DEEP_FUSION] Texture settings synchronized: atlas={self.workspace.config.texture_atlas_size}px")

    def _start_worker(self, status):
        self.worker.log_message.connect(self._log); self.worker.completed.connect(self._on_task_complete); self.worker.failed.connect(self._on_task_failed)
        self.processing = True; self.cancel_button.setVisible(True); self.cancel_button.setEnabled(True); self.status_text.setText(status); self._update_state(); self.worker.start()

    def _on_task_complete(self, task, payload):
        self.processing = False; self.cancel_button.setVisible(False); self.action_button.setEnabled(True)
        if task == "diagnostics":
            self.workspace = payload["workspace"]; self.completed_stages.add("diagnostics"); self.last_result["diagnostics"] = payload["passes"]
            self.reference_combo.clear()
            for scan_pass in self.workspace.passes: self.reference_combo.addItem(scan_pass.name, scan_pass.pass_id)
            self.status_text.setText("Scan diagnostics complete"); self._advance_to(1); return
        if task == "alignment":
            self.completed_stages.add("alignment"); self.last_result["alignment"] = payload; self.unlocked_stage = max(self.unlocked_stage, 2)
            self._show_registered_scans(); self._show_alignment_results(); self.status_text.setText("Alignment complete — inspect results or continue")
        elif task == "point_fusion":
            self.completed_stages.add("point_fusion"); self.last_result["point_fusion"] = payload; self.unlocked_stage = max(self.unlocked_stage, 3)
            self._show_fused_points(payload); self._show_analysis(payload); self.status_text.setText("Point fusion complete — inspect the unified evidence cloud")
        elif task == "surface":
            self.completed_stages.add("surface"); self.last_result["surface"] = payload; self.unlocked_stage = max(self.unlocked_stage, 4)
            self._show_fused_geometry(payload); self.rerun_surface.setVisible(True); self.surface_summary.setText(f"{payload.reconstructed_face_count:,} faces · quality {payload.overall_geometry_quality:.0%} · {payload.validation_review_region_count} review region(s)")
            self._prepare_cleanup_workspace(payload.validated_mesh_path); self.status_text.setText("Surface reconstruction complete — inspect or continue")
        elif task == "cleanup_validation":
            if not payload.summary.ready_for_appearance_processing and not self.cleanup_allow_review.isChecked():
                self.status_text.setText("Texture blocked by cleanup validation"); QMessageBox.warning(self, "Texture is blocked", "Cleaned geometry still has critical or readiness-blocking defects. Enable Allow provisional review only for expert inspection.")
            else:
                self.allow_review.setChecked(self.cleanup_allow_review.isChecked()); self.completed_stages.add("cleanup"); self.last_result["cleanup"] = payload; self._advance_to(5); self.refresh_reconstruction_state(); return
        elif task == "photo_registration":
            self.completed_stages.add("photo_registration"); self.last_result["photo_registration"] = payload; self.status_text.setText("Photogrammetry registered — inspect correspondence quality before baking")
        elif task == "texture":
            self.completed_stages.update(("texture", "quality")); self.last_result.update(payload); self.unlocked_stage = 6; quality = payload["quality"].summary
            self.final_summary.setText(f"Geometry {quality.geometry:.0%}\nCompleteness {quality.completeness:.0%}\nTexture coverage {quality.texture_coverage:.0%}\nCritical defects {quality.critical_defect_count}")
            self.status_text.setText("Tour ready" if quality.tour_ready else f"{quality.blocking_issue_count} blocking issue(s) require review"); self.report_button.setVisible(True)
        self._update_state()

    def _on_task_failed(self, _task, message):
        self.processing = False; self.cancel_button.setVisible(False); self._update_state(); self.status_text.setText("Processing stopped — view diagnostics"); self._log(f"[ERROR] {message}"); QMessageBox.critical(self, "Deep Mesh Fusion", message)

    def _show_alignment_results(self):
        registrations = [item.registration for item in self.workspace.passes if item.registration and item.registration.method != "reference"]
        accepted = [item for item in registrations if item.accepted]
        fitness = float(np.mean([item.fitness for item in accepted])) if accepted else 0.0
        rejected = len(registrations) - len(accepted)
        self.analysis_requires_review = rejected > 0 or fitness < .60
        self.alignment_summary.setText(f"Accepted passes: {len(accepted) + 1}/{len(registrations) + 1}\nMean fitness: {fitness:.0%}\nExcluded passes: {rejected}")
        self.retry_alignment.setVisible(True); self.export_registered.setVisible(len(accepted) + 1 >= 2)

    def _show_fused_points(self, result):
        loaded = load_point_cloud(result.fused_cloud_path, self._log)
        if not loaded.success or loaded.cloud is None: raise ValueError("Fused point cloud could not be displayed")
        for scan_id in self.scan_previews: self.viewport.set_layer_visible(scan_id, False)
        self.viewport.clear_mesh(); self.viewport.remove_layer("fused-geometry")
        self.viewport.set_layer("fused-geometry", np.asarray(loaded.cloud.points), QColor("#00E676"), True, True); self.viewport.fit_to_layers()
        self.rerun_fusion.setVisible(True)

    def _export_registered_cloud(self, downsampled):
        if not self.workspace or "alignment" not in self.completed_stages: return
        suggested = "registered_scans_unified_voxel.ply" if downsampled else "registered_scans_unified.ply"
        path, _ = QFileDialog.getSaveFileName(self, "Export registered point cloud", suggested, "PLY point cloud (*.ply)")
        if not path: return
        try:
            exported = self.workspace.export_registered_cloud(path, voxel_downsampled=downsampled)
            self.status_text.setText(f"Exported {Path(exported).name}")
        except Exception as error:
            QMessageBox.critical(self, "Point-cloud export failed", str(error))

    def _prepare_cleanup_workspace(self, source_path):
        cleanup_dir = self.workspace.root / "derived" / "cleanup"; cleanup_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_original_path = cleanup_dir / "reconstructed_original.ply"; self.cleanup_working_path = cleanup_dir / "cleanup_working.ply"
        shutil.copy2(source_path, self.cleanup_original_path); shutil.copy2(source_path, self.cleanup_working_path)
        self.cleanup_status.setText("Ready. Operations are cumulative; Reset restores the reconstructed surface.")

    def _run_cleanup_operation(self, operation):
        if self.processing or not hasattr(self, "cleanup_working_path") or not self.cleanup_working_path.is_file(): return
        output = self.cleanup_working_path.with_name("cleanup_next.ply")
        if operation == "cleanup_reduce":
            worker_operation = "cleanup"; params = {"target_reduction_pct": self.face_reduction.value(), "remove_duplicates": True, "repair_nonmanifold": False, "close_holes": False, "full_cleanup": False}
        elif operation == "repair_nonmanifold":
            worker_operation = "cleanup"; params = {"target_reduction_pct": 0, "remove_duplicates": False, "repair_nonmanifold": True, "close_holes": False, "full_cleanup": False}
        elif operation == "close_holes":
            worker_operation = "cleanup"; params = {"target_reduction_pct": 0, "remove_duplicates": False, "repair_nonmanifold": False, "close_holes": True, "max_hole_size": self.max_hole_size.value(), "full_cleanup": False}
        elif operation == "merge":
            worker_operation = "merge_by_distance"; params = {"threshold_pct": self.merge_threshold.value(), "bbox_diagonal": 0.0}
        else:
            worker_operation = "smooth_taubin"; factor = self.smooth_factor.value(); params = {"lambda_factor": factor, "mu_factor": -(factor + .01), "iterations": self.smooth_iterations.value()}
        from pipeline_manager import MeshOperationWorker
        self.worker = None; self._active_cleanup_operation = operation; self.mesh_op_worker = MeshOperationWorker(worker_operation, str(self.cleanup_working_path), str(output), params, self)
        self.mesh_op_worker.log_message.connect(self._log); self.mesh_op_worker.finished.connect(self._on_cleanup_complete)
        self.processing = True; self.status_text.setText(f"Running {operation.replace('_', ' ')}…"); self._update_state(); self.mesh_op_worker.start()

    def _on_cleanup_complete(self, success, output_path, message):
        self.processing = False
        if success and output_path and Path(output_path).is_file():
            Path(output_path).replace(self.cleanup_working_path); self._show_mesh_path(self.cleanup_working_path)
            self.cleanup_history.append(self._active_cleanup_operation)
            self.completed_stages.discard("cleanup"); self._invalidate_after(4)
            self.cleanup_status.setText(f"Applied {self._active_cleanup_operation.replace('_', ' ')}. {message}")
        else:
            self.cleanup_status.setText(f"Cleanup failed: {message}"); QMessageBox.critical(self, "Mesh cleanup failed", message)
        self._update_state()

    def _reset_cleanup(self):
        if hasattr(self, "cleanup_original_path") and self.cleanup_original_path.is_file():
            shutil.copy2(self.cleanup_original_path, self.cleanup_working_path); self._show_mesh_path(self.cleanup_working_path)
            self.cleanup_history.clear()
            self.completed_stages.discard("cleanup"); self._invalidate_after(4); self.cleanup_status.setText("Cleanup reset to the reconstructed surface."); self._update_state()

    def _show_mesh_path(self, path):
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(path)); source_faces = len(mesh.triangles)
        if source_faces > FusionViewport.MAX_PREVIEW_FACES: mesh = mesh.simplify_quadric_decimation(FusionViewport.MAX_PREVIEW_FACES)
        self.viewport.remove_layer("fused-geometry"); self.viewport.set_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), source_faces)

    def _proceed_cleanup(self):
        if not hasattr(self, "cleanup_working_path") or not self.cleanup_working_path.is_file(): return
        self.worker = DeepMeshFusionTaskWorker("cleanup_validation", workspace=self.workspace, mesh_path=str(self.cleanup_working_path), parent=self)
        self._start_worker("Revalidating cleaned geometry…")

    def _show_analysis(self, result):
        payload = json.loads(Path(result.evidence_map_path).read_text(encoding="utf-8")); summary = payload.get("summary", {})
        registrations = [item.registration for item in self.workspace.passes if item.registration and item.registration.method != "reference"]
        alignment = float(np.mean([item.fitness if item.accepted else 0.0 for item in registrations])) if registrations else 1.0
        conflicts = result.conflict_region_count / max(result.region_count, 1)
        agreement = 1.0 - conflicts
        self.analysis_requires_review = alignment < .60 or agreement < .75 or any(not item.accepted for item in registrations)
        self.metric_labels["Coverage"].setText(self._quality_word(float(summary.get("overlap_ratio", 0))))
        self.metric_labels["Alignment"].setText(f"{self._quality_word(alignment)} · {alignment:.0%}")
        self.metric_labels["Agreement"].setText(self._quality_word(agreement)); self.metric_labels["Confidence"].setText(f"{result.mean_confidence:.0%}"); self.analysis_panel.setVisible(True)

    def _show_registered_scans(self):
        """Replace raw preview layers with each accepted source-to-reference transform."""
        preview_by_path = {str(Path(item.source_path).resolve()): item for item in self.scan_previews.values()}
        for scan_pass in self.workspace.passes:
            preview = preview_by_path.get(str(Path(scan_pass.source_path).resolve()))
            if preview is None or scan_pass.registration is None: continue
            registration = scan_pass.registration; row = self.scan_rows.get(preview.scan_id)
            if registration.accepted:
                transform = np.asarray(registration.transform, dtype=float)
                registered = np.asarray(preview.points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]
                self.viewport.set_layer(preview.scan_id, registered, preview.color, preview.visible, preview.scan_id == self.selected_scan_id)
                if row:
                    quality = "Reference" if registration.method == "reference" else f"Registered · {registration.fitness:.0%}"
                    row.file_label.setText(f"{Path(preview.source_path).name} · {quality}")
                    row.file_label.setStyleSheet("color:#00E676;font-size:10px")
            else:
                preview.visible = False; self.viewport.set_layer_visible(preview.scan_id, False)
                if row:
                    row.visibility.blockSignals(True); row.visibility.setChecked(False); row.visibility.blockSignals(False)
                    row.file_label.setText(f"{Path(preview.source_path).name} · Alignment review")
                    row.file_label.setStyleSheet("color:#FFB74D;font-size:10px")
                    row.setToolTip(registration.message)
        self.viewport.fit_to_layers()

    def _show_fused_geometry(self, result):
        try:
            mesh, source_face_count, method = self._validated_preview_mesh(result.validated_mesh_path, result.fused_cloud_path)
            for scan_id in self.scan_previews: self.viewport.set_layer_visible(scan_id, False)
            self.viewport.remove_layer("fused-geometry")
            self.viewport.set_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), source_face_count)
            self._log(f"[DEEP_FUSION] Displaying solid {method} preview: {source_face_count:,} source faces, {len(mesh.triangles):,} viewport faces")
        except Exception as error:
            self._log(f"[WARNING] Solid mesh preview failed ({error}); displaying the fused evidence points instead.")
            loaded = load_point_cloud(result.fused_cloud_path, self._log)
            if loaded.cloud is not None and len(loaded.cloud.points):
                for scan_id in self.scan_previews: self.viewport.set_layer_visible(scan_id, False)
                self.viewport.set_layer("fused-geometry", np.asarray(loaded.cloud.points), QColor("#00E676"), True, True); self.viewport.fit_to_layers()
        self.metric_labels["Confidence"].setText(f"{result.mean_confidence:.0%}")

    def _validated_preview_mesh(self, validated_path, fused_cloud_path):
        """Load the architecture-aware validated mesh, with conservative Poisson fallback."""
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(validated_path)) if Path(validated_path).is_file() else o3d.geometry.TriangleMesh()
        method = "validated architecture-aware surface"
        if mesh.is_empty() or not mesh.has_triangles():
            cloud = o3d.io.read_point_cloud(str(fused_cloud_path))
            if cloud.is_empty(): raise ValueError("Neither validated mesh faces nor fused points are available")
            preview_voxel = max(float(self.voxel.value()), .01)
            cloud = cloud.voxel_down_sample(preview_voxel)
            if not cloud.has_normals():
                cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=preview_voxel * 3, max_nn=50))
                try: cloud.orient_normals_consistent_tangent_plane(30)
                except RuntimeError: cloud.normalize_normals()
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=8, scale=1.05, linear_fit=True)
            densities = np.asarray(densities)
            if len(densities): mesh.remove_vertices_by_mask(densities < np.quantile(densities, .02))
            mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
            method = "Poisson fallback surface"
        mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles(); mesh.remove_duplicated_vertices(); mesh.remove_unreferenced_vertices()
        source_face_count = len(mesh.triangles)
        if not source_face_count: raise ValueError("Reconstruction produced no valid triangles")
        if source_face_count > FusionViewport.MAX_PREVIEW_FACES:
            mesh = mesh.simplify_quadric_decimation(FusionViewport.MAX_PREVIEW_FACES)
            mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh, source_face_count, method

    @staticmethod
    def _quality_word(value): return "Excellent" if value >= .90 else "Good" if value >= .75 else "Review"

    def _update_state(self):
        count = len(self.scan_previews); self.scan_count_label.setText(f"{count} scan{'s' if count != 1 else ''}")
        display_stage = min(max(self.current_stage, 0), len(PIPELINE_STAGES) - 1); self.stage_counter.setText(f"Stage {display_stage + 1} of {len(PIPELINE_STAGES)}")
        stage_keys = ("diagnostics", "alignment", "point_fusion", "surface", "cleanup", "texture", "quality")
        for index, label in enumerate(self.stage_labels):
            completed = stage_keys[index] in self.completed_stages
            if completed: text, color, background = f"✓ {PIPELINE_STAGES[index]}", "#00E676", "#1B382B"
            elif index == display_stage: text, color, background = f"● {PIPELINE_STAGES[index]}", "#00E676", "#242424"
            else: text, color, background = f"○ {PIPELINE_STAGES[index]}", "#777777", "#1A1A1A"
            label.setText(text); label.setEnabled(not self.processing and index <= self.unlocked_stage)
            label.setStyleSheet(f"QToolButton{{color:{color};background:{background};padding:6px;border-radius:4px;font-size:10px;border:1px solid #2B2B2B}}")
        for index, panel in enumerate(self.stage_panels):
            panel.setVisible(index == display_stage); panel.setEnabled(not self.processing)
        if self.processing:
            task_labels = {"diagnostics": "Analyzing Scans…", "alignment": "Aligning Point Clouds…", "point_fusion": "Fusing Point Clouds…", "surface": "Reconstructing Surface…", "cleanup_validation": "Validating Cleanup…", "photo_registration": "Registering Photogrammetry…", "texture": "Baking Texture…"}
            self.action_button.setText(task_labels.get(getattr(self.worker, "task", ""), "Processing…"))
            self.action_button.setEnabled(False)
            self._set_action_style("processing")
            return
        enabled = True
        if display_stage == 0:
            self.action_button.setText("Continue"); enabled = count >= 2
            if self.workspace is None: self.status_text.setText(f"Ready to analyze {count} scans" if enabled else "Add at least two independent scan passes")
        elif display_stage == 1: self.action_button.setText("Continue" if "alignment" in self.completed_stages else "Align Point Clouds"); enabled = self.workspace is not None
        elif display_stage == 2: self.action_button.setText("Continue" if "point_fusion" in self.completed_stages else "Fuse Point Clouds"); enabled = "alignment" in self.completed_stages
        elif display_stage == 3: self.action_button.setText("Continue" if "surface" in self.completed_stages else "Reconstruct Surface"); enabled = "point_fusion" in self.completed_stages
        elif display_stage == 4: self.action_button.setText("Proceed"); enabled = "surface" in self.completed_stages
        elif display_stage == 5:
            if "texture" in self.completed_stages: self.action_button.setText("Continue")
            elif "photo_registration" in self.completed_stages: self.action_button.setText("Bake Texture")
            else: self.action_button.setText("Register Photogrammetry")
            enabled = "cleanup" in self.completed_stages and (self.photogrammetry_paths is not None or not self.photogrammetry_loaded)
        else: self.action_button.setText("Open Quality Report"); enabled = "quality" in self.completed_stages
        self.action_button.setEnabled(enabled)
        self._set_action_style("ready" if self.action_button.isEnabled() else "blocked")
        self._persist_workflow_state()

    def _set_action_style(self, state):
        if state == "processing":
            colors = "background:#FFB74D;color:#121212;border:1px solid #FFB74D"
            hover = "background:#FFB74D;color:#121212"
        elif state == "blocked":
            colors = "background:#2A2A2A;color:#777777;border:1px solid #3A3A3A"
            hover = "background:#2A2A2A;color:#777777"
        else:
            colors = "background:#00E676;color:#121212;border:1px solid #00E676"
            hover = "background:#00C853;color:#121212"
        self.action_button.setStyleSheet(
            f"QPushButton{{{colors};font-weight:700;padding:8px 18px}}"
            f"QPushButton:hover{{{hover}}}"
            f"QPushButton:disabled{{{colors}}}"
        )

    def _input_problems(self):
        previews = list(self.scan_previews.values()); problems = []
        if len(previews) < 2: problems.append("Add at least two independent .PLY scan passes.")
        problems.extend(f"Scan pass is missing: {item.source_path}" for item in previews if not Path(item.source_path).is_file())
        if not self.workspace_edit.text().strip(): problems.append("Choose a derived workspace directory in Advanced Settings.")
        return problems

    def _new_workspace_path(self):
        # A stable derived path is required for File > Recover Last Session.
        # add_pass() rewrites only the manifest/derived workspace and never the sources.
        return Path(self.workspace_edit.text()).resolve()

    def _cancel(self):
        if self.worker and self.worker.isRunning(): self.worker.requestInterruption(); self.cancel_button.setEnabled(False); self._log("Cancellation requested; the active native operation will finish safely.")

    def _open_report(self):
        quality = self.last_result.get("quality"); path = getattr(quality, "html_report_path", None)
        if path and Path(path).is_file(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    def _log(self, message): self.console.append(str(message))


DeepMeshFusionWorkspaceWidget = DeepMeshFusionPanel
