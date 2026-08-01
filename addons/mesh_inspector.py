"""
Mesh Inspector & Statistics Add-on for Proximap Mesh Editor.
Displays vertex counts, face counts, bounding dimensions, and mesh metrics for selected scene objects.
"""

from typing import Optional
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame, QGridLayout
)
from PySide6.QtCore import Qt

from addons.addon_base import ProximapAddon


class MeshInspectorAddon(ProximapAddon):
    addon_id = "mesh_inspector"
    addon_name = "Mesh Inspector & Statistics"
    addon_version = "1.0.0"
    addon_description = "Calculates vertex counts, face counts, and bounding dimensions for selected meshes in the Mesh Editor."
    addon_author = "Proximap Team"
    addon_category = "Mesh Editor"
    dependencies = ["trimesh"]

    def __init__(self):
        super().__init__()
        self._panel_widget = None

    def register(self, mesh_editor) -> None:
        super().register(mesh_editor)
        if hasattr(mesh_editor, "viewport") and hasattr(mesh_editor.viewport, "selection_changed"):
            try:
                mesh_editor.viewport.selection_changed.connect(self.update_stats)
                mesh_editor.viewport.transform_changed.connect(self.update_stats)
            except Exception:
                pass

    def unregister(self, mesh_editor) -> None:
        if self.mesh_editor and hasattr(self.mesh_editor, "viewport"):
            try:
                self.mesh_editor.viewport.selection_changed.disconnect(self.update_stats)
                self.mesh_editor.viewport.transform_changed.disconnect(self.update_stats)
            except Exception:
                pass
        super().unregister(mesh_editor)

    def get_panel_widget(self, parent: Optional[QWidget] = None) -> Optional[QWidget]:
        if self._panel_widget is not None:
            return self._panel_widget

        panel = QFrame(parent)
        panel.setObjectName("MeshInspectorPanel")
        panel.setStyleSheet("""
            QFrame#MeshInspectorPanel {
                background-color: #121212;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 8px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 11px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hdr_label = QLabel("🔍 Mesh Inspector", panel)
        hdr_label.setStyleSheet("color: #00E676; font-weight: bold; font-size: 12px;")
        layout.addWidget(hdr_label)

        grid = QGridLayout()
        grid.setSpacing(4)

        lbl_style = "color: #aaaaaa; font-weight: bold;"
        val_style = "color: #ffffff; font-size: 11px;"

        grid.addWidget(QLabel("Vertices:", panel), 0, 0)
        self.lbl_verts = QLabel("0", panel)
        self.lbl_verts.setStyleSheet(val_style)
        grid.addWidget(self.lbl_verts, 0, 1)

        grid.addWidget(QLabel("Faces (Triangles):", panel), 1, 0)
        self.lbl_faces = QLabel("0", panel)
        self.lbl_faces.setStyleSheet(val_style)
        grid.addWidget(self.lbl_faces, 1, 1)

        grid.addWidget(QLabel("Dimensions (X/Y/Z):", panel), 2, 0)
        self.lbl_dims = QLabel("0 x 0 x 0 m", panel)
        self.lbl_dims.setStyleSheet(val_style)
        grid.addWidget(self.lbl_dims, 2, 1)

        layout.addLayout(grid)

        btn_refresh = QPushButton("Refresh Mesh Stats", panel)
        btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #262626;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #00E676;
            }
        """)
        btn_refresh.clicked.connect(self.update_stats)
        layout.addWidget(btn_refresh)

        self._panel_widget = panel
        self.update_stats()
        return self._panel_widget

    def update_stats(self, *args):
        if not self._panel_widget:
            return

        if not self.mesh_editor or not hasattr(self.mesh_editor, "viewport"):
            self._reset_labels()
            return

        selected_objs = self.mesh_editor.viewport.scene.selected_objects
        if not selected_objs:
            self._reset_labels("No selection")
            return

        total_verts = 0
        total_faces = 0
        min_bound = np.array([float('inf'), float('inf'), float('inf')])
        max_bound = np.array([float('-inf'), float('-inf'), float('-inf')])

        for obj in selected_objs:
            if hasattr(obj, "mesh_data") and obj.mesh_data is not None:
                verts = obj.mesh_data.vertices
                faces = obj.mesh_data.faces
                total_verts += len(verts)
                total_faces += len(faces)

                if len(verts) > 0:
                    # Apply local matrix transformation for scale/bounding calculation
                    scale = np.array(obj.scale)
                    scaled_verts = verts * scale
                    min_bound = np.minimum(min_bound, scaled_verts.min(axis=0))
                    max_bound = np.maximum(max_bound, scaled_verts.max(axis=0))

        if total_verts == 0 or np.isinf(min_bound[0]):
            self._reset_labels()
            return

        dims = max_bound - min_bound
        self.lbl_verts.setText(f"{total_verts:,}")
        self.lbl_faces.setText(f"{total_faces:,}")
        self.lbl_dims.setText(f"{dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} m")

    def _reset_labels(self, status: str = "Select a mesh"):
        if self._panel_widget:
            self.lbl_verts.setText("0")
            self.lbl_faces.setText("0")
            self.lbl_dims.setText(status)


# Mandatory addon instance
addon = MeshInspectorAddon()
