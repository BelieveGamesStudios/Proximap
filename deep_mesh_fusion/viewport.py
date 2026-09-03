from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import OpenGL.GL as gl
from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QSizePolicy

from mesh_editor.scene import Mesh, Object
from mesh_editor.viewport import MeshEditorViewport


class DeepMeshFusionViewport(MeshEditorViewport):
    """Locked Mesh Editor viewport for evidence and reconstructed-surface review."""

    ply_files_dropped = Signal(list)
    MAX_POINTS_PER_LAYER = 18_000
    MAX_PREVIEW_FACES = 20_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("deepMeshFusionViewport")
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAcceptDrops(True)
        self.set_transform_enabled(False)
        self.picking_enabled = False
        self.layers: Dict[str, Dict] = {}
        self.mesh_layer = None
        self._mesh_object = None
        self._point_program = None
        self._point_buffers = {}
        self._drop_active = False

    def set_transform_enabled(self, _enabled: bool):
        """Keep this inspection viewport locked even if a shared caller requests editing."""
        super().set_transform_enabled(False)

    def _pick_object_at(self, _mouse_x: float, _mouse_y: float, shift_held: bool = False):
        """Deep Mesh Fusion never performs GPU picking or ray-based object selection."""
        return None

    def initializeGL(self):
        super().initializeGL()
        self._point_program = self._compile_and_link_shaders(
            """#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 view;
uniform mat4 proj;
uniform float pointSize;
void main() {
    gl_Position = proj * view * vec4(aPos, 1.0);
    gl_PointSize = pointSize;
}
""",
            """#version 330 core
uniform vec4 pointColor;
out vec4 fragColor;
void main() { fragColor = pointColor; }
""",
        )
        for layer_id in self.layers:
            self._upload_layer(layer_id)

    def set_layer(self, layer_id: str, points, color: QColor, visible=True, selected=False):
        points = np.asarray(points, dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1)] if len(points) else points.reshape(0, 3)
        if len(points) > self.MAX_POINTS_PER_LAYER:
            points = points[np.linspace(0, len(points) - 1, self.MAX_POINTS_PER_LAYER, dtype=int)]
        self.layers[layer_id] = {
            "points": np.ascontiguousarray(points), "color": QColor(color),
            "visible": bool(visible), "selected": bool(selected),
        }
        if self.isValid() and self._point_program is not None:
            self.makeCurrent(); self._upload_layer(layer_id); self.doneCurrent()
        self.update()

    def remove_layer(self, layer_id):
        self.layers.pop(layer_id, None)
        handles = self._point_buffers.pop(layer_id, None)
        if handles and self.isValid():
            self.makeCurrent()
            gl.glDeleteVertexArrays(1, [handles[0]]); gl.glDeleteBuffers(1, [handles[1]])
            self.doneCurrent()
        self.fit_to_layers()

    def set_layer_visible(self, layer_id, visible):
        if layer_id in self.layers:
            self.layers[layer_id]["visible"] = bool(visible); self.update()

    def select_layer(self, layer_id):
        # Selection is visual-only. It never enters Mesh Editor object selection,
        # so no picking, raycasting, or transformation path can be activated.
        for key, layer in self.layers.items():
            layer["selected"] = key == layer_id
        self.update()

    def set_mesh(self, vertices, faces, source_face_count=None):
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("A viewport mesh requires vertices (N, 3) and triangular faces (M, 3)")
        valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1); faces = faces[valid]
        self.mesh_layer = {
            "vertices": vertices, "faces": faces,
            "source_face_count": int(source_face_count if source_face_count is not None else len(faces)),
        }
        normals = self._vertex_normals(vertices, faces)
        mesh = Mesh(vertices, faces.astype(np.uint32).ravel(), normals, np.min(vertices, axis=0), np.max(vertices, axis=0))
        self._discard_mesh_object()
        self._mesh_object = Object("Validated Fused Surface", mesh)
        self.scene.add_object(self._mesh_object)
        self.fit_to_layers(); self.mesh_inspector.update_stats(); self.update()

    def clear_mesh(self):
        self._discard_mesh_object(); self.mesh_layer = None
        self.mesh_inspector.update_stats(); self.fit_to_layers()

    def _discard_mesh_object(self):
        if self._mesh_object is None:
            return
        self.scene.remove_object(self._mesh_object)
        if self.isValid() and self._mesh_object.mesh.vao is not None:
            self.makeCurrent(); self._mesh_object.mesh.cleanup(); self.doneCurrent()
        self._mesh_object = None

    @staticmethod
    def _vertex_normals(vertices, faces):
        normals = np.zeros_like(vertices, dtype=np.float32)
        if len(faces):
            triangles = vertices[faces]
            face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            for corner in range(3):
                np.add.at(normals, faces[:, corner], face_normals)
            lengths = np.linalg.norm(normals, axis=1); valid = lengths > 1e-12
            normals[valid] /= lengths[valid, None]
        return normals

    def fit_to_layers(self):
        visible = [layer["points"] for layer in self.layers.values() if layer["visible"] and len(layer["points"])]
        if self.mesh_layer is not None and len(self.mesh_layer["vertices"]):
            visible.append(self.mesh_layer["vertices"])
        if visible:
            points = np.vstack(visible); lower = np.min(points, axis=0); upper = np.max(points, axis=0)
            extent = max(float(np.max(upper - lower)), 1e-3)
            self.camera.target = ((lower + upper) * .5).astype(np.float32)
            self.camera.distance = max(extent * 2.2, 1.0)
            self.camera.ortho_scale = max(extent * 1.35, 1.0)
            self.notify_camera_changed()
        self.update()

    def paintGL(self):
        for obj in self.scene.objects:
            if obj.mesh.vao is None:
                obj.mesh.setup_buffers()
        super().paintGL()
        if self._point_program is None:
            return
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE); gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(self._point_program)
        aspect = self.width() / max(self.height(), 1)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._point_program, "view"), 1, gl.GL_FALSE, self.camera.get_view_matrix())
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._point_program, "proj"), 1, gl.GL_FALSE, self.camera.get_projection_matrix(aspect))
        for layer_id, layer in self.layers.items():
            if not layer["visible"] or not len(layer["points"]):
                continue
            if layer_id not in self._point_buffers:
                self._upload_layer(layer_id)
            vao, _vbo, count = self._point_buffers[layer_id]
            color = layer["color"]
            gl.glUniform4f(gl.glGetUniformLocation(self._point_program, "pointColor"), color.redF(), color.greenF(), color.blueF(), 1.0)
            gl.glUniform1f(gl.glGetUniformLocation(self._point_program, "pointSize"), 3.0 if layer["selected"] else 2.0)
            gl.glBindVertexArray(vao); gl.glDrawArrays(gl.GL_POINTS, 0, count)
        gl.glBindVertexArray(0); gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)

    def _upload_layer(self, layer_id):
        points = self.layers[layer_id]["points"]
        handles = self._point_buffers.get(layer_id)
        if handles is None:
            vao, vbo = gl.glGenVertexArrays(1), gl.glGenBuffers(1)
        else:
            vao, vbo = handles[:2]
        gl.glBindVertexArray(vao); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, points.nbytes, points, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindVertexArray(0); self._point_buffers[layer_id] = (vao, vbo, len(points))

    def mousePressEvent(self, event: QMouseEvent):
        # Plain left-click is deliberately inert. Mesh Editor navigation remains
        # available through MMB and Alt+LMB, but object interaction is locked.
        if event.button() == Qt.MouseButton.LeftButton and not (event.modifiers() & Qt.KeyboardModifier.AltModifier):
            event.accept(); return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        blocked = {Qt.Key.Key_G, Qt.Key.Key_R, Qt.Key.Key_S, Qt.Key.Key_Delete, Qt.Key.Key_Backspace}
        ctrl_edit = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier) and event.key() in {Qt.Key.Key_A, Qt.Key.Key_Z, Qt.Key.Key_Y}
        if event.key() in blocked or ctrl_edit:
            event.accept(); return
        super().keyPressEvent(event)

    @staticmethod
    def _ply_paths_from_mime(mime: QMimeData):
        if not mime or not mime.hasUrls():
            return []
        return [str(Path(url.toLocalFile()).resolve()) for url in mime.urls()
                if url.isLocalFile() and Path(url.toLocalFile()).is_file() and Path(url.toLocalFile()).suffix.lower() == ".ply"]

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

    def cleanupGL(self):
        try:
            for vao, vbo, _count in self._point_buffers.values():
                gl.glDeleteVertexArrays(1, [vao]); gl.glDeleteBuffers(1, [vbo])
            if self._point_program:
                gl.glDeleteProgram(self._point_program)
        except Exception:
            pass
        self._point_buffers.clear(); self._point_program = None
        super().cleanupGL()
