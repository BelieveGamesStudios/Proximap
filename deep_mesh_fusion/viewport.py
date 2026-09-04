from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict

import numpy as np
import OpenGL.GL as gl
import pyrr
from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QSizePolicy

from mesh_editor.scene import Mesh, Object
from mesh_editor.viewport import MeshEditorViewport


class _PointLayerTransformProxy:
    """Mesh-editor compatible transform target for one point-cloud layer."""

    def __init__(self, layer_id: str, points):
        self.layer_id = layer_id
        self.name = layer_id
        self.position = np.zeros(3, dtype=np.float32)
        self.rotation = np.zeros(3, dtype=np.float32)
        self.scale = np.ones(3, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        self._local_min = np.min(points, axis=0) if len(points) else np.zeros(3, dtype=np.float32)
        self._local_max = np.max(points, axis=0) if len(points) else np.zeros(3, dtype=np.float32)

    def set_matrix(self, matrix):
        import trimesh

        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("A point-layer transform must be a finite 4x4 matrix")
        self.position = matrix[:3, 3].astype(np.float32)
        angles = trimesh.transformations.euler_from_matrix(matrix, "sxyz")
        self.rotation = -np.degrees(angles).astype(np.float32)
        self.scale = np.ones(3, dtype=np.float32)

    def matrix(self):
        import trimesh

        angles = np.radians(-self.rotation)
        translation = trimesh.transformations.translation_matrix(self.position)
        rotation = trimesh.transformations.euler_matrix(*angles, axes="sxyz")
        return np.asarray(translation @ rotation, dtype=float)

    def get_model_matrix(self):
        return self.matrix().T.astype(np.float32)

    def get_world_aabb(self):
        lower, upper = self._local_min, self._local_max
        corners = np.asarray([
            (x, y, z) for x in (lower[0], upper[0])
            for y in (lower[1], upper[1]) for z in (lower[2], upper[2])
        ], dtype=float)
        matrix = self.matrix()
        moved = corners @ matrix[:3, :3].T + matrix[:3, 3]
        return np.min(moved, axis=0), np.max(moved, axis=0)


class DeepMeshFusionViewport(MeshEditorViewport):
    """Mesh Editor viewport with isolated point-layer alignment support."""

    ply_files_dropped = Signal(list)
    layer_transform_changed = Signal(str, object)
    MAX_POINTS_PER_LAYER = 18_000
    MAX_PREVIEW_FACES = 20_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("deepMeshFusionViewport")
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAcceptDrops(True)
        self._alignment_proxy = None
        self.set_transform_enabled(False)
        self.picking_enabled = False
        self.layers: Dict[str, Dict] = {}
        self.mesh_layer = None
        self._mesh_object = None
        self._point_program = None
        self._point_buffers = {}
        self._drop_active = False
        self.selection_overlay = None
        self.transform_changed.connect(self._sync_alignment_proxy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.selection_overlay is not None:
            self.selection_overlay.setGeometry(self.rect())

    def attach_selection_overlay(self, overlay):
        self.selection_overlay = overlay
        overlay.setGeometry(self.rect())
        overlay.raise_()

    def project_layer_points(self, layer_id: str):
        """Return screen pixels and a front-facing mask for one rendered point layer."""
        layer = self.layers[layer_id]
        matrix = np.asarray(layer.get("transform", np.eye(4)), dtype=float)
        points = np.asarray(layer["points"], dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
        return self.project_points(points)

    def project_points(self, points):
        """Project world-space points to viewport pixels."""
        points = np.asarray(points, dtype=float)
        world = np.column_stack((points, np.ones(len(points), dtype=float)))
        aspect = self.width() / max(self.height(), 1)
        vp = pyrr.matrix44.multiply(self.camera.get_view_matrix(), self.camera.get_projection_matrix(aspect))
        # pyrr's matrix helpers use row vectors (clip = world · view · projection).
        clip = world @ np.asarray(vp, dtype=float)
        valid = clip[:, 3] > 1e-6
        screen = np.zeros((len(points), 2), dtype=np.float32)
        ndc = clip[valid, :3] / clip[valid, 3, None]
        screen[valid, 0] = (ndc[:, 0] + 1.0) * .5 * self.width()
        screen[valid, 1] = (1.0 - ndc[:, 1]) * .5 * self.height()
        valid_indices = np.flatnonzero(valid)
        inside = (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (np.abs(ndc[:, 2]) <= 1)
        visible = np.zeros(len(points), dtype=bool); visible[valid_indices[inside]] = True
        return screen, visible

    def set_transform_enabled(self, enabled: bool):
        """Only permit transforms while a moving point layer is active."""
        super().set_transform_enabled(bool(enabled and self._alignment_proxy is not None))

    def _pick_object_at(self, _mouse_x: float, _mouse_y: float, shift_held: bool = False):
        """Deep Mesh Fusion never performs GPU picking or ray-based object selection."""
        return None

    def initializeGL(self):
        super().initializeGL()
        self._point_program = self._compile_and_link_shaders(
            """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;
uniform float pointSize;
out vec3 vertexColor;
void main() {
    gl_Position = proj * view * model * vec4(aPos, 1.0);
    gl_PointSize = pointSize;
    vertexColor = aColor;
}
""",
            """#version 330 core
uniform vec4 pointColor;
uniform bool hasVertexColors;
in vec3 vertexColor;
out vec4 fragColor;
void main() { fragColor = hasVertexColors ? vec4(vertexColor, 1.0) : pointColor; }
""",
        )
        for layer_id in self.layers:
            self._upload_layer(layer_id)

    def set_layer(self, layer_id: str, points, color: QColor, visible=True, selected=False, vertex_colors=None):
        points = np.asarray(points, dtype=np.float32)
        colors = None if vertex_colors is None else np.asarray(vertex_colors, dtype=np.float32)
        if colors is not None and (colors.ndim != 2 or colors.shape != points.shape):
            colors = None
        if colors is not None and colors.size and float(np.nanmax(colors)) > 1.0:
            colors = colors / 255.0
        finite = np.isfinite(points).all(axis=1) if len(points) else np.zeros(0, dtype=bool)
        points = points[finite] if len(points) else points.reshape(0, 3)
        if colors is not None:
            colors = np.clip(colors[finite], 0.0, 1.0)
        if len(points) > self.MAX_POINTS_PER_LAYER:
            indices = np.linspace(0, len(points) - 1, self.MAX_POINTS_PER_LAYER, dtype=int)
            points = points[indices]
            if colors is not None: colors = colors[indices]
        self.layers[layer_id] = {
            "points": np.ascontiguousarray(points), "color": QColor(color),
            "vertex_colors": None if colors is None else np.ascontiguousarray(colors),
            "visible": bool(visible), "selected": bool(selected),
            "transform": np.asarray(
                self.layers.get(layer_id, {}).get("transform", np.eye(4)), dtype=float
            ),
        }
        if self.isValid() and self._point_program is not None:
            self.makeCurrent(); self._upload_layer(layer_id); self.doneCurrent()
        self.update()

    def remove_layer(self, layer_id):
        if self._alignment_proxy is not None and self._alignment_proxy.layer_id == layer_id:
            self.end_layer_alignment()
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
        for key, layer in self.layers.items():
            layer["selected"] = key == layer_id
        self.update()

    def begin_layer_alignment(self, layer_id: str, transform=None):
        if layer_id not in self.layers:
            raise KeyError(f"Unknown point-cloud layer: {layer_id}")
        self.end_layer_alignment()
        proxy = _PointLayerTransformProxy(layer_id, self.layers[layer_id]["points"])
        proxy.set_matrix(transform if transform is not None else self.layers[layer_id]["transform"])
        self._alignment_proxy = proxy
        self.layers[layer_id]["transform"] = proxy.matrix()
        self.scene.selected_objects = [proxy]
        self.scene.active_object = proxy
        super().set_transform_enabled(True)
        self.gizmo.operation = "translate"
        self.gizmo.space = "global"
        self.select_layer(layer_id)
        self.update()

    def end_layer_alignment(self):
        if self._alignment_proxy is not None:
            layer_id = self._alignment_proxy.layer_id
            if layer_id in self.layers:
                self.layers[layer_id]["transform"] = self._alignment_proxy.matrix()
        self._alignment_proxy = None
        super().set_transform_enabled(False)

    def set_alignment_tool(self, operation: str):
        if operation not in {"translate", "rotate"}:
            raise ValueError("Point-cloud alignment supports translate and rotate only")
        if self._alignment_proxy is None:
            return
        self.gizmo.operation = operation
        self.tool_changed.emit(operation)
        self.update()

    def layer_transform(self, layer_id: str):
        if self._alignment_proxy is not None and self._alignment_proxy.layer_id == layer_id:
            return self._alignment_proxy.matrix().copy()
        return np.asarray(self.layers[layer_id]["transform"], dtype=float).copy()

    def set_layer_transform(self, layer_id: str, transform):
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("A point-layer transform must be a finite 4x4 matrix")
        self.layers[layer_id]["transform"] = matrix.copy()
        if self._alignment_proxy is not None and self._alignment_proxy.layer_id == layer_id:
            self._alignment_proxy.set_matrix(matrix)
        self.update()

    def _sync_alignment_proxy(self, obj):
        if obj is None or obj is not self._alignment_proxy:
            return
        matrix = obj.matrix()
        self.layers[obj.layer_id]["transform"] = matrix
        self.layer_transform_changed.emit(obj.layer_id, matrix.copy())

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
        visible = []
        for layer in self.layers.values():
            if not layer["visible"] or not len(layer["points"]):
                continue
            matrix = np.asarray(layer.get("transform", np.eye(4)), dtype=float)
            visible.append(layer["points"] @ matrix[:3, :3].T + matrix[:3, 3])
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
        draw_gizmo = bool(self.transform_enabled and self.enable_gizmo and self.scene.selected_object is not None)
        previous_gizmo = self.enable_gizmo
        self.enable_gizmo = False
        super().paintGL()
        self.enable_gizmo = previous_gizmo
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
            model = np.asarray(layer.get("transform", np.eye(4)), dtype=np.float32).T
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._point_program, "model"), 1, gl.GL_FALSE, model)
            gl.glUniform4f(gl.glGetUniformLocation(self._point_program, "pointColor"), color.redF(), color.greenF(), color.blueF(), 1.0)
            gl.glUniform1i(gl.glGetUniformLocation(self._point_program, "hasVertexColors"), 1 if layer["vertex_colors"] is not None else 0)
            gl.glUniform1f(gl.glGetUniformLocation(self._point_program, "pointSize"), 3.0 if layer["selected"] else 2.0)
            gl.glBindVertexArray(vao); gl.glDrawArrays(gl.GL_POINTS, 0, count)
        gl.glBindVertexArray(0); gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)
        if draw_gizmo and self._alignment_proxy is not None:
            pivot = self._get_gizmo_pivot()
            if pivot is not None:
                self.gizmo.draw(pivot, self.camera, self.width(), self.height())

    def _upload_layer(self, layer_id):
        layer = self.layers[layer_id]; points = layer["points"]
        colors = layer["vertex_colors"]
        if colors is None:
            colors = np.zeros_like(points, dtype=np.float32)
        interleaved = np.ascontiguousarray(np.column_stack((points, colors)), dtype=np.float32)
        handles = self._point_buffers.get(layer_id)
        if handles is None:
            vao, vbo = gl.glGenVertexArrays(1), gl.glGenBuffers(1)
        else:
            vao, vbo = handles[:2]
        gl.glBindVertexArray(vao); gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, gl.GL_STATIC_DRAW)
        stride = 6 * np.dtype(np.float32).itemsize
        gl.glEnableVertexAttribArray(0); gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(1); gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * np.dtype(np.float32).itemsize))
        gl.glBindVertexArray(0); self._point_buffers[layer_id] = (vao, vbo, len(points))

    def mousePressEvent(self, event: QMouseEvent):
        if (self._alignment_proxy is None and event.button() == Qt.MouseButton.LeftButton
                and not (event.modifiers() & Qt.KeyboardModifier.AltModifier)):
            event.accept(); return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if self._alignment_proxy is not None and event.key() in {Qt.Key.Key_G, Qt.Key.Key_R}:
            self.set_alignment_tool("translate" if event.key() == Qt.Key.Key_G else "rotate")
            event.accept(); return
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
