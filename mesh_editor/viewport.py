import os
import sys
import time
import numpy as np
import pyrr
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Signal, QPoint, QRectF
from PySide6.QtGui import QMouseEvent, QKeyEvent, QWheelEvent, QAction, QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent

import OpenGL.GL as gl
from PySide6.QtCore import QThread
from mesh_editor.scene import Camera, Scene, Object
from mesh_editor.gizmo import Gizmo

class MeshLoadWorker(QThread):
    """Background worker for loading and parsing 3D meshes off the main UI thread."""
    finished = Signal(object, str)  # (mesh_data_list, file_path)
    error = Signal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        try:
            import trimesh
            from PIL import Image
            loaded = trimesh.load(self.file_path, process=False)
            results = []
            if isinstance(loaded, trimesh.Scene):
                for name, geom in loaded.geometry.items():
                    data = self._extract_geom_data(geom)
                    results.append((name, data))
            else:
                data = self._extract_geom_data(loaded)
                name = os.path.basename(self.file_path)
                results.append((name, data))
            self.finished.emit(results, self.file_path)
        except Exception as e:
            self.error.emit(str(e))

    def _extract_geom_data(self, geom):
        verts = np.asarray(geom.vertices, dtype=np.float32)
        faces = np.asarray(geom.faces, dtype=np.uint32)
        normals = np.asarray(geom.vertex_normals, dtype=np.float32) if hasattr(geom, 'vertex_normals') and geom.vertex_normals is not None else np.zeros_like(verts)
        uvs = np.asarray(geom.visual.uv, dtype=np.float32) if hasattr(geom, 'visual') and hasattr(geom.visual, 'uv') and geom.visual.uv is not None else None
        
        img = None
        if hasattr(geom, 'visual') and hasattr(geom.visual, 'material') and hasattr(geom.visual.material, 'image') and geom.visual.material.image is not None:
            img = geom.visual.material.image.copy()

        bounds = geom.bounds if hasattr(geom, 'bounds') and geom.bounds is not None else (verts.min(axis=0), verts.max(axis=0))
        return {
            'vertices': verts,
            'indices': faces,
            'normals': normals,
            'texcoords': uvs,
            'texture_data': img,
            'aabb_min': bounds[0],
            'aabb_max': bounds[1]
        }

class GizmoPivot:
    """A virtual object representing the pivot / median center of selected objects."""
    def __init__(self, position, rotation, scale):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)
        
    def get_model_matrix(self) -> np.ndarray:
        import trimesh
        rx = np.radians(-self.rotation[0])
        ry = np.radians(-self.rotation[1])
        rz = np.radians(-self.rotation[2])
        t_mat = trimesh.transformations.translation_matrix(self.position)
        r_mat = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
        s_mat = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0])
        col_major = t_mat @ r_mat @ s_mat
        return col_major.T.astype(np.float32)

class MeshEditorViewport(QOpenGLWidget):
    """3D OpenGL Viewport for Mesh Editing. Integrates camera controls, grid, and ImGuizmo."""
    selection_changed = Signal(object) # Emits the newly selected Object (or None)
    transform_changed = Signal(object) # Emits the selected Object when transformed by the gizmo
    transform_committed = Signal(object) # Emits (states_dict, operation_str) tuple when gizmo drag ends
    undo_requested = Signal() # Emits when Ctrl+Z shortcut is pressed
    redo_requested = Signal() # Emits when Ctrl+Y or Ctrl+Shift+Z is pressed
    delete_pressed = Signal() # Emits when Delete/Backspace key is pressed
    tool_changed = Signal(object) # Emits active gizmo OPERATION when changed by hotkey
    camera_changed = Signal(object) # Emits Camera when it transforms/snaps

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        
        # State containers
        self.camera = Camera()
        self.scene = Scene()
        self.gizmo = Gizmo()
        self._gizmo_dragging = False
        
        # Box selection states
        self._is_box_selecting = False
        self._box_select_start = QPoint()
        self._box_select_end = QPoint()
        
        # Viewport customizable properties (defaults)
        self.bg_color = np.array([0.18, 0.18, 0.18], dtype=np.float32)       # Blender-style dark-grey background
        self.grid_color_1 = np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float32)  # Major grid lines
        self.grid_color_2 = np.array([0.25, 0.25, 0.25, 0.4], dtype=np.float32)  # Minor grid lines
        self.axis_color_x = np.array([0.8, 0.2, 0.2, 1.0], dtype=np.float32)   # X-axis (Red)
        self.axis_color_y = np.array([0.2, 0.8, 0.2, 1.0], dtype=np.float32)   # Y-axis (Green)
        self.grid_fade = 60.0
        self.grid_thickness = 1.0
        self.grid_subdivisions = 10.0
        self.invert_y = False
        
        # Orbital Navigation mouse states
        self.last_mouse_pos = QPoint()
        self.is_orbiting = False
        self.is_panning = False
        self.is_zooming = False
        
        # GPU buffers & shader programs
        self.object_program = None
        self.grid_program = None
        self.screen_program = None
        
        # Grid quad VAO
        self.grid_vao = None
        self.grid_vbo = None
        self.grid_ebo = None

        # Screen quad VAO/VBO for RTT compositing
        self.screen_vao = None
        self.screen_vbo = None

        # Offscreen FBO (Render-to-Texture) Handles
        self.fbo = None
        self.fbo_texture = None
        self.fbo_depth_stencil = None
        self.fbo_width = 0
        self.fbo_height = 0

        # Dynamic Resolution Scaling Constants & State
        self.render_scale = 1.0
        self.TARGET_FRAME_TIME_MS = 16.6  # 60 FPS target
        self.MIN_RENDER_SCALE = 0.5
        self.MAX_RENDER_SCALE = 1.0
        self.SCALE_STEP = 0.05
        self._last_frame_time_ms = 16.0

        # Shading modes: 0 = Flat (Unlit), 1 = Flux (Blinn-Phong), 2 = Prism (PBR Stub)
        self.shading_mode = 1
        # Independent Wireframe rasterization toggle
        self.wireframe = False

        # Navigation Gizmo Overlay Widget
        from mesh_editor.nav_gizmo import NavGizmoWidget
        self.nav_gizmo = NavGizmoWidget(self)
        self.nav_gizmo.snap_requested.connect(self._on_nav_gizmo_snap_requested)
        # Sync initial orientation
        self.nav_gizmo.update_orientation(self.camera.yaw, self.camera.pitch)

    def set_shading_mode(self, mode: int):
        """Set shading mode: 0 = Flat, 1 = Flux, 2 = Prism."""
        self.shading_mode = max(0, min(2, int(mode)))
        self.update()

    def set_wireframe_mode(self, enabled: bool):
        """Toggle Wire (Wireframe rasterization mode)."""
        self.wireframe = bool(enabled)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'nav_gizmo') and self.nav_gizmo:
            margin = 10
            self.nav_gizmo.move(self.width() - self.nav_gizmo.width() - margin, margin)

    def _on_nav_gizmo_snap_requested(self, view_name):
        self.camera.snap_to_view(view_name)
        self.notify_camera_changed()
        self.update()

    def notify_camera_changed(self):
        if hasattr(self, 'nav_gizmo') and self.nav_gizmo:
            self.nav_gizmo.update_orientation(self.camera.yaw, self.camera.pitch)
        self.camera_changed.emit(self.camera)

    def initializeGL(self):
        # Initialize OpenGL context
        print(f"[VP] initializeGL widget size=({self.width()},{self.height()})")
        
        # Load Shaders from disk with PyInstaller frozen bundle support
        if getattr(sys, 'frozen', False):
            meipass = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            shaders_dir = os.path.join(meipass, "mesh_editor", "shaders")
            if not os.path.exists(shaders_dir):
                shaders_dir = os.path.join(os.path.dirname(sys.executable), "mesh_editor", "shaders")
            if not os.path.exists(shaders_dir):
                shaders_dir = os.path.join(meipass, "shaders")
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            shaders_dir = os.path.join(base_dir, "shaders")
        
        with open(os.path.join(shaders_dir, "object.vert"), "r") as f:
            obj_vert_src = f.read()
        with open(os.path.join(shaders_dir, "object.frag"), "r") as f:
            obj_frag_src = f.read()
        with open(os.path.join(shaders_dir, "grid.vert"), "r") as f:
            grid_vert_src = f.read()
        with open(os.path.join(shaders_dir, "grid.frag"), "r") as f:
            grid_frag_src = f.read()
            
        self.object_program = self._compile_and_link_shaders(obj_vert_src, obj_frag_src)
        self.grid_program = self._compile_and_link_shaders(grid_vert_src, grid_frag_src)
        
        # Setup infinite grid quad and screen quad
        self._setup_grid_quad()
        self._setup_screen_quad()
        
        # Initialize custom gizmo inside this GL context
        self.gizmo.initialize(shaders_dir)
        
        # Enable depth testing
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

    def _get_gizmo_pivot(self):
        selected_objs = self.scene.selected_objects
        if not selected_objs:
            return None
        active_obj = self.scene.active_object or selected_objs[-1]
        
        # Compute median center of world bounding boxes
        world_min = np.array([float('inf'), float('inf'), float('inf')], dtype=np.float32)
        world_max = np.array([-float('inf'), -float('inf'), -float('inf')], dtype=np.float32)
        for obj in selected_objs:
            omin, omax = obj.get_world_aabb()
            world_min = np.minimum(world_min, omin)
            world_max = np.maximum(world_max, omax)
        median_center = (world_min + world_max) * 0.5
        
        return GizmoPivot(median_center, active_obj.rotation, active_obj.scale)

    def _setup_fbo(self, w: int, h: int):
        w = max(1, w)
        h = max(1, h)
        if self.fbo is not None and self.fbo_width == w and self.fbo_height == h:
            return

        self._cleanup_fbo()

        self.fbo_width = w
        self.fbo_height = h

        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        self.fbo_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.fbo_texture, 0)

        self.fbo_depth_stencil = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.fbo_depth_stencil)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, w, h)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self.fbo_depth_stencil)

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            print(f"[VP] FBO setup status incomplete: {status}")

        default_fbo = self.defaultFramebufferObject()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, default_fbo)

    def _cleanup_fbo(self):
        try:
            if self.fbo is not None:
                gl.glDeleteFramebuffers(1, [self.fbo])
                self.fbo = None
            if self.fbo_texture is not None:
                gl.glDeleteTextures(1, [self.fbo_texture])
                self.fbo_texture = None
            if self.fbo_depth_stencil is not None:
                gl.glDeleteRenderbuffers(1, [self.fbo_depth_stencil])
                self.fbo_depth_stencil = None
        except Exception:
            pass

    def _setup_screen_quad(self):
        vert_src = """#version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoords;
        out vec2 TexCoords;
        void main() {
            TexCoords = aTexCoords;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }"""
        frag_src = """#version 330 core
        out vec4 FragColor;
        in vec2 TexCoords;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoords);
        }"""
        self.screen_program = self._compile_and_link_shaders(vert_src, frag_src)

        quad_verts = np.array([
            -1.0,  1.0,  0.0, 1.0,
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,

            -1.0,  1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0
        ], dtype=np.float32)

        self.screen_vao = gl.glGenVertexArrays(1)
        self.screen_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.screen_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.screen_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad_verts.nbytes, quad_verts, gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(8))
        gl.glBindVertexArray(0)

    def paintGL(self):
        t0 = time.perf_counter()

        w_physical = getattr(self, '_fb_width', self.width())
        h_physical = getattr(self, '_fb_height', self.height())

        fbo_w = max(1, int(w_physical * self.render_scale))
        fbo_h = max(1, int(h_physical * self.render_scale))

        self._setup_fbo(fbo_w, fbo_h)

        # 1. Bind Offscreen FBO (Render-To-Texture Pass)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, fbo_w, fbo_h)

        gl.glClearColor(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        
        w_logical = self.width()
        h_logical = self.height()
        aspect = w_logical / max(h_logical, 1.0)
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix(aspect)
        
        # 2. Render Scene Objects using Master Shader (Flat=0, Flux=1, Prism=2)
        gl.glUseProgram(self.object_program)
        
        eye = self.camera.get_position()
        light_dir = eye - self.camera.target
        norm_light_dir = light_dir / max(np.linalg.norm(light_dir), 1e-6)
        
        gl.glUniform3fv(gl.glGetUniformLocation(self.object_program, "lightDir"), 1, norm_light_dir)
        gl.glUniform3fv(gl.glGetUniformLocation(self.object_program, "eyePos"), 1, eye)
        gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "uLightingMode"), self.shading_mode)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.object_program, "view"), 1, gl.GL_FALSE, view_matrix)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.object_program, "proj"), 1, gl.GL_FALSE, proj_matrix)
        
        # Wireframe mode rasterization toggle (Wire)
        if self.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        for obj in self.scene.objects:
            model_matrix = obj.get_model_matrix()
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.object_program, "model"), 1, gl.GL_FALSE, model_matrix)
            
            color = np.array([0.45, 0.55, 0.65], dtype=np.float32)
            gl.glUniform3fv(gl.glGetUniformLocation(self.object_program, "objectColor"), 1, color)
            gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "useOverrideColor"), 0)
            
            has_tex = obj.mesh.texture_id is not None
            gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "useTexture"), 1 if has_tex else 0)
            gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "textureSampler"), 0)
            
            is_selected = obj in self.scene.selected_objects
            is_active = self.scene.active_object == obj
            
            if is_selected:
                gl.glEnable(gl.GL_STENCIL_TEST)
                gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_REPLACE)
                gl.glStencilFunc(gl.GL_ALWAYS, 1, 0xFF)
                gl.glStencilMask(0xFF)
                gl.glClear(gl.GL_STENCIL_BUFFER_BIT)
                
            obj.mesh.draw()
            
            if is_selected:
                gl.glStencilFunc(gl.GL_NOTEQUAL, 1, 0xFF)
                gl.glStencilMask(0x00)
                gl.glDepthMask(gl.GL_FALSE)
                
                outline_model = obj.get_outline_model_matrix(1.015)
                gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.object_program, "model"), 1, gl.GL_FALSE, outline_model)
                
                gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "useOverrideColor"), 1)
                highlight_color = np.array([0.0, 1.0, 0.61, 1.0], dtype=np.float32) if is_active else np.array([0.1, 0.42, 0.27, 1.0], dtype=np.float32)
                gl.glUniform4fv(gl.glGetUniformLocation(self.object_program, "overrideColor"), 1, highlight_color)
                
                obj.mesh.draw()
                
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_STENCIL_TEST)

        # Reset polygon mode to GL_FILL for infinite grid, gizmos, and screen blit
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                
        # 3. Render Infinite Grid
        gl.glUseProgram(self.grid_program)
        
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_program, "view"), 1, gl.GL_FALSE, view_matrix)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_program, "proj"), 1, gl.GL_FALSE, proj_matrix)
        inv_view = pyrr.matrix44.inverse(view_matrix)
        inv_proj = pyrr.matrix44.inverse(proj_matrix)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_program, "invView"), 1, gl.GL_FALSE, inv_view)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_program, "invProj"), 1, gl.GL_FALSE, inv_proj)
        
        gl.glUniform4fv(gl.glGetUniformLocation(self.grid_program, "gridColor1"), 1, self.grid_color_1)
        gl.glUniform4fv(gl.glGetUniformLocation(self.grid_program, "gridColor2"), 1, self.grid_color_2)
        gl.glUniform4fv(gl.glGetUniformLocation(self.grid_program, "axisColorX"), 1, self.axis_color_x)
        gl.glUniform4fv(gl.glGetUniformLocation(self.grid_program, "axisColorY"), 1, self.axis_color_y)
        gl.glUniform1f(gl.glGetUniformLocation(self.grid_program, "gridFade"), self.grid_fade)
        gl.glUniform1f(gl.glGetUniformLocation(self.grid_program, "gridThickness"), self.grid_thickness)
        gl.glUniform1f(gl.glGetUniformLocation(self.grid_program, "subdivisions"), self.grid_subdivisions)
        
        gl.glBindVertexArray(self.grid_vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)
        
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        
        # 4. Render Custom Gizmo Overlay
        if self.scene.selected_object is not None:
            pivot = self._get_gizmo_pivot()
            if pivot is not None:
                self.gizmo.draw(pivot, self.camera, w_logical, h_logical)

        # 5. Composite Offscreen FBO Texture to Native Widget Canvas
        default_fbo = self.defaultFramebufferObject()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, default_fbo)
        gl.glViewport(0, 0, w_physical, h_physical)

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glUseProgram(self.screen_program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo_texture)
        gl.glUniform1i(gl.glGetUniformLocation(self.screen_program, "screenTexture"), 0)

        gl.glBindVertexArray(self.screen_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)

        # Dynamic Resolution Scaling Metrics
        frame_ms = (time.perf_counter() - t0) * 1000.0
        self._last_frame_time_ms = frame_ms
        if frame_ms > self.TARGET_FRAME_TIME_MS + 2.0:
            self.render_scale = max(self.MIN_RENDER_SCALE, self.render_scale - self.SCALE_STEP)
        elif frame_ms < self.TARGET_FRAME_TIME_MS - 4.0:
            self.render_scale = min(self.MAX_RENDER_SCALE, self.render_scale + self.SCALE_STEP)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._is_box_selecting:
            from PySide6.QtGui import QPainter, QColor, QPen
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            pen = QPen(QColor(0, 255, 156, 204), 1)
            painter.setPen(pen)
            
            brush = QColor(0, 255, 156, 15)
            painter.setBrush(brush)
            
            x1, y1 = self._box_select_start.x(), self._box_select_start.y()
            x2, y2 = self._box_select_end.x(), self._box_select_end.y()
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)
            painter.end()

    def resizeGL(self, w, h):
        self._fb_width = w
        self._fb_height = h
        print(f"[VP] resizeGL fb=({w},{h})")
        gl.glViewport(0, 0, w, h)

    def cleanupGL(self):
        self.gizmo.shutdown()
        self._cleanup_fbo()
        
        try:
            if self.object_program:
                gl.glDeleteProgram(self.object_program)
            if self.grid_program:
                gl.glDeleteProgram(self.grid_program)
            if self.screen_program:
                gl.glDeleteProgram(self.screen_program)
            if self.grid_vao:
                gl.glDeleteVertexArrays(1, [self.grid_vao])
                gl.glDeleteBuffers(1, [self.grid_vbo])
                gl.glDeleteBuffers(1, [self.grid_ebo])
            if self.screen_vao:
                gl.glDeleteVertexArrays(1, [self.screen_vao])
                gl.glDeleteBuffers(1, [self.screen_vbo])
        except Exception:
            pass
            
        for obj in self.scene.objects:
            obj.mesh.cleanup()

        # Reset input states

    def closeEvent(self, event):
        self.makeCurrent()
        self.cleanupGL()
        self.doneCurrent()
        super().closeEvent(event)

    # Drag and drop event handlers - forward to parent
    def dragEnterEvent(self, event: QDragEnterEvent):
        parent = self.parent()
        while parent:
            if hasattr(parent, "dragEnterEvent"):
                parent.dragEnterEvent(event)
                if event.isAccepted():
                    return
            parent = parent.parent()
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        parent = self.parent()
        while parent:
            if hasattr(parent, "dragMoveEvent"):
                parent.dragMoveEvent(event)
                if event.isAccepted():
                    return
            parent = parent.parent()
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        parent = self.parent()
        while parent:
            if hasattr(parent, "dragLeaveEvent"):
                parent.dragLeaveEvent(event)
                return
            parent = parent.parent()

    def dropEvent(self, event: QDropEvent):
        parent = self.parent()
        while parent:
            if hasattr(parent, "dropEvent"):
                parent.dropEvent(event)
                if event.isAccepted():
                    return
            parent = parent.parent()
        event.ignore()

    # Event handlers
    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        dx = pos.x() - self.last_mouse_pos.x()
        dy = pos.y() - self.last_mouse_pos.y()
        if self.invert_y:
            dy = -dy
        
        if self._gizmo_dragging:
            self.gizmo.update_drag(pos.x(), pos.y(), self._drag_pivot, self.camera, self.width(), self.height())
            
            # Propagate pivot modifications to all selected objects
            import trimesh
            if self.gizmo.operation == "translate":
                delta_pos = self._drag_pivot.position - self._drag_start_pivot_pos
                for obj in self.scene.selected_objects:
                    obj.position = self._drag_start_positions[obj] + delta_pos
            elif self.gizmo.operation == "scale":
                ratio = self._drag_pivot.scale / np.maximum(self._drag_start_pivot_scale, 1e-6)
                for obj in self.scene.selected_objects:
                    obj.scale = self._drag_start_scales[obj] * ratio
            elif self.gizmo.operation == "rotate":
                rx = np.radians(-self._drag_pivot.rotation[0])
                ry = np.radians(-self._drag_pivot.rotation[1])
                rz = np.radians(-self._drag_pivot.rotation[2])
                r_new_pivot = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
                
                # r_delta @ start_pivot_r_mat = r_new_pivot => r_delta = r_new_pivot @ inv(start_pivot_r_mat)
                r_delta = r_new_pivot @ np.linalg.inv(self._drag_start_pivot_r_mat)
                R_delta_3x3 = r_delta[:3, :3]
                
                for obj in self.scene.selected_objects:
                    # Rotate position around median pivot start pos
                    obj.position = self._drag_start_pivot_pos + R_delta_3x3 @ (self._drag_start_positions[obj] - self._drag_start_pivot_pos)
                    # Rotate orientation matrix
                    obj_r_new = r_delta @ self._drag_start_r_mats[obj]
                    ex, ey, ez = trimesh.transformations.euler_from_matrix(obj_r_new, 'sxyz')
                    obj.rotation = -np.degrees([ex, ey, ez]).astype(np.float32)
                    
            self.transform_changed.emit(self.scene.active_object)
            self.update()
        elif self._is_box_selecting:
            self._box_select_end = pos.toPoint()
            self.update()
        else:
            # Hover check for cursor change or highlight
            if self.scene.selected_object is not None:
                pivot = self._get_gizmo_pivot()
                if pivot is not None:
                    old_hover = self.gizmo.hovered_handle
                    self.gizmo.hovered_handle = self.gizmo.check_hover(pos.x(), pos.y(), pivot, self.camera, self.width(), self.height())
                    if self.gizmo.hovered_handle != old_hover:
                        self.update()
                    
            if self.is_orbiting:
                # Orbiting navigation (Middle click drag or Alt + LMB drag)
                try:
                    from preferences_dialog import load_preferences
                    invert = load_preferences().get("invert_mouse_rotation", True)
                except Exception:
                    invert = True
                pitch_mult = 0.005 if invert else -0.005
                self.camera.orbit(-dx * 0.005, dy * pitch_mult)
                self.notify_camera_changed()
                self.update()
            elif self.is_panning:
                # Panning navigation (Shift + Middle drag or Alt + Shift + LMB drag)
                self.camera.pan(dx, dy)
                self.notify_camera_changed()
                self.update()
            elif self.is_zooming:
                # Zooming navigation (Alt + Ctrl + LMB drag)
                zoom_speed = self.camera.distance * 0.005 if self.camera.is_perspective else self.camera.ortho_scale * 0.005
                self.camera.zoom(-dy * zoom_speed)
                self.notify_camera_changed()
                self.update()
                
        self.last_mouse_pos = pos.toPoint()
        super().mouseMoveEvent(event)
 
    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        self.last_mouse_pos = pos.toPoint()
        
        # 1. First check if mouse clicked the gizmo
        if self.scene.selected_object is not None and event.button() == Qt.MouseButton.LeftButton:
            pivot = self._get_gizmo_pivot()
            if pivot is not None:
                dragged = self.gizmo.begin_drag(pos.x(), pos.y(), pivot, self.camera, self.width(), self.height())
                if dragged:
                    self._gizmo_dragging = True
                    # Record start states for propagation
                    self._drag_pivot = pivot
                    self._drag_start_pivot_pos = np.copy(pivot.position)
                    self._drag_start_pivot_scale = np.copy(pivot.scale)
                    
                    import trimesh
                    rx = np.radians(-pivot.rotation[0])
                    ry = np.radians(-pivot.rotation[1])
                    rz = np.radians(-pivot.rotation[2])
                    self._drag_start_pivot_r_mat = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
                    
                    self._drag_start_positions = {obj: np.copy(obj.position) for obj in self.scene.selected_objects}
                    self._drag_start_rotations = {obj: np.copy(obj.rotation) for obj in self.scene.selected_objects}
                    self._drag_start_scales = {obj: np.copy(obj.scale) for obj in self.scene.selected_objects}
                    self._drag_start_r_mats = {}
                    for obj in self.scene.selected_objects:
                        orx = np.radians(-obj.rotation[0])
                        ory = np.radians(-obj.rotation[1])
                        orz = np.radians(-obj.rotation[2])
                        self._drag_start_r_mats[obj] = trimesh.transformations.euler_matrix(orx, ory, orz, 'sxyz')
                        
                    self.update()
                    super().mousePressEvent(event)
                    return
        
        # 2. Handle viewport cameras or selection if not captured
        # Check for Alt modifier for trackpad navigation fallbacks
        if event.modifiers() & Qt.KeyboardModifier.AltModifier and event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.is_zooming = True
            elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.is_panning = True
            else:
                self.is_orbiting = True
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Shift+MMB = Pan, MMB = Orbit
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.is_panning = True
            else:
                self.is_orbiting = True
        elif event.button() == Qt.MouseButton.LeftButton:
            # Raycasting selection against objects
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            old_selection = list(self.scene.selected_objects)
            hit_obj = self.scene.perform_picking(pos.x(), pos.y(), self.width(), self.height(), self.camera, shift_held=shift)
            
            if hit_obj is None and not shift:
                # Clicked empty space: start box selection
                self._is_box_selecting = True
                self._box_select_start = pos.toPoint()
                self._box_select_end = pos.toPoint()
            
            if list(self.scene.selected_objects) != old_selection:
                self.selection_changed.emit(self.scene.active_object)
            self.update()
                
        super().mousePressEvent(event)
 
    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._gizmo_dragging:
            self.gizmo.end_drag()
            self._gizmo_dragging = False
            if hasattr(self, '_drag_start_positions') and hasattr(self, '_drag_start_rotations') and hasattr(self, '_drag_start_scales'):
                states = {}
                for obj in self.scene.selected_objects:
                    if obj in self._drag_start_positions and obj in self._drag_start_rotations and obj in self._drag_start_scales:
                        pos_b = self._drag_start_positions[obj]
                        rot_b = self._drag_start_rotations[obj]
                        scale_b = self._drag_start_scales[obj]
                        pos_a = np.copy(obj.position)
                        rot_a = np.copy(obj.rotation)
                        scale_a = np.copy(obj.scale)
                        if not (np.array_equal(pos_b, pos_a) and np.array_equal(rot_b, rot_a) and np.array_equal(scale_b, scale_a)):
                            states[obj] = (pos_b, rot_b, scale_b, pos_a, rot_a, scale_a)
                if states:
                    self.transform_committed.emit((states, self.gizmo.operation))
            self.update()
            
        if self._is_box_selecting:
            self._is_box_selecting = False
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            old_selection = list(self.scene.selected_objects)
            self.scene.perform_box_picking(
                self._box_select_start.x(), self._box_select_start.y(),
                self._box_select_end.x(), self._box_select_end.y(),
                self.width(), self.height(), self.camera, shift_held=shift
            )
            if list(self.scene.selected_objects) != old_selection:
                self.selection_changed.emit(self.scene.active_object)
            self.update()

        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and (self.is_orbiting or self.is_panning or self.is_zooming)):
            self.is_orbiting = False
            self.is_panning = False
            self.is_zooming = False
            
        super().mouseReleaseEvent(event)
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        delta_y = event.angleDelta().y() / 120.0
        # Scale zoom speed based on distance
        zoom_speed = self.camera.distance * 0.1 if self.camera.is_perspective else self.camera.ortho_scale * 0.1
        self.camera.zoom(delta_y * zoom_speed)
        self.notify_camera_changed()
        self.update()
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        
        if ctrl and key == Qt.Key.Key_Z:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.redo_requested.emit()
            else:
                self.undo_requested.emit()
            event.accept()
            return
        elif ctrl and key == Qt.Key.Key_Y:
            self.redo_requested.emit()
            event.accept()
            return
        elif ctrl and key == Qt.Key.Key_A:
            # Select all
            self.scene.selected_objects = list(self.scene.objects)
            self.scene.active_object = self.scene.objects[-1] if self.scene.objects else None
            self.selection_changed.emit(self.scene.active_object)
            self.update()
            event.accept()
            return
        elif key == Qt.Key.Key_Escape:
            # Deselect all
            self.scene.selected_objects = []
            self.scene.active_object = None
            self.selection_changed.emit(None)
            self.update()
            event.accept()
            return
            
        if key == Qt.Key.Key_G:
            # G: Translate
            self.gizmo.operation = "translate"
            self.tool_changed.emit(self.gizmo.operation)
            self.update()
        elif key == Qt.Key.Key_R:
            # R: Rotate
            self.gizmo.operation = "rotate"
            self.tool_changed.emit(self.gizmo.operation)
            self.update()
        elif key == Qt.Key.Key_S:
            # S: Scale
            self.gizmo.operation = "scale"
            self.tool_changed.emit(self.gizmo.operation)
            self.update()
        elif key in [Qt.Key.Key_Delete, Qt.Key.Key_Backspace]:
            # Delete/Backspace: Remove Selected Object
            self.delete_pressed.emit()
            event.accept()
            return
        
        # Camera snap shortcuts (keys 1,3,5,7 and numpad counterparts)
        if key == Qt.Key.Key_1:
            self.camera.snap_to_view("back" if ctrl else "front")
            self.notify_camera_changed()
            self.update()
            event.accept()
            return
        elif key == Qt.Key.Key_3:
            self.camera.snap_to_view("left" if ctrl else "right")
            self.notify_camera_changed()
            self.update()
            event.accept()
            return
        elif key == Qt.Key.Key_7:
            self.camera.snap_to_view("bottom" if ctrl else "top")
            self.notify_camera_changed()
            self.update()
            event.accept()
            return
        elif key == Qt.Key.Key_5:
            self.camera.is_perspective = not self.camera.is_perspective
            self.notify_camera_changed()
            self.update()
            event.accept()
            return
        elif key == Qt.Key.Key_F:
            if self.scene.selected_object is not None:
                self.camera.frame_object(self.scene.selected_object)
                self.notify_camera_changed()
                self.update()
                event.accept()
                return
                
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        super().keyReleaseEvent(event)

    # Shaders compile/link helper
    def _compile_and_link_shaders(self, vert_src: str, frag_src: str) -> int:
        def compile_shader(shader_type, source):
            shader = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                log = gl.glGetShaderInfoLog(shader).decode('utf-8')
                raise RuntimeError(f"Shader compilation failed:\n{log}")
            return shader

        vert = compile_shader(gl.GL_VERTEX_SHADER, vert_src)
        frag = compile_shader(gl.GL_FRAGMENT_SHADER, frag_src)
        
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vert)
        gl.glAttachShader(program, frag)
        gl.glLinkProgram(program)
        
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            log = gl.glGetProgramInfoLog(program).decode('utf-8')
            raise RuntimeError(f"Shader linking failed:\n{log}")
            
        gl.glDeleteShader(vert)
        gl.glDeleteShader(frag)
        return program

    # Create full screen quad geometry in Z=0 plane for infinite grid unprojection
    def _setup_grid_quad(self):
        # Triangle strip vertices (bottom-left, bottom-right, top-left, top-right)
        vertices = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
             1.0,  1.0, 0.0
        ], dtype=np.float32)
        
        self.grid_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.grid_vao)
        
        self.grid_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.grid_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        
        gl.glBindVertexArray(0)
