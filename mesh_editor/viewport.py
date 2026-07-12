import os
import time
import numpy as np
import pyrr
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Signal, QPoint, QRectF
from PySide6.QtGui import QMouseEvent, QKeyEvent, QWheelEvent, QAction

import OpenGL.GL as gl
from mesh_editor.scene import Camera, Scene, Object
from mesh_editor.gizmo import Gizmo

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
    delete_pressed = Signal() # Emits when Delete/Backspace key is pressed
    tool_changed = Signal(object) # Emits active gizmo OPERATION when changed by hotkey
    camera_changed = Signal(object) # Emits Camera when it transforms/snaps

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        
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
        
        # Grid quad VAO
        self.grid_vao = None
        self.grid_vbo = None
        self.grid_ebo = None

        # Navigation Gizmo Overlay Widget
        from mesh_editor.nav_gizmo import NavGizmoWidget
        self.nav_gizmo = NavGizmoWidget(self)
        self.nav_gizmo.snap_requested.connect(self._on_nav_gizmo_snap_requested)
        # Sync initial orientation
        self.nav_gizmo.update_orientation(self.camera.yaw, self.camera.pitch)

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
        print("[Viewport] Initializing OpenGL...")
        
        # Load Shaders from disk
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
        
        # Setup infinite grid full-screen quad VAO
        self._setup_grid_quad()
        
        # Initialize custom gizmo inside this GL context
        self.gizmo.initialize(shaders_dir)
        
        # Scene starts empty on startup. Primitives/meshes can be imported using the Import button.
        pass
        
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

    def paintGL(self):
        # 1. Clear background
        gl.glClearColor(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Ensure clean opaque state before rendering objects
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        
        # Retrieve physical vs logical sizes
        dpr = self.devicePixelRatioF()
        w_logical = self.width()
        h_logical = self.height()
        w_physical = int(w_logical * dpr)
        h_physical = int(h_logical * dpr)
        
        gl.glViewport(0, 0, w_physical, h_physical)
        
        aspect = w_logical / max(h_logical, 1.0)
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix(aspect)
        
        # 2. Render Scene Objects
        gl.glUseProgram(self.object_program)
        
        # Setup global lighting direction (directional light moving with the camera eye)
        eye = self.camera.get_position()
        light_dir = eye - self.camera.target
        # Normalize direction
        norm_light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Pass camera and lighting uniforms
        gl.glUniform3fv(gl.glGetUniformLocation(self.object_program, "lightDir"), 1, norm_light_dir)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.object_program, "view"), 1, gl.GL_FALSE, view_matrix)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.object_program, "proj"), 1, gl.GL_FALSE, proj_matrix)
        
        # Render each object
        for obj in self.scene.objects:
            model_matrix = obj.get_model_matrix()
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.object_program, "model"), 1, gl.GL_FALSE, model_matrix)
            
            # Object base color (greyish blue)
            color = np.array([0.45, 0.55, 0.65], dtype=np.float32)
            gl.glUniform3fv(gl.glGetUniformLocation(self.object_program, "objectColor"), 1, color)
            gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "useOverrideColor"), 0)
            
            # Pass texture uniforms
            has_tex = obj.mesh.texture_id is not None
            gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "useTexture"), 1 if has_tex else 0)
            gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "textureSampler"), 0)
            
            # Draw standard geometry
            obj.mesh.draw()
            
            # Draw wireframe selection highlight if selected
            is_selected = obj in self.scene.selected_objects
            is_active = self.scene.active_object == obj
            if is_selected:
                gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glPolygonOffset(-1.5, -1.5) # Shift depth forward slightly to prevent z-fighting
                
                # Draw thick wireframe outlines (Blender selection style)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glLineWidth(2.0)
                
                gl.glUniform1i(gl.glGetUniformLocation(self.object_program, "useOverrideColor"), 1)
                if is_active:
                    highlight_color = np.array([0.0, 1.0, 0.61, 1.0], dtype=np.float32) # Bright green
                else:
                    highlight_color = np.array([0.1, 0.42, 0.27, 1.0], dtype=np.float32) # Muted teal-green
                gl.glUniform4fv(gl.glGetUniformLocation(self.object_program, "overrideColor"), 1, highlight_color)
                
                obj.mesh.draw()
                
                # Restore defaults
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glLineWidth(1.0)
                
        # 3. Render Infinite Grid (Semi-transparent overlay)
        gl.glUseProgram(self.grid_program)
        
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # View matrices
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_program, "view"), 1, gl.GL_FALSE, view_matrix)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_program, "proj"), 1, gl.GL_FALSE, proj_matrix)
        # Inverse view/projection matrices for screen-space unprojection
        inv_view = pyrr.matrix44.inverse(view_matrix)
        inv_proj = pyrr.matrix44.inverse(proj_matrix)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_program, "invView"), 1, gl.GL_FALSE, inv_view)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_program, "invProj"), 1, gl.GL_FALSE, inv_proj)
        
        # Grid parameters
        gl.glUniform4fv(gl.glGetUniformLocation(self.grid_program, "gridColor1"), 1, self.grid_color_1)
        gl.glUniform4fv(gl.glGetUniformLocation(self.grid_program, "gridColor2"), 1, self.grid_color_2)
        gl.glUniform4fv(gl.glGetUniformLocation(self.grid_program, "axisColorX"), 1, self.axis_color_x)
        gl.glUniform4fv(gl.glGetUniformLocation(self.grid_program, "axisColorY"), 1, self.axis_color_y)
        gl.glUniform1f(gl.glGetUniformLocation(self.grid_program, "gridFade"), self.grid_fade)
        gl.glUniform1f(gl.glGetUniformLocation(self.grid_program, "gridThickness"), self.grid_thickness)
        gl.glUniform1f(gl.glGetUniformLocation(self.grid_program, "subdivisions"), self.grid_subdivisions)
        
        # Draw full screen quad
        gl.glBindVertexArray(self.grid_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        
        # 4. Render Custom Gizmo Overlay
        if self.scene.selected_object is not None:
            pivot = self._get_gizmo_pivot()
            if pivot is not None:
                self.gizmo.draw(pivot, self.camera, w_logical, h_logical)

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
        gl.glViewport(0, 0, w, h)

    def cleanupGL(self):
        # Shut down custom gizmo
        self.gizmo.shutdown()
        
        # Clean up shaders and VAOs
        try:
            if self.object_program:
                gl.glDeleteProgram(self.object_program)
            if self.grid_program:
                gl.glDeleteProgram(self.grid_program)
            if self.grid_vao:
                gl.glDeleteVertexArrays(1, [self.grid_vao])
                gl.glDeleteBuffers(1, [self.grid_vbo])
                gl.glDeleteBuffers(1, [self.grid_ebo])
        except Exception:
            pass
            
        # Clean up scene mesh VBOs/VAOs
        for obj in self.scene.objects:
            obj.mesh.cleanup()

        # Reset input states

    def closeEvent(self, event):
        self.makeCurrent()
        self.cleanupGL()
        self.doneCurrent()
        super().closeEvent(event)

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
                self.camera.orbit(-dx * 0.005, -dy * 0.005)
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
        
        if ctrl and key == Qt.Key.Key_A:
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
        vertices = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
             1.0,  1.0, 0.0,
            -1.0,  1.0, 0.0
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype=np.uint32)
        
        self.grid_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.grid_vao)
        
        self.grid_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.grid_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        
        self.grid_ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.grid_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        
        gl.glBindVertexArray(0)
