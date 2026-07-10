import time
import numpy as np
from imgui_bundle import imgui, imguizmo
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QKeyEvent, QWheelEvent

class ImGuiBridge:
    """Manages the manual ImGui context, event mapping from Qt, and ImGuizmo updates."""
    def __init__(self):
        self.imgui_ctx = None
        self.last_time = time.time()
        
        # Gizmo settings
        self.current_operation = imguizmo.im_guizmo.OPERATION.translate
        self.current_mode = imguizmo.im_guizmo.MODE.local
        self.use_snapping = False
        self.snap_translation = 0.5   # 0.5 units
        self.snap_rotation = 15.0      # 15 degrees
        self.snap_scale = 0.1         # 10% increments

    def initialize(self, glsl_version="#version 150") -> bool:
        # Create context
        self.imgui_ctx = imgui.create_context()
        imgui.set_current_context(self.imgui_ctx)
        
        # Setup config
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        
        # Initialize OpenGL3 Backend
        success = imgui.backends.opengl3_init(glsl_version)
        print(f"[ImGuiBridge] OpenGL3 backend initialized: {success}")
        return success

    def shutdown(self):
        if self.imgui_ctx:
            print("[ImGuiBridge] Shutting down ImGui context and backend...")
            imgui.set_current_context(self.imgui_ctx)
            imgui.backends.opengl3_shutdown()
            imgui.destroy_context(self.imgui_ctx)
            self.imgui_ctx = None

    def new_frame(self, w_logical: float, h_logical: float, dpr: float):
        if not self.imgui_ctx:
            return
        imgui.set_current_context(self.imgui_ctx)
        
        io = imgui.get_io()
        
        # Update time delta
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        io.delta_time = max(dt, 0.0001)
        
        # Update sizes
        io.display_size = (w_logical, h_logical)
        io.display_framebuffer_scale = (dpr, dpr)
        
        # Begin Frame
        imgui.backends.opengl3_new_frame()
        imgui.new_frame()
        imguizmo.im_guizmo.begin_frame()

    def render(self):
        if not self.imgui_ctx:
            return
        imgui.set_current_context(self.imgui_ctx)
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def reset_input_state(self):
        """Resets pressed keys and mouse states in ImGui to prevent stuck inputs on tab-switch."""
        if self.imgui_ctx:
            imgui.set_current_context(self.imgui_ctx)
            io = imgui.get_io()
            io.clear_input_mouse()
            io.clear_events_queue()

    def want_capture(self) -> bool:
        """Returns True if ImGui or ImGuizmo wants to capture the mouse input."""
        if not self.imgui_ctx:
            return False
        imgui.set_current_context(self.imgui_ctx)
        io = imgui.get_io()
        return io.want_capture_mouse or imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()

    # Event forwards
    def process_mouse_move(self, x: float, y: float) -> bool:
        if not self.imgui_ctx:
            return False
        imgui.set_current_context(self.imgui_ctx)
        io = imgui.get_io()
        io.add_mouse_pos_event(x, y)
        return self.want_capture()

    def process_mouse_button(self, button: Qt.MouseButton, is_pressed: bool, x: float, y: float) -> bool:
        if not self.imgui_ctx:
            return False
        imgui.set_current_context(self.imgui_ctx)
        io = imgui.get_io()
        
        io.add_mouse_pos_event(x, y)
        btn_idx = self._map_qt_button(button)
        if btn_idx is not None:
            io.add_mouse_button_event(btn_idx, is_pressed)
            
        return self.want_capture()

    def process_wheel(self, delta_x: float, delta_y: float) -> bool:
        if not self.imgui_ctx:
            return False
        imgui.set_current_context(self.imgui_ctx)
        io = imgui.get_io()
        io.add_mouse_wheel_event(delta_x, delta_y)
        return self.want_capture()

    def process_key(self, key_val: Qt.Key, is_pressed: bool, text: str = "") -> bool:
        if not self.imgui_ctx:
            return False
        imgui.set_current_context(self.imgui_ctx)
        io = imgui.get_io()
        
        key = self._map_qt_key(key_val)
        if key is not None:
            io.add_key_event(key, is_pressed)
            
        if is_pressed and text:
            for char in text:
                if char.isprintable():
                    io.add_input_character(ord(char))
                    
        return io.want_capture_keyboard

    def draw_gizmo(self, obj, camera, width_logical: float, height_logical: float) -> bool:
        """
        Draws the ImGuizmo at the target object's position/rotation/scale.
        Modifies coordinates in-place if manipulated.
        """
        if not self.imgui_ctx or obj is None:
            return False
            
        imgui.set_current_context(self.imgui_ctx)
        
        # 1. Configure rect bounds in logical screen coordinates
        imguizmo.im_guizmo.set_rect(0, 0, width_logical, height_logical)
        imguizmo.im_guizmo.set_orthographic(not camera.is_perspective)
        
        # 2. Get Camera view/projection matrices (row-major flat arrays of 16 floats)
        aspect = width_logical / max(height_logical, 1.0)
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect)
        
        imgui_view = imguizmo.im_guizmo.Matrix16(list(view_mat.flatten()))
        imgui_proj = imguizmo.im_guizmo.Matrix16(list(proj_mat.flatten()))
        
        # 3. Get Object model matrix and transpose to column-major for ImGuizmo
        model_mat = obj.get_model_matrix()
        col_major = model_mat.T
        imgui_model = imguizmo.im_guizmo.Matrix16(list(col_major.flatten()))
        
        # 4. Handle snapping
        snap_val = None
        if self.use_snapping:
            if self.current_operation == imguizmo.im_guizmo.OPERATION.translate:
                snap_val = imguizmo.im_guizmo.Matrix3([self.snap_translation] * 3)
            elif self.current_operation == imguizmo.im_guizmo.OPERATION.rotate:
                snap_val = imguizmo.im_guizmo.Matrix3([self.snap_rotation] * 3)
            elif self.current_operation == imguizmo.im_guizmo.OPERATION.scale:
                snap_val = imguizmo.im_guizmo.Matrix3([self.snap_scale] * 3)
                
        # 5. Render & manipulate the gizmo
        modified = imguizmo.im_guizmo.manipulate(
            imgui_view,
            imgui_proj,
            self.current_operation,
            self.current_mode,
            imgui_model,
            None,
            snap_val
        )
        
        if modified:
            # 6. Decompose back to components using trimesh (avoids coordinate system mismatch & Euler singularities)
            import trimesh
            modified_col_major = np.array(list(imgui_model.values), dtype=np.float32).reshape(4, 4)
            scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(modified_col_major)
            
            obj.position = np.array(trans, dtype=np.float32)
            obj.rotation = np.degrees(-angles).astype(np.float32)
            obj.scale = np.array(scale, dtype=np.float32)
            
        return modified

    def _map_qt_button(self, button):
        if button == Qt.MouseButton.LeftButton:
            return 0
        elif button == Qt.MouseButton.RightButton:
            return 1
        elif button == Qt.MouseButton.MiddleButton:
            return 2
        return None

    def _map_qt_key(self, key):
        mapping = {
            Qt.Key.Key_Tab: imgui.Key.tab,
            Qt.Key.Key_Left: imgui.Key.left_arrow,
            Qt.Key.Key_Right: imgui.Key.right_arrow,
            Qt.Key.Key_Up: imgui.Key.up_arrow,
            Qt.Key.Key_Down: imgui.Key.down_arrow,
            Qt.Key.Key_PageUp: imgui.Key.page_up,
            Qt.Key.Key_PageDown: imgui.Key.page_down,
            Qt.Key.Key_Home: imgui.Key.home,
            Qt.Key.Key_End: imgui.Key.end,
            Qt.Key.Key_Insert: imgui.Key.insert,
            Qt.Key.Key_Delete: imgui.Key.delete,
            Qt.Key.Key_Backspace: imgui.Key.backspace,
            Qt.Key.Key_Space: imgui.Key.space,
            Qt.Key.Key_Enter: imgui.Key.enter,
            Qt.Key.Key_Return: imgui.Key.enter,
            Qt.Key.Key_Escape: imgui.Key.escape,
            Qt.Key.Key_Control: imgui.Key.left_ctrl,
            Qt.Key.Key_Shift: imgui.Key.left_shift,
            Qt.Key.Key_Alt: imgui.Key.left_alt,
            Qt.Key.Key_A: imgui.Key.a,
            Qt.Key.Key_B: imgui.Key.b,
            Qt.Key.Key_C: imgui.Key.c,
            Qt.Key.Key_D: imgui.Key.d,
            Qt.Key.Key_E: imgui.Key.e,
            Qt.Key.Key_F: imgui.Key.f,
            Qt.Key.Key_G: imgui.Key.g,
            Qt.Key.Key_H: imgui.Key.h,
            Qt.Key.Key_I: imgui.Key.i,
            Qt.Key.Key_J: imgui.Key.j,
            Qt.Key.Key_K: imgui.Key.k,
            Qt.Key.Key_L: imgui.Key.l,
            Qt.Key.Key_M: imgui.Key.m,
            Qt.Key.Key_N: imgui.Key.n,
            Qt.Key.Key_O: imgui.Key.o,
            Qt.Key.Key_P: imgui.Key.p,
            Qt.Key.Key_Q: imgui.Key.q,
            Qt.Key.Key_R: imgui.Key.r,
            Qt.Key.Key_S: imgui.Key.s,
            Qt.Key.Key_T: imgui.Key.t,
            Qt.Key.Key_U: imgui.Key.u,
            Qt.Key.Key_V: imgui.Key.v,
            Qt.Key.Key_W: imgui.Key.w,
            Qt.Key.Key_X: imgui.Key.x,
            Qt.Key.Key_Y: imgui.Key.y,
            Qt.Key.Key_Z: imgui.Key.z,
        }
        return mapping.get(key, None)
