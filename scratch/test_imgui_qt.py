import sys
import time
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent

import OpenGL.GL as gl
from imgui_bundle import imgui

class TestOpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.last_time = time.time()
        self.imgui_ctx = None

    def initializeGL(self):
        print("initializeGL() start")
        # 1. Create ImGui Context
        self.imgui_ctx = imgui.create_context()
        imgui.set_current_context(self.imgui_ctx)
        
        # 2. Configure ImGui IO
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        
        # 3. Initialize ImGui OpenGL3 Backend
        success = imgui.backends.opengl3_init("#version 150")
        print(f"ImGui OpenGL3 Backend initialized: {success}")
        
        # Enable basic GL states
        gl.glClearColor(0.1, 0.1, 0.15, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        if not self.imgui_ctx:
            return
            
        imgui.set_current_context(self.imgui_ctx)
        
        # Feed time delta and display size
        io = imgui.get_io()
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        io.delta_time = max(dt, 0.0001)
        
        # Set display size to logical widget size
        w_logical = self.width()
        h_logical = self.height()
        io.display_size = (w_logical, h_logical)
        
        # Handle framebuffer scaling for High DPI
        dpr = self.devicePixelRatioF()
        io.display_framebuffer_scale = (dpr, dpr)
        
        # Start ImGui Frame
        imgui.backends.opengl3_new_frame()
        imgui.new_frame()
        
        # Create a simple ImGui window overlay
        imgui.set_next_window_pos((10, 10), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((320, 240), imgui.Cond_.first_use_ever)
        
        imgui.begin("ImGui + PySide6 Bridge Test")
        imgui.text("This window is rendered natively inside QOpenGLWidget.")
        imgui.text(f"Logical Size: {w_logical}x{h_logical}")
        imgui.text(f"Device Pixel Ratio: {dpr}")
        imgui.text(f"Framebuffer Scale: {io.display_framebuffer_scale.x}, {io.display_framebuffer_scale.y}")
        
        imgui.spacing()
        if imgui.button("Click Me!"):
            print(">>> IMGUI BUTTON CLICKED! <<<", flush=True)
            
        imgui.spacing()
        imgui.text(f"Mouse Pos: {io.mouse_pos.x:.1f}, {io.mouse_pos.y:.1f}")
        imgui.text(f"Want Capture Mouse: {io.want_capture_mouse}")
        imgui.text(f"Want Capture Keyboard: {io.want_capture_keyboard}")
        
        imgui.end()
        
        # Render ImGui draw lists
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)
        if self.imgui_ctx:
            imgui.set_current_context(self.imgui_ctx)
            io = imgui.get_io()
            # w, h are physical pixels, we divide by DPR to set logical display size
            dpr = self.devicePixelRatioF()
            io.display_size = (w / dpr, h / dpr)
            io.display_framebuffer_scale = (dpr, dpr)

    def cleanupGL(self):
        if self.imgui_ctx:
            print("Shutting down ImGui context and backend...")
            imgui.set_current_context(self.imgui_ctx)
            imgui.backends.opengl3_shutdown()
            imgui.destroy_context(self.imgui_ctx)
            self.imgui_ctx = None

    def closeEvent(self, event):
        self.makeCurrent()
        self.cleanupGL()
        self.doneCurrent()
        super().closeEvent(event)

    # Event Mapping
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.imgui_ctx:
            imgui.set_current_context(self.imgui_ctx)
            io = imgui.get_io()
            io.add_mouse_pos_event(event.position().x(), event.position().y())
            # print(f"Mouse Move to: {event.position().x()}, {event.position().y()}", flush=True)
        self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if self.imgui_ctx:
            imgui.set_current_context(self.imgui_ctx)
            io = imgui.get_io()
            # Set position first to ensure synchronization
            io.add_mouse_pos_event(event.position().x(), event.position().y())
            btn = self._map_qt_button(event.button())
            if btn is not None:
                io.add_mouse_button_event(btn, True)
                print(f"Mouse Press: btn={btn} at ({event.position().x()}, {event.position().y()})", flush=True)
        self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.imgui_ctx:
            imgui.set_current_context(self.imgui_ctx)
            io = imgui.get_io()
            # Set position first to ensure synchronization
            io.add_mouse_pos_event(event.position().x(), event.position().y())
            btn = self._map_qt_button(event.button())
            if btn is not None:
                io.add_mouse_button_event(btn, False)
                print(f"Mouse Release: btn={btn} at ({event.position().x()}, {event.position().y()})", flush=True)
        self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self.imgui_ctx:
            imgui.set_current_context(self.imgui_ctx)
            io = imgui.get_io()
            delta_x = event.angleDelta().x() / 120.0
            delta_y = event.angleDelta().y() / 120.0
            io.add_mouse_wheel_event(delta_x, delta_y)
        self.update()
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if self.imgui_ctx:
            imgui.set_current_context(self.imgui_ctx)
            io = imgui.get_io()
            key = self._map_qt_key(event.key())
            if key is not None:
                io.add_key_event(key, True)
            
            text = event.text()
            if text:
                for char in text:
                    if char.isprintable():
                        io.add_input_character(ord(char))
        self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if self.imgui_ctx:
            imgui.set_current_context(self.imgui_ctx)
            io = imgui.get_io()
            key = self._map_qt_key(event.key())
            if key is not None:
                io.add_key_event(key, False)
        self.update()
        super().keyReleaseEvent(event)

    def _map_qt_button(self, button):
        if button == Qt.LeftButton:
            return 0
        elif button == Qt.RightButton:
            return 1
        elif button == Qt.MiddleButton:
            return 2
        return None

    def _map_qt_key(self, key):
        mapping = {
            Qt.Key_Tab: imgui.Key.tab,
            Qt.Key_Left: imgui.Key.left_arrow,
            Qt.Key_Right: imgui.Key.right_arrow,
            Qt.Key_Up: imgui.Key.up_arrow,
            Qt.Key_Down: imgui.Key.down_arrow,
            Qt.Key_PageUp: imgui.Key.page_up,
            Qt.Key_PageDown: imgui.Key.page_down,
            Qt.Key_Home: imgui.Key.home,
            Qt.Key_End: imgui.Key.end,
            Qt.Key_Insert: imgui.Key.insert,
            Qt.Key_Delete: imgui.Key.delete,
            Qt.Key_Backspace: imgui.Key.backspace,
            Qt.Key_Space: imgui.Key.space,
            Qt.Key_Enter: imgui.Key.enter,
            Qt.Key_Return: imgui.Key.enter,
            Qt.Key_Escape: imgui.Key.escape,
            Qt.Key_Control: imgui.Key.left_ctrl,
            Qt.Key_Shift: imgui.Key.left_shift,
            Qt.Key_Alt: imgui.Key.left_alt,
            Qt.Key_A: imgui.Key.a,
            Qt.Key_B: imgui.Key.b,
            Qt.Key_C: imgui.Key.c,
            Qt.Key_D: imgui.Key.d,
            Qt.Key_E: imgui.Key.e,
            Qt.Key_F: imgui.Key.f,
            Qt.Key_G: imgui.Key.g,
            Qt.Key_H: imgui.Key.h,
            Qt.Key_I: imgui.Key.i,
            Qt.Key_J: imgui.Key.j,
            Qt.Key_K: imgui.Key.k,
            Qt.Key_L: imgui.Key.l,
            Qt.Key_M: imgui.Key.m,
            Qt.Key_N: imgui.Key.n,
            Qt.Key_O: imgui.Key.o,
            Qt.Key_P: imgui.Key.p,
            Qt.Key_Q: imgui.Key.q,
            Qt.Key_R: imgui.Key.r,
            Qt.Key_S: imgui.Key.s,
            Qt.Key_T: imgui.Key.t,
            Qt.Key_U: imgui.Key.u,
            Qt.Key_V: imgui.Key.v,
            Qt.Key_W: imgui.Key.w,
            Qt.Key_X: imgui.Key.x,
            Qt.Key_Y: imgui.Key.y,
            Qt.Key_Z: imgui.Key.z,
        }
        return mapping.get(key, None)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImGui + PySide6 Bridge Test")
        self.resize(800, 600)
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.gl_widget = TestOpenGLWidget(self)
        layout.addWidget(self.gl_widget)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.gl_widget.update)
        self.timer.start(16)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
