import os
import sys
import ctypes

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Fix taskbar icon grouping on Windows
    if sys.platform == 'win32':
        myappid = 'proximaxr.proximap.photogrammetry.1.0'
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    from PySide6.QtWidgets import QApplication, QSplashScreen
    from PySide6.QtCore import Qt, QThread, Signal, QObject
    from PySide6.QtGui import QIcon, QPixmap, QSurfaceFormat

    # Configure default surface format before QApplication
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # Resolve app icon path
    base_dir = get_base_dir()
    icon_path = os.path.join(base_dir, "app_icon.ico")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(base_dir, "public", "app_icon.png")

    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Show splash screen immediately
    splash = None
    splash_path = os.path.join(base_dir, "public", "app_icon.png")
    if not os.path.exists(splash_path):
        splash_path = icon_path

    if os.path.exists(splash_path):
        pixmap = QPixmap(splash_path)
        scaled_pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(scaled_pixmap)
        splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        splash.show()
        splash.showMessage("Starting Proximap...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        app.processEvents()

    class BackgroundLoader(QThread):
        loaded = Signal(object)
        progress = Signal(str)

        def run(self):
            if splash:
                self.progress.emit("Initializing hardware profile...")
            import hardware_profiler
            hardware_profiler.initialize()

            if splash:
                self.progress.emit("Loading application core...")
            import main_window as mw
            self.loaded.emit(mw)

    class AppManager(QObject):
        def __init__(self, splash, icon_path):
            super().__init__()
            self.splash = splash
            self.icon_path = icon_path
            self.loader = BackgroundLoader()
            self.loader.loaded.connect(self.on_loaded)
            self.loader.progress.connect(self.on_progress)
            self.window = None

        def start(self):
            self.loader.start()

        def on_progress(self, msg):
            if self.splash:
                self.splash.showMessage(msg, Qt.AlignBottom | Qt.AlignCenter, Qt.white)

        def on_loaded(self, mw):
            if self.splash:
                self.splash.showMessage("Launching user interface...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
            self.window = mw.MainWindow()
            if os.path.exists(self.icon_path):
                self.window.setWindowIcon(QIcon(self.icon_path))
            self.window.show()
            if self.splash:
                self.splash.finish(self.window)

    manager = AppManager(splash, icon_path)
    manager.start()

    sys.exit(app.exec())
