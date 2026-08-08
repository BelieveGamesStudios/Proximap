"""
pg_chrome.py - Top Chrome Bar & Segmented Control components for Polyground Mesh Editor.
Provides a modern, Blender-inspired chrome header bar with modular segmented controls.
"""

import sys
import os
import webbrowser
from PySide6.QtCore import Qt, Signal, QSize, QUrl
from PySide6.QtGui import QColor, QFont, QAction, QIcon, QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, 
    QComboBox, QToolButton, QMenu, QFrame, QSizePolicy, QDialog
)


class SegmentedControl(QWidget):
    """
    A modern pill-shaped segmented button control for mode/tab switching.
    """
    index_changed = Signal(int)
    segment_selected = Signal(str)

    def __init__(self, items: list[str], default_index: int = 0, small: bool = False, parent=None):
        super().__init__(parent)
        self.items = items
        self._current_index = default_index
        self._small = small
        self._buttons: list[QPushButton] = []
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Outer container styling
        height = 24 if self._small else 28
        self.setFixedHeight(height)
        self.setStyleSheet("""
            SegmentedControl {
                background-color: #1A1A1A;
                border: 1px solid #2B2B2B;
                border-radius: 4px;
            }
        """)

        font_size = "10px" if self._small else "12px"
        padding = "0 8px" if self._small else "0 14px"

        for i, text in enumerate(self.items):
            btn = QPushButton(text, self)
            btn.setCheckable(True)
            btn.setChecked(i == self._current_index)
            btn.setFixedHeight(height - 4)
            btn.setCursor(Qt.PointingHandCursor)
            
            # Styling for buttons
            self._update_btn_style(btn, i == self._current_index, font_size, padding)
            
            # Connect click
            btn.clicked.connect(lambda checked, idx=i: self.set_current_index(idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

    def _update_btn_style(self, btn: QPushButton, is_active: bool, font_size: str, padding: str):
        if is_active:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #00E676;
                    color: #121212;
                    border: none;
                    border-radius: 3px;
                    font-family: 'Inter', sans-serif;
                    font-size: {font_size};
                    font-weight: bold;
                    padding: {padding};
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: #9E9E9E;
                    border: none;
                    border-radius: 3px;
                    font-family: 'Inter', sans-serif;
                    font-size: {font_size};
                    font-weight: normal;
                    padding: {padding};
                }}
                QPushButton:hover {{
                    background-color: #262626;
                    color: #E0E0E0;
                }}
            """)

    def set_current_index(self, index: int):
        if 0 <= index < len(self._buttons) and index != self._current_index:
            self._current_index = index
            font_size = "10px" if self._small else "12px"
            padding = "0 8px" if self._small else "0 14px"
            
            for i, btn in enumerate(self._buttons):
                is_active = (i == index)
                btn.setChecked(is_active)
                self._update_btn_style(btn, is_active, font_size, padding)

            self.index_changed.emit(index)
            self.segment_selected.emit(self.items[index])

    def current_index(self) -> int:
        return self._current_index

    def current_text(self) -> str:
        return self.items[self._current_index]


class SaveStateIndicator(QWidget):
    """Small dot indicator showing save state (emerald = saved, amber = unsaved)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self._saved = True
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.dot = QLabel(self)
        self.dot.setFixedSize(8, 8)
        layout.addWidget(self.dot, 0, Qt.AlignCenter)
        self.set_saved(True)

    def set_saved(self, saved: bool):
        self._saved = saved
        color = "#00E676" if saved else "#FFB74D"
        tooltip = "Session Saved" if saved else "Unsaved Changes"
        self.dot.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                border-radius: 4px;
            }}
        """)
        self.setToolTip(tooltip)


class PolygroundChromeBar(QFrame):
    """
    Top application chrome bar spanning the top of the main window.
    Features 3-region architecture (left / center / right):
    - Left: App Menu button (☰ Proximap) + QMenuBar (File / Edit / Window / Help)
    - Center (Mathematically anchored midpoint): Segmented controls (Polyground | 3D Reconstruction, Object | Edit | Sculpt)
    - Right: Layout Preset Dropdown, Save State Indicator & Profile Icon
    """
    tab_switch_requested = Signal(int)
    mode_changed = Signal(str)
    layout_preset_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PolygroundChromeBar")
        self.setFixedHeight(40)
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            #PolygroundChromeBar {
                background-color: #161616;
                border-bottom: 1px solid #2A2A2A;
            }
        """)

        # ---------------------------------------------------------------------
        # 1. Left Region: Proximap Logo (About only) + Mode Notch + QMenuBar
        # ---------------------------------------------------------------------
        self.left_region = QWidget(self)
        left_layout = QHBoxLayout(self.left_region)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Proximap Logo Button (face.png icon)
        self.logo_btn = QToolButton(self.left_region)
        self.logo_btn.setText("")  # Icon only, no spelled out text
        self.logo_btn.setPopupMode(QToolButton.InstantPopup)
        self.logo_btn.setCursor(Qt.PointingHandCursor)
        self.logo_btn.setToolTip("Proximap")
        self.logo_btn.setFixedSize(30, 28)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        icon_path = os.path.join(base_dir, "public", "face.png")
        if os.path.exists(icon_path):
            self.logo_btn.setIcon(QIcon(icon_path))
            self.logo_btn.setIconSize(QSize(22, 22))

        self.logo_btn.setStyleSheet("""
            QToolButton {
                background-color: #1A1A1A;
                border: 1px solid #00E676;
                border-radius: 6px;
                padding: 2px;
            }
            QToolButton:hover {
                background-color: #262626;
                border-color: #00FF88;
            }
            QToolButton::menu-indicator {
                image: none;
            }
        """)

        # Dropdown menu for Logo Button: Contains ONLY "About Proximap"
        self.logo_menu = QMenu(self)
        self.logo_menu.setStyleSheet("""
            QMenu {
                background-color: #1E1E1E;
                color: #E0E0E0;
                border: 1px solid #333333;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 3px;
                font-weight: bold;
            }
            QMenu::item:selected {
                background-color: #00E676;
                color: #121212;
            }
        """)
        about_action = self.logo_menu.addAction("About Proximap")
        about_action.triggered.connect(self._show_about_dialog)
        self.logo_btn.setMenu(self.logo_menu)

        # Backward compatibility reference
        self.app_menu_btn = self.logo_btn

        # Top Menu Bar (File / Edit / Window / Help)
        self.menu_bar = QMenuBar(self.left_region)
        self.menu_bar.setFixedHeight(28)

        # Set green custom logo icon
        logo_pix = QPixmap(18, 18)
        logo_pix.fill(Qt.transparent)
        p = QPainter(logo_pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#00E676"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 4, 18, 10, 5, 5)
        p.end()
        self.app_menu_btn.setIcon(QIcon(logo_pix))

        # App Logo Popup Menu
        self.app_menu = QMenu(self.app_menu_btn)
        self.app_menu.setStyleSheet("""
            QMenu {
                background-color: #1E1E1E;
                color: #CCCCCC;
                border: 1px solid #333333;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #2D2D2D;
                color: #00E676;
            }
        """)

        a_about = self.app_menu.addAction("About Proximap")
        a_about.triggered.connect(self._show_about_dialog)
        self.app_menu.addSeparator()

        a_pref = self.app_menu.addAction("Preferences…")
        a_pref.setShortcut("Ctrl+,")
        win = self.window()
        if win and hasattr(win, "_open_preferences_dialog"):
            a_pref.triggered.connect(win._open_preferences_dialog)
        else:
            a_pref.triggered.connect(lambda: print("[CHROME] Preferences clicked"))

        self.app_menu.addSeparator()
        a_quit = self.app_menu.addAction("Quit Proximap")
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(QApplication.instance().quit)

        self.app_menu_btn.setMenu(self.app_menu)
        left_layout.addWidget(self.app_menu_btn)

        # Application QMenuBar (File, Edit, Window, Help)
        self.menu_bar = QMenuBar(self.left_region)
        self.menu_bar.setFixedHeight(28)
        self.menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: transparent;
                color: #CCCCCC;
                font-size: 12px;
                spacing: 6px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QMenuBar::item:selected {
                background-color: #262626;
                color: #FFFFFF;
            }
            QMenu {
                background-color: #1E1E1E;
                color: #CCCCCC;
                border: 1px solid #333333;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #2D2D2D;
                color: #00E676;
            }
        """)

        self.file_menu = self.menu_bar.addMenu("File")
        self.edit_menu = self.menu_bar.addMenu("Edit")
        self.window_menu = self.menu_bar.addMenu("Window")
        self.help_menu = self.menu_bar.addMenu("Help")

        left_layout.addWidget(self.menu_bar)


        # ---------------------------------------------------------------------
        # 2. Center Region: Navigation Tabs
        # ---------------------------------------------------------------------
        self.center_region = QWidget(self)
        center_layout = QHBoxLayout(self.center_region)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        self.tab_segmented = SegmentedControl(["3D Reconstruction", "Polyground"], default_index=default_idx, parent=self.center_region)
        self.tab_segmented.index_changed.connect(self.tab_switch_requested.emit)
        center_layout.addWidget(self.tab_segmented)

        self.mode_segmented = self.mode_notch


        # ---------------------------------------------------------------------
        # 3. Right Region: Layout Presets
        # ---------------------------------------------------------------------
        self.right_region = QWidget(self)
        right_layout = QHBoxLayout(self.right_region)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        preset_label = QLabel("Layout:", self.right_region)
        preset_label.setStyleSheet("color: #888888; font-size: 11px;")
        right_layout.addWidget(preset_label)

        self.preset_combo = QComboBox(self.right_region)
        self.preset_combo.setFixedHeight(26)
        self.preset_combo.setCursor(Qt.PointingHandCursor)
        self.preset_combo.setStyleSheet("""
            QComboBox {
                background-color: #222222;
                color: #CCCCCC;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
                min-width: 90px;
            }
            QComboBox:hover {
                border-color: #444444;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E1E;
                color: #CCCCCC;
                selection-background-color: #00E676;
                selection-color: #121212;
            }
        """)
        self.preset_combo.activated.connect(self._on_preset_activated)
        right_layout.addWidget(self.preset_combo)
        self.save_indicator = None

        self.populate_layout_presets(["Default"], "Default")

        if default_idx == 0:
            self.right_region.setVisible(False)

        self._update_region_geometries()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_region_geometries()

    def _update_region_geometries(self):
        """Calculates exact positions to anchor center_region to top bar midpoint."""
        self.left_region.adjustSize()
        if self.right_region.isVisible():
            self.right_region.adjustSize()
        self.center_region.adjustSize()

        h = self.height()

        # Left region anchored left (10px margin)
        left_h = self.left_region.height()
        self.left_region.move(10, (h - left_h) // 2)

        # Right region anchored right (10px margin)
        right_w = self.right_region.width() if self.right_region.isVisible() else 0
        right_h = self.right_region.height() if self.right_region.isVisible() else 0
        right_x = max(0, self.width() - right_w - 10)
        if self.right_region.isVisible():
            self.right_region.move(right_x, (h - right_h) // 2)

        # Center region mathematically anchored to horizontal midpoint of window
        center_w = self.center_region.width()
        center_h = self.center_region.height()
        center_x = (self.width() - center_w) // 2

        # Prevent overlapping when squeezed
        left_bound = 10 + self.left_region.width() + 10
        right_bound = right_x - 10 if self.right_region.isVisible() else self.width() - 10
        if center_x < left_bound:
            center_x = left_bound
        elif center_x + center_w > right_bound:
            center_x = max(left_bound, right_bound - center_w)

        self.center_region.move(center_x, (h - center_h) // 2)

    def _show_about_dialog(self):
        """Redirects to Proximap GitHub repository link."""
        url = "https://github.com/BelieveGamesStudios/Proximap"
        if not QDesktopServices.openUrl(QUrl(url)):
            webbrowser.open(url)

    @property
    def mode_notch(self):
        """Dynamic reference to the Object Mode notch widget inside ViewportDockContainer."""
        win = self.window()
        if win and hasattr(win, "polyground_workspace") and win.polyground_workspace:
            vp = getattr(win.polyground_workspace, "viewport_container", None)
            if vp and hasattr(vp, "mode_notch"):
                return vp.mode_notch
        return None

    def attach_native_mac_menu(self, window):
        """macOS compatibility shim to replicate app menu on native menu bar if running on Darwin."""
        if sys.platform == "darwin" and window:
            from PySide6.QtWidgets import QMenuBar
            native_bar = window.menuBar()
            if native_bar:
                native_bar.clear()
                for menu in [self.file_menu, self.edit_menu, self.window_menu, self.help_menu]:
                    native_bar.addMenu(menu)


