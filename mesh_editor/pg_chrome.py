"""
pg_chrome.py - Top Chrome Bar & Segmented Control components for Polyground Mesh Editor.
Provides a modern, Blender-inspired chrome header bar with modular segmented controls.
"""

import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QAction
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

        # Proximap Logo Button
        self.logo_btn = QToolButton(self.left_region)
        self.logo_btn.setText(" ✦ Proximap ")
        self.logo_btn.setPopupMode(QToolButton.InstantPopup)
        self.logo_btn.setCursor(Qt.PointingHandCursor)
        self.logo_btn.setStyleSheet("""
            QToolButton {
                background-color: #1A1A1A;
                color: #00E676;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #00E676;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QToolButton:hover {
                background-color: #242424;
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

        # Mode Dropdown Notch directly under/beside logo button
        self.mode_notch = QComboBox(self.left_region)
        self.mode_notch.addItems(["Object Mode", "Edit Mode", "Sculpt Mode"])
        self.mode_notch.setFixedHeight(26)
        self.mode_notch.setCursor(Qt.PointingHandCursor)
        self.mode_notch.setStyleSheet("""
            QComboBox {
                background-color: #222222;
                color: #00E676;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 2px 8px;
                min-width: 96px;
            }
            QComboBox:hover {
                border-color: #00E676;
                background-color: #2A2A2A;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E1E;
                color: #CCCCCC;
                selection-background-color: #00E676;
                selection-color: #121212;
            }
        """)
        self.mode_notch.currentTextChanged.connect(lambda txt: self.mode_changed.emit(txt.replace(" Mode", "")))

        # Top Menu Bar (File / Edit / Window / Help)
        from PySide6.QtWidgets import QMenuBar
        self.menu_bar = QMenuBar(self.left_region)
        self.menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: transparent;
                color: #CCCCCC;
                font-family: 'Inter', sans-serif;
                font-size: 12px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 10px;
                border-radius: 3px;
            }
            QMenuBar::item:selected {
                background-color: #2A2A2A;
                color: #00E676;
            }
            QMenu {
                background-color: #1E1E1E;
                color: #E0E0E0;
                border: 1px solid #333333;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #00E676;
                color: #121212;
            }
            QMenu::separator {
                height: 1px;
                background-color: #333333;
                margin: 4px 0px;
            }
        """)

        # Menus
        self.file_menu = self.menu_bar.addMenu("File")
        self.edit_menu = self.menu_bar.addMenu("Edit")
        self.window_menu = self.menu_bar.addMenu("Window")
        self.help_menu = self.menu_bar.addMenu("Help")

        left_layout.addWidget(self.logo_btn)
        left_layout.addWidget(self.mode_notch)
        left_layout.addWidget(self.menu_bar)

        # ---------------------------------------------------------------------
        # 2. Center Region: Sole Midpoint Anchored Control (Polyground | 3D Rec)
        # ---------------------------------------------------------------------
        self.center_region = QWidget(self)
        center_layout = QHBoxLayout(self.center_region)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # Primary Segmented Toggle: Polyground | 3D Reconstruction
        self.tab_segmented = SegmentedControl(["Polyground", "3D Reconstruction"], default_index=0, parent=self.center_region)
        self.tab_segmented.index_changed.connect(self.tab_switch_requested.emit)
        center_layout.addWidget(self.tab_segmented)

        # Backward compatibility reference
        self.mode_segmented = self.mode_notch


        # ---------------------------------------------------------------------
        # 3. Right Region: Layout Presets, Save Indicator, Profile
        # ---------------------------------------------------------------------
        self.right_region = QWidget(self)
        right_layout = QHBoxLayout(self.right_region)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        preset_label = QLabel("Layout:", self.right_region)
        preset_label.setStyleSheet("color: #888888; font-size: 11px;")
        right_layout.addWidget(preset_label)

        self.preset_combo = QComboBox(self.right_region)
        self.preset_combo.addItems(["Default", "Sculpt", "Custom…"])
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
                min-width: 80px;
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
        self.preset_combo.currentTextChanged.connect(self.layout_preset_changed.emit)
        right_layout.addWidget(self.preset_combo)

        # Save State Indicator
        self.save_indicator = SaveStateIndicator(self.right_region)
        right_layout.addWidget(self.save_indicator)

        # Profile Icon
        profile_icon = QLabel(" P ", self.right_region)
        profile_icon.setFixedSize(24, 24)
        profile_icon.setAlignment(Qt.AlignCenter)
        profile_icon.setStyleSheet("""
            QLabel {
                background-color: #2B2B2B;
                color: #00E676;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #00E676;
                border-radius: 12px;
            }
        """)
        profile_icon.setToolTip("ProximaXR Profile")
        right_layout.addWidget(profile_icon)

        # Initial layout update
        self._update_region_geometries()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_region_geometries()

    def _update_region_geometries(self):
        """Calculates exact positions to anchor center_region to top bar midpoint."""
        self.left_region.adjustSize()
        self.right_region.adjustSize()
        self.center_region.adjustSize()

        h = self.height()

        # Left region anchored left (10px margin)
        left_h = self.left_region.height()
        self.left_region.move(10, (h - left_h) // 2)

        # Right region anchored right (10px margin)
        right_w = self.right_region.width()
        right_h = self.right_region.height()
        right_x = max(0, self.width() - right_w - 10)
        self.right_region.move(right_x, (h - right_h) // 2)

        # Center region mathematically anchored to horizontal midpoint of window
        center_w = self.center_region.width()
        center_h = self.center_region.height()
        center_x = (self.width() - center_w) // 2

        # Prevent overlapping when squeezed
        left_bound = 10 + self.left_region.width() + 10
        right_bound = right_x - 10
        if center_x < left_bound:
            center_x = left_bound
        elif center_x + center_w > right_bound:
            center_x = max(left_bound, right_bound - center_w)

        self.center_region.move(center_x, (h - center_h) // 2)

    def _show_about_dialog(self):
        """Displays modal About Proximap dialog window."""
        dialog = QDialog(self.window())
        dialog.setWindowTitle("About Proximap")
        dialog.setFixedSize(380, 240)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1A1A1A;
                color: #E0E0E0;
                border: 1px solid #333333;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        brand = QLabel("✦ PROXIMAP", dialog)
        brand.setStyleSheet("font-size: 18px; font-weight: bold; color: #00E676; letter-spacing: 1px;")

        org = QLabel("ProximaXR Spatial Technologies", dialog)
        org.setStyleSheet("font-size: 13px; font-weight: bold; color: #FFFFFF;")

        desc = QLabel("v1.0.0 — Spatial Computing & 3D Mesh Editor Suite\nHigh-Performance Reconstruction & Photogrammetry Workbench.", dialog)
        desc.setStyleSheet("font-size: 11px; color: #AAAAAA;")
        desc.setWordWrap(True)

        contact = QLabel("Contact: fumz@proximaxr.space", dialog)
        contact.setStyleSheet("font-size: 11px; color: #00E676;")

        btn_close = QPushButton("Close", dialog)
        btn_close.setFixedSize(90, 30)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #00E676;
                color: #121212;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #00C853;
            }
        """)
        btn_close.clicked.connect(dialog.accept)

        layout.addWidget(brand)
        layout.addWidget(org)
        layout.addWidget(desc)
        layout.addWidget(contact)
        layout.addStretch()
        layout.addWidget(btn_close, 0, Qt.AlignRight)

        dialog.exec()

    def attach_native_mac_menu(self, window):
        """macOS compatibility shim to replicate app menu on native menu bar if running on Darwin."""
        if sys.platform == "darwin" and window:
            from PySide6.QtWidgets import QMenuBar
            native_bar = window.menuBar()
            if native_bar:
                native_bar.clear()
                for menu in [self.file_menu, self.edit_menu, self.window_menu, self.help_menu]:
                    native_bar.addMenu(menu)


