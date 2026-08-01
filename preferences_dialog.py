import os
import sys
import json
import subprocess
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QScrollArea, QWidget, QCheckBox, QFrame, QGroupBox, QTabWidget,
    QFileDialog, QMessageBox, QComboBox, QColorDialog
)

from addon_manager import get_user_addons_dir, get_builtin_addons_dir

PREFERENCES_FILE = os.path.join(os.path.expanduser("~"), ".proximap", "preferences.json")


def load_preferences() -> dict:
    defaults = {
        "startup_tab": "3D Reconstruction",
        "autosave_history": True,
        "default_snapping": False,
        "camera_mode": "Arcball Camera",
        "invert_mouse_rotation": True,
        "show_controls": True,
        "reconstruction_bg_color": "#1A1A1A",
        "polyground_bg_color": "#1E1E1E",
        "polyground_grid_color": "#333333"
    }
    if os.path.exists(PREFERENCES_FILE):
        try:
            with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                defaults.update(data)
        except Exception as e:
            print(f"[WARNING] Failed to load preferences: {e}")
    return defaults


def save_preferences(data: dict):
    try:
        os.makedirs(os.path.dirname(PREFERENCES_FILE), exist_ok=True)
        current = load_preferences()
        current.update(data)
        with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to save preferences: {e}")


class AddonItemWidget(QFrame):
    def __init__(self, addon_class, addon_manager, main_window, parent=None):
        super().__init__(parent)
        self.addon_class = addon_class
        self.addon_id = getattr(addon_class, "addon_id", "")
        self.addon_manager = addon_manager
        self.main_window = main_window

        self.setStyleSheet("""
            QFrame {
                background-color: #212121;
                border: 1px solid #2C2C2C;
                border-radius: 6px;
                margin-bottom: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header row: Checkbox, Name, Version, Category badge
        header_layout = QHBoxLayout()

        self.chk_enable = QCheckBox()
        self.chk_enable.setChecked(self.addon_manager.is_enabled(self.addon_id))
        self.chk_enable.toggled.connect(self._on_toggled)

        name_str = getattr(addon_class, "addon_name", self.addon_id)
        version_str = getattr(addon_class, "addon_version", "1.0.0")
        author_str = getattr(addon_class, "addon_author", "Unknown")
        category_str = getattr(addon_class, "addon_category", "General")
        desc_str = getattr(addon_class, "addon_description", "")
        deps_str = getattr(addon_class, "dependencies", [])

        lbl_title = QLabel(f"<b>{name_str}</b>  <span style='color: #888888; font-size: 11px;'>v{version_str}</span>")
        lbl_title.setStyleSheet("border: none; color: #FFFFFF; font-size: 13px;")

        lbl_category = QLabel(category_str)
        lbl_category.setStyleSheet("""
            background-color: #162B20;
            color: #00E676;
            border: 1px solid #00E676;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: bold;
        """)

        header_layout.addWidget(self.chk_enable)
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(lbl_category)

        layout.addLayout(header_layout)

        # Description
        lbl_desc = QLabel(desc_str)
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #AAAAAA; font-size: 11px; border: none; margin-left: 24px;")
        layout.addWidget(lbl_desc)

        # Meta & Dependencies row
        meta_layout = QHBoxLayout()
        meta_layout.setContentsMargins(24, 0, 0, 0)

        lbl_author = QLabel(f"Author: {author_str}")
        lbl_author.setStyleSheet("color: #666666; font-size: 10px; border: none;")
        meta_layout.addWidget(lbl_author)

        if deps_str:
            dep_status = []
            for dep in deps_str:
                try:
                    __import__(dep)
                    dep_status.append(f"<span style='color: #00E676;'>✓ {dep}</span>")
                except ImportError:
                    dep_status.append(f"<span style='color: #FF5252;'>✗ {dep}</span>")
            lbl_deps = QLabel("Dependencies: " + ", ".join(dep_status))
            lbl_deps.setStyleSheet("font-size: 10px; border: none; margin-left: 12px;")
            meta_layout.addWidget(lbl_deps)

        meta_layout.addStretch()
        layout.addLayout(meta_layout)

    def _on_toggled(self, checked: bool):
        if checked:
            success = self.addon_manager.enable_addon(self.addon_id)
            if not success:
                self.chk_enable.blockSignals(True)
                self.chk_enable.setChecked(False)
                self.chk_enable.blockSignals(False)
        else:
            self.addon_manager.disable_addon(self.addon_id)

        if self.main_window and hasattr(self.main_window, "update_addon_panels"):
            self.main_window.update_addon_panels()


class PreferencesDialog(QDialog):
    def __init__(self, addon_manager, main_window, parent=None):
        super().__init__(parent)
        self.addon_manager = addon_manager
        self.main_window = main_window
        self.setWindowTitle("Proximap Preferences")
        self.resize(760, 560)
        self.prefs = load_preferences()

        self.setStyleSheet("""
            QDialog {
                background-color: #1A1A1A;
                color: #FFFFFF;
            }
            QTabWidget::pane {
                border: 1px solid #333333;
                background-color: #141414;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #242424;
                color: #AAAAAA;
                border: 1px solid #333333;
                padding: 8px 16px;
                font-weight: bold;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #141414;
                color: #00E676;
                border-bottom-color: #141414;
            }
            QTabBar::tab:hover {
                color: #FFFFFF;
            }
            QLineEdit {
                background-color: #242424;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #00E676;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header Title
        title_lbl = QLabel("Proximap Preferences")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #00E676;")
        main_layout.addWidget(title_lbl)

        # Tabbed Widget Layout
        self.tabs = QTabWidget(self)

        # ---------------------------------------------------------------------
        # TAB 1: General
        # ---------------------------------------------------------------------
        self.general_tab = QWidget()
        gen_layout = QVBoxLayout(self.general_tab)
        gen_layout.setContentsMargins(16, 16, 16, 16)
        gen_layout.setSpacing(16)

        grp_rec_gen = QGroupBox("3D Reconstruction Settings", self.general_tab)
        grp_rec_gen.setStyleSheet("QGroupBox { font-weight: bold; color: #00E676; border: 1px solid #333; border-radius: 4px; margin-top: 10px; padding-top: 15px; }")
        rec_gen_layout = QHBoxLayout(grp_rec_gen)
        rec_gen_layout.setContentsMargins(12, 12, 12, 12)
        rec_gen_layout.setSpacing(12)

        lbl_startup = QLabel("Default Startup Workspace:", grp_rec_gen)
        lbl_startup.setStyleSheet("color: #FFFFFF; font-size: 13px; font-weight: normal;")

        self.combo_startup = QComboBox(grp_rec_gen)
        self.combo_startup.addItems(["3D Reconstruction", "Polyground"])
        self.combo_startup.setFixedHeight(30)
        self.combo_startup.setCursor(Qt.PointingHandCursor)
        self.combo_startup.setCurrentText(self.prefs.get("startup_tab", "3D Reconstruction"))
        self.combo_startup.setStyleSheet("""
            QComboBox {
                background-color: #242424;
                color: #FFFFFF;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px 10px;
                min-width: 160px;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E1E;
                color: #FFFFFF;
                selection-background-color: #00E676;
                selection-color: #121212;
            }
        """)
        self.combo_startup.currentTextChanged.connect(self._on_startup_preference_changed)

        rec_gen_layout.addWidget(lbl_startup)
        rec_gen_layout.addWidget(self.combo_startup)
        rec_gen_layout.addStretch()

        gen_layout.addWidget(grp_rec_gen)

        grp_pg_gen = QGroupBox("Polyground Settings", self.general_tab)
        grp_pg_gen.setStyleSheet("QGroupBox { font-weight: bold; color: #00E676; border: 1px solid #333; border-radius: 4px; margin-top: 10px; padding-top: 15px; }")
        pg_gen_layout = QVBoxLayout(grp_pg_gen)
        pg_gen_layout.setContentsMargins(12, 12, 12, 12)
        pg_gen_layout.setSpacing(10)

        chk_autosave = QCheckBox("Enable Undo/Redo History Persistence", grp_pg_gen)
        chk_autosave.setChecked(self.prefs.get("autosave_history", True))
        chk_autosave.setStyleSheet("color: #FFFFFF;")
        chk_autosave.toggled.connect(lambda val: save_preferences({"autosave_history": val}))
        pg_gen_layout.addWidget(chk_autosave)

        chk_snapping = QCheckBox("Default Snapping Mode: Enabled", grp_pg_gen)
        chk_snapping.setChecked(self.prefs.get("default_snapping", False))
        chk_snapping.setStyleSheet("color: #FFFFFF;")
        chk_snapping.toggled.connect(lambda val: save_preferences({"default_snapping": val}))
        pg_gen_layout.addWidget(chk_snapping)

        gen_layout.addWidget(grp_pg_gen)
        gen_layout.addStretch()

        self.tabs.addTab(self.general_tab, "General")

        # ---------------------------------------------------------------------
        # TAB 2: Navigation
        # ---------------------------------------------------------------------
        self.navigation_tab = QWidget()
        nav_layout = QVBoxLayout(self.navigation_tab)
        nav_layout.setContentsMargins(16, 16, 16, 16)
        nav_layout.setSpacing(16)

        # 3D Reconstruction Navigation Group Box
        grp_rec_nav = QGroupBox("3D Reconstruction Navigation", self.navigation_tab)
        grp_rec_nav.setStyleSheet("QGroupBox { font-weight: bold; color: #00E676; border: 1px solid #333; border-radius: 4px; margin-top: 10px; padding-top: 15px; }")
        rec_nav_layout = QVBoxLayout(grp_rec_nav)
        rec_nav_layout.setContentsMargins(12, 12, 12, 12)
        rec_nav_layout.setSpacing(12)

        row_cam = QHBoxLayout()
        lbl_cam = QLabel("Camera Tracking Style:", grp_rec_nav)
        lbl_cam.setStyleSheet("color: #FFFFFF; font-size: 13px;")

        self.combo_cam = QComboBox(grp_rec_nav)
        self.combo_cam.addItems(["Arcball Camera", "Turntable Camera"])
        self.combo_cam.setFixedHeight(30)
        self.combo_cam.setCursor(Qt.PointingHandCursor)
        self.combo_cam.setCurrentText(self.prefs.get("camera_mode", "Arcball Camera"))
        self.combo_cam.setStyleSheet(self.combo_startup.styleSheet())
        self.combo_cam.currentTextChanged.connect(self._on_camera_mode_changed)

        row_cam.addWidget(lbl_cam)
        row_cam.addWidget(self.combo_cam)
        row_cam.addStretch()
        rec_nav_layout.addLayout(row_cam)

        self.chk_invert_rot = QCheckBox("Invert Mouse Y-Rotation", grp_rec_nav)
        self.chk_invert_rot.setChecked(self.prefs.get("invert_mouse_rotation", True))
        self.chk_invert_rot.setStyleSheet("color: #FFFFFF;")
        self.chk_invert_rot.toggled.connect(self._on_invert_rotation_changed)
        rec_nav_layout.addWidget(self.chk_invert_rot)

        self.chk_show_controls = QCheckBox("Show Viewport Navigation Controls Overlay", grp_rec_nav)
        self.chk_show_controls.setChecked(self.prefs.get("show_controls", True))
        self.chk_show_controls.setStyleSheet("color: #FFFFFF;")
        self.chk_show_controls.toggled.connect(self._on_show_controls_changed)
        rec_nav_layout.addWidget(self.chk_show_controls)

        nav_layout.addWidget(grp_rec_nav)

        # Polyground Navigation Group Box
        grp_pg_nav = QGroupBox("Polyground Navigation", self.navigation_tab)
        grp_pg_nav.setStyleSheet("QGroupBox { font-weight: bold; color: #00E676; border: 1px solid #333; border-radius: 4px; margin-top: 10px; padding-top: 15px; }")
        pg_nav_layout = QVBoxLayout(grp_pg_nav)
        pg_nav_layout.setContentsMargins(12, 12, 12, 12)
        pg_nav_layout.setSpacing(12)

        self.chk_pg_invert_rot = QCheckBox("Invert Polyground Mouse Orbit Rotation", grp_pg_nav)
        self.chk_pg_invert_rot.setChecked(self.prefs.get("invert_mouse_rotation", True))
        self.chk_pg_invert_rot.setStyleSheet("color: #FFFFFF;")
        self.chk_pg_invert_rot.toggled.connect(self._on_invert_rotation_changed)
        pg_nav_layout.addWidget(self.chk_pg_invert_rot)

        nav_layout.addWidget(grp_pg_nav)
        nav_layout.addStretch()

        self.tabs.addTab(self.navigation_tab, "Navigation")

        # ---------------------------------------------------------------------
        # TAB 3: Themes
        # ---------------------------------------------------------------------
        self.themes_tab = QWidget()
        themes_layout = QVBoxLayout(self.themes_tab)
        themes_layout.setContentsMargins(16, 16, 16, 16)
        themes_layout.setSpacing(16)

        # 3D Reconstruction Theme Group Box
        grp_rec_theme = QGroupBox("3D Reconstruction Theme", self.themes_tab)
        grp_rec_theme.setStyleSheet("QGroupBox { font-weight: bold; color: #00E676; border: 1px solid #333; border-radius: 4px; margin-top: 10px; padding-top: 15px; }")
        rec_theme_layout = QVBoxLayout(grp_rec_theme)
        rec_theme_layout.setContentsMargins(12, 12, 12, 12)
        rec_theme_layout.setSpacing(12)

        rec_bg = self.prefs.get("reconstruction_bg_color", "#1A1A1A")
        row_rec_bg = QHBoxLayout()
        lbl_rec_bg = QLabel("3D Reconstruction Background:", grp_rec_theme)
        lbl_rec_bg.setStyleSheet("color: #FFFFFF; font-size: 13px;")

        self.btn_rec_bg = QPushButton(rec_bg, grp_rec_theme)
        self.btn_rec_bg.setStyleSheet(f"background-color: {rec_bg}; color: #FFFFFF; font-weight: bold; border: 1px solid #555;")
        self.btn_rec_bg.clicked.connect(self._choose_rec_bg_color)

        row_rec_bg.addWidget(lbl_rec_bg)
        row_rec_bg.addWidget(self.btn_rec_bg)
        row_rec_bg.addStretch()
        rec_theme_layout.addLayout(row_rec_bg)

        themes_layout.addWidget(grp_rec_theme)

        # Polyground Theme Group Box
        grp_pg_theme = QGroupBox("Polyground Theme", self.themes_tab)
        grp_pg_theme.setStyleSheet("QGroupBox { font-weight: bold; color: #00E676; border: 1px solid #333; border-radius: 4px; margin-top: 10px; padding-top: 15px; }")
        pg_theme_layout = QVBoxLayout(grp_pg_theme)
        pg_theme_layout.setContentsMargins(12, 12, 12, 12)
        pg_theme_layout.setSpacing(12)

        pg_bg = self.prefs.get("polyground_bg_color", "#1E1E1E")
        row_pg_bg = QHBoxLayout()
        lbl_pg_bg = QLabel("Polyground Background:", grp_pg_theme)
        lbl_pg_bg.setStyleSheet("color: #FFFFFF; font-size: 13px;")

        self.btn_pg_bg = QPushButton(pg_bg, grp_pg_theme)
        self.btn_pg_bg.setStyleSheet(f"background-color: {pg_bg}; color: #FFFFFF; font-weight: bold; border: 1px solid #555;")
        self.btn_pg_bg.clicked.connect(self._choose_pg_bg_color)

        row_pg_bg.addWidget(lbl_pg_bg)
        row_pg_bg.addWidget(self.btn_pg_bg)
        row_pg_bg.addStretch()
        pg_theme_layout.addLayout(row_pg_bg)

        pg_grid = self.prefs.get("polyground_grid_color", "#333333")
        row_pg_grid = QHBoxLayout()
        lbl_pg_grid = QLabel("Polyground Grid Color:", grp_pg_theme)
        lbl_pg_grid.setStyleSheet("color: #FFFFFF; font-size: 13px;")

        self.btn_pg_grid = QPushButton(pg_grid, grp_pg_theme)
        self.btn_pg_grid.setStyleSheet(f"background-color: {pg_grid}; color: #FFFFFF; font-weight: bold; border: 1px solid #555;")
        self.btn_pg_grid.clicked.connect(self._choose_pg_grid_color)

        row_pg_grid.addWidget(lbl_pg_grid)
        row_pg_grid.addWidget(self.btn_pg_grid)
        row_pg_grid.addStretch()
        pg_theme_layout.addLayout(row_pg_grid)

        themes_layout.addWidget(grp_pg_theme)
        themes_layout.addStretch()

        self.tabs.addTab(self.themes_tab, "Themes")

        # ---------------------------------------------------------------------
        # TAB 4: Add-ons
        # ---------------------------------------------------------------------
        self.addons_tab = QWidget()
        addons_layout = QVBoxLayout(self.addons_tab)
        addons_layout.setContentsMargins(12, 12, 12, 12)
        addons_layout.setSpacing(12)

        ctrl_layout = QHBoxLayout()

        self.search_box = QLineEdit(self.addons_tab)
        self.search_box.setPlaceholderText("Search add-ons...")
        self.search_box.textChanged.connect(self._filter_addons)
        ctrl_layout.addWidget(self.search_box)

        btn_install_addon = QPushButton("Install Add-on...", self.addons_tab)
        btn_install_addon.setStyleSheet("""
            QPushButton {
                background-color: #00E676;
                color: #121212;
                font-weight: bold;
                border: 1px solid #00E676;
            }
            QPushButton:hover {
                background-color: #00C853;
            }
        """)
        btn_install_addon.setToolTip("Install an add-on from a .zip package or .py file")
        btn_install_addon.clicked.connect(self._install_addon_clicked)
        ctrl_layout.addWidget(btn_install_addon)

        btn_open_folder = QPushButton("Open Add-ons Folder", self.addons_tab)
        btn_open_folder.setToolTip("Open user add-ons directory (~/.proximap/addons)")
        btn_open_folder.clicked.connect(self._open_user_addons_folder)
        ctrl_layout.addWidget(btn_open_folder)

        btn_refresh = QPushButton("Refresh List", self.addons_tab)
        btn_refresh.clicked.connect(self.populate_addons)
        ctrl_layout.addWidget(btn_refresh)

        addons_layout.addLayout(ctrl_layout)

        # Scroll Area for Addons
        self.scroll_area = QScrollArea(self.addons_tab)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: 1px solid #2B2B2B; background-color: #141414; }")

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(8, 8, 8, 8)
        self.scroll_layout.setAlignment(Qt.AlignTop)

        self.scroll_area.setWidget(self.scroll_widget)
        addons_layout.addWidget(self.scroll_area)

        # Add-ons Tab Footer info
        user_dir = get_user_addons_dir()
        lbl_dir = QLabel(f"User Add-ons Directory: {user_dir}", self.addons_tab)
        lbl_dir.setStyleSheet("color: #666666; font-size: 11px;")
        addons_layout.addWidget(lbl_dir)

        self.tabs.addTab(self.addons_tab, "Add-ons")

        main_layout.addWidget(self.tabs)

        # Bottom Close action
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.accept)
        footer_layout.addWidget(btn_close)

        main_layout.addLayout(footer_layout)

        self.addon_widgets = []
        self.populate_addons()

    def populate_addons(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.addon_widgets.clear()
        self.addon_manager.refresh_discovery()

        addons = self.addon_manager.discovered_addons.values()
        if not addons:
            lbl_empty = QLabel("No add-ons found.")
            lbl_empty.setStyleSheet("color: #888888; font-size: 13px; padding: 20px;")
            self.scroll_layout.addWidget(lbl_empty)
            return

        for addon_cls in addons:
            item_widget = AddonItemWidget(addon_cls, self.addon_manager, self.main_window, self.scroll_widget)
            self.scroll_layout.addWidget(item_widget)
            self.addon_widgets.append(item_widget)

    def _install_addon_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Install Add-on Package or Script",
            "",
            "Proximap Add-on Package (*.zip *.py)"
        )
        if not file_path:
            return

        success, message = self.addon_manager.install_addon_from_file(file_path)
        if success:
            QMessageBox.information(self, "Add-on Installation", message)
            self.populate_addons()
            if self.main_window and hasattr(self.main_window, "update_addon_panels"):
                self.main_window.update_addon_panels()
        else:
            QMessageBox.warning(self, "Add-on Installation Failed", message)

    def _filter_addons(self, text: str):
        query = text.lower().strip()
        for item_widget in self.addon_widgets:
            name = getattr(item_widget.addon_class, "addon_name", "").lower()
            desc = getattr(item_widget.addon_class, "addon_description", "").lower()
            cat = getattr(item_widget.addon_class, "addon_category", "").lower()
            match = query in name or query in desc or query in cat
            item_widget.setVisible(match)

    def _open_user_addons_folder(self):
        path = get_user_addons_dir()
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _on_startup_preference_changed(self, text: str):
        save_preferences({"startup_tab": text})

    def _on_camera_mode_changed(self, text: str):
        save_preferences({"camera_mode": text})
        main_win = self.main_window
        if main_win:
            idx = 0 if text == "Arcball Camera" else 1
            if hasattr(main_win, "viewer_widget") and hasattr(main_win.viewer_widget, "cam_select"):
                main_win.viewer_widget.cam_select.setCurrentIndex(idx)
            if hasattr(main_win, "_on_camera_changed"):
                main_win._on_camera_changed(idx)

    def _on_invert_rotation_changed(self, checked: bool):
        save_preferences({"invert_mouse_rotation": checked})
        if hasattr(self, "chk_invert_rot"):
            self.chk_invert_rot.blockSignals(True)
            self.chk_invert_rot.setChecked(checked)
            self.chk_invert_rot.blockSignals(False)
        if hasattr(self, "chk_pg_invert_rot"):
            self.chk_pg_invert_rot.blockSignals(True)
            self.chk_pg_invert_rot.setChecked(checked)
            self.chk_pg_invert_rot.blockSignals(False)
        main_win = self.main_window
        if main_win and hasattr(main_win, "view") and main_win.view and hasattr(main_win.view, "camera") and hasattr(main_win.view.camera, "flip"):
            main_win.view.camera.flip = (False, checked, False)

    def _on_show_controls_changed(self, checked: bool):
        save_preferences({"show_controls": checked})
        main_win = self.main_window
        if main_win:
            if hasattr(main_win, "viewer_widget") and hasattr(main_win.viewer_widget, "show_controls_cb"):
                main_win.viewer_widget.show_controls_cb.blockSignals(True)
                main_win.viewer_widget.show_controls_cb.setChecked(checked)
                main_win.viewer_widget.show_controls_cb.blockSignals(False)
            if hasattr(main_win, "overlay_label") and main_win.overlay_label:
                main_win.overlay_label.setVisible(checked)
                if checked and hasattr(main_win, "_update_overlay_content"):
                    main_win._update_overlay_content()
                    if hasattr(main_win, "_position_overlay"):
                        main_win._position_overlay()

    def _choose_rec_bg_color(self):
        curr = self.prefs.get("reconstruction_bg_color", "#1A1A1A")
        color = QColorDialog.getColor(curr, self, "3D Reconstruction Background Color")
        if color.isValid():
            hex_c = color.name()
            self.btn_rec_bg.setText(hex_c)
            self.btn_rec_bg.setStyleSheet(f"background-color: {hex_c}; color: #FFFFFF; font-weight: bold; border: 1px solid #555;")
            save_preferences({"reconstruction_bg_color": hex_c})
            if self.main_window and hasattr(self.main_window, "viewport_bg_color"):
                self.main_window.viewport_bg_color = hex_c
                if hasattr(self.main_window, "canvas") and self.main_window.canvas:
                    self.main_window.canvas.bgcolor = hex_c
                    self.main_window.canvas.update()

    def _choose_pg_bg_color(self):
        curr = self.prefs.get("polyground_bg_color", "#1E1E1E")
        color = QColorDialog.getColor(curr, self, "Polyground Background Color")
        if color.isValid():
            hex_c = color.name()
            self.btn_pg_bg.setText(hex_c)
            self.btn_pg_bg.setStyleSheet(f"background-color: {hex_c}; color: #FFFFFF; font-weight: bold; border: 1px solid #555;")
            save_preferences({"polyground_bg_color": hex_c})
            if self.main_window and hasattr(self.main_window, "polyground_workspace") and self.main_window.polyground_workspace:
                vp = getattr(self.main_window.polyground_workspace, "viewport", None)
                if vp and hasattr(vp, "bg_color"):
                    r, g, b = color.redF(), color.greenF(), color.blueF()
                    import numpy as np
                    vp.bg_color = np.array([r, g, b], dtype=np.float32)
                    vp.update()

    def _choose_pg_grid_color(self):
        curr = self.prefs.get("polyground_grid_color", "#333333")
        color = QColorDialog.getColor(curr, self, "Polyground Grid Color")
        if color.isValid():
            hex_c = color.name()
            self.btn_pg_grid.setText(hex_c)
            self.btn_pg_grid.setStyleSheet(f"background-color: {hex_c}; color: #FFFFFF; font-weight: bold; border: 1px solid #555;")
            save_preferences({"polyground_grid_color": hex_c})
            if self.main_window and hasattr(self.main_window, "polyground_workspace") and self.main_window.polyground_workspace:
                vp = getattr(self.main_window.polyground_workspace, "viewport", None)
                if vp and hasattr(vp, "grid_color_1"):
                    r, g, b = color.redF(), color.greenF(), color.blueF()
                    import numpy as np
                    vp.grid_color_1 = np.array([r, g, b, 1.0], dtype=np.float32)
                    vp.update()
