import os
import sys
import subprocess
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QScrollArea, QWidget, QCheckBox, QFrame, QGroupBox, QTabWidget,
    QFileDialog, QMessageBox
)

from addon_manager import get_user_addons_dir, get_builtin_addons_dir


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
        
        # TAB 1: Add-ons
        self.addons_tab = QWidget()
        addons_layout = QVBoxLayout(self.addons_tab)
        addons_layout.setContentsMargins(12, 12, 12, 12)
        addons_layout.setSpacing(12)

        # Top Control bar in Addons tab
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

        # TAB 2: General Settings
        self.general_tab = QWidget()
        gen_layout = QVBoxLayout(self.general_tab)
        gen_layout.setContentsMargins(16, 16, 16, 16)
        gen_layout.setSpacing(16)

        grp_editor = QGroupBox("Mesh Editor Preferences", self.general_tab)
        grp_editor.setStyleSheet("QGroupBox { font-weight: bold; color: #00E676; border: 1px solid #333; border-radius: 4px; margin-top: 10px; padding-top: 15px; }")
        grp_layout = QVBoxLayout(grp_editor)

        chk_autosave = QCheckBox("Enable Undo/Redo History Persistence", grp_editor)
        chk_autosave.setChecked(True)
        chk_autosave.setStyleSheet("color: #FFFFFF;")
        grp_layout.addWidget(chk_autosave)

        chk_snapping = QCheckBox("Default Snapping Mode: Enabled", grp_editor)
        chk_snapping.setChecked(False)
        chk_snapping.setStyleSheet("color: #FFFFFF;")
        grp_layout.addWidget(chk_snapping)

        gen_layout.addWidget(grp_editor)
        gen_layout.addStretch()

        self.tabs.addTab(self.general_tab, "General")

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
