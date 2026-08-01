"""
pg_panels.py - Modular panel components for the Polyground Mesh Editor docking workspace.
Contains OutlinerPanel, PropertiesPanel, StatsChip, and ToolShelfWidget.
"""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QScrollArea, QListWidget, QSizePolicy, QToolButton
)


class OutlinerPanel(QWidget):
    """Outliner panel content displaying scene object hierarchy."""
    def __init__(self, outliner_list: QListWidget, btn_delete: QPushButton, parent=None):
        super().__init__(parent)
        self.outliner_list = outliner_list
        self.btn_delete = btn_delete
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header
        header = QLabel("SCENE OUTLINER", self)
        header.setStyleSheet("color: #888888; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(header)

        # Add list widget
        self.outliner_list.setStyleSheet("""
            QListWidget {
                background-color: #141414;
                color: #E0E0E0;
                border: 1px solid #282828;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px 10px;
                border-radius: 3px;
                margin-bottom: 2px;
            }
            QListWidget::item:selected {
                background-color: #1E3A2B;
                color: #00E676;
                border: 1px solid #00E676;
            }
            QListWidget::item:hover:!selected {
                background-color: #222222;
            }
        """)
        layout.addWidget(self.outliner_list)

        # Delete button at bottom
        if self.btn_delete:
            self.btn_delete.setStyleSheet("""
                QPushButton {
                    background-color: #2A1A1A;
                    color: #FF5252;
                    border: 1px solid #5A2A2A;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #3A1A1A;
                    border-color: #FF5252;
                }
                QPushButton:disabled {
                    background-color: #181818;
                    color: #555555;
                    border-color: #262626;
                }
            """)
            layout.addWidget(self.btn_delete)


class PropertiesPanel(QWidget):
    """Properties panel content displaying transform spinboxes and add-on tools."""
    def __init__(self, properties_box: QWidget, addon_container: QWidget, parent=None):
        super().__init__(parent)
        self.properties_box = properties_box
        self.addon_container = addon_container
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # Transform Properties Container
        if self.properties_box:
            self.properties_box.setStyleSheet("""
                QWidget {
                    background-color: #1A1A1A;
                    border: 1px solid #282828;
                    border-radius: 4px;
                }
                QLabel {
                    color: #CCCCCC;
                    font-size: 11px;
                }
                QDoubleSpinBox {
                    background-color: #121212;
                    color: #00E676;
                    border: 1px solid #333333;
                    border-radius: 3px;
                    padding: 3px;
                    font-family: 'JetBrains Mono', 'Courier New', monospace;
                    font-size: 11px;
                }
                QDoubleSpinBox:focus {
                    border-color: #00E676;
                }
            """)
            content_layout.addWidget(self.properties_box)

        # Add-on container
        if self.addon_container:
            content_layout.addWidget(self.addon_container)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)


class StatsChip(QFrame):
    """
    Monospace stats overlay chip positioned in the corner of the OpenGL Viewport.
    Displays Vertices, Faces, and Triangles updated via a 4Hz QTimer.
    """
    def __init__(self, get_stats_cb, parent=None):
        super().__init__(parent)
        self.get_stats_cb = get_stats_cb
        self.setObjectName("StatsChip")
        self._init_ui()

        # Update timer at 4Hz (250ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(250)

    def _init_ui(self):
        self.setStyleSheet("""
            #StatsChip {
                background-color: rgba(18, 18, 18, 0.78);
                border: 1px solid #2A2A2A;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QLabel {
                color: #A0A0A0;
                font-family: 'JetBrains Mono', 'Courier New', monospace;
                font-size: 10px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        self.lbl_stats = QLabel("Verts: 0  |  Faces: 0  |  Tris: 0", self)
        layout.addWidget(self.lbl_stats)

    def update_stats(self):
        if callable(self.get_stats_cb):
            verts, faces, tris = self.get_stats_cb()
            self.lbl_stats.setText(f"Verts: {verts:,}  |  Faces: {faces:,}  |  Tris: {tris:,}")


class ToolShelfWidget(QWidget):
    """
    Collapsible left tool shelf panel for mode-based tools (Object, Edit, Sculpt).
    """
    mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode = "Object"
        self._expanded = False
        self.tools_layout = QVBoxLayout()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.setFixedWidth(52)  # Default icon-only width

        # Background styling
        self.setStyleSheet("""
            ToolShelfWidget {
                background-color: #161616;
                border-right: 1px solid #2A2A2A;
            }
            QToolButton {
                background-color: #202020;
                color: #CCCCCC;
                border: 1px solid #2D2D2D;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
            }
            QToolButton:hover {
                background-color: #2C2C2C;
                border-color: #00E676;
            }
            QToolButton:checked {
                background-color: #00E676;
                color: #121212;
                border-color: #00E676;
                font-weight: bold;
            }
        """)

        # Tools container layout
        self.tools_container = QWidget(self)
        self.tools_layout = QVBoxLayout(self.tools_container)
        self.tools_layout.setContentsMargins(0, 0, 0, 0)
        self.tools_layout.setSpacing(6)
        layout.addWidget(self.tools_container)

        layout.addStretch()

        # Bottom expand/collapse toggle
        self.btn_toggle_expand = QToolButton(self)
        self.btn_toggle_expand.setText("»")
        self.btn_toggle_expand.setToolTip("Expand / Collapse Tool Shelf")
        self.btn_toggle_expand.clicked.connect(self._toggle_expand)
        layout.addWidget(self.btn_toggle_expand, 0, Qt.AlignCenter)

        self.set_mode("Object")

    def _toggle_expand(self):
        self._expanded = not self._expanded
        self.setFixedWidth(130 if self._expanded else 52)
        self.btn_toggle_expand.setText("«" if self._expanded else "»")
        self.set_mode(self._current_mode)

    def set_mode(self, mode: str):
        self._current_mode = mode
        # Clear existing tools
        while self.tools_layout.count():
            child = self.tools_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if mode == "Object":
            self._build_object_tools()
        else:
            self._build_coming_soon(mode)

    def _build_object_tools(self):
        tools = [
            ("Translate", "✢", "Select & Move (G)"),
            ("Rotate", "🔄", "Rotate (R)"),
            ("Scale", "⤢", "Scale (S)"),
        ]
        
        for name, icon, tooltip in tools:
            btn = QToolButton(self)
            btn.setCheckable(True)
            btn.setText(f"{icon} {name}" if self._expanded else icon)
            btn.setToolTip(tooltip)
            btn.setFixedHeight(34)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.tools_layout.addWidget(btn)

    def _build_coming_soon(self, mode: str):
        lbl = QLabel(f"— {mode} —\nComing soon", self)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #666666; font-size: 10px; padding: 10px 2px;")
        self.tools_layout.addWidget(lbl)
