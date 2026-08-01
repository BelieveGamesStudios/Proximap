"""
pg_workspace.py - Qt Advanced Docking System (QADS) workspace container for Polyground.
Hosts the CDockManager, dock widgets (Viewport, Outliner, Properties, Timeline, ToolShelf),
and manages layout persistence.
"""

import os
import json
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QComboBox
import PySide6QtAds as ads

from mesh_editor.pg_panels import OutlinerPanel, PropertiesPanel, StatsChip, ToolShelfWidget
from mesh_editor.pg_timeline import TimelineWidget


class ViewportDockContainer(QWidget):
    """
    Wrapper container for the QOpenGLWidget viewport, StatsChip corner overlay,
    and Object Mode / Edit Mode dropdown notch overlay anchored in top-left.
    """
    mode_changed = Signal(str)

    def __init__(self, viewport_widget: QWidget, get_stats_cb, parent=None):
        super().__init__(parent)
        self.viewport_widget = viewport_widget
        self.get_stats_cb = get_stats_cb
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewport_widget)

        # Mode Selector Notch anchored in top-left corner directly under + Proximap logo
        self.mode_notch = QComboBox(self)
        self.mode_notch.addItems(["Object Mode", "Edit Mode"])
        self.mode_notch.setFixedHeight(26)
        self.mode_notch.setCursor(Qt.PointingHandCursor)
        self.mode_notch.setStyleSheet("""
            QComboBox {
                background-color: rgba(28, 28, 28, 230);
                color: #FFFFFF;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid rgba(80, 80, 80, 200);
                border-radius: 4px;
                padding: 2px 10px;
                min-width: 95px;
            }
            QComboBox:hover {
                border-color: #00E676;
                background-color: rgba(40, 40, 40, 250);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 16px;
                border-left: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1E1E1E;
                color: #CCCCCC;
                selection-background-color: #00E676;
                selection-color: #121212;
                border: 1px solid #3A3A3A;
                outline: 0;
            }
        """)
        self.mode_notch.currentTextChanged.connect(
            lambda txt: self.mode_changed.emit(txt.replace(" Mode", ""))
        )
        self.mode_notch.show()

        # Stats Overlay Chip anchored in bottom-left corner
        self.stats_chip = StatsChip(self.get_stats_cb, self)
        self.stats_chip.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 12
        # Position Mode Notch overlay in top-left corner directly under + Proximap
        self.mode_notch.move(margin, margin)
        self.mode_notch.raise_()

        # Position stats chip in bottom-left with 12px margin
        chip_size = self.stats_chip.sizeHint()
        y_pos = self.height() - chip_size.height() - margin
        self.stats_chip.move(margin, max(margin, y_pos))
        self.stats_chip.raise_()


class PolygroundWorkspace(QWidget):
    """
    QADS workspace container hosting all Polyground panels in a reconfigurable layout.
    """
    def __init__(self, mesh_editor_coordinator, parent=None):
        super().__init__(parent)
        self.editor = mesh_editor_coordinator
        self.is_initialized = False
        
        self.dock_manager: ads.CDockManager = None
        self.docks = {}
        
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create QADS CDockManager and add to layout
        self.dock_manager = ads.CDockManager(self)
        layout.addWidget(self.dock_manager)
        
        # Style QADS title bars and splitters
        self.dock_manager.setStyleSheet("""
            ads--CDockManager {
                background-color: #121212;
            }
            ads--CDockAreaWidget {
                background-color: #1A1A1A;
                border: 1px solid #262626;
            }
            ads--CDockAreaTitleBar {
                background-color: #181818;
                border-bottom: 1px solid #282828;
                min-height: 26px;
            }
            ads--CDockAreaTitleBar QLabel {
                color: #CCCCCC;
                font-family: 'Inter', sans-serif;
                font-size: 11px;
                font-weight: bold;
                padding-left: 6px;
            }
            ads--CDockSplitter {
                background-color: #242424;
            }
        """)

    def _get_stats(self):
        if hasattr(self.editor, 'viewport') and hasattr(self.editor.viewport, 'scene'):
            scene = self.editor.viewport.scene
            v, f, t = 0, 0, 0
            for obj in scene.objects:
                if hasattr(obj, 'vertices') and obj.vertices is not None:
                    v += len(obj.vertices)
                if hasattr(obj, 'faces') and obj.faces is not None:
                    f += len(obj.faces)
                    t += len(obj.faces)
            return v, f, t
        return 0, 0, 0

    def initialize(self):
        if self.is_initialized:
            return
        self.is_initialized = True

        # 1. Viewport Dock (Non-floatable to prevent QOpenGLWidget context destruction)
        self.viewport_container = ViewportDockContainer(self.editor.viewport, self._get_stats, self)
        self.viewport_container.mode_changed.connect(self._on_mode_changed)
        viewport_dock = ads.CDockWidget("3D Viewport", self)
        viewport_dock.setWidget(self.viewport_container)
        viewport_dock.setFeature(ads.CDockWidget.DockWidgetFloatable, False)
        viewport_dock.setFeature(ads.CDockWidget.DockWidgetClosable, False)

        # 2. Outliner Dock
        outliner_panel = OutlinerPanel(self.editor.get_outliner_widget(), self.editor.btn_delete_mesh, self)
        outliner_dock = ads.CDockWidget("Outliner", self)
        outliner_dock.setWidget(outliner_panel)

        # 3. Properties Dock
        properties_panel = PropertiesPanel(self.editor.get_properties_widget(), self.editor.get_addon_widget(), self)
        properties_dock = ads.CDockWidget("Properties", self)
        properties_dock.setWidget(properties_panel)

        # 4. Timeline Dock
        self.timeline_widget = TimelineWidget(parent=self)
        timeline_dock = ads.CDockWidget("Timeline", self)
        timeline_dock.setWidget(self.timeline_widget)

        # 5. Tool Shelf Dock
        self.tool_shelf_widget = ToolShelfWidget(parent=self)
        tool_shelf_dock = ads.CDockWidget("Tool Shelf", self)
        tool_shelf_dock.setWidget(self.tool_shelf_widget)

        # Store references
        self.docks = {
            "viewport": viewport_dock,
            "outliner": outliner_dock,
            "properties": properties_dock,
            "timeline": timeline_dock,
            "tool_shelf": tool_shelf_dock
        }

        # Try restoring saved layout or build default layout
        if not self.restore_layout():
            self._build_default_layout()

    def _build_default_layout(self):
        """Builds default arrangement of docks explicitly using setCentralWidget."""
        if not self.dock_manager or "viewport" not in self.docks:
            return
            
        viewport_dock = self.docks["viewport"]
        outliner_dock = self.docks["outliner"]
        properties_dock = self.docks["properties"]
        timeline_dock = self.docks["timeline"]
        tool_shelf_dock = self.docks["tool_shelf"]

        # Set central widget
        central_area = self.dock_manager.setCentralWidget(viewport_dock)
        
        # Add side and bottom docks
        outliner_area = self.dock_manager.addDockWidget(ads.RightDockWidgetArea, outliner_dock, central_area)
        self.dock_manager.addDockWidget(ads.BottomDockWidgetArea, properties_dock, outliner_area)
        self.dock_manager.addDockWidget(ads.BottomDockWidgetArea, timeline_dock, central_area)
        self.dock_manager.addDockWidget(ads.LeftDockWidgetArea, tool_shelf_dock, central_area)

        tool_shelf_dock.toggleView(False)



    def _on_mode_changed(self, mode: str):
        if hasattr(self, 'tool_shelf_widget') and self.tool_shelf_widget:
            self.tool_shelf_widget.set_mode(mode)

    def save_layout(self):
        """Saves current QADS layout state to ~/.proximap/dock_layout.json."""
        if not self.dock_manager:
            return
        try:
            config_dir = os.path.join(os.path.expanduser("~"), ".proximap")
            os.makedirs(config_dir, exist_ok=True)
            layout_path = os.path.join(config_dir, "dock_layout.json")

            state_bytes = self.dock_manager.saveState()
            state_hex = state_bytes.toHex().data().decode("utf-8")
            
            with open(layout_path, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "state": state_hex}, f)
        except Exception as e:
            print(f"[WARNING] Failed to save dock layout: {e}")

    def restore_layout(self) -> bool:
        """Restores QADS layout state from ~/.proximap/dock_layout.json if available."""
        if not self.dock_manager:
            return False
        try:
            layout_path = os.path.join(os.path.expanduser("~"), ".proximap", "dock_layout.json")
            if os.path.exists(layout_path):
                with open(layout_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                state_hex = data.get("state", "")
                if state_hex:
                    from PySide6.QtCore import QByteArray
                    state_bytes = QByteArray.fromHex(state_hex.encode("utf-8"))
                    return self.dock_manager.restoreState(state_bytes)
        except Exception as e:
            print(f"[WARNING] Could not restore dock layout: {e}")
        return False
