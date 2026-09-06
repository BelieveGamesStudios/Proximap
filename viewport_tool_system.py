"""
viewport_tool_system.py — Unity EditorWindow-Style Tool System & Movable Toolbox for Proximap Viewport

Provides:
- BaseEditorTool: Abstract base class defining the standard Unity-like EditorWindow lifecycle.
- EditorWindowHostModal: Fixed overlay modal hosting the active tool's GUI.
- FloatingToolboxWidget: Movable vertical floating toolbar with icon-only tool buttons.
- TransformToolWindow: Scaffolded 3D Transform tool window.
- MeshCleanupToolWindow: Scaffolded Mesh Cleanup & Repair tool window.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint, QSize, QObject
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QGridLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QGroupBox, QStackedWidget, QAbstractSpinBox, QSlider
)


def get_base_dir() -> str:
    """Returns absolute path to the Proximap root directory."""
    return os.path.dirname(os.path.abspath(__file__))


def load_tinted_svg_icon(svg_path: Optional[str], color_hex: str = "#E0E0E0", size: int = 24) -> QIcon:
    """Renders and tints an SVG file into a crisp QIcon."""
    if not svg_path:
        return QIcon()

    if not os.path.isabs(svg_path):
        svg_path = os.path.join(get_base_dir(), svg_path)

    if not os.path.exists(svg_path):
        return QIcon()

    renderer = QSvgRenderer(svg_path)
    if not renderer.isValid():
        return QIcon(svg_path)

    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background

    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color_hex))
    painter.end()

    return QIcon(pixmap)


# ---------------------------------------------------------------------------
# Stepper Widget: Wraps a SpinBox with dedicated '-' and '+' buttons
# ---------------------------------------------------------------------------
class SpinBoxStepper(QWidget):
    """
    Wraps a QSpinBox or QDoubleSpinBox with dedicated '-' and '+' push buttons.
    Hides native arrow buttons to provide reliable, touch/click-friendly targets.
    """
    def __init__(self, spinbox, parent=None):
        super().__init__(parent)
        self.spin = spinbox
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setAlignment(Qt.AlignCenter)
        self.spin.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                background-color: #1E1E1E;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 3px;
                padding: 2px 4px;
                font-size: 11px;
                min-height: 20px;
                max-width: 90px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        btn_style = """
            QPushButton {
                background-color: #2D2D2D;
                color: #00E676;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #444444;
                border-radius: 3px;
                min-width: 22px;
                max-width: 22px;
                min-height: 22px;
                max-height: 22px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #383838;
                border-color: #00E676;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #00E676;
                color: #121212;
            }
            QPushButton:disabled {
                background-color: #1C1C1C;
                color: #555555;
                border-color: #2A2A2A;
            }
        """

        self.btn_minus = QPushButton("-", self)
        self.btn_minus.setCursor(Qt.PointingHandCursor)
        self.btn_minus.setStyleSheet(btn_style)
        self.btn_minus.clicked.connect(self._step_down)

        self.btn_plus = QPushButton("+", self)
        self.btn_plus.setCursor(Qt.PointingHandCursor)
        self.btn_plus.setStyleSheet(btn_style)
        self.btn_plus.clicked.connect(self._step_up)

        layout.addWidget(self.btn_minus)
        layout.addWidget(self.spin, stretch=1)
        layout.addWidget(self.btn_plus)

    def _step_down(self):
        self.spin.stepBy(-1)

    def _step_up(self):
        self.spin.stepBy(1)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self.spin.setEnabled(enabled)
        self.btn_minus.setEnabled(enabled)
        self.btn_plus.setEnabled(enabled)


# ===========================================================================
# 1. Base Editor Tool (Unity EditorWindow Pattern)
# ===========================================================================

class BaseEditorTool(QObject):
    """
    Base class for all Viewport Tool Windows (Unity EditorWindow pattern).
    Each tool subclass defines its identification, icon, data requirements, and lifecycle hooks.
    """

    def __init__(self, tool_id: str, title: str, icon_path: str, tooltip: str = "",
                 requires_data: bool = True, empty_message: str = "No loaded point or mesh data"):
        super().__init__()
        self.tool_id = tool_id
        self.title = title
        self.icon_path = icon_path
        self.tooltip = tooltip or title
        self.requires_data = requires_data
        self.empty_message = empty_message
        self.context: Optional[Dict[str, Any]] = None
        self.is_active: bool = False
        self.has_data: bool = True

    def on_enable(self, context: Optional[Dict[str, Any]] = None):
        """Called when the tool window is opened or switched to."""
        self.context = context or {}
        self.is_active = True

    @abstractmethod
    def on_gui(self, layout: QVBoxLayout, parent: QWidget):
        """
        Builds the tool's GUI inside the provided layout and parent widget.
        Subclasses should populate `layout` with their widgets.
        """
        pass

    def on_disable(self):
        """Called when the tool window is closed or switched away from."""
        self.is_active = False

    def on_update(self):
        """Optional tick/update hook called during live scene changes or animation frames."""
        pass

    def on_data_state_changed(self, has_data: bool):
        """Hook called whenever scene data availability changes (loaded or cleared)."""
        self.has_data = has_data


# ===========================================================================
# 2. Fixed Modal Window Host (EditorWindowHostModal)
# ===========================================================================

class EditorWindowHostModal(QFrame):
    """
    Viewport overlay that dynamically hosts the active BaseEditorTool's UI.
    Attached directly alongside the movable FloatingToolboxWidget so it always follows it.
    """

    tool_opened = Signal(str)      # tool_id
    tool_closed = Signal(str)      # tool_id
    tool_switched = Signal(str, str)  # (old_tool_id, new_tool_id)
    gizmo_toggled = Signal(bool)

    def __init__(self, parent=None, default_x: int = 70, default_y: int = 20, min_width: int = 345,
                 data_provider: Optional[callable] = None):
        super().__init__(parent)
        self.default_x = default_x
        self.default_y = default_y
        self.min_width = min_width
        self.data_provider = data_provider
        self.toolbox: Optional['FloatingToolboxWidget'] = None
        self._is_dragging = False
        self._drag_pos = QPoint()

        self.current_tool: Optional[BaseEditorTool] = None
        self.registered_tools: Dict[str, BaseEditorTool] = {}
        self.tool_pages: Dict[str, Dict[str, Any]] = {}

        self.setObjectName("EditorWindowHostModal")
        self.setFixedWidth(self.min_width)
        self.setStyleSheet("""
            QFrame#EditorWindowHostModal {
                background-color: rgba(20, 20, 20, 245);
                border: 1px solid #3A3A3A;
                border-radius: 8px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 11px;
                background: transparent;
                border: none;
            }
            QLabel:disabled {
                color: #555555;
            }
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 11px;
                font-weight: bold;
                color: #00E676;
            }
            QGroupBox:disabled {
                border-color: #222222;
                color: #555555;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #161616;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #00E676;
            }
            QDoubleSpinBox:disabled, QSpinBox:disabled {
                background-color: #111111;
                color: #555555;
                border-color: #262626;
            }
            QCheckBox {
                color: #d0d0d0;
                font-size: 11px;
                spacing: 6px;
            }
            QCheckBox:disabled {
                color: #555555;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid #555555;
                background-color: #1a1a1a;
            }
            QCheckBox::indicator:checked {
                background-color: #00E676;
                border-color: #00E676;
            }
            QCheckBox::indicator:disabled {
                border-color: #333333;
                background-color: #141414;
            }
            QPushButton.action-btn {
                background-color: #242424;
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton.action-btn:hover {
                background-color: #333333;
                border-color: #00E676;
                color: #ffffff;
            }
            QPushButton.action-btn:disabled {
                background-color: #161616;
                color: #555555;
                border-color: #262626;
            }
            QPushButton.primary-btn {
                background-color: #00A854;
                color: #ffffff;
                border: 1px solid #00E676;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton.primary-btn:hover {
                background-color: #00C864;
            }
            QPushButton.primary-btn:disabled {
                background-color: #1a3a2a;
                color: #555555;
                border-color: #263a2e;
            }
        """)

        # Main vertical layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 10, 12, 12)
        self.main_layout.setSpacing(8)

        # Header bar
        self.header_widget = QWidget(self)
        self.header_widget.setStyleSheet("background: transparent; border: none;")
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        # Header tool icon
        self.header_icon_label = QLabel(self.header_widget)
        self.header_icon_label.setFixedSize(18, 18)
        self.header_icon_label.setScaledContents(True)
        header_layout.addWidget(self.header_icon_label)

        # Header title
        self.title_label = QLabel("Tool Window", self.header_widget)
        self.title_label.setStyleSheet("color: #00E676; font-size: 12px; font-weight: bold;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        # Show Gizmo toggle (top right opposite the Transform title)
        self.gizmo_toggle = QCheckBox("Show Gizmo", self.header_widget)
        self.gizmo_toggle.setChecked(False)
        self.gizmo_toggle.setCursor(Qt.PointingHandCursor)
        self.gizmo_toggle.setStyleSheet("""
            QCheckBox {
                color: #888888;
                font-size: 11px;
                font-weight: 500;
                spacing: 5px;
            }
            QCheckBox:hover {
                color: #e0e0e0;
            }
            QCheckBox:checked {
                color: #00E676;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border-radius: 3px;
                border: 1px solid #444444;
                background-color: #1a1a1a;
            }
            QCheckBox::indicator:hover {
                border-color: #00E676;
            }
            QCheckBox::indicator:checked {
                background-color: #00E676;
                border-color: #00E676;
            }
        """)
        self.gizmo_toggle.toggled.connect(self._on_gizmo_toggled)
        header_layout.addWidget(self.gizmo_toggle)

        # Close button
        self.close_btn = QPushButton("✕", self.header_widget)
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #888888;
                border: none;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #00E676;
            }
        """)
        self.close_btn.clicked.connect(self.close_tool)
        header_layout.addWidget(self.close_btn)

        self.main_layout.addWidget(self.header_widget)

        # Stacked container for all registered tools
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.setStyleSheet("background: transparent; border: none;")

        self.main_layout.addWidget(self.stacked_widget)

        self.hide()

    def set_toolbox(self, toolbox: 'FloatingToolboxWidget'):
        """Connects the toolbox instance so the modal always follows it."""
        self.toolbox = toolbox

    def set_data_provider(self, provider: callable):
        """Sets the data availability callback."""
        self.data_provider = provider

    def register_tool(self, tool: BaseEditorTool):
        """Registers a BaseEditorTool instance."""
        self.registered_tools[tool.tool_id] = tool

    def open_tool(self, tool: BaseEditorTool, context: Optional[Dict[str, Any]] = None):
        """Opens or switches to the given tool."""
        old_tool_id = self.current_tool.tool_id if self.current_tool else None

        if self.current_tool and self.current_tool != tool:
            self.current_tool.on_disable()

        self.current_tool = tool
        self.title_label.setText(tool.title)

        # Set header icon
        if tool.icon_path:
            icon = load_tinted_svg_icon(tool.icon_path, color_hex="#00E676", size=18)
            self.header_icon_label.setPixmap(icon.pixmap(18, 18))
            self.header_icon_label.show()
        else:
            self.header_icon_label.hide()

        # Build tool widget on first visit, or reuse existing widget
        if tool.tool_id not in self.tool_pages:
            tool_page = QWidget(self.stacked_widget)
            tool_page.setStyleSheet("background: transparent; border: none;")
            page_layout = QVBoxLayout(tool_page)
            page_layout.setContentsMargins(0, 4, 0, 0)
            page_layout.setSpacing(8)

            # 1. Interactive content container
            content_widget = QWidget(tool_page)
            content_widget.setStyleSheet("background: transparent; border: none;")
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(6)

            tool.on_gui(content_layout, content_widget)
            page_layout.addWidget(content_widget)

            # 2. Empty data state notice banner
            empty_banner = QFrame(tool_page)
            empty_banner.setObjectName("EmptyDataBanner")
            empty_banner.setStyleSheet("""
                QFrame#EmptyDataBanner {
                    background-color: rgba(255, 170, 0, 0.08);
                    border: 1px solid rgba(255, 170, 0, 0.25);
                    border-radius: 5px;
                }
            """)
            banner_layout = QHBoxLayout(empty_banner)
            banner_layout.setContentsMargins(8, 6, 8, 6)
            banner_layout.setSpacing(6)

            banner_label = QLabel(tool.empty_message, empty_banner)
            banner_label.setStyleSheet("color: #E5B25D; font-size: 11px; font-weight: 500; background: transparent; border: none;")

            banner_layout.addWidget(banner_label, 1)

            page_layout.addWidget(empty_banner)

            self.stacked_widget.addWidget(tool_page)
            self.tool_pages[tool.tool_id] = {
                'page': tool_page,
                'content': content_widget,
                'banner': empty_banner,
                'tool': tool
            }

        # Switch to tool page
        self.stacked_widget.setCurrentWidget(self.tool_pages[tool.tool_id]['page'])

        # Show gizmo toggle only for tools that support 3D gizmos (e.g. transform)
        if tool.tool_id == "transform":
            self.gizmo_toggle.show()
        else:
            self.gizmo_toggle.hide()

        # Update data state (grey-out / banner)
        self.refresh_data_state()

        # Call enable lifecycle hook
        tool.on_enable(context)

        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()

        if old_tool_id and old_tool_id != tool.tool_id:
            self.tool_switched.emit(old_tool_id, tool.tool_id)
        else:
            self.tool_opened.emit(tool.tool_id)

    def _on_gizmo_toggled(self, checked: bool):
        self.gizmo_toggled.emit(checked)

    def refresh_data_state(self):
        """Updates the active tool's empty state and greyed-out controls based on data availability."""
        has_data = self.data_provider() if callable(self.data_provider) else True

        for tool_id, page_info in self.tool_pages.items():
            tool = page_info['tool']
            content_w = page_info['content']
            banner_w = page_info['banner']

            if tool.requires_data:
                content_w.setEnabled(has_data)
                banner_w.setVisible(not has_data)
            else:
                content_w.setEnabled(True)
                banner_w.setVisible(False)

            tool.on_data_state_changed(has_data)

        if self.isVisible():
            self.reposition()

    def close_tool(self):
        """Closes the active tool window."""
        if self.current_tool:
            tool_id = self.current_tool.tool_id
            self.current_tool.on_disable()
            self.current_tool = None
            if hasattr(self, 'gizmo_toggle') and self.gizmo_toggle.isChecked():
                self.gizmo_toggle.setChecked(False)
            self.hide()
            self.tool_closed.emit(tool_id)
        else:
            self.hide()

    def toggle_tool(self, tool: BaseEditorTool, context: Optional[Dict[str, Any]] = None):
        """Toggles a tool on/off."""
        if self.isVisible() and self.current_tool == tool:
            self.close_tool()
        else:
            self.open_tool(tool, context)

    def reposition(self):
        """Anchors and attaches the modal directly alongside the floating toolbox."""
        self.adjustSize()
        modal_w = self.min_width

        # Calculate dynamic height based on current active page
        current_page = self.stacked_widget.currentWidget() if hasattr(self, 'stacked_widget') else None
        if current_page:
            content_h = current_page.sizeHint().height()
            header_h = self.header_widget.sizeHint().height()
            modal_h = max(90, header_h + content_h + 24)
        else:
            modal_h = max(self.sizeHint().height(), self.height())

        if self.toolbox and self.parentWidget():
            parent_w = self.parentWidget().width()
            parent_h = self.parentWidget().height()
            tb_x = self.toolbox.x()
            tb_y = self.toolbox.y()
            tb_w = self.toolbox.width()

            gap = 6
            # Try placing to the right of the toolbox first
            target_x = tb_x + tb_w + gap
            if target_x + modal_w > parent_w - 8:
                # If not enough room on the right, flip to left of toolbox
                target_x = tb_x - modal_w - gap
                if target_x < 8:
                    target_x = max(8, min(tb_x + tb_w + gap, parent_w - modal_w - 8))

            # Align top with toolbar, clamped inside vertical viewport bounds
            target_y = max(8, min(tb_y, parent_h - modal_h - 8))
            self.setGeometry(int(target_x), int(target_y), int(modal_w), int(modal_h))
        else:
            self.setGeometry(self.default_x, self.default_y, self.min_width, modal_h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.toolbox:
            self._is_dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, '_is_dragging', False) and (event.buttons() & Qt.LeftButton) and self.toolbox and self.parentWidget():
            current_mouse = event.globalPosition().toPoint()
            new_modal_pos = current_mouse - self._drag_pos
            delta_x = new_modal_pos.x() - self.x()
            delta_y = new_modal_pos.y() - self.y()

            parent_rect = self.parentWidget().rect()
            new_tb_x = max(0, min(self.toolbox.x() + delta_x, parent_rect.width() - self.toolbox.width()))
            new_tb_y = max(0, min(self.toolbox.y() + delta_y, parent_rect.height() - self.toolbox.height()))
            self.toolbox.move(new_tb_x, new_tb_y)
            self.reposition()
            self._drag_pos = current_mouse - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._is_dragging = False
        super().mouseReleaseEvent(event)


# ===========================================================================
# 3. Movable Floating Vertical Toolbox (FloatingToolboxWidget)
# ===========================================================================

class FloatingToolboxWidget(QFrame):
    """
    Movable vertical toolbox overlay containing icon-only buttons.
    Supports mouse dragging and extensible tool registration.
    The active tool modal attaches directly to it and follows it everywhere.
    """

    def __init__(self, parent=None, host_modal: Optional[EditorWindowHostModal] = None, initial_x: int = 16, initial_y: int = 20):
        super().__init__(parent)
        self.host_modal = host_modal
        self.initial_x = initial_x
        self.initial_y = initial_y

        if self.host_modal:
            self.host_modal.set_toolbox(self)

        self._drag_pos = QPoint()
        self._is_dragging = False
        self.tool_buttons: Dict[str, QPushButton] = {}
        self.registered_tools: Dict[str, BaseEditorTool] = {}

        self.setObjectName("FloatingToolboxWidget")
        self.setFixedWidth(44)
        self.setStyleSheet("""
            QFrame#FloatingToolboxWidget {
                background-color: rgba(20, 20, 20, 230);
                border: 1px solid #3A3A3A;
                border-radius: 8px;
            }
            QPushButton.toolbox-btn {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 4px;
            }
            QPushButton.toolbox-btn:hover {
                background-color: rgba(255, 255, 255, 0.1);
                border-color: #555555;
            }
            QPushButton.toolbox-btn:checked, QPushButton.toolbox-btn.active {
                background-color: rgba(0, 230, 118, 0.18);
                border: 1px solid #00E676;
            }
        """)

        # Vertical layout
        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(4, 6, 4, 6)
        self.vbox.setSpacing(6)

        # Drag handle / grip indicator at top
        self.grip = QLabel(self)
        self.grip.setFixedHeight(6)
        self.grip.setStyleSheet("background: #444444; border-radius: 2px; margin: 0 8px;")
        self.grip.setCursor(Qt.SizeAllCursor)
        self.vbox.addWidget(self.grip)

        # Hook into host modal signals to update active button states
        if self.host_modal:
            self.host_modal.tool_opened.connect(self._on_modal_tool_opened)
            self.host_modal.tool_closed.connect(self._on_modal_tool_closed)
            self.host_modal.tool_switched.connect(self._on_modal_tool_switched)

        self.setGeometry(self.initial_x, self.initial_y, 44, 100)

    def register_tool(self, tool: BaseEditorTool, context_provider: Optional[callable] = None) -> QPushButton:
        """
        Extensible registration: adds a new icon button to the vertical toolbox.
        """
        self.registered_tools[tool.tool_id] = tool
        if self.host_modal:
            self.host_modal.register_tool(tool)

        btn = QPushButton(self)
        btn.setProperty("class", "toolbox-btn")
        btn.setFixedSize(34, 34)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(tool.tooltip or tool.title)

        # Load SVG icon
        icon = load_tinted_svg_icon(tool.icon_path, color_hex="#E0E0E0", size=22)
        btn.setIcon(icon)
        btn.setIconSize(QSize(22, 22))

        # Connect click to toggle tool in the host modal
        def on_btn_clicked():
            ctx = context_provider() if context_provider else None
            if self.host_modal:
                self.host_modal.toggle_tool(tool, context=ctx)

        btn.clicked.connect(on_btn_clicked)

        self.vbox.addWidget(btn)
        self.tool_buttons[tool.tool_id] = btn

        # Adjust height
        self.adjustSize()
        self.resize(44, self.sizeHint().height())

        return btn

    def set_tool_visible(self, tool_id: str, visible: bool):
        """Idempotently shows or hides a tool button in the floating sidebar."""
        btn = self.tool_buttons.get(tool_id)
        if not btn:
            return
        if (not btn.isHidden()) == visible:
            return
        btn.setVisible(visible)

        # If hiding the currently active tool, close the host modal
        if not visible and self.host_modal and self.host_modal.current_tool:
            if self.host_modal.current_tool.tool_id == tool_id:
                self.host_modal.close_tool()

        # Recalculate layout size
        self.adjustSize()
        self.resize(44, self.sizeHint().height())

        # Clamp toolbox within parent boundaries
        if self.parentWidget():
            parent_w = self.parentWidget().width()
            parent_h = self.parentWidget().height()
            cur_x = max(0, min(self.x(), max(0, parent_w - self.width())))
            cur_y = max(0, min(self.y(), max(0, parent_h - self.height())))
            self.move(cur_x, cur_y)

        # Reposition host window if open
        if self.host_modal and self.host_modal.isVisible():
            self.host_modal.reposition()

    def _on_modal_tool_opened(self, tool_id: str):
        self._update_button_highlights(active_tool_id=tool_id)

    def _on_modal_tool_closed(self, tool_id: str):
        self._update_button_highlights(active_tool_id=None)

    def _on_modal_tool_switched(self, old_tool_id: str, new_tool_id: str):
        self._update_button_highlights(active_tool_id=new_tool_id)

    def _update_button_highlights(self, active_tool_id: Optional[str]):
        for tid, btn in self.tool_buttons.items():
            is_active = (tid == active_tool_id)
            tool = self.registered_tools.get(tid)
            if is_active:
                btn.setStyleSheet("""
                    background-color: rgba(0, 230, 118, 0.2);
                    border: 1px solid #00E676;
                    border-radius: 6px;
                """)
                if tool and tool.icon_path:
                    btn.setIcon(load_tinted_svg_icon(tool.icon_path, color_hex="#00E676", size=22))
            else:
                btn.setStyleSheet("")
                if tool and tool.icon_path:
                    btn.setIcon(load_tinted_svg_icon(tool.icon_path, color_hex="#E0E0E0", size=22))

    # --- Mouse dragging implementation ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.raise_()
            if self.host_modal and self.host_modal.isVisible():
                self.host_modal.raise_()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging and (event.buttons() & Qt.LeftButton):
            new_pos = event.globalPosition().toPoint() - self._drag_pos
            # Clamp inside parent boundaries
            if self.parentWidget():
                parent_rect = self.parentWidget().rect()
                max_x = max(0, parent_rect.width() - self.width())
                max_y = max(0, parent_rect.height() - self.height())
                clamped_x = max(0, min(new_pos.x(), max_x))
                clamped_y = max(0, min(new_pos.y(), max_y))
                self.move(clamped_x, clamped_y)
            else:
                self.move(new_pos)

            # Move attached tool modal synchronously alongside the toolbox
            if self.host_modal and self.host_modal.isVisible():
                self.host_modal.reposition()
                self.host_modal.raise_()
            self.raise_()
            event.accept()
    def mouseReleaseEvent(self, event):
        self._is_dragging = False
        super().mouseReleaseEvent(event)


# ===========================================================================
# 4. Unity-Style Transform Controls (Draggable Axis & Rows)
# ===========================================================================

class DraggableAxisLabel(QLabel):
    """
    Color-coded axis label (X, Y, Z) that scrubs the associated spinbox value when dragged horizontally.
    """
    def __init__(self, axis_text: str, color_hex: str, spinbox: QDoubleSpinBox, parent=None):
        super().__init__(axis_text, parent)
        self.axis_text = axis_text
        self.color_hex = color_hex
        self.spinbox = spinbox
        self._is_dragging = False
        self._drag_start_x = 0
        self._start_value = 0.0

        self.setFixedWidth(16)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.SizeHorCursor)
        self.setStyleSheet(f"""
            QLabel {{
                color: {self.color_hex};
                font-weight: bold;
                font-size: 11px;
                background-color: transparent;
                border: none;
                padding-left: 2px;
            }}
            QLabel:hover {{
                color: #FFFFFF;
                background-color: {self.color_hex};
                border-radius: 2px;
            }}
            QLabel:disabled {{
                color: #555555;
                background-color: transparent;
            }}
        """)
        self.setToolTip(f"Drag horizontally to adjust {axis_text} (Hold Shift for fine tuning)")

    def mousePressEvent(self, event):
        if not self.isEnabled() or not self.spinbox.isEnabled():
            return
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_start_x = event.globalPosition().x()
            self._start_value = self.spinbox.value()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging and (event.buttons() & Qt.LeftButton):
            delta = event.globalPosition().x() - self._drag_start_x

            # Sensitivity step modifiers
            if event.modifiers() & Qt.ShiftModifier:
                step = 0.01  # Fine control
            elif event.modifiers() & Qt.ControlModifier:
                step = 1.0   # Fast control
            else:
                step = 0.1   # Standard control

            new_val = self._start_value + (delta * step)
            new_val = max(self.spinbox.minimum(), min(self.spinbox.maximum(), new_val))
            self.spinbox.setValue(round(new_val, 2))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._is_dragging = False
        super().mouseReleaseEvent(event)


class DraggableAxisSpinBox(QFrame):
    """
    A compound Unity-style numeric input box with a draggable color-coded axis label prefix.
    """
    valueChanged = Signal(float)

    def __init__(self, axis_text: str, color_hex: str, default_val: float = 0.0,
                 min_val: float = -99999.0, max_val: float = 99999.0, decimals: int = 2,
                 suffix: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("DraggableAxisSpinBox")
        self.setFixedHeight(24)
        self.setStyleSheet("""
            QFrame#DraggableAxisSpinBox {
                background-color: #141414;
                border: 1px solid #333333;
                border-radius: 3px;
            }
            QFrame#DraggableAxisSpinBox:hover {
                border-color: #555555;
            }
            QFrame#DraggableAxisSpinBox:focus-within {
                border-color: #00E676;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(1, 0, 3, 0)
        layout.setSpacing(2)

        # Spinbox
        self.spin = QDoubleSpinBox(self)
        self.spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.spin.setRange(min_val, max_val)
        self.spin.setDecimals(decimals)
        self.spin.setValue(default_val)
        if suffix:
            self.spin.setSuffix(suffix)
        self.spin.setStyleSheet("""
            QDoubleSpinBox {
                background: transparent;
                border: none;
                color: #e0e0e0;
                font-size: 11px;
                padding: 0px;
            }
            QDoubleSpinBox:focus {
                color: #ffffff;
            }
        """)
        self.spin.valueChanged.connect(self.valueChanged.emit)

        # Draggable Label Prefix
        self.label = DraggableAxisLabel(axis_text, color_hex, self.spin, self)

        layout.addWidget(self.label)
        layout.addWidget(self.spin, 1)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, val: float):
        self.spin.setValue(val)


class UnityTransformRow(QWidget):
    """
    A single Unity Inspector transform row:
    [Title]  [X: val] [Y: val] [Z: val] [↺]
    """
    valuesChanged = Signal(float, float, float)

    def __init__(self, title: str, default_vals=(0.0, 0.0, 0.0), suffix: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(5)

        # Row Title Label
        self.title_label = QLabel(title, self)
        self.title_label.setFixedWidth(56)
        self.title_label.setStyleSheet("color: #b0b0b0; font-size: 11px; font-weight: 500;")
        layout.addWidget(self.title_label)

        # X, Y, Z fields
        self.x_box = DraggableAxisSpinBox("X", "#FF5252", default_val=default_vals[0], suffix=suffix, parent=self)
        self.y_box = DraggableAxisSpinBox("Y", "#00E676", default_val=default_vals[1], suffix=suffix, parent=self)
        self.z_box = DraggableAxisSpinBox("Z", "#448AFF", default_val=default_vals[2], suffix=suffix, parent=self)

        self.x_box.valueChanged.connect(self._on_axis_changed)
        self.y_box.valueChanged.connect(self._on_axis_changed)
        self.z_box.valueChanged.connect(self._on_axis_changed)

        layout.addWidget(self.x_box, 1)
        layout.addWidget(self.y_box, 1)
        layout.addWidget(self.z_box, 1)

        # Reset button (↺)
        self.reset_btn = QPushButton("↺", self)
        self.reset_btn.setFixedSize(18, 22)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setToolTip(f"Reset {title} to (0, 0, 0)")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #666666;
                border: none;
                font-size: 12px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                color: #00E676;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_values)
        layout.addWidget(self.reset_btn)

    def _on_axis_changed(self, _=0.0):
        self.valuesChanged.emit(*self.get_values())

    def set_values(self, x: float, y: float, z: float):
        self.x_box.blockSignals(True)
        self.y_box.blockSignals(True)
        self.z_box.blockSignals(True)
        self.x_box.setValue(round(float(x), 2))
        self.y_box.setValue(round(float(y), 2))
        self.z_box.setValue(round(float(z), 2))
        self.x_box.blockSignals(False)
        self.y_box.blockSignals(False)
        self.z_box.blockSignals(False)
        self.valuesChanged.emit(*self.get_values())

    def reset_values(self):
        self.set_values(0.0, 0.0, 0.0)

    def get_values(self):
        return (self.x_box.value(), self.y_box.value(), self.z_box.value())


class TransformToolWindow(BaseEditorTool):
    """
    Unity-Style Transform Inspector Window:
    Position: [X: 0.00] [Y: 0.00] [Z: 0.00] [↺]
    Rotation: [X: 0.00°] [Y: 0.00°] [Z: 0.00°] [↺]
    """
    transform_changed = Signal(object)  # Emits 4x4 np.ndarray T
    transform_applied = Signal(object)  # Emits 4x4 np.ndarray T
    transform_reset = Signal()

    def __init__(self):
        icon_path = os.path.join("interface element", "transform.svg")
        super().__init__(
            tool_id="transform",
            title="Transform",
            icon_path=icon_path,
            tooltip="Transform (Position & Rotation)",
            requires_data=True,
            empty_message="No loaded point or mesh data"
        )
        self.position_row: Optional[UnityTransformRow] = None
        self.rotation_row: Optional[UnityTransformRow] = None
        self._updating_fields = False

    def on_enable(self, context: Optional[Dict[str, Any]] = None):
        super().on_enable(context)

    def on_gui(self, layout: QVBoxLayout, parent: QWidget):
        """Constructs the ultra-clean Unity Transform UI controls."""
        self.position_row = UnityTransformRow("Position", default_vals=(0.0, 0.0, 0.0), parent=parent)
        self.rotation_row = UnityTransformRow("Rotation", default_vals=(0.0, 0.0, 0.0), suffix="°", parent=parent)

        self.position_row.valuesChanged.connect(self._on_transform_values_changed)
        self.rotation_row.valuesChanged.connect(self._on_transform_values_changed)

        layout.addWidget(self.position_row)
        layout.addWidget(self.rotation_row)

        # Action Buttons Row (Reset All, Apply)
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 4, 0, 0)
        action_row.setSpacing(8)

        self.btn_reset_all = QPushButton("Reset All", parent)
        self.btn_reset_all.setProperty("class", "action-btn")
        self.btn_reset_all.setCursor(Qt.PointingHandCursor)
        self.btn_reset_all.clicked.connect(self.reset_all)

        self.btn_apply = QPushButton("Apply", parent)
        self.btn_apply.setProperty("class", "primary-btn")
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.clicked.connect(self.apply_transform)

        action_row.addWidget(self.btn_reset_all)
        action_row.addWidget(self.btn_apply)

        layout.addLayout(action_row)

    def has_unapplied_changes(self) -> bool:
        """Returns True if any translation or rotation values are non-zero."""
        if not self.position_row or not self.rotation_row:
            return False
        tx, ty, tz = self.position_row.get_values()
        rx, ry, rz = self.rotation_row.get_values()
        return any(abs(v) > 1e-4 for v in (tx, ty, tz, rx, ry, rz))

    def set_transform_values(self, pos=None, rot=None):
        """Sets numeric values programmatically (e.g. from viewport gizmo dragging)."""
        self._updating_fields = True
        if pos is not None and self.position_row:
            self.position_row.set_values(pos[0], pos[1], pos[2])
        if rot is not None and self.rotation_row:
            self.rotation_row.set_values(rot[0], rot[1], rot[2])
        self._updating_fields = False
        self.transform_changed.emit(self.get_transform_matrix())

    def _on_transform_values_changed(self, *_):
        if not self._updating_fields:
            self.transform_changed.emit(self.get_transform_matrix())

    def reset_all(self):
        self._updating_fields = True
        if self.position_row:
            self.position_row.reset_values()
        if self.rotation_row:
            self.rotation_row.reset_values()
        self._updating_fields = False
        self.transform_reset.emit()

    def apply_transform(self):
        mat = self.get_transform_matrix()
        self.transform_applied.emit(mat)
        self._updating_fields = True
        if self.position_row:
            self.position_row.reset_values()
        if self.rotation_row:
            self.rotation_row.reset_values()
        self._updating_fields = False

    def get_transform_matrix(self) -> np.ndarray:
        if not self.position_row or not self.rotation_row:
            return np.eye(4, dtype=np.float32)

        tx, ty, tz = self.position_row.get_values()
        rx_deg, ry_deg, rz_deg = self.rotation_row.get_values()

        rx = np.radians(rx_deg)
        ry = np.radians(ry_deg)
        rz = np.radians(rz_deg)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ], dtype=np.float32)

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ], dtype=np.float32)

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        R = Rz @ Ry @ Rx

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def on_disable(self):
        super().on_disable()


class MeshCleanupToolWindow(BaseEditorTool):
    """
    Concrete Tool: Mesh Cleanup & Optimization.
    Provides Face Reduction with a scrollable container, precision stepper, and background decimation.
    """

    apply_reduction_requested = Signal(float)
    revert_reduction_requested = Signal()
    apply_close_holes_requested = Signal(int)
    revert_close_holes_requested = Signal()
    apply_repair_nonmanifold_requested = Signal()
    revert_repair_nonmanifold_requested = Signal()
    apply_remove_duplicates_requested = Signal()
    revert_remove_duplicates_requested = Signal()
    apply_merge_vertices_requested = Signal(float, float)
    revert_merge_vertices_requested = Signal()
    apply_smooth_requested = Signal(float, int)
    revert_smooth_requested = Signal()
    revert_requested = Signal()
    retexture_requested = Signal()

    def __init__(self):
        icon_path = os.path.join("interface element", "clean up.svg")
        super().__init__(
            tool_id="cleanup",
            title="Mesh Cleanup",
            icon_path=icon_path,
            tooltip="Clean & Optimize 3D Mesh (Face Reduction, Holes, Non-Manifold, Duplicates, Merge Vertices, Smooth)"
        )
        self.spin_reduction = None
        self.stepper_reduction = None
        self.apply_btn = None
        self.revert_btn = None
        self.spin_hole = None
        self.stepper_hole = None
        self.apply_hole_btn = None
        self.revert_hole_btn = None
        self.apply_nonmanifold_btn = None
        self.revert_nonmanifold_btn = None
        self.apply_duplicates_btn = None
        self.revert_duplicates_btn = None
        self.btn_unit_pct = None
        self.btn_unit_abs = None
        self.spin_merge = None
        self.stepper_merge = None
        self.merge_equiv_label = None
        self.apply_merge_btn = None
        self.revert_merge_btn = None
        self.merge_unit_mode = "pct"
        self.bbox_diagonal = 1.0
        self.spin_smooth_lambda = None
        self.stepper_smooth_lambda = None
        self.spin_smooth_iter = None
        self.stepper_smooth_iter = None
        self.apply_smooth_btn = None
        self.revert_smooth_btn = None
        self.btn_retexture = None
        self.initial_hole_size = 30

    def on_enable(self, context: Optional[Dict[str, Any]] = None):
        super().on_enable(context)
        if context and "bbox_diagonal" in context:
            self.set_bbox_diagonal(float(context["bbox_diagonal"]))

    def on_gui(self, layout: QVBoxLayout, parent: QWidget):
        """Constructs the scrollable Mesh Cleanup UI controls."""
        # Scroll Area to ensure the tool window is scrollable
        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumHeight(240)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #181818;
                width: 6px;
                margin: 0px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #444444;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00E676;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        container = QWidget()
        container.setStyleSheet("background: transparent; border: none;")
        content_layout = QVBoxLayout(container)
        content_layout.setContentsMargins(0, 0, 4, 0)
        content_layout.setSpacing(10)

        # ---------------- Section 1: Face Reduction ----------------
        section_lbl = QLabel("Face Reduction", container)
        section_lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold; background: transparent; border: none;")
        content_layout.addWidget(section_lbl)

        # Input field row: Label + Stepper with percentage value
        input_container = QWidget(container)
        input_container.setStyleSheet("background: transparent; border: none;")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        lbl_reduc = QLabel("Face Reduction:", input_container)
        lbl_reduc.setStyleSheet("color: #e0e0e0; font-size: 11px; background: transparent; border: none;")

        self.spin_reduction = QSpinBox(input_container)
        self.spin_reduction.setRange(1, 99)
        self.spin_reduction.setValue(50)
        self.spin_reduction.setSuffix(" %")
        self.spin_reduction.setSingleStep(5)
        self.spin_reduction.setStyleSheet("""
            QSpinBox {
                background-color: #161616;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QSpinBox:focus {
                border-color: #00E676;
            }
        """)

        self.stepper_reduction = SpinBoxStepper(self.spin_reduction, input_container)

        input_layout.addWidget(lbl_reduc)
        input_layout.addWidget(self.stepper_reduction)
        content_layout.addWidget(input_container)

        # Action Buttons Row: Revert and Apply for Face Reduction
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.revert_btn = QPushButton("Revert", container)
        self.revert_btn.setProperty("class", "action-btn")
        self.revert_btn.setEnabled(False)
        self.revert_btn.setCursor(Qt.PointingHandCursor)
        self.revert_btn.clicked.connect(self._on_revert_clicked)

        self.apply_btn = QPushButton("Apply", container)
        self.apply_btn.setProperty("class", "primary-btn")
        self.apply_btn.setCursor(Qt.PointingHandCursor)
        self.apply_btn.clicked.connect(self._on_apply_clicked)

        action_row.addWidget(self.revert_btn)
        action_row.addWidget(self.apply_btn)
        content_layout.addLayout(action_row)

        # ---------------- Section 2: Close Holes ----------------
        section2_lbl = QLabel("Close Holes", container)
        section2_lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold; background: transparent; border: none; margin-top: 6px;")
        content_layout.addWidget(section2_lbl)

        # Input field row: Label + Stepper with max hole size (faces)
        hole_input_container = QWidget(container)
        hole_input_container.setStyleSheet("background: transparent; border: none;")
        hole_input_layout = QHBoxLayout(hole_input_container)
        hole_input_layout.setContentsMargins(0, 0, 0, 0)
        hole_input_layout.setSpacing(8)

        lbl_hole = QLabel("Max Hole Size (faces):", hole_input_container)
        lbl_hole.setStyleSheet("color: #e0e0e0; font-size: 11px; background: transparent; border: none;")

        self.spin_hole = QSpinBox(hole_input_container)
        self.spin_hole.setRange(5, 5000)
        self.spin_hole.setValue(getattr(self, 'initial_hole_size', 30))
        self.spin_hole.setSingleStep(10)
        self.spin_hole.setStyleSheet("""
            QSpinBox {
                background-color: #161616;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QSpinBox:focus {
                border-color: #00E676;
            }
        """)

        self.stepper_hole = SpinBoxStepper(self.spin_hole, hole_input_container)

        hole_input_layout.addWidget(lbl_hole)
        hole_input_layout.addWidget(self.stepper_hole)
        content_layout.addWidget(hole_input_container)

        # Action Buttons Row: Revert and Apply for Close Holes
        hole_action_row = QHBoxLayout()
        hole_action_row.setSpacing(8)

        self.revert_hole_btn = QPushButton("Revert", container)
        self.revert_hole_btn.setProperty("class", "action-btn")
        self.revert_hole_btn.setEnabled(False)
        self.revert_hole_btn.setCursor(Qt.PointingHandCursor)
        self.revert_hole_btn.clicked.connect(self._on_revert_hole_clicked)

        self.apply_hole_btn = QPushButton("Apply", container)
        self.apply_hole_btn.setProperty("class", "primary-btn")
        self.apply_hole_btn.setCursor(Qt.PointingHandCursor)
        self.apply_hole_btn.clicked.connect(self._on_apply_hole_clicked)

        hole_action_row.addWidget(self.revert_hole_btn)
        hole_action_row.addWidget(self.apply_hole_btn)
        content_layout.addLayout(hole_action_row)

        # ---------------- Section 3: Repair Non-Manifold ----------------
        section3_lbl = QLabel("Repair Non-Manifold", container)
        section3_lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold; background: transparent; border: none; margin-top: 6px;")
        content_layout.addWidget(section3_lbl)

        # Action Buttons Row: Revert and Apply for Repair Non-Manifold
        nm_action_row = QHBoxLayout()
        nm_action_row.setSpacing(8)

        self.revert_nonmanifold_btn = QPushButton("Revert", container)
        self.revert_nonmanifold_btn.setProperty("class", "action-btn")
        self.revert_nonmanifold_btn.setEnabled(False)
        self.revert_nonmanifold_btn.setCursor(Qt.PointingHandCursor)
        self.revert_nonmanifold_btn.clicked.connect(self._on_revert_nonmanifold_clicked)

        self.apply_nonmanifold_btn = QPushButton("Apply", container)
        self.apply_nonmanifold_btn.setProperty("class", "primary-btn")
        self.apply_nonmanifold_btn.setCursor(Qt.PointingHandCursor)
        self.apply_nonmanifold_btn.clicked.connect(self._on_apply_nonmanifold_clicked)

        nm_action_row.addWidget(self.revert_nonmanifold_btn)
        nm_action_row.addWidget(self.apply_nonmanifold_btn)
        content_layout.addLayout(nm_action_row)

        # ---------------- Section 4: Remove Duplicates ----------------
        section4_lbl = QLabel("Remove Duplicate Faces / Vertices", container)
        section4_lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold; background: transparent; border: none; margin-top: 6px;")
        content_layout.addWidget(section4_lbl)

        # Action Buttons Row: Revert and Apply for Remove Duplicates
        dup_action_row = QHBoxLayout()
        dup_action_row.setSpacing(8)

        self.revert_duplicates_btn = QPushButton("Revert", container)
        self.revert_duplicates_btn.setProperty("class", "action-btn")
        self.revert_duplicates_btn.setEnabled(False)
        self.revert_duplicates_btn.setCursor(Qt.PointingHandCursor)
        self.revert_duplicates_btn.clicked.connect(self._on_revert_duplicates_clicked)

        self.apply_duplicates_btn = QPushButton("Apply", container)
        self.apply_duplicates_btn.setProperty("class", "primary-btn")
        self.apply_duplicates_btn.setCursor(Qt.PointingHandCursor)
        self.apply_duplicates_btn.clicked.connect(self._on_apply_duplicates_clicked)

        dup_action_row.addWidget(self.revert_duplicates_btn)
        dup_action_row.addWidget(self.apply_duplicates_btn)
        content_layout.addLayout(dup_action_row)

        # ---------------- Section 5: Merge Close Vertices ----------------
        section5_lbl = QLabel("Merge Close Vertices", container)
        section5_lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold; background: transparent; border: none; margin-top: 6px;")
        content_layout.addWidget(section5_lbl)

        # Unit Toggle Buttons (% vs Absolute)
        unit_toggle_row = QHBoxLayout()
        unit_toggle_row.setContentsMargins(0, 0, 0, 0)
        unit_toggle_row.setSpacing(4)

        self.btn_unit_pct = QPushButton("%", container)
        self.btn_unit_pct.setCheckable(True)
        self.btn_unit_pct.setChecked(True)
        self.btn_unit_pct.setStyleSheet("""
            QPushButton {
                font-size: 10px; padding: 3px 8px; border-radius: 3px;
                background-color: #00E676; color: #121212; border: 1px solid #00E676; font-weight: bold;
            }
            QPushButton:!checked {
                background-color: #2D2D2D; color: #aaaaaa; border: 1px solid #444444; font-weight: normal;
            }
        """)

        self.btn_unit_abs = QPushButton("Absolute", container)
        self.btn_unit_abs.setCheckable(True)
        self.btn_unit_abs.setChecked(False)
        self.btn_unit_abs.setStyleSheet("""
            QPushButton {
                font-size: 10px; padding: 3px 8px; border-radius: 3px;
                background-color: #00E676; color: #121212; border: 1px solid #00E676; font-weight: bold;
            }
            QPushButton:!checked {
                background-color: #2D2D2D; color: #aaaaaa; border: 1px solid #444444; font-weight: normal;
            }
        """)

        self.btn_unit_pct.clicked.connect(lambda: self._set_merge_unit("pct"))
        self.btn_unit_abs.clicked.connect(lambda: self._set_merge_unit("abs"))

        unit_toggle_row.addWidget(self.btn_unit_pct)
        unit_toggle_row.addWidget(self.btn_unit_abs)
        unit_toggle_row.addStretch()
        content_layout.addLayout(unit_toggle_row)

        # Merge threshold input row
        merge_input_container = QWidget(container)
        merge_input_container.setStyleSheet("background: transparent; border: none;")
        merge_input_layout = QHBoxLayout(merge_input_container)
        merge_input_layout.setContentsMargins(0, 0, 0, 0)
        merge_input_layout.setSpacing(8)

        lbl_merge = QLabel("Distance Threshold:", merge_input_container)
        lbl_merge.setStyleSheet("color: #e0e0e0; font-size: 11px; background: transparent; border: none;")

        self.spin_merge = QDoubleSpinBox(merge_input_container)
        self.spin_merge.setRange(0.001, 50.0)
        self.spin_merge.setSingleStep(0.05)
        self.spin_merge.setDecimals(3)
        self.spin_merge.setSuffix(" %")
        self.spin_merge.setValue(0.2)
        self.spin_merge.valueChanged.connect(self._on_merge_value_changed)
        self.spin_merge.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #161616;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QDoubleSpinBox:focus {
                border-color: #00E676;
            }
        """)

        self.stepper_merge = SpinBoxStepper(self.spin_merge, merge_input_container)

        merge_input_layout.addWidget(lbl_merge)
        merge_input_layout.addWidget(self.stepper_merge)
        content_layout.addWidget(merge_input_container)

        self.merge_equiv_label = QLabel("Equivalent distance: ≈ 0.00000 units (abs)", container)
        self.merge_equiv_label.setStyleSheet("color: #00E676; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        content_layout.addWidget(self.merge_equiv_label)

        # Action Buttons Row: Revert and Apply for Merge Vertices
        merge_action_row = QHBoxLayout()
        merge_action_row.setSpacing(8)

        self.revert_merge_btn = QPushButton("Revert", container)
        self.revert_merge_btn.setProperty("class", "action-btn")
        self.revert_merge_btn.setEnabled(False)
        self.revert_merge_btn.setCursor(Qt.PointingHandCursor)
        self.revert_merge_btn.clicked.connect(self._on_revert_merge_clicked)

        self.apply_merge_btn = QPushButton("Apply", container)
        self.apply_merge_btn.setProperty("class", "primary-btn")
        self.apply_merge_btn.setCursor(Qt.PointingHandCursor)
        self.apply_merge_btn.clicked.connect(self._on_apply_merge_clicked)

        merge_action_row.addWidget(self.revert_merge_btn)
        merge_action_row.addWidget(self.apply_merge_btn)
        content_layout.addLayout(merge_action_row)

        self._update_merge_equiv_label()

        # ---------------- Section 6: Smooth Mesh ----------------
        section6_lbl = QLabel("Smooth Mesh (Taubin)", container)
        section6_lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: bold; background: transparent; border: none; margin-top: 6px;")
        content_layout.addWidget(section6_lbl)

        # Lambda input row
        lambda_container = QWidget(container)
        lambda_container.setStyleSheet("background: transparent; border: none;")
        lambda_layout = QHBoxLayout(lambda_container)
        lambda_layout.setContentsMargins(0, 0, 0, 0)
        lambda_layout.setSpacing(8)

        lbl_lambda = QLabel("Smoothing Factor (λ):", lambda_container)
        lbl_lambda.setStyleSheet("color: #e0e0e0; font-size: 11px; background: transparent; border: none;")

        self.spin_smooth_lambda = QDoubleSpinBox(lambda_container)
        self.spin_smooth_lambda.setRange(0.01, 1.00)
        self.spin_smooth_lambda.setSingleStep(0.05)
        self.spin_smooth_lambda.setDecimals(2)
        self.spin_smooth_lambda.setValue(0.50)
        self.spin_smooth_lambda.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #161616;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QDoubleSpinBox:focus {
                border-color: #00E676;
            }
        """)

        self.stepper_smooth_lambda = SpinBoxStepper(self.spin_smooth_lambda, lambda_container)

        lambda_layout.addWidget(lbl_lambda)
        lambda_layout.addWidget(self.stepper_smooth_lambda)
        content_layout.addWidget(lambda_container)

        # Iterations input row
        iter_container = QWidget(container)
        iter_container.setStyleSheet("background: transparent; border: none;")
        iter_layout = QHBoxLayout(iter_container)
        iter_layout.setContentsMargins(0, 0, 0, 0)
        iter_layout.setSpacing(8)

        lbl_iter = QLabel("Iterations (steps):", iter_container)
        lbl_iter.setStyleSheet("color: #e0e0e0; font-size: 11px; background: transparent; border: none;")

        self.spin_smooth_iter = QSpinBox(iter_container)
        self.spin_smooth_iter.setRange(1, 100)
        self.spin_smooth_iter.setSingleStep(1)
        self.spin_smooth_iter.setValue(10)
        self.spin_smooth_iter.setStyleSheet("""
            QSpinBox {
                background-color: #161616;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QSpinBox:focus {
                border-color: #00E676;
            }
        """)

        self.stepper_smooth_iter = SpinBoxStepper(self.spin_smooth_iter, iter_container)

        iter_layout.addWidget(lbl_iter)
        iter_layout.addWidget(self.stepper_smooth_iter)
        content_layout.addWidget(iter_container)

        # Action Buttons Row: Revert and Apply for Smooth Mesh
        smooth_action_row = QHBoxLayout()
        smooth_action_row.setSpacing(8)

        self.revert_smooth_btn = QPushButton("Revert", container)
        self.revert_smooth_btn.setProperty("class", "action-btn")
        self.revert_smooth_btn.setEnabled(False)
        self.revert_smooth_btn.setCursor(Qt.PointingHandCursor)
        self.revert_smooth_btn.clicked.connect(self._on_revert_smooth_clicked)

        self.apply_smooth_btn = QPushButton("Apply", container)
        self.apply_smooth_btn.setProperty("class", "primary-btn")
        self.apply_smooth_btn.setCursor(Qt.PointingHandCursor)
        self.apply_smooth_btn.clicked.connect(self._on_apply_smooth_clicked)

        smooth_action_row.addWidget(self.revert_smooth_btn)
        smooth_action_row.addWidget(self.apply_smooth_btn)
        content_layout.addLayout(smooth_action_row)

        content_layout.addStretch()

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # ---------------- Retexture Button (Fixed at Bottom) ----------------
        self.btn_retexture = QPushButton("Retexture", parent)
        self.btn_retexture.setToolTip("Reruns OpenMVS texture projection on the modified mesh geometry.")
        self.btn_retexture.setCursor(Qt.PointingHandCursor)
        self.btn_retexture.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 7px 14px; font-weight: bold;
                background-color: #00C853; color: #121212;
                border: 1px solid #00C853; border-radius: 4px;
            }
            QPushButton:hover { background-color: #00E676; }
            QPushButton:disabled { background-color: #2D2D2D; color: #666666; border-color: #444444; }
        """)
        self.btn_retexture.clicked.connect(self._on_retexture_clicked)
        layout.addWidget(self.btn_retexture)

    def _on_apply_clicked(self):
        if self.spin_reduction:
            val = float(self.spin_reduction.value())
            self.apply_reduction_requested.emit(val)

    def _on_revert_clicked(self):
        self.revert_reduction_requested.emit()
        self.revert_requested.emit()

    def _on_apply_hole_clicked(self):
        if self.spin_hole:
            val = int(self.spin_hole.value())
            self.apply_close_holes_requested.emit(val)

    def _on_revert_hole_clicked(self):
        self.revert_close_holes_requested.emit()

    def _on_apply_nonmanifold_clicked(self):
        self.apply_repair_nonmanifold_requested.emit()

    def _on_revert_nonmanifold_clicked(self):
        self.revert_repair_nonmanifold_requested.emit()

    def _on_apply_duplicates_clicked(self):
        self.apply_remove_duplicates_requested.emit()

    def _on_revert_duplicates_clicked(self):
        self.revert_remove_duplicates_requested.emit()

    def _set_merge_unit(self, unit: str):
        self.merge_unit_mode = unit
        if unit == "pct":
            if self.btn_unit_pct:
                self.btn_unit_pct.setChecked(True)
            if self.btn_unit_abs:
                self.btn_unit_abs.setChecked(False)
            if self.spin_merge:
                self.spin_merge.blockSignals(True)
                self.spin_merge.setRange(0.001, 50.0)
                self.spin_merge.setSingleStep(0.05)
                self.spin_merge.setDecimals(3)
                self.spin_merge.setSuffix(" %")
                self.spin_merge.setValue(0.2)
                self.spin_merge.blockSignals(False)
        else:
            if self.btn_unit_pct:
                self.btn_unit_pct.setChecked(False)
            if self.btn_unit_abs:
                self.btn_unit_abs.setChecked(True)
            if self.spin_merge:
                self.spin_merge.blockSignals(True)
                self.spin_merge.setRange(0.00001, 10.0)
                self.spin_merge.setSingleStep(0.001)
                self.spin_merge.setDecimals(5)
                self.spin_merge.setSuffix(" units")
                equiv_val = 0.002 * getattr(self, 'bbox_diagonal', 1.0)
                self.spin_merge.setValue(max(0.00001, equiv_val))
                self.spin_merge.blockSignals(False)
        self._update_merge_equiv_label()

    def _on_merge_value_changed(self, val: float):
        self._update_merge_equiv_label()

    def _update_merge_equiv_label(self):
        if not self.merge_equiv_label or not self.spin_merge:
            return
        val = self.spin_merge.value()
        diag = max(0.0001, getattr(self, 'bbox_diagonal', 1.0))
        if getattr(self, 'merge_unit_mode', 'pct') == "pct":
            equiv_abs = (val / 100.0) * diag
            self.merge_equiv_label.setText(f"Equivalent distance: ≈ {equiv_abs:.5f} units (abs)")
        else:
            equiv_pct = (val / diag) * 100.0 if diag > 0 else 0.0
            self.merge_equiv_label.setText(f"Equivalent percentage: ≈ {equiv_pct:.3f} % of diagonal")

    def set_bbox_diagonal(self, bbox_diagonal: float):
        self.bbox_diagonal = max(0.0001, bbox_diagonal)
        self._update_merge_equiv_label()

    def _on_apply_merge_clicked(self):
        if self.spin_merge:
            val = float(self.spin_merge.value())
            diag = max(0.0001, getattr(self, 'bbox_diagonal', 1.0))
            if getattr(self, 'merge_unit_mode', 'pct') == "pct":
                threshold_pct = val
            else:
                threshold_pct = (val / diag) * 100.0
            self.apply_merge_vertices_requested.emit(threshold_pct, diag)

    def _on_revert_merge_clicked(self):
        self.revert_merge_vertices_requested.emit()

    def _on_apply_smooth_clicked(self):
        if self.spin_smooth_lambda and self.spin_smooth_iter:
            lambda_val = float(self.spin_smooth_lambda.value())
            iter_val = int(self.spin_smooth_iter.value())
            self.apply_smooth_requested.emit(lambda_val, iter_val)

    def _on_revert_smooth_clicked(self):
        self.revert_smooth_requested.emit()

    def _on_retexture_clicked(self):
        self.retexture_requested.emit()

    def set_revert_enabled(self, enabled: bool):
        if self.revert_btn:
            self.revert_btn.setEnabled(enabled)

    def set_holes_revert_enabled(self, enabled: bool):
        if self.revert_hole_btn:
            self.revert_hole_btn.setEnabled(enabled)

    def set_nonmanifold_revert_enabled(self, enabled: bool):
        if self.revert_nonmanifold_btn:
            self.revert_nonmanifold_btn.setEnabled(enabled)

    def set_duplicates_revert_enabled(self, enabled: bool):
        if self.revert_duplicates_btn:
            self.revert_duplicates_btn.setEnabled(enabled)

    def set_merge_revert_enabled(self, enabled: bool):
        if self.revert_merge_btn:
            self.revert_merge_btn.setEnabled(enabled)

    def set_smooth_revert_enabled(self, enabled: bool):
        if self.revert_smooth_btn:
            self.revert_smooth_btn.setEnabled(enabled)

    def set_busy(self, is_busy: bool):
        if is_busy:
            # All cleanup actions share one backup: only the latest completed
            # action may enable Revert, and no revert may run during a worker.
            self.set_revert_enabled(False)
            self.set_holes_revert_enabled(False)
            self.set_nonmanifold_revert_enabled(False)
            self.set_duplicates_revert_enabled(False)
            self.set_merge_revert_enabled(False)
            self.set_smooth_revert_enabled(False)
        if self.apply_btn:
            self.apply_btn.setEnabled(not is_busy)
        if self.stepper_reduction:
            self.stepper_reduction.setEnabled(not is_busy)
        if self.apply_hole_btn:
            self.apply_hole_btn.setEnabled(not is_busy)
        if self.stepper_hole:
            self.stepper_hole.setEnabled(not is_busy)
        if self.apply_nonmanifold_btn:
            self.apply_nonmanifold_btn.setEnabled(not is_busy)
        if self.stepper_merge:
            self.stepper_merge.setEnabled(not is_busy)
        if self.apply_merge_btn:
            self.apply_merge_btn.setEnabled(not is_busy)
        if self.apply_duplicates_btn:
            self.apply_duplicates_btn.setEnabled(not is_busy)
        if self.apply_smooth_btn:
            self.apply_smooth_btn.setEnabled(not is_busy)
        if self.stepper_smooth_lambda:
            self.stepper_smooth_lambda.setEnabled(not is_busy)
        if self.stepper_smooth_iter:
            self.stepper_smooth_iter.setEnabled(not is_busy)
        if self.btn_retexture:
            self.btn_retexture.setEnabled(not is_busy)

    def on_disable(self):
        super().on_disable()


class MeshCutToolWindow(BaseEditorTool):
    """
    Concrete Tool: Cut & Crop Mesh.
    Provides Bounding Box Crop, Box Selection, and Lasso Selection.
    """
    mode_changed = Signal(str)            # 'crop_box', 'box', 'lasso', 'none'
    crop_requested = Signal()              # RealityScan Bounding Box Crop
    reset_crop_requested = Signal()        # Reset Bounding Box
    delete_selection_requested = Signal()  # Delete selected vertices/faces
    clear_selection_requested = Signal()   # Clear active selection
    invert_selection_requested = Signal()  # Invert selection
    retexture_requested = Signal()        # Retexture modified mesh

    def __init__(self):
        icon_path = os.path.join("interface element", "scissors.svg")
        super().__init__(
            tool_id="cut",
            title="Cut & Select",
            icon_path=icon_path,
            tooltip="Cut & Select 3D Mesh (Bounding Box Crop, Box Select, Lasso Select)"
        )
        self.current_mode = "crop_box"
        self.btn_mode_crop = None
        self.btn_mode_box = None
        self.btn_mode_lasso = None
        self.crop_panel = None
        self.select_panel = None
        self.select_hint_label = None
        self.btn_crop = None
        self.btn_reset_crop = None
        self.btn_delete_selection = None
        self.btn_clear_selection = None
        self.btn_invert_selection = None
        self.btn_retexture = None

    def on_enable(self, context: Optional[Dict[str, Any]] = None):
        super().on_enable(context)
        self.mode_changed.emit(self.current_mode)

    def on_disable(self):
        super().on_disable()
        self.mode_changed.emit('none')

    def on_gui(self, layout: QVBoxLayout, parent: QWidget):
        # 1. Mode Switcher (Bounding Box | Box Select | Lasso Select)
        mode_btn_row = QHBoxLayout()
        mode_btn_row.setContentsMargins(0, 0, 0, 0)
        mode_btn_row.setSpacing(4)

        mode_style = """
            QPushButton {
                font-size: 10px;
                padding: 4px 6px;
                background-color: #2D2D2D;
                color: #AAAAAA;
                border: 1px solid #444444;
                border-radius: 4px;
                font-weight: normal;
            }
            QPushButton:checked {
                background-color: #1B382B;
                color: #00E676;
                border: 1px solid #00E676;
                font-weight: bold;
            }
            QPushButton:hover:!checked {
                background-color: #383838;
                color: #FFFFFF;
            }
        """

        self.btn_mode_crop = QPushButton("Bounding Box", parent)
        self.btn_mode_crop.setCheckable(True)
        self.btn_mode_crop.setChecked(self.current_mode == "crop_box")
        self.btn_mode_crop.setStyleSheet(mode_style)
        self.btn_mode_crop.setCursor(Qt.PointingHandCursor)

        self.btn_mode_box = QPushButton("Box Select", parent)
        self.btn_mode_box.setCheckable(True)
        self.btn_mode_box.setChecked(self.current_mode == "box")
        self.btn_mode_box.setStyleSheet(mode_style)
        self.btn_mode_box.setCursor(Qt.PointingHandCursor)

        self.btn_mode_lasso = QPushButton("Lasso Select", parent)
        self.btn_mode_lasso.setCheckable(True)
        self.btn_mode_lasso.setChecked(self.current_mode == "lasso")
        self.btn_mode_lasso.setStyleSheet(mode_style)
        self.btn_mode_lasso.setCursor(Qt.PointingHandCursor)

        self.btn_mode_crop.clicked.connect(lambda: self._set_mode("crop_box"))
        self.btn_mode_box.clicked.connect(lambda: self._set_mode("box"))
        self.btn_mode_lasso.clicked.connect(lambda: self._set_mode("lasso"))

        mode_btn_row.addWidget(self.btn_mode_crop)
        mode_btn_row.addWidget(self.btn_mode_box)
        mode_btn_row.addWidget(self.btn_mode_lasso)
        layout.addLayout(mode_btn_row)

        # 2. Bounding Box Mode Panel
        self.crop_panel = QWidget(parent)
        self.crop_panel.setStyleSheet("background: transparent; border: none;")
        crop_layout = QVBoxLayout(self.crop_panel)
        crop_layout.setContentsMargins(0, 6, 0, 4)
        crop_layout.setSpacing(8)

        crop_info = QLabel("Adjust the 3D bounding box corners and planes in the viewport to enclose the region of interest.", self.crop_panel)
        crop_info.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        crop_info.setWordWrap(True)
        crop_layout.addWidget(crop_info)

        crop_action_row = QHBoxLayout()
        crop_action_row.setSpacing(8)

        self.btn_crop = QPushButton("Crop", self.crop_panel)
        self.btn_crop.setToolTip("Deletes all mesh vertices/faces outside the crop box")
        self.btn_crop.setProperty("class", "primary-btn")
        self.btn_crop.setCursor(Qt.PointingHandCursor)
        self.btn_crop.clicked.connect(self.crop_requested.emit)

        self.btn_reset_crop = QPushButton("Reset", self.crop_panel)
        self.btn_reset_crop.setToolTip("Resets bounding box bounds to the initial mesh boundaries")
        self.btn_reset_crop.setProperty("class", "action-btn")
        self.btn_reset_crop.setCursor(Qt.PointingHandCursor)
        self.btn_reset_crop.clicked.connect(self.reset_crop_requested.emit)

        crop_action_row.addWidget(self.btn_crop)
        crop_action_row.addWidget(self.btn_reset_crop)
        crop_layout.addLayout(crop_action_row)

        layout.addWidget(self.crop_panel)

        # 3. Selection Mode Panel (Box / Lasso)
        self.select_panel = QWidget(parent)
        self.select_panel.setStyleSheet("background: transparent; border: none;")
        select_layout = QVBoxLayout(self.select_panel)
        select_layout.setContentsMargins(0, 6, 0, 4)
        select_layout.setSpacing(8)

        self.select_hint_label = QLabel("Hold Ctrl + Left Click & Drag in viewport to select vertices.", self.select_panel)
        self.select_hint_label.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        self.select_hint_label.setWordWrap(True)
        select_layout.addWidget(self.select_hint_label)

        select_action_row1 = QHBoxLayout()
        select_action_row1.setSpacing(8)

        self.btn_delete_selection = QPushButton("Delete Selected", self.select_panel)
        self.btn_delete_selection.setToolTip("Deletes all selected mesh triangles/vertices")
        self.btn_delete_selection.setCursor(Qt.PointingHandCursor)
        self.btn_delete_selection.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 5px 12px; font-weight: bold;
                background-color: #E53935; color: #ffffff;
                border: 1px solid #E53935; border-radius: 4px;
            }
            QPushButton:hover { background-color: #D32F2F; }
            QPushButton:disabled { background-color: #2D2D2D; color: #666666; border-color: #444444; }
        """)
        self.btn_delete_selection.clicked.connect(self.delete_selection_requested.emit)

        self.btn_invert_selection = QPushButton("Invert", self.select_panel)
        self.btn_invert_selection.setToolTip("Inverts the current selection highlighting")
        self.btn_invert_selection.setProperty("class", "action-btn")
        self.btn_invert_selection.setCursor(Qt.PointingHandCursor)
        self.btn_invert_selection.clicked.connect(self.invert_selection_requested.emit)

        self.btn_clear_selection = QPushButton("Clear", self.select_panel)
        self.btn_clear_selection.setToolTip("Clears the current selection highlighting")
        self.btn_clear_selection.setProperty("class", "action-btn")
        self.btn_clear_selection.setCursor(Qt.PointingHandCursor)
        self.btn_clear_selection.clicked.connect(self.clear_selection_requested.emit)

        select_action_row1.addWidget(self.btn_delete_selection)
        select_action_row1.addWidget(self.btn_invert_selection)
        select_action_row1.addWidget(self.btn_clear_selection)
        select_layout.addLayout(select_action_row1)

        layout.addWidget(self.select_panel)

        # 4. Retexture Button at Bottom
        self.btn_retexture = QPushButton("Retexture", parent)
        self.btn_retexture.setToolTip("Reruns OpenMVS texture projection on the modified mesh geometry.")
        self.btn_retexture.setCursor(Qt.PointingHandCursor)
        self.btn_retexture.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 7px 14px; font-weight: bold;
                background-color: #00C853; color: #121212;
                border: 1px solid #00C853; border-radius: 4px;
            }
            QPushButton:hover { background-color: #00E676; }
            QPushButton:disabled { background-color: #2D2D2D; color: #666666; border-color: #444444; }
        """)
        self.btn_retexture.clicked.connect(self.retexture_requested.emit)
        layout.addWidget(self.btn_retexture)

        self._update_panel_visibility()

    def _set_mode(self, mode: str):
        self.current_mode = mode
        if self.btn_mode_crop:
            self.btn_mode_crop.setChecked(mode == "crop_box")
        if self.btn_mode_box:
            self.btn_mode_box.setChecked(mode == "box")
        if self.btn_mode_lasso:
            self.btn_mode_lasso.setChecked(mode == "lasso")
        self._update_panel_visibility()
        self.mode_changed.emit(mode)

    def _update_panel_visibility(self):
        is_crop = (self.current_mode == "crop_box")
        if self.crop_panel:
            self.crop_panel.setVisible(is_crop)
        if self.select_panel:
            self.select_panel.setVisible(not is_crop)
        if self.select_hint_label:
            if self.current_mode == "lasso":
                self.select_hint_label.setText("Hold Ctrl + Left Click & Drag in viewport to draw a lasso selection around vertices.")
            else:
                self.select_hint_label.setText("Hold Ctrl + Left Click & Drag in viewport to box select vertices.")

    def set_busy(self, is_busy: bool):
        if self.btn_crop:
            self.btn_crop.setEnabled(not is_busy)
        if self.btn_reset_crop:
            self.btn_reset_crop.setEnabled(not is_busy)
        if self.btn_delete_selection:
            self.btn_delete_selection.setEnabled(not is_busy)
        if self.btn_invert_selection:
            self.btn_invert_selection.setEnabled(not is_busy)
        if self.btn_clear_selection:
            self.btn_clear_selection.setEnabled(not is_busy)
        if self.btn_retexture:
            self.btn_retexture.setEnabled(not is_busy)
        if self.btn_mode_crop:
            self.btn_mode_crop.setEnabled(not is_busy)
        if self.btn_mode_box:
            self.btn_mode_box.setEnabled(not is_busy)
        if self.btn_mode_lasso:
            self.btn_mode_lasso.setEnabled(not is_busy)


class CameraRegistrationToolWindow(BaseEditorTool):
    """
    Concrete Tool: Camera Registration & Display.
    Provides camera wireframe frustum toggling, photo plane toggling,
    frustum scaling, and photo opacity adjustments.
    """
    cameras_toggled = Signal(bool)
    photos_toggled = Signal(bool)
    scale_changed = Signal(float)
    opacity_changed = Signal(float)

    def __init__(self):
        icon_path = os.path.join("interface element", "point cloud.svg")
        super().__init__(
            tool_id="camera_registration",
            title="Camera Registration",
            icon_path=icon_path,
            tooltip="Camera Registration & Display (Frustums, Image Planes, Scale & Opacity)",
            requires_data=True,
            empty_message="No registered cameras found in current sparse model"
        )
        self.btn_show_cameras = None
        self.btn_show_photos = None
        self.scale_slider = None
        self.scale_val_lbl = None
        self.opacity_slider = None
        self.opacity_val_lbl = None

        self._cameras_checked = True
        self._photos_checked = True
        self._scale_val = 1.0
        self._opacity_val = 0.85

    def on_gui(self, layout: QVBoxLayout, parent: QWidget):
        # 1. Display Toggles
        toggle_box = QGroupBox("Display", parent)
        toggle_layout = QHBoxLayout(toggle_box)
        toggle_layout.setContentsMargins(8, 12, 8, 8)
        toggle_layout.setSpacing(8)

        toggle_style = """
            QPushButton {
                font-size: 11px;
                padding: 5px 10px;
                background-color: #2D2D2D;
                color: #AAAAAA;
                border: 1px solid #444444;
                border-radius: 4px;
                font-weight: normal;
            }
            QPushButton:checked {
                background-color: #1B382B;
                color: #00E676;
                border: 1px solid #00E676;
                font-weight: bold;
            }
            QPushButton:hover:!checked {
                background-color: #383838;
                color: #FFFFFF;
            }
        """

        self.btn_show_cameras = QPushButton("Cameras", toggle_box)
        self.btn_show_cameras.setCheckable(True)
        self.btn_show_cameras.setChecked(self._cameras_checked)
        self.btn_show_cameras.setStyleSheet(toggle_style)
        self.btn_show_cameras.setCursor(Qt.PointingHandCursor)
        self.btn_show_cameras.toggled.connect(self._on_cameras_toggled)

        self.btn_show_photos = QPushButton("Photos", toggle_box)
        self.btn_show_photos.setCheckable(True)
        self.btn_show_photos.setChecked(self._photos_checked)
        self.btn_show_photos.setStyleSheet(toggle_style)
        self.btn_show_photos.setCursor(Qt.PointingHandCursor)
        self.btn_show_photos.toggled.connect(self._on_photos_toggled)

        toggle_layout.addWidget(self.btn_show_cameras)
        toggle_layout.addWidget(self.btn_show_photos)
        layout.addWidget(toggle_box)

        # 2. Camera Settings (Scale & Opacity Sliders)
        settings_box = QGroupBox("Settings", parent)
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setContentsMargins(8, 12, 8, 8)
        settings_layout.setSpacing(10)

        lbl_style = "color: #CCCCCC; font-size: 11px;"
        slider_style = """
            QSlider::groove:horizontal {
                background: #333333;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00E676;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #00E676;
                border-radius: 2px;
            }
        """

        # Frustum Scale
        scale_header = QLabel("Frustum Scale", settings_box)
        scale_header.setStyleSheet(lbl_style)
        settings_layout.addWidget(scale_header)

        scale_row = QHBoxLayout()
        self.scale_slider = QSlider(Qt.Horizontal, settings_box)
        self.scale_slider.setRange(20, 300)
        self.scale_slider.setValue(int(round(self._scale_val * 100)))
        self.scale_slider.setStyleSheet(slider_style)
        self.scale_slider.valueChanged.connect(self._on_scale_slider_changed)

        self.scale_val_lbl = QLabel(f"{self._scale_val:.1f}x", settings_box)
        self.scale_val_lbl.setStyleSheet(lbl_style)
        self.scale_val_lbl.setFixedWidth(36)

        scale_row.addWidget(self.scale_slider)
        scale_row.addWidget(self.scale_val_lbl)
        settings_layout.addLayout(scale_row)

        # Photo Opacity
        opacity_header = QLabel("Photo Opacity", settings_box)
        opacity_header.setStyleSheet(lbl_style)
        settings_layout.addWidget(opacity_header)

        opacity_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal, settings_box)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(round(self._opacity_val * 100)))
        self.opacity_slider.setStyleSheet(slider_style)
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)

        self.opacity_val_lbl = QLabel(f"{int(round(self._opacity_val * 100))}%", settings_box)
        self.opacity_val_lbl.setStyleSheet(lbl_style)
        self.opacity_val_lbl.setFixedWidth(36)

        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_val_lbl)
        settings_layout.addLayout(opacity_row)

        layout.addWidget(settings_box)

        # 3. Instruction label
        hint_lbl = QLabel("Click any camera frustum in the 3D viewport to inspect the photo and position.", parent)
        hint_lbl.setStyleSheet("color: #888888; font-size: 10px; padding-top: 4px;")
        hint_lbl.setWordWrap(True)
        layout.addWidget(hint_lbl)

    def _on_cameras_toggled(self, checked: bool):
        self._cameras_checked = checked
        self.cameras_toggled.emit(checked)

    def _on_photos_toggled(self, checked: bool):
        self._photos_checked = checked
        self.photos_toggled.emit(checked)

    def _on_scale_slider_changed(self, value: int):
        self._scale_val = value / 100.0
        if self.scale_val_lbl:
            self.scale_val_lbl.setText(f"{self._scale_val:.1f}x")
        self.scale_changed.emit(self._scale_val)

    def _on_opacity_slider_changed(self, value: int):
        self._opacity_val = value / 100.0
        if self.opacity_val_lbl:
            self.opacity_val_lbl.setText(f"{value}%")
        self.opacity_changed.emit(self._opacity_val)

    def set_busy(self, is_busy: bool):
        if self.btn_show_cameras:
            self.btn_show_cameras.setEnabled(not is_busy)
        if self.btn_show_photos:
            self.btn_show_photos.setEnabled(not is_busy)
        if self.scale_slider:
            self.scale_slider.setEnabled(not is_busy)
        if self.opacity_slider:
            self.opacity_slider.setEnabled(not is_busy)

