import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QMouseEvent

class NavGizmoWidget(QWidget):
    """
    Blender-style 3D view navigation gizmo drawn on screen with QPainter.
    Displays X (Red), Y (Green), Z (Blue) axes and their negative counterparts.
    Clicking any axis snaps the viewport camera to that canonical view.
    Supports both Z-up (Mesh Editor OpenGL) and Y-up (VisPy 3D Reconstruction) coordinate systems.
    """
    snap_requested = Signal(str)  # Emits the view name: "front", "back", "right", "left", "top", "bottom"

    def __init__(self, parent=None, coord_system: str = "z-up"):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self.setMouseTracking(True)
        self.coord_system = coord_system.lower()
        
        # Camera angles
        self.yaw = 0.0
        self.pitch = 0.0
        
        # Custom camera basis vectors (if supplied directly, e.g. from VisPy)
        self._custom_basis = None  # (right, up, v_dir)
        
        # Gizmo configuration
        self.radius = 35.0  # Axis line length in pixels
        self.dot_radius_pos = 9.0
        self.dot_radius_neg = 6.0
        
        # Hover state
        self.hovered_axis = None  # None or tuple (name, is_pos)
        
        self._init_axes()

    def _init_axes(self):
        """Initializes 3D axis vectors and canonical view mappings based on coordinate system."""
        if self.coord_system == "y-up":
            # VisPy coordinate system (Y-up, X-right, Z-front)
            self.axes = {
                "X": (np.array([1.0, 0.0, 0.0], dtype=np.float32), QColor(255, 60, 50), "+X", "right"),
                "-X": (np.array([-1.0, 0.0, 0.0], dtype=np.float32), QColor(180, 50, 45), "-X", "left"),
                "Y": (np.array([0.0, 1.0, 0.0], dtype=np.float32), QColor(50, 200, 80), "+Y", "top"),
                "-Y": (np.array([0.0, -1.0, 0.0], dtype=np.float32), QColor(40, 150, 65), "-Y", "bottom"),
                "Z": (np.array([0.0, 0.0, 1.0], dtype=np.float32), QColor(30, 140, 255), "+Z", "front"),
                "-Z": (np.array([0.0, 0.0, -1.0], dtype=np.float32), QColor(25, 100, 190), "-Z", "back"),
            }
        else:
            # Z-up coordinate system (Mesh Editor / Blender)
            self.axes = {
                "X": (np.array([1.0, 0.0, 0.0], dtype=np.float32), QColor(255, 60, 50), "+X", "right"),
                "-X": (np.array([-1.0, 0.0, 0.0], dtype=np.float32), QColor(180, 50, 45), "-X", "left"),
                "Y": (np.array([0.0, 1.0, 0.0], dtype=np.float32), QColor(50, 200, 80), "+Y", "back"),
                "-Y": (np.array([0.0, -1.0, 0.0], dtype=np.float32), QColor(40, 150, 65), "-Y", "front"),
                "Z": (np.array([0.0, 0.0, 1.0], dtype=np.float32), QColor(30, 140, 255), "+Z", "top"),
                "-Z": (np.array([0.0, 0.0, -1.0], dtype=np.float32), QColor(25, 100, 190), "-Z", "bottom"),
            }

    def set_coordinate_system(self, coord_system: str):
        """Switch between 'z-up' and 'y-up' coordinate conventions."""
        self.coord_system = coord_system.lower()
        self._init_axes()
        self.update()

    def update_orientation(self, yaw: float, pitch: float):
        """Updates internal camera angles (Z-up standard) and redraws."""
        self._custom_basis = None
        self.yaw = yaw
        self.pitch = pitch
        self.update()

    def update_from_vispy(self, azimuth_deg: float, elevation_deg: float):
        """
        Updates gizmo orientation directly from VisPy TurntableCamera (Y-up standard).
        Computes the camera basis vectors in real-time.
        """
        self.coord_system = "y-up"
        az = np.radians(azimuth_deg)
        el = np.radians(elevation_deg)
        
        cos_el = np.cos(el)
        sin_el = np.sin(el)
        cos_az = np.cos(az)
        sin_az = np.sin(az)
        
        # Camera right vector: [cos_az, 0, -sin_az]
        right = np.array([cos_az, 0.0, -sin_az], dtype=np.float32)
        
        # Camera up vector: [-sin_el * sin_az, cos_el, -sin_el * cos_az]
        up = np.array([-sin_el * sin_az, cos_el, -sin_el * cos_az], dtype=np.float32)
        
        # View direction (from camera towards target at origin):
        v_dir = np.array([-cos_el * sin_az, -sin_el, -cos_el * cos_az], dtype=np.float32)
        
        self._custom_basis = (right, up, v_dir)
        self.update()

    def _get_projected_axes(self):
        """
        Projects the 3D axes to 2D widget coordinates using either custom basis or yaw/pitch.
        Returns a sorted list of axis dicts ordered from back-to-front (largest depth to smallest depth).
        """
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        
        if self._custom_basis is not None:
            right, up, v_dir = self._custom_basis
        else:
            cos_pitch = np.cos(self.pitch)
            sin_pitch = np.sin(self.pitch)
            cos_yaw = np.cos(self.yaw)
            sin_yaw = np.sin(self.yaw)
            
            # Camera right vector: [cos_yaw, sin_yaw, 0]
            right = np.array([cos_yaw, sin_yaw, 0.0], dtype=np.float32)
            
            # Camera up vector: [-sin_yaw * sin_pitch, cos_yaw * sin_pitch, cos_pitch]
            up = np.array([-sin_yaw * sin_pitch, cos_yaw * sin_pitch, cos_pitch], dtype=np.float32)
            
            # View direction (pointing from camera to target):
            v_dir = np.array([-cos_pitch * sin_yaw, cos_pitch * cos_yaw, -sin_pitch], dtype=np.float32)
        
        projected = []
        for name, (vec, color, label, view_name) in self.axes.items():
            is_pos = not name.startswith("-")
            
            # Project onto screen plane using right and up vectors
            dx = float(np.dot(vec, right))
            dy = float(np.dot(vec, up))
            
            # Depth along camera view direction (positive = away, negative = towards viewer)
            depth = float(np.dot(vec, v_dir))
            
            # Screen space mapping (Qt Y goes down)
            sx = center_x + self.radius * dx
            sy = center_y - self.radius * dy
            
            projected.append({
                "name": name,
                "sx": sx,
                "sy": sy,
                "depth": depth,
                "color": color,
                "view_name": view_name,
                "is_pos": is_pos,
                "label": label
            })
            
        # Sort by depth descending (back to front) so we paint items in the background first
        projected.sort(key=lambda item: item["depth"], reverse=True)
        return projected

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill widget area with slight dark circular mask for background overlay
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(30, 30, 30, 80)))
        painter.drawEllipse(QPointF(center_x, center_y), 45.0, 45.0)
        
        # Outer boundary ring
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(center_x, center_y), 44.0, 44.0)

        # Get sorted projected axes
        axes_list = self._get_projected_axes()
        
        # Font settings for label rendering
        font = QFont("Outfit", 8, QFont.Weight.Bold)
        painter.setFont(font)
        
        # 1. Draw Axis lines first (only for positive axes, or all depending on view)
        for item in axes_list:
            # We draw lines from center to the dot
            # Lines are drawn only for positive axes to keep it clean (like Blender)
            if item["is_pos"]:
                # Fade color if it is pointing away (depth > 0)
                alpha = 80 if item["depth"] > 0 else 255
                color = QColor(item["color"])
                color.setAlpha(alpha)
                
                painter.setPen(QPen(color, 2.0))
                painter.drawLine(QPointF(center_x, center_y), QPointF(item["sx"], item["sy"]))
                
        # 2. Draw dots on top
        for item in axes_list:
            name = item["name"]
            sx = item["sx"]
            sy = item["sy"]
            is_pos = item["is_pos"]
            color = QColor(item["color"])
            
            is_hovered = (self.hovered_axis == name)
            
            # Decide dot radius and styling based on positive/negative axis and hover state
            if is_pos:
                r = self.dot_radius_pos + (2.0 if is_hovered else 0.0)
                # If pointing away (depth > 0), fade it slightly
                alpha = 150 if item["depth"] > 0 else 255
                color.setAlpha(alpha)
                
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(sx, sy), r, r)
                
                # Draw letter label inside positive axis dots
                label_color = QColor(255, 255, 255, 255 if item["depth"] <= 0 else 180)
                painter.setPen(QPen(label_color, 1.0))
                
                # Centered text drawing
                text = item["label"].replace("+", "")
                text_rect = painter.fontMetrics().boundingRect(text)
                painter.drawText(QPointF(sx - text_rect.width() / 2.0, sy + text_rect.height() / 3.0), text)
            else:
                # Negative axis
                r = self.dot_radius_neg + (1.5 if is_hovered else 0.0)
                
                # Transparent filled dot with color border
                alpha_border = 120 if item["depth"] > 0 else 220
                color.setAlpha(alpha_border)
                
                painter.setPen(QPen(color, 1.5))
                # Fill color is dark transparent
                painter.setBrush(QBrush(QColor(20, 20, 20, 180)))
                painter.drawEllipse(QPointF(sx, sy), r, r)

        painter.end()

    def _hit_test(self, pos):
        """Returns the name of the axis under the mouse coordinates, if any."""
        px = pos.x()
        py = pos.y()
        
        # Search starting from front-most items (index reversed in projected list)
        axes_list = self._get_projected_axes()
        # Front-most items are at the end of the depth sorted list
        for item in reversed(axes_list):
            sx = item["sx"]
            sy = item["sy"]
            is_pos = item["is_pos"]
            r = self.dot_radius_pos if is_pos else self.dot_radius_neg
            
            # Padding for easier clicking
            click_r = r + 4.0
            
            dist = np.hypot(px - sx, py - sy)
            if dist <= click_r:
                return item["name"], item["view_name"]
                
        return None, None

    def mouseMoveEvent(self, event: QMouseEvent):
        name, view_name = self._hit_test(event.position())
        if name != self.hovered_axis:
            self.hovered_axis = name
            if name is not None:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.unsetCursor()
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self.hovered_axis is not None:
            self.hovered_axis = None
            self.unsetCursor()
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            name, view_name = self._hit_test(event.position())
            if view_name is not None:
                self.snap_requested.emit(view_name)
                event.accept()
                return
        super().mousePressEvent(event)
