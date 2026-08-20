# selection_overlay.py
# Transparent 2D screen overlay for interactive Box and Lasso selection over the 3D viewport

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QCoreApplication
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF, QMouseEvent, QWheelEvent


class SelectionOverlayWidget(QWidget):
    """
    Transparent overlay widget positioned directly on top of the VisPy OpenGL canvas.
    Handles mouse drag events for Box and Lasso selection modes when Control (Ctrl) is held,
    and draws real-time screen-space selection marquee. If Control is not held, mouse events
    are forwarded directly to the underlying 3D canvas widget for camera orbit, pan, and zoom.
    """
    shape_changed = Signal(object)      # Emits ('box', (x0, y0, x1, y1)) or ('lasso', [[x, y], ...])
    selection_committed = Signal()     # Emits when mouse is released to finalize shape
    selection_cleared = Signal()       # Emits when selection is cleared

    def __init__(self, parent=None, underlying_widget=None):
        super().__init__(parent)
        self._underlying_widget = underlying_widget
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        self._mode = 'none'  # 'none', 'box', 'lasso'
        self._is_drawing = False
        self._start_pos = None
        self._current_pos = None
        self._lasso_points = []

    def set_underlying_widget(self, widget):
        """Set or update the underlying 3D viewport widget for event forwarding."""
        self._underlying_widget = widget

    def set_mode(self, mode: str):
        """Set the active selection tool ('none', 'box', 'lasso')."""
        self._mode = mode
        self.clear()
        self.setVisible(mode in ['box', 'lasso'])

    def get_mode(self) -> str:
        return self._mode

    def clear(self):
        """Reset the current shape and clear visual selection drawing."""
        self._is_drawing = False
        self._start_pos = None
        self._current_pos = None
        self._lasso_points = []
        self.update()

    def _forward_event(self, event):
        """Forward mouse/wheel event to underlying 3D OpenGL viewport canvas."""
        if self._underlying_widget is not None:
            QCoreApplication.sendEvent(self._underlying_widget, event)

    def mousePressEvent(self, event: QMouseEvent):
        if self._mode not in ['box', 'lasso']:
            self._forward_event(event)
            return

        # Selection only triggers when Control (Ctrl) is held down
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ControlModifier):
            self._is_drawing = True
            pos = event.position()
            self._start_pos = pos
            self._current_pos = pos
            if self._mode == 'lasso':
                self._lasso_points = [[float(pos.x()), float(pos.y())]]
            else:
                self._lasso_points = []

            self.update()
            event.accept()
        else:
            # Without Ctrl, forward event to VisPy canvas for camera orbit/pan
            self._is_drawing = False
            self._forward_event(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._is_drawing or self._mode not in ['box', 'lasso']:
            self._forward_event(event)
            return

        if event.buttons() & Qt.LeftButton:
            pos = event.position()
            self._current_pos = pos
            if self._mode == 'lasso':
                if not self._lasso_points:
                    self._lasso_points.append([float(pos.x()), float(pos.y())])
                else:
                    last = self._lasso_points[-1]
                    dx = pos.x() - last[0]
                    dy = pos.y() - last[1]
                    if (dx * dx + dy * dy) >= 4.0:  # 2 px threshold
                        self._lasso_points.append([float(pos.x()), float(pos.y())])

            # Repaint shape overlay smoothly
            self.update()
            event.accept()
        else:
            self._is_drawing = False
            self._forward_event(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._is_drawing or self._mode not in ['box', 'lasso']:
            self._forward_event(event)
            return

        if event.button() == Qt.LeftButton:
            self._is_drawing = False
            pos = event.position()
            self._current_pos = pos

            # Execute point projection & hit-test only on mouse release
            if self._mode == 'lasso':
                if len(self._lasso_points) >= 3:
                    self.shape_changed.emit(('lasso', list(self._lasso_points)))
            elif self._mode == 'box' and self._start_pos is not None:
                x0, y0 = float(self._start_pos.x()), float(self._start_pos.y())
                x1, y1 = float(pos.x()), float(pos.y())
                self.shape_changed.emit(('box', (x0, y0, x1, y1)))

            # Clear 2D marquee representation once hit-test is dispatched
            self._start_pos = None
            self._current_pos = None
            self._lasso_points = []

            self.selection_committed.emit()
            self.update()
            event.accept()
        else:
            self._is_drawing = False
            self._forward_event(event)

    def wheelEvent(self, event: QWheelEvent):
        # Forward scroll wheel events to VisPy canvas for zooming
        self._forward_event(event)

    def paintEvent(self, event):
        if self._mode not in ['box', 'lasso']:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Style: Dashed vibrant green outline with subtle green fill
        pen = QPen(QColor("#00E676"), 1.5, Qt.DashLine)
        brush = QBrush(QColor(0, 230, 118, 35))
        painter.setPen(pen)
        painter.setBrush(brush)

        if self._mode == 'box' and self._start_pos is not None and self._current_pos is not None:
            rect = QRectF(self._start_pos, self._current_pos).normalized()
            if rect.width() > 1 or rect.height() > 1:
                painter.drawRect(rect)
        elif self._mode == 'lasso' and len(self._lasso_points) >= 2:
            qpoints = [QPointF(x, y) for x, y in self._lasso_points]
            poly = QPolygonF(qpoints)
            painter.drawPolygon(poly)

        painter.end()
