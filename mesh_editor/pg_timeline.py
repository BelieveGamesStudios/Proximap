"""
pg_timeline.py - Blender-style Timeline dock scaffold & custom canvas component.
Provides transport controls, full-height animation timeline grid, time ruler, and interactive scrubbing.
"""

from dataclasses import dataclass, field
from PySide6.QtCore import Qt, Signal, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF, QFontMetrics
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSpinBox, QFrame, QSizePolicy, QToolButton, QMenu
)


@dataclass
class Keyframe:
    frame: int
    value: float


@dataclass
class AnimationClip:
    tracks: dict[str, list[Keyframe]] = field(default_factory=dict)
    start_frame: int = 1
    end_frame: int = 250
    fps: int = 24


class TimelineState(QWidget):
    """Data model and playback controller for the animation timeline."""
    frame_changed = Signal(int)
    playback_toggled = Signal(bool)

    def __init__(self, clip: AnimationClip = None, parent=None):
        super().__init__(parent)
        self.clip = clip or AnimationClip()
        self.current_frame = 1
        self.is_playing = False
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.set_fps(self.clip.fps)

    def set_fps(self, fps: int):
        self.clip.fps = max(1, fps)
        interval_ms = int(1000.0 / self.clip.fps)
        self.timer.setInterval(interval_ms)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.playback_toggled.emit(self.is_playing)

    def set_frame(self, frame: int):
        clamped = max(self.clip.start_frame, min(self.clip.end_frame, frame))
        if clamped != self.current_frame:
            self.current_frame = clamped
            self.frame_changed.emit(self.current_frame)

    def _on_tick(self):
        next_f = self.current_frame + 1
        if next_f > self.clip.end_frame:
            next_f = self.clip.start_frame
        self.set_frame(next_f)


class BlenderTimelineCanvas(QWidget):
    """
    Full-height Blender-style timeline canvas featuring:
    - Top frame ruler with tick marks and numbers (1, 12, 24, 36...)
    - Full-height vertical grid lines spanning the entire viewport height
    - Blender-blue playhead line with top frame number badge
    - Scrubbing via click/drag
    - Clean empty grid by default (no default keyframes)
    """
    def __init__(self, timeline_state: TimelineState, parent=None):
        super().__init__(parent)
        self.timeline_state = timeline_state
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._dragging = False

        self.timeline_state.frame_changed.connect(lambda f: self.update())

    def _frame_from_pos(self, x_pos: float) -> int:
        width = self.width()
        if width <= 0:
            return self.timeline_state.clip.start_frame
        start_f = self.timeline_state.clip.start_frame
        end_f = self.timeline_state.clip.end_frame
        total_f = max(1, end_f - start_f)
        
        margin = 16.0
        usable_w = max(1.0, width - 2 * margin)
        ratio = max(0.0, min(1.0, (x_pos - margin) / usable_w))
        return int(round(start_f + ratio * total_f))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            target_f = self._frame_from_pos(event.position().x())
            self.timeline_state.set_frame(target_f)

    def mouseMoveEvent(self, event):
        if self._dragging:
            target_f = self._frame_from_pos(event.position().x())
            self.timeline_state.set_frame(target_f)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Colors (Blender Dark Theme)
        bg_color = QColor("#1E1E1E")
        ruler_bg = QColor("#181818")
        ruler_border = QColor("#282828")
        grid_line_main = QColor("#2D2D2D")
        grid_line_sub = QColor("#242424")
        text_color = QColor("#888888")
        playhead_blue = QColor("#4772B3")  # Authentic Blender playhead blue
        playhead_text = QColor("#FFFFFF")

        # 1. Canvas Background
        painter.fillRect(rect, bg_color)

        # Ruler Header Height
        ruler_h = 24

        start_f = self.timeline_state.clip.start_frame
        end_f = self.timeline_state.clip.end_frame
        total_f = max(1, end_f - start_f)

        margin = 16.0
        usable_w = max(1.0, width - 2 * margin)
        px_per_frame = usable_w / float(total_f)

        # Determine tick step (1, 5, 12, 24, 48, 60, etc.)
        min_label_dist = 42.0
        raw_step = min_label_dist / max(0.001, px_per_frame)
        if raw_step <= 1:
            step = 1
        elif raw_step <= 5:
            step = 5
        elif raw_step <= 12:
            step = 12
        elif raw_step <= 24:
            step = 24
        elif raw_step <= 48:
            step = 48
        elif raw_step <= 60:
            step = 60
        else:
            step = 120

        # 2. Draw Full-Height Vertical Grid Lines
        sub_step = max(1, step // 4)
        for f in range(start_f, end_f + 1, sub_step):
            x = margin + (f - start_f) * px_per_frame
            is_main = (f % step == 0) or (f == start_f) or (f == end_f)
            pen_col = grid_line_main if is_main else grid_line_sub
            pen_w = 1.0 if is_main else 0.5
            painter.setPen(QPen(pen_col, pen_w))
            painter.drawLine(QPointF(x, ruler_h), QPointF(x, height))

        # 3. Draw Ruler Header Strip
        ruler_rect = QRectF(0, 0, width, ruler_h)
        painter.fillRect(ruler_rect, ruler_bg)
        painter.setPen(QPen(ruler_border, 1.0))
        painter.drawLine(QPointF(0, ruler_h), QPointF(width, ruler_h))

        # 4. Ruler Ticks & Frame Labels
        font = QFont("Inter", 8)
        font.setBold(False)
        painter.setFont(font)
        fm = QFontMetrics(font)

        for f in range(start_f, end_f + 1, step):
            x = margin + (f - start_f) * px_per_frame
            # Tick mark in ruler
            painter.setPen(QPen(QColor("#444444"), 1.0))
            painter.drawLine(QPointF(x, ruler_h - 6), QPointF(x, ruler_h))

            # Text label
            label = str(f)
            tw = fm.horizontalAdvance(label)
            tx = x - (tw / 2.0)
            ty = ruler_h - 8
            painter.setPen(text_color)
            painter.drawText(QPointF(tx, ty), label)

        # 5. Draw Keyframe Diamonds ONLY if real tracks exist in clip
        if self.timeline_state.clip.tracks:
            painter.setPen(QPen(QColor("#00E676"), 1))
            painter.setBrush(QBrush(QColor("#00E676")))
            d_size = 4.0
            track_y = ruler_h + 30.0
            for track_name, kf_list in self.timeline_state.clip.tracks.items():
                for kf in kf_list:
                    if start_f <= kf.frame <= end_f:
                        kx = margin + (kf.frame - start_f) * px_per_frame
                        diamond = QPolygonF([
                            QPointF(kx, track_y - d_size),
                            QPointF(kx + d_size, track_y),
                            QPointF(kx, track_y + d_size),
                            QPointF(kx - d_size, track_y)
                        ])
                        painter.drawPolygon(diamond)

        # 6. Draw Playhead Line and Top Badge
        cur_f = self.timeline_state.current_frame
        px = margin + (cur_f - start_f) * px_per_frame

        # Vertical line spanning full height
        painter.setPen(QPen(playhead_blue, 1.5))
        painter.drawLine(QPointF(px, ruler_h), QPointF(px, height))

        # Top Badge (Pill with frame number)
        badge_str = str(cur_f)
        badge_w = max(20.0, fm.horizontalAdvance(badge_str) + 10.0)
        badge_h = 18.0
        badge_x = px - (badge_w / 2.0)
        badge_y = (ruler_h - badge_h) / 2.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(playhead_blue))
        painter.drawRoundedRect(QRectF(badge_x, badge_y, badge_w, badge_h), 3.0, 3.0)

        # Badge Text
        painter.setPen(playhead_text)
        b_font = QFont("Inter", 8, QFont.Weight.Bold)
        painter.setFont(b_font)
        b_fm = QFontMetrics(b_font)
        btw = b_fm.horizontalAdvance(badge_str)
        b_tx = px - (btw / 2.0)
        b_ty = badge_y + 13.0
        painter.drawText(QPointF(b_tx, b_ty), badge_str)


class TimelineWidget(QWidget):
    """
    Timeline dock widget featuring Blender transport controls, frame spinboxes, and full-height canvas.
    """
    def __init__(self, timeline_state: TimelineState = None, parent=None):
        super().__init__(parent)
        self.state = timeline_state or TimelineState(parent=self)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setStyleSheet("""
            TimelineWidget {
                background-color: #161616;
                border-top: 1px solid #2A2A2A;
            }
            QPushButton {
                background-color: #202020;
                color: #CCCCCC;
                border: 1px solid #2D2D2D;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2C2C2C;
                border-color: #4772B3;
            }
            QSpinBox {
                background-color: #202020;
                color: #CCCCCC;
                border: 1px solid #2D2D2D;
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 11px;
            }
        """)

        # 1. Transport Bar Header (Blender layout)
        transport_bar = QFrame(self)
        transport_bar.setFixedHeight(32)
        transport_bar.setStyleSheet("background-color: #181818; border-bottom: 1px solid #242424;")
        t_layout = QHBoxLayout(transport_bar)
        t_layout.setContentsMargins(8, 0, 8, 0)
        t_layout.setSpacing(6)

        # Left Header Menus (View, Marker, Playback)
        for menu_title in ["View ▾", "Marker ▾", "Playback ▾"]:
            btn_menu = QToolButton(transport_bar)
            btn_menu.setText(menu_title)
            btn_menu.setFixedHeight(22)
            btn_menu.setCursor(Qt.PointingHandCursor)
            btn_menu.setStyleSheet("""
                QToolButton {
                    font-size: 11px;
                    color: #AAAAAA;
                    border: none;
                    background: transparent;
                    padding: 0 4px;
                }
                QToolButton:hover {
                    color: #FFFFFF;
                    background-color: #262626;
                    border-radius: 3px;
                }
            """)
            t_layout.addWidget(btn_menu)

        t_layout.addStretch()

        # Center Transport Controls
        btn_jump_start = QPushButton("|<", transport_bar)
        btn_prev = QPushButton("<", transport_bar)
        self.btn_play = QPushButton("▶", transport_bar)
        btn_next = QPushButton(">", transport_bar)
        btn_jump_end = QPushButton(">|", transport_bar)

        for b in [btn_jump_start, btn_prev, self.btn_play, btn_next, btn_jump_end]:
            b.setFixedWidth(26)
            b.setFixedHeight(22)
            b.setCursor(Qt.PointingHandCursor)
            t_layout.addWidget(b)

        btn_jump_start.clicked.connect(lambda: self.state.set_frame(self.state.clip.start_frame))
        btn_prev.clicked.connect(lambda: self.state.set_frame(self.state.current_frame - 1))
        self.btn_play.clicked.connect(self.state.toggle_play)
        btn_next.clicked.connect(lambda: self.state.set_frame(self.state.current_frame + 1))
        btn_jump_end.clicked.connect(lambda: self.state.set_frame(self.state.clip.end_frame))

        self.state.playback_toggled.connect(lambda playing: self.btn_play.setText("⏸" if playing else "▶"))

        t_layout.addStretch()

        # Right Controls: Current Frame, Start & End Spinboxes
        lbl_start = QLabel("Start:", transport_bar)
        lbl_start.setStyleSheet("color: #888888; font-size: 11px;")
        t_layout.addWidget(lbl_start)

        sp_start = QSpinBox(transport_bar)
        sp_start.setRange(0, 10000)
        sp_start.setValue(self.state.clip.start_frame)
        sp_start.setFixedWidth(55)
        sp_start.setFixedHeight(22)
        sp_start.valueChanged.connect(lambda v: self._update_start_frame(v))
        t_layout.addWidget(sp_start)

        self.sp_frame = QSpinBox(transport_bar)
        self.sp_frame.setRange(0, 10000)
        self.sp_frame.setValue(self.state.current_frame)
        self.sp_frame.setFixedWidth(55)
        self.sp_frame.setFixedHeight(22)
        self.sp_frame.setStyleSheet("""
            QSpinBox {
                background-color: #252525;
                color: #4772B3;
                font-weight: bold;
                border: 1px solid #4772B3;
                border-radius: 3px;
            }
        """)
        self.sp_frame.valueChanged.connect(self.state.set_frame)
        t_layout.addWidget(self.sp_frame)

        self.state.frame_changed.connect(lambda f: self._on_state_frame_changed(f))

        lbl_end = QLabel("End:", transport_bar)
        lbl_end.setStyleSheet("color: #888888; font-size: 11px;")
        t_layout.addWidget(lbl_end)

        sp_end = QSpinBox(transport_bar)
        sp_end.setRange(1, 10000)
        sp_end.setValue(self.state.clip.end_frame)
        sp_end.setFixedWidth(55)
        sp_end.setFixedHeight(22)
        sp_end.valueChanged.connect(lambda v: self._update_end_frame(v))
        t_layout.addWidget(sp_end)

        layout.addWidget(transport_bar)

        # 2. Main Full-Height Timeline Canvas View
        self.canvas = BlenderTimelineCanvas(self.state, self)
        layout.addWidget(self.canvas, 1)

    def _update_start_frame(self, val: int):
        self.state.clip.start_frame = val
        self.canvas.update()

    def _update_end_frame(self, val: int):
        self.state.clip.end_frame = val
        self.canvas.update()

    def _on_state_frame_changed(self, f: int):
        self.sp_frame.blockSignals(True)
        self.sp_frame.setValue(f)
        self.sp_frame.blockSignals(False)
