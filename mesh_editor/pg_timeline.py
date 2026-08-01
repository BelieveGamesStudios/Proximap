"""
pg_timeline.py - Reusable Timeline dock scaffold & TrackRow keyframe component.
Provides transport controls, animation timeline state, and custom-painted track rows.
"""

from dataclasses import dataclass, field
from PySide6.QtCore import Qt, Signal, QTimer, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSpinBox, QScrollArea, QFrame, QSizePolicy
)


@dataclass
class Keyframe:
    frame: int
    value: float


@dataclass
class AnimationClip:
    tracks: dict[str, list[Keyframe]] = field(default_factory=dict)
    start_frame: int = 0
    end_frame: int = 100
    fps: int = 24


class TimelineState(QWidget):
    """Data model and playback controller for the animation timeline."""
    frame_changed = Signal(int)
    playback_toggled = Signal(bool)

    def __init__(self, clip: AnimationClip = None, parent=None):
        super().__init__(parent)
        self.clip = clip or AnimationClip()
        self.current_frame = 0
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


class TrackRowCanvas(QWidget):
    """
    Custom-painted timeline canvas widget drawing time ticks, playhead, and keyframe diamonds.
    """
    def __init__(self, track_name: str, keyframes: list[Keyframe], timeline_state: TimelineState, parent=None):
        super().__init__(parent)
        self.track_name = track_name
        self.keyframes = keyframes
        self.timeline_state = timeline_state
        self.setFixedHeight(30)
        self.setCursor(Qt.PointingHandCursor)

        self.timeline_state.frame_changed.connect(lambda f: self.update())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            width = self.width()
            start_f = self.timeline_state.clip.start_frame
            end_f = self.timeline_state.clip.end_frame
            total_f = max(1, end_f - start_f)
            
            click_x = event.position().x()
            ratio = max(0.0, min(1.0, click_x / width))
            target_f = int(start_f + ratio * total_f)
            self.timeline_state.set_frame(target_f)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # 1. Background
        painter.fillRect(rect, QColor("#141414"))

        # 2. Time Ticks
        start_f = self.timeline_state.clip.start_frame
        end_f = self.timeline_state.clip.end_frame
        total_f = max(1, end_f - start_f)
        px_per_frame = width / float(total_f)

        painter.setPen(QPen(QColor("#2A2A2A"), 1))
        step = 5 if total_f <= 100 else 10
        for f in range(start_f, end_f + 1, step):
            x = (f - start_f) * px_per_frame
            painter.drawLine(int(x), 0, int(x), height)

        # 3. Draw Keyframe Diamonds
        painter.setPen(QPen(QColor("#00E676"), 1))
        painter.setBrush(QBrush(QColor("#00E676")))
        
        d_size = 5.0
        for kf in self.keyframes:
            if start_f <= kf.frame <= end_f:
                kx = (kf.frame - start_f) * px_per_frame
                ky = height / 2.0
                diamond = QPolygonF([
                    (kx, ky - d_size),
                    (kx + d_size, ky),
                    (kx, ky + d_size),
                    (kx - d_size, ky)
                ])
                painter.drawPolygon(diamond)

        # 4. Draw Playhead Line
        cur_f = self.timeline_state.current_frame
        px = (cur_f - start_f) * px_per_frame
        painter.setPen(QPen(QColor("#00E676"), 2))
        painter.drawLine(int(px), 0, int(px), height)


class TrackRow(QWidget):
    """
    Reusable TrackRow component containing a left track label and right TrackRowCanvas.
    """
    def __init__(self, track_name: str, keyframes: list[Keyframe], timeline_state: TimelineState, parent=None):
        super().__init__(parent)
        self.track_name = track_name
        self.keyframes = keyframes
        self.timeline_state = timeline_state
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left label panel
        left_panel = QFrame(self)
        left_panel.setFixedWidth(140)
        left_panel.setStyleSheet("background-color: #1A1A1A; border-right: 1px solid #282828;")
        left_layout = QHBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 0, 10, 0)

        lbl = QLabel(self.track_name, left_panel)
        lbl.setStyleSheet("color: #CCCCCC; font-size: 11px;")
        left_layout.addWidget(lbl)

        layout.addWidget(left_panel)

        # Right custom paint canvas
        self.canvas = TrackRowCanvas(self.track_name, self.keyframes, self.timeline_state, self)
        layout.addWidget(self.canvas, 1)


class TimelineWidget(QWidget):
    """
    Timeline dock widget featuring transport controls, frame spinboxes, and track rows.
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
                border-color: #00E676;
            }
        """)

        # 1. Transport Bar
        transport_bar = QFrame(self)
        transport_bar.setFixedHeight(34)
        transport_bar.setStyleSheet("background-color: #181818; border-bottom: 1px solid #282828;")
        t_layout = QHBoxLayout(transport_bar)
        t_layout.setContentsMargins(10, 0, 10, 0)
        t_layout.setSpacing(8)

        # Tab sub-header buttons
        for mode in ["Keying", "View", "Marker"]:
            btn_mode = QPushButton(mode, transport_bar)
            btn_mode.setFixedHeight(22)
            btn_mode.setStyleSheet("font-size: 10px; color: #888888; border: none; background: transparent;")
            t_layout.addWidget(btn_mode)

        t_layout.addStretch()

        # Playback buttons
        btn_jump_start = QPushButton("|<", transport_bar)
        btn_prev = QPushButton("<", transport_bar)
        self.btn_play = QPushButton("▶", transport_bar)
        btn_next = QPushButton(">", transport_bar)
        btn_jump_end = QPushButton(">|", transport_bar)

        for b in [btn_jump_start, btn_prev, self.btn_play, btn_next, btn_jump_end]:
            b.setFixedWidth(28)
            t_layout.addWidget(b)

        btn_jump_start.clicked.connect(lambda: self.state.set_frame(self.state.clip.start_frame))
        btn_prev.clicked.connect(lambda: self.state.set_frame(self.state.current_frame - 1))
        self.btn_play.clicked.connect(self.state.toggle_play)
        btn_next.clicked.connect(lambda: self.state.set_frame(self.state.current_frame + 1))
        btn_jump_end.clicked.connect(lambda: self.state.set_frame(self.state.clip.end_frame))

        self.state.playback_toggled.connect(lambda playing: self.btn_play.setText("⏸" if playing else "▶"))

        # Frame counter & limits
        t_layout.addStretch()

        lbl_start = QLabel("Start:", transport_bar)
        lbl_start.setStyleSheet("color: #888888; font-size: 10px;")
        t_layout.addWidget(lbl_start)

        sp_start = QSpinBox(transport_bar)
        sp_start.setRange(0, 10000)
        sp_start.setValue(self.state.clip.start_frame)
        sp_start.setFixedWidth(55)
        sp_start.valueChanged.connect(lambda v: setattr(self.state.clip, 'start_frame', v))
        t_layout.addWidget(sp_start)

        self.lbl_frame = QLabel(f"{self.state.current_frame:04d}", transport_bar)
        self.lbl_frame.setStyleSheet("""
            QLabel {
                background-color: #111111;
                color: #00E676;
                font-family: 'JetBrains Mono', 'Courier New', monospace;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #00E676;
                border-radius: 3px;
                padding: 2px 6px;
            }
        """)
        t_layout.addWidget(self.lbl_frame)

        self.state.frame_changed.connect(lambda f: self.lbl_frame.setText(f"{f:04d}"))

        lbl_end = QLabel("End:", transport_bar)
        lbl_end.setStyleSheet("color: #888888; font-size: 10px;")
        t_layout.addWidget(lbl_end)

        sp_end = QSpinBox(transport_bar)
        sp_end.setRange(1, 10000)
        sp_end.setValue(self.state.clip.end_frame)
        sp_end.setFixedWidth(55)
        sp_end.valueChanged.connect(lambda v: setattr(self.state.clip, 'end_frame', v))
        t_layout.addWidget(sp_end)

        layout.addWidget(transport_bar)

        # 2. Track Rows Area
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout(tracks_container)
        self.tracks_layout.setContentsMargins(0, 0, 0, 0)
        self.tracks_layout.setSpacing(1)

        # Demo initial track row
        demo_keyframes = [Keyframe(0, 0.0), Keyframe(25, 2.0), Keyframe(60, -1.5)]
        row1 = TrackRow("Active Object / Pos.X", demo_keyframes, self.state, tracks_container)
        self.tracks_layout.addWidget(row1)

        self.tracks_layout.addStretch()
        scroll.setWidget(tracks_container)
        layout.addWidget(scroll)
