"""Application composition: widgets depend on snapshots, not model internals."""

import time
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QKeySequence, QLinearGradient, QPainter, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from ..controller import InferenceController
from .canvas import CanvasPanel
from .network import NetworkView
from .prediction import PredictionPanel
from .theme import BACKGROUND, MUTED, TEXT, color, font


class Background(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, color("#0b172d"))
        gradient.setColorAt(0.45, color(BACKGROUND))
        gradient.setColorAt(1, color("#050c19"))
        painter.fillRect(self.rect(), gradient)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self, model_path: Path):
        super().__init__()
        self.setWindowTitle("Digit Recognition")
        self.resize(1440, 850)
        self.setMinimumSize(1100, 660)
        root = Background()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(42, 38, 34, 23)
        layout.setSpacing(20)

        title = QLabel("Digit Recognition")
        title.setFont(font(31, QFont.Weight.Medium))
        title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        layout.addWidget(title)

        content = QHBoxLayout()
        content.setSpacing(18)
        self.canvas_panel = CanvasPanel()
        self.network = NetworkView()
        self.predictions = PredictionPanel()
        content.addWidget(self.canvas_panel, 34)
        content.addWidget(self.network, 57)
        content.addWidget(self.predictions, 24)
        layout.addLayout(content, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("Loading model…")
        self.status_label.setWordWrap(True)
        self.status_label.setFont(font(12))
        self.status_label.setStyleSheet(f"color: {MUTED}; background: transparent;")
        footer.addWidget(self.status_label, 1)
        signature = QLabel("CNN  /  MNIST")
        signature.setFont(font(11))
        signature.setStyleSheet("color: #4b6083; background: transparent;")
        footer.addWidget(signature)
        layout.addLayout(footer)

        self.controller = InferenceController(self.canvas_panel.canvas, model_path, self)
        self.controller.prediction_ready.connect(self.network.set_prediction)
        self.controller.prediction_ready.connect(self.predictions.set_prediction)
        self.controller.status_changed.connect(self.status_label.setText)

        self.clear_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.clear_shortcut.activated.connect(self.canvas_panel.canvas.clear)
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.canvas_panel.canvas.undo)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._animate)
        self.last_frame = time.perf_counter()
        self.timer.start()
        self.controller.start()

    def _animate(self):
        now = time.perf_counter()
        dt = min(now - self.last_frame, 0.1)
        self.last_frame = now
        if self.isMinimized():
            return
        self.network.advance(dt)
        self.predictions.advance(dt)

    def closeEvent(self, event):
        self.timer.stop()
        self.controller.stop()
        super().closeEvent(event)
