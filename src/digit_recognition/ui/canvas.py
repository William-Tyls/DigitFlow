"""Mouse input and stroke history. The canvas emits changes, not predictions."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .theme import CYAN, MUTED, color, font


class DrawingCanvas(QWidget):
    changed = Signal()
    invalidated = Signal()
    RASTER_SIZE = 560

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("drawingCanvas")
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.image = QImage(self.RASTER_SIZE, self.RASTER_SIZE, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.black)
        self.history: list[QImage] = []
        self.drawing = False
        self.has_ink = False
        self.last_point = QPointF()
        self.brush_width = 24.0

    def pixels(self) -> np.ndarray:
        gray = self.image.convertToFormat(QImage.Format.Format_Grayscale8)
        buffer = np.frombuffer(gray.constBits(), dtype=np.uint8)
        return buffer.reshape(gray.height(), gray.bytesPerLine())[:, :gray.width()].copy()

    def clear(self) -> None:
        self.image.fill(Qt.GlobalColor.black)
        self.history.clear()
        self.has_ink = False
        self.drawing = False
        self.invalidated.emit()
        self.changed.emit()
        self.update()

    def undo(self) -> None:
        if self.history:
            self.image = self.history.pop()
            self.has_ink = bool(np.any(self.pixels()))
            self.drawing = False
            self.invalidated.emit()
            self.changed.emit()
            self.update()

    def _position(self, event) -> QPointF:
        point = event.position()
        return QPointF(point.x() * self.RASTER_SIZE / self.width(),
                       point.y() * self.RASTER_SIZE / self.height())

    def _stroke(self, start: QPointF, end: QPointF) -> None:
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(Qt.GlobalColor.white, self.brush_width,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                            Qt.PenJoinStyle.RoundJoin))
        if start == end:
            painter.drawPoint(start)
        else:
            painter.drawLine(start, end)
        painter.end()
        self.has_ink = True
        self.changed.emit()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.history.append(self.image.copy())
            self.history = self.history[-30:]
            self.drawing = True
            self.last_point = self._position(event)
            self._stroke(self.last_point, self.last_point)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drawing:
            point = self._position(event)
            self._stroke(self.last_point, point)
            self.last_point = point
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self._stroke(self.last_point, self._position(event))
            self.drawing = False
            event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing |
                               QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setPen(QPen(color("#608bd2", 0.75), 1))
        painter.setBrush(color("#081224"))
        painter.drawRoundedRect(rect, 22, 22)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Screen)
        painter.drawImage(rect.adjusted(3, 3, -3, -3), self.image)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if not self.has_ink:
            painter.setPen(color(MUTED, 0.65))
            painter.setFont(font(18))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Draw a digit")
        painter.end()


class CanvasPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.canvas = DrawingCanvas()
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_button.setFixedSize(184, 44)
        self.clear_button.clicked.connect(self.canvas.clear)
        self.clear_button.setStyleSheet(f"""
            QPushButton {{ color: #d7e5ff; background: #0b1830;
                border: 1px solid #294b7c; border-radius: 22px; font-size: 17px; }}
            QPushButton:hover {{ border-color: {CYAN}; background: #11253d; }}
            QPushButton:pressed {{ background: #18364b; }}
        """)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(25)
        layout.addStretch()
        layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.clear_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()

    def resizeEvent(self, event):
        size = max(100, min(self.width(), self.height() - 100))
        self.canvas.setFixedSize(size, size)
        super().resizeEvent(event)
