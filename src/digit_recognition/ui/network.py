"""An aggregated CNN view drawn from actual feature maps and activations.

Three fixed feature channels per convolution block are visible. Each dense
node represents eight hidden units; edge intensity is their summed absolute
contribution to an output. Particles indicate flow, not execution timing.
"""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QImage, QLinearGradient, QPainter, QPainterPath,
                          QPen, QTransform)
from PySide6.QtWidgets import QWidget

from .theme import CYAN, PURPLE, color, glow


def feature_image(values: np.ndarray, hue: str) -> QImage:
    """NumPy color mapping, with ownership detached from the temporary array."""
    values = np.maximum(values, 0)
    peak = float(values.max())
    normalized = np.sqrt(values / peak) if peak > 1e-8 else np.zeros_like(values)
    rgb = np.array([int(hue[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    background = np.array([9, 20, 42], dtype=np.float32)
    pixels = np.ascontiguousarray(
        (background + normalized[..., None] * (rgb - background)).astype(np.uint8)
    )
    height, width = values.shape
    return QImage(pixels.data, width, height, pixels.strides[0],
                  QImage.Format.Format_RGB888).copy()


class NetworkView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(420)
        self.phase = 0.0
        self.active = False
        self.winner = None
        self.hidden = np.zeros(8, dtype=np.float32)
        self.edges = np.zeros((10, 8), dtype=np.float32)
        self.probabilities = np.zeros(10, dtype=np.float32)
        self.images = [[feature_image(np.zeros((14, 14)), CYAN) for _ in range(3)],
                       [feature_image(np.zeros((7, 7)), PURPLE) for _ in range(3)]]

    def set_prediction(self, prediction) -> None:
        self.active = prediction is not None
        self.winner = prediction.digit if prediction is not None else None
        if prediction is None:
            self.hidden.fill(0)
            self.edges.fill(0)
            self.probabilities.fill(0)
            for group, hue in enumerate((CYAN, PURPLE)):
                self.images[group] = [feature_image(np.zeros((14, 14)), hue) for _ in range(3)]
        else:
            self.images = [[feature_image(values, hue) for values in maps]
                           for maps, hue in zip(prediction.feature_maps, (CYAN, PURPLE))]
            grouped = prediction.hidden_activations.reshape(8, 8).mean(axis=1)
            self.hidden = grouped / max(float(grouped.max()), 1e-8)
            self.edges = np.abs(prediction.contributions).reshape(10, 8, 8).sum(axis=2)
            self.edges /= max(float(self.edges.max()), 1e-8)
            self.probabilities = prediction.probabilities.copy()
        self.update()

    def advance(self, dt: float) -> None:
        if self.active:
            self.phase = (self.phase + dt * 0.25) % 1.0
            self.update()

    def _connection(self, painter: QPainter, start: QPointF, end: QPointF,
                    hue: str, intensity: float, offset: float, particle: bool = True) -> None:
        curve = QPainterPath(start)
        bend = (end.x() - start.x()) * 0.48
        curve.cubicTo(start + QPointF(bend, 0), end - QPointF(bend, 0), end)
        alpha = (0.12 + intensity * 0.47) if self.active else 0.09
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color(hue, alpha), 0.6 + intensity * 0.65))
        painter.drawPath(curve)
        if self.active and particle:
            point = curve.pointAtPercent((self.phase + offset) % 1)
            glow(painter, point, 7, hue, 0.24 + intensity * 0.15)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color(hue, 0.65 + intensity * 0.3))
            painter.drawEllipse(point, 1.3, 1.3)

    def _tiles(self, painter: QPainter, group: int, x: float, hue: str) -> None:
        for index in range(3):
            left = x + index * 27
            top = 197 + index * 16
            # A simple affine shear gives the floating planes a restrained depth.
            transform = QTransform(1, -0.28, 0, 1, left, top)
            painter.save()
            painter.setTransform(transform, combine=True)
            rect = QRectF(0, 0, 86, 198)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.setOpacity(0.93 if self.active else 0.5)
            painter.drawImage(rect, self.images[group][index])
            painter.setOpacity(1)
            painter.setPen(QPen(color(hue, 0.55 if self.active else 0.24), 0.8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
            # A faint cell grid emphasizes image pixels without explanatory labels.
            painter.setPen(QPen(color(hue, 0.06), 0.4))
            for column in range(1, 14):
                painter.drawLine(QPointF(column * 86 / 14, 0), QPointF(column * 86 / 14, 198))
            for row in range(1, 14):
                painter.drawLine(QPointF(0, row * 198 / 14), QPointF(86, row * 198 / 14))
            painter.restore()

    def _node(self, painter: QPainter, point: QPointF, radius: float,
              hue: str, intensity: float) -> None:
        if self.active:
            glow(painter, point, radius * 2.8, hue, 0.12 + intensity * 0.35)
        gradient = QLinearGradient(point - QPointF(radius, radius), point + QPointF(radius, radius))
        gradient.setColorAt(0, color(hue, 0.65 + intensity * 0.35 if self.active else 0.23))
        gradient.setColorAt(1, color("#182843" if not self.active else "#373166"))
        painter.setBrush(gradient)
        painter.setPen(QPen(color(hue, 0.55 + intensity * 0.45 if self.active else 0.35), 1.2))
        painter.drawEllipse(point, radius, radius)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = min(self.width() / 620, self.height() / 610)
        painter.translate((self.width() - 620 * scale) / 2,
                          (self.height() - 610 * scale) / 2)
        painter.scale(scale, scale)
        hidden_points = [QPointF(460, 179 + i * 34) for i in range(8)]
        output_points = [QPointF(595, 146 + i * 34) for i in range(10)]
        for index in range(7):
            self._connection(painter, QPointF(-10, 250 + index * 13),
                             QPointF(110, 258 + index * 12), CYAN, 0.38,
                             index / 7, particle=index % 2 == 0)
        for index in range(5):
            self._connection(painter, QPointF(213, 235 + index * 28),
                             QPointF(288, 245 + index * 22), PURPLE, 0.3,
                             index / 5, particle=index % 2 == 0)
        for index, point in enumerate(hidden_points):
            self._connection(painter, QPointF(378, 260 + index * 12), point,
                             PURPLE, float(self.hidden[index]) * 0.6, index / 8)
        # Keep only the three strongest grouped contributions per output visible.
        for digit, end in enumerate(output_points):
            strongest = np.argsort(self.edges[digit])[-3:]
            for group in strongest:
                intensity = float(self.edges[digit, group])
                is_winner = digit == self.winner
                self._connection(painter, hidden_points[group], end,
                                 CYAN if is_winner else PURPLE,
                                 intensity * (1 if is_winner else 0.35),
                                 digit * 0.08 + group * 0.11,
                                 particle=is_winner)
        self._tiles(painter, 0, 80, CYAN)
        self._tiles(painter, 1, 265, PURPLE)
        for index, point in enumerate(hidden_points):
            self._node(painter, point, 9.5, PURPLE, float(self.hidden[index]))
        for digit, point in enumerate(output_points):
            self._node(painter, point, 8, CYAN if digit == self.winner else PURPLE,
                       float(self.probabilities[digit]))
        painter.end()
