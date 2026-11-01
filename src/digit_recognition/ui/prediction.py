"""Live probabilities with smooth bars and a gently crossfading main digit."""

import math

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .theme import CYAN, MUTED, PURPLE, TEXT, color, font, glow


class PredictionPanel(QWidget):
    DIGIT_FADE_SECONDS = 0.18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current = np.zeros(10, dtype=np.float32)
        self.target = self.current.copy()
        self.winner: int | None = None
        self._digit_opacities = np.zeros(10, dtype=np.float32)
        self._fade_start = self._digit_opacities.copy()
        self._fade_elapsed = self.DIGIT_FADE_SECONDS
        self.setMinimumWidth(220)

    def set_prediction(self, prediction) -> None:
        if prediction is None:
            self.target.fill(0)
            self.current.fill(0)
            self.winner = None
            # Clearing is immediate; an old digit must never fade back in.
            self._digit_opacities.fill(0)
            self._fade_start.fill(0)
            self._fade_elapsed = self.DIGIT_FADE_SECONDS
        else:
            self.target = prediction.probabilities.copy()
            if prediction.digit != self.winner:
                # Retarget from the visible blend, including mid-animation.
                # Repeated predictions of the same digit do not restart it.
                self._fade_start = self._digit_opacities.copy()
                self._fade_elapsed = 0.0
            self.winner = prediction.digit
        self.update()

    def advance(self, dt: float) -> None:
        if self.winner is not None and self._fade_elapsed < self.DIGIT_FADE_SECONDS:
            self._fade_elapsed = min(self.DIGIT_FADE_SECONDS, self._fade_elapsed + dt)
            progress = self._fade_elapsed / self.DIGIT_FADE_SECONDS
            blend = progress * progress * (3 - 2 * progress)
            self._digit_opacities = self._fade_start * (1 - blend)
            self._digit_opacities[self.winner] += blend
            self.update()
        if np.max(np.abs(self.target - self.current)) > 0.0001:
            self.current += (self.target - self.current) * (1 - math.exp(-dt * 15))
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # A virtual canvas maintains the composition across window sizes.
        scale = min(self.width() / 260, self.height() / 610)
        painter.translate((self.width() - 260 * scale) / 2,
                          (self.height() - 610 * scale) / 2)
        painter.scale(scale, scale)
        painter.setPen(QPen(color("#264264", 0.8), 1))
        painter.drawLine(0, 0, 0, 610)
        painter.setFont(font(21))
        painter.setPen(color(TEXT))
        painter.drawText(QRectF(30, 0, 220, 35), "Prediction")
        if self.winner is not None:
            glow(painter, QRectF(30, 55, 210, 145).center(), 105, CYAN,
                 0.10 * float(self._digit_opacities.sum()))
        painter.setPen(color(CYAN if self.winner is not None else MUTED,
                             1 if self.winner is not None else 0.45))
        painter.setFont(font(146, weight=QFont.Weight.Light))
        digit_rect = QRectF(28, 33, 220, 172)
        if self.winner is None:
            painter.drawText(digit_rect, Qt.AlignmentFlag.AlignCenter, "—")
        else:
            painter.save()
            for digit, opacity in enumerate(self._digit_opacities):
                if opacity > 0:
                    painter.setOpacity(float(opacity))
                    painter.drawText(digit_rect, Qt.AlignmentFlag.AlignCenter, str(digit))
            painter.restore()
        painter.setFont(font(33))
        painter.drawText(QRectF(30, 202, 220, 48), Qt.AlignmentFlag.AlignCenter,
                         f"{self.target[self.winner]:.1%}" if self.winner is not None else "")
        for digit in range(10):
            y = 286 + digit * 31
            active = digit == self.winner
            hue = CYAN if active else PURPLE
            painter.setFont(font(16))
            painter.setPen(color(CYAN if active else TEXT, 1 if self.winner is not None else 0.45))
            painter.drawText(QRectF(31, y - 5, 19, 24), Qt.AlignmentFlag.AlignVCenter, str(digit))
            rect = QRectF(60, y + 1, 133, 11)
            painter.setPen(QPen(color("#203652", 0.8), 0.8))
            painter.setBrush(color("#0e1b30"))
            painter.drawRoundedRect(rect, 3, 3)
            bar_width = float(self.current[digit]) * rect.width()
            if bar_width > 0.05:
                if active:
                    glow(painter, rect.center(), 46, CYAN, 0.1)
                gradient = QLinearGradient(rect.topLeft(), rect.topRight())
                gradient.setColorAt(0, color(hue, 0.75))
                gradient.setColorAt(1, color(hue))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(gradient)
                painter.drawRoundedRect(QRectF(rect.x(), rect.y(), bar_width, rect.height()), 2, 2)
            painter.setFont(font(13))
            painter.setPen(color(CYAN if active else MUTED))
            value = f"{self.target[digit]:.1%}" if self.winner is not None else "—"
            painter.drawText(QRectF(197, y - 5, 60, 24), Qt.AlignmentFlag.AlignRight |
                             Qt.AlignmentFlag.AlignVCenter, value)
        painter.end()
