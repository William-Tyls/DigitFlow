"""Colors, typography and small drawing helpers."""

from functools import lru_cache

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QRadialGradient

BACKGROUND = "#070f20"
TEXT = "#e0e9ff"
MUTED = "#7387ab"
CYAN = "#64f0ff"
PURPLE = "#ab95ff"


def color(value: str, alpha: float = 1.0) -> QColor:
    result = QColor(value)
    result.setAlphaF(max(0.0, min(1.0, alpha)))
    return result


@lru_cache(maxsize=1)
def _font_family() -> str:
    installed = set(QFontDatabase.families())
    return next((name for name in ("Inter", "Helvetica Neue", "Segoe UI", "DejaVu Sans", "Arial")
                 if name in installed), QFont().family())


def font(size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    result = QFont(_font_family())
    result.setStyleHint(QFont.StyleHint.SansSerif)
    result.setPixelSize(size)
    result.setWeight(weight)
    return result


def glow(painter: QPainter, point: QPointF, radius: float, hue: str,
         strength: float = 0.4) -> None:
    gradient = QRadialGradient(point, radius)
    gradient.setColorAt(0, color(hue, strength))
    gradient.setColorAt(0.4, color(hue, strength * 0.25))
    gradient.setColorAt(1, color(hue, 0))
    painter.setPen(color(hue, 0))
    painter.setBrush(gradient)
    painter.drawEllipse(point, radius, radius)
