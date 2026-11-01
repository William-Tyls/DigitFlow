"""Render a real application screenshot after drawing through Qt mouse events.

Run from the project root:
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/capture_preview.py
"""

import time

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPainterPath
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from digit_recognition.config import DEFAULT_MODEL, PROJECT_ROOT
from digit_recognition.ui.window import MainWindow


def wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    if not predicate():
        raise RuntimeError("Timed out waiting for the live application.")


def main():
    app = QApplication([])
    window = MainWindow(DEFAULT_MODEL)
    window.show()
    try:
        wait_until(lambda: window.controller.ready)
        canvas = window.canvas_panel.canvas
        path = QPainterPath()
        path.moveTo(0.66, 0.17)
        path.cubicTo(0.44, 0.23, 0.25, 0.47, 0.29, 0.69)
        path.cubicTo(0.31, 0.87, 0.58, 0.88, 0.69, 0.71)
        path.cubicTo(0.85, 0.44, 0.49, 0.42, 0.30, 0.65)
        points = [path.pointAtPercent(i / 99) for i in range(100)]
        def point(value):
            return QPoint(round(value.x() * canvas.width()), round(value.y() * canvas.height()))
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=point(points[0]))
        for value in points[1:]:
            QTest.mouseMove(canvas, point(value), delay=0)
            QApplication.processEvents()
            time.sleep(0.01)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=point(points[-1]))
        wait_until(lambda: window.predictions.winner is not None and not window.controller.busy
                   and window.controller.submitted_revision == window.controller.revision)
        QTest.qWait(350)
        target = PROJECT_ROOT / "artifacts" / "app-preview.png"
        if not window.grab().save(str(target)):
            raise RuntimeError(f"Could not write {target}")
        print(f"Prediction: {window.predictions.winner}; probabilities: "
              f"{window.predictions.target.tolist()}")
        print(f"Screenshot: {target}")
        window.resize(1100, 660)
        QTest.qWait(100)
        window.grab().save(str(target.with_name("app-preview-small.png")))
        canvas.clear()
        QTest.qWait(150)
        window.grab().save(str(target.with_name("app-empty.png")))
    finally:
        window.close()


if __name__ == "__main__":
    main()
