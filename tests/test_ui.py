import time

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from digit_recognition.config import DEFAULT_MODEL
from digit_recognition.ui.window import MainWindow


def wait_until(condition, timeout=10):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        QTest.qWait(20)
    assert condition()


def test_mouse_input_live_prediction_clear_undo_and_resize(app):
    window = MainWindow(DEFAULT_MODEL)
    window.show()
    try:
        wait_until(lambda: window.controller.ready)
        canvas = window.canvas_panel.canvas
        center = canvas.width() // 2
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(center, 80))
        for y in range(90, 260, 10):
            QTest.mouseMove(canvas, QPoint(center, y), delay=5)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(center, 260))
        wait_until(lambda: window.predictions.winner is not None)
        assert window.network.active
        assert np.sum(window.predictions.target) > 0.999
        assert canvas.has_ink
        canvas.undo()
        assert not canvas.has_ink
        assert window.predictions.winner is None
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(center, 100))
        QTest.mouseClick(window.canvas_panel.clear_button, Qt.MouseButton.LeftButton)
        QTest.qWait(150)
        assert window.predictions.winner is None
        assert not window.network.active
        assert not canvas.pixels().any()
        window.resize(1100, 660)
        QTest.qWait(30)
        assert not window.grab().isNull()
        for widget in (canvas, window.network, window.predictions):
            assert widget.isVisible()
    finally:
        window.close()
    assert not window.controller.thread.isRunning()


def test_missing_model_keeps_ui_usable(app, tmp_path):
    window = MainWindow(tmp_path / "missing.pt")
    window.show()
    try:
        wait_until(lambda: "Model not found" in window.status_label.text())
        assert window.canvas_panel.clear_button.isEnabled()
        assert window.predictions.winner is None
    finally:
        window.close()
