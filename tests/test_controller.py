from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, Signal

from digit_recognition.controller import InferenceController


class CanvasStub(QObject):
    changed = Signal()
    invalidated = Signal()
    has_ink = True

    def pixels(self):
        return np.full((28, 28), 255, dtype=np.uint8)


def test_latest_frame_is_retained_without_growing_a_queue(app):
    canvas = CanvasStub()
    controller = InferenceController(canvas, Path("unused.pt"))
    controller.ready = True
    requests = []
    controller.request.connect(lambda revision, pixels: requests.append(revision))
    controller._submit()
    for _ in range(20):
        canvas.changed.emit()
        controller._submit()
    assert requests == [0]
    controller._completed(0, SimpleNamespace(elapsed_ms=1))
    controller._submit()
    assert requests == [0, 20]
    controller.stop()


def test_clearing_blocks_an_in_flight_result(app):
    canvas = CanvasStub()
    controller = InferenceController(canvas, Path("unused.pt"))
    controller.ready = True
    delivered = []
    controller.prediction_ready.connect(delivered.append)
    controller._submit()
    canvas.has_ink = False
    canvas.invalidated.emit()
    canvas.changed.emit()
    controller._completed(0, SimpleNamespace(elapsed_ms=1))
    assert delivered and all(value is None for value in delivered)
    # Even after a new stroke begins, the pre-clear result cannot come back.
    canvas.has_ink = True
    canvas.changed.emit()
    controller._completed(0, SimpleNamespace(elapsed_ms=1))
    assert all(value is None for value in delivered)
    controller.stop()


def test_continuous_drawing_does_not_starve_live_results(app):
    canvas = CanvasStub()
    controller = InferenceController(canvas, Path("unused.pt"))
    controller.ready = True
    delivered = []
    controller.prediction_ready.connect(delivered.append)
    controller._submit()
    canvas.changed.emit()
    snapshot = SimpleNamespace(elapsed_ms=3)
    controller._completed(0, snapshot)
    assert delivered == [snapshot]
    controller.stop()
