"""Latest-frame scheduling and inference on a dedicated Qt worker thread."""

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from .config import INFERENCE_INTERVAL_MS
from .inference import ModelLoadError, Predictor

logger = logging.getLogger(__name__)


class InferenceWorker(QObject):
    loaded = Signal()
    completed = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, model_path: Path):
        super().__init__()
        self.model_path = model_path
        self.predictor = None

    @Slot()
    def load(self):
        try:
            self.predictor = Predictor(self.model_path)
        except ModelLoadError as exc:
            logger.exception("Model initialization failed")
            self.failed.emit(-1, str(exc))
        else:
            self.loaded.emit()

    @Slot(int, object)
    def predict(self, revision: int, pixels):
        try:
            result = self.predictor.predict(pixels)
        except (RuntimeError, ValueError) as exc:
            logger.exception("Inference failed")
            self.failed.emit(revision, f"Prediction failed: {exc}")
        else:
            self.completed.emit(revision, result)


class InferenceController(QObject):
    request = Signal(int, object)
    prediction_ready = Signal(object)
    status_changed = Signal(str)

    def __init__(self, canvas, model_path: Path, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.revision = 0
        self.submitted_revision = -1
        self.minimum_revision = 0
        self.busy = False
        self.ready = False
        self.stopping = False
        self.thread = QThread(self)
        self.worker = InferenceWorker(model_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.load)
        self.thread.finished.connect(self.worker.deleteLater)
        self.request.connect(self.worker.predict)
        self.worker.loaded.connect(self._loaded)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.canvas.changed.connect(self._changed)
        self.canvas.invalidated.connect(self._invalidate)
        self.timer = QTimer(self)
        self.timer.setInterval(INFERENCE_INTERVAL_MS)
        self.timer.timeout.connect(self._submit)

    def start(self):
        self.status_changed.emit("Loading model…")
        self.thread.start()
        self.timer.start()

    @Slot()
    def _loaded(self):
        self.ready = True
        self.status_changed.emit("Draw a single digit")
        self._submit()

    @Slot()
    def _invalidate(self):
        # Clearing/undoing is a barrier: an in-flight result must not restore it.
        self.minimum_revision = self.revision + 1
        self.prediction_ready.emit(None)

    @Slot()
    def _changed(self):
        self.revision += 1
        if not self.canvas.has_ink:
            self.prediction_ready.emit(None)
            if self.ready:
                self.status_changed.emit("Draw a single digit")

    @Slot()
    def _submit(self):
        if (self.stopping or not self.ready or self.busy or not self.canvas.has_ink
                or self.revision == self.submitted_revision):
            return
        self.busy = True
        self.submitted_revision = self.revision
        self.request.emit(self.revision, self.canvas.pixels())

    @Slot(int, object)
    def _completed(self, revision, prediction):
        self.busy = False
        if self.stopping:
            return
        # During a stroke, show the latest completed snapshot instead of starving
        # the display whenever the mouse moves faster than inference completes.
        if revision >= self.minimum_revision and self.canvas.has_ink:
            self.prediction_ready.emit(prediction)
            self.status_changed.emit(
                f"Live inference · {prediction.elapsed_ms:.0f} ms" if prediction else "Draw a single digit"
            )

    @Slot(int, str)
    def _failed(self, revision, message):
        self.busy = False
        if revision == -1:
            self.ready = False
            self.timer.stop()
        if revision == -1 or revision == self.revision:
            self.status_changed.emit(message)

    def stop(self):
        self.stopping = True
        self.timer.stop()
        self.thread.quit()
        # A single small CPU forward pass is allowed to finish before destruction.
        self.thread.wait()
