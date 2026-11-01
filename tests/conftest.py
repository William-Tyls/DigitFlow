import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from digit_recognition.config import DEFAULT_MODEL
from digit_recognition.inference import Predictor


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="session")
def predictor():
    return Predictor(DEFAULT_MODEL)
