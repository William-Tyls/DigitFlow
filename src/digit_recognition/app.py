"""Desktop entry point. Launch with `python -m digit_recognition`."""

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .config import DEFAULT_MODEL
from .ui.window import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="Live handwritten digit recognition")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to a trained digit-cnn-v1 checkpoint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = QApplication([sys.argv[0]])
    app.setApplicationName("Digit Recognition")
    app.setOrganizationName("Digit Lab")
    app.setStyle("Fusion")
    window = MainWindow(args.model)
    window.show()
    sys.exit(app.exec())
