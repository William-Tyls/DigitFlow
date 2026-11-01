"""Shared paths and inference settings; no GUI or ML imports."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "mnist_cnn.pt"
DEFAULT_DATA = PROJECT_ROOT / "data"
IMAGE_SIZE = 28
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
INFERENCE_INTERVAL_MS = 65
