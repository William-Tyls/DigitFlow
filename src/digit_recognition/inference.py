"""Model loading and immutable inference snapshots, independent of the UI."""

from dataclasses import dataclass
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .config import MNIST_MEAN, MNIST_STD
from .model import DigitCNN
from .preprocessing import prepare_digit


class ModelLoadError(RuntimeError):
    """A missing or incompatible model with an actionable user message."""


@dataclass(frozen=True)
class Prediction:
    probabilities: np.ndarray
    feature_maps: tuple[np.ndarray, np.ndarray]
    hidden_activations: np.ndarray
    contributions: np.ndarray
    input_image: np.ndarray
    elapsed_ms: float

    @property
    def digit(self) -> int:
        return int(np.argmax(self.probabilities))


class Predictor:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise ModelLoadError(
                "Model not found. Run 'python -m digit_recognition.train' first."
            )
        # This model is small; one CPU thread avoids thread-pool overhead.
        torch.set_num_threads(1)
        self.model = DigitCNN()
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            if checkpoint.get("architecture") != "digit-cnn-v1":
                raise ValueError("Unsupported model architecture.")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.channels = checkpoint.get("visual_channels", [[0, 5, 10], [0, 10, 20]])
            if len(self.channels) != 2 or any(
                len(ids) != 3 or any(not isinstance(i, int) or not 0 <= i < n for i in ids)
                for ids, n in zip(self.channels, (16, 32))
            ):
                raise ValueError("Invalid visualization channels.")
        except (OSError, RuntimeError, KeyError, IndexError, ValueError, TypeError, AttributeError,
                EOFError, pickle.UnpicklingError) as exc:
            raise ModelLoadError(
                "Could not load the model. Retrain it with 'python -m digit_recognition.train'."
            ) from exc
        self.model.eval()
        self.weights = self.model.output.weight.detach().numpy().copy()

    @torch.inference_mode()
    def predict(self, pixels: np.ndarray) -> Prediction | None:
        start = time.perf_counter()
        prepared = prepare_digit(pixels)
        if prepared is None:
            return None
        tensor = torch.from_numpy((prepared - MNIST_MEAN) / MNIST_STD)[None, None]
        logits, first, second, hidden = self.model.forward_with_features(tensor)
        activations = hidden[0].numpy().copy()
        return Prediction(
            probabilities=logits.softmax(dim=1)[0].numpy().copy(),
            feature_maps=(first[0, self.channels[0]].numpy().copy(),
                          second[0, self.channels[1]].numpy().copy()),
            hidden_activations=activations,
            contributions=self.weights * activations[None, :],
            input_image=prepared,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
