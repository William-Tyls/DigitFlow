"""GUI-independent, NumPy-based handwriting normalization."""

import numpy as np
from PIL import Image

from .config import IMAGE_SIZE


def prepare_digit(pixels: np.ndarray) -> np.ndarray | None:
    """Fit white-on-black ink into MNIST's 20 px box, then center its mass.

    Input is a 2-D uint8 grayscale image. Empty canvases return None rather
    than producing an arbitrary prediction. The returned float32 image is
    in [0, 1]; model normalization is applied separately by the predictor.
    """
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional uint8 grayscale image.")
    yy, xx = np.nonzero(pixels > 20)
    if xx.size < 4:
        return None
    crop = pixels[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    height, width = crop.shape
    scale = 20.0 / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    ink = np.asarray(Image.fromarray(crop).resize(size, Image.Resampling.LANCZOS))
    result = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    left = (IMAGE_SIZE - size[0]) // 2
    top = (IMAGE_SIZE - size[1]) // 2
    result[top : top + size[1], left : left + size[0]] = ink / 255.0
    rows, cols = np.indices(result.shape)
    mass = result.sum()
    dx = int(round(13.5 - float((cols * result).sum() / mass)))
    dy = int(round(13.5 - float((rows * result).sum() / mass)))
    # Translate without wrapping cropped pixels to the opposite edge.
    centered = np.zeros_like(result)
    sx, sy = max(0, -dx), max(0, -dy)
    tx, ty = max(0, dx), max(0, dy)
    w, h = IMAGE_SIZE - abs(dx), IMAGE_SIZE - abs(dy)
    centered[ty : ty + h, tx : tx + w] = result[sy : sy + h, sx : sx + w]
    return centered
