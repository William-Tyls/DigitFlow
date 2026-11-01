# DigitFlow — Digit Recognition

A Python desktop application for recognizing handwritten digits as you draw. View live CNN feature maps, animated connections, and prediction probabilities for digits 0–9 in a minimalist English interface with a dark background and cyan and violet accents.

![Actual application screenshot](artifacts/app-preview.png)

## Run the Application

The repository includes a trained model. If the project's virtual environment is already configured, run the following command from the project directory. For a fresh checkout, complete **Installation** below first.

```bash
.venv/bin/python main.py
```

You can also run `.venv/bin/python -m digit_recognition`, or use `digit-recognition` after activating the virtual environment.

- **Left-click and drag:** draw a single digit and see predictions update while writing.
- **Clear / Esc:** clear the canvas and prediction results.
- **Cmd+Z (macOS) / Ctrl+Z (Windows, Linux):** undo the last stroke.

Once installed, the application loads the bundled model and works offline. Handwriting is processed locally and is never uploaded. Drawing does not retrain the model.

## Installation

Python 3.11 or later is required; Python 3.12 is recommended and has been tested. A browser, Node.js, and a dedicated GPU are not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python main.py
```

On Windows, activate the environment with `.venv\Scripts\activate` in Command Prompt or `.\.venv\Scripts\Activate.ps1` in PowerShell. If you only need to run the application, install it with `python -m pip install -e .` instead of including the development extras.

All direct dependencies are declared in `pyproject.toml`. The exact versions and transitive dependencies from the tested environment are recorded in `requirements-lock.txt`. To reproduce those versions on a compatible system:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

| Dependency | Purpose |
| --- | --- |
| NumPy | Handwriting cropping, center-of-mass calculations, array transformations, activation aggregation, and feature-map color mapping |
| Pillow | Resizing handwriting images while preserving their aspect ratio |
| PySide6 | Native windows, mouse input, custom drawing, animation, and background threads |
| PyTorch | CNN training and CPU inference |
| Torchvision | Loading the MNIST dataset |
| certifi | Trusted HTTPS CA certificates for dataset downloads during training |
| pytest (dev) | Automated tests; Qt interaction tests use QtTest bundled with PySide6 |
| setuptools, wheel (build) | Python package builds and editable installation |

The tested environment is **macOS arm64 / Python 3.12.7**. Package wheel availability may differ on other platforms. For installation on another platform, prefer the standard installation command so dependencies can be resolved from `pyproject.toml`.

## Model and Training

`artifacts/mnist_cnn.pt` contains the trained model weights.

```text
28×28 grayscale
→ Conv(1→16, 3×3) → ReLU → MaxPool(2)
→ Conv(16→32, 3×3) → ReLU → MaxPool(2)
→ Flatten(1568) → Dense(64) → ReLU
→ Dropout(0.15, training only) → Dense(10)
```

Preprocessing crops the handwriting, fits it within a 20×20 box, pads it to 28×28, centers its mass, and normalizes it using the MNIST parameters. Blank input produces no prediction. Inference uses `eval()`, `inference_mode()`, and Softmax.

The bundled model was trained for 6 epochs with random seed 42, using 55,000 training samples and 5,000 separate validation samples. Weights were selected using validation accuracy, then evaluated on 10,000 test samples. Validation accuracy was **98.78%**, and test accuracy was **98.93%**. Training and evaluation records are available in `artifacts/mnist_cnn.json`. These MNIST results do not guarantee the same accuracy on mouse-drawn digits; incomplete strokes may also receive high model probabilities.

To retrain the model:

```bash
python -m digit_recognition.train --epochs 6 --device cpu
```

The first training run requires an internet connection to download the public MNIST dataset. Subsequent runs use the cache in `data/`. Available arguments include `--device auto|cpu|mps|cuda`, `--seed`, `--batch-size`, `--data-dir`, and `--output`. By default, training replaces the local weights. Restart the application to load the updated model.

To specify a model path: `python main.py --model artifacts/mnist_cnn.pt`.

## Project Structure

```text
main.py                       # Application entry point
pyproject.toml                # Runtime, development, and build dependencies
requirements-lock.txt         # Exact versions from the tested environment
src/digit_recognition/
  config.py                   # Paths and shared constants
  preprocessing.py            # NumPy preprocessing, independent of Qt
  model.py                    # CNN definition, independent of Qt
  train.py                    # Standalone training workflow
  inference.py                # Model loading and inference snapshots, independent of Qt
  controller.py               # Background inference, latest-frame scheduling, and stale-result handling
  app.py                      # Argument parsing and application lifecycle
  ui/
    canvas.py                 # Handwriting input, clearing, and undo
    network.py                # Feature maps, grouped nodes, and connection animations
    prediction.py             # Probability bars and recognition results
    theme.py                  # Colors, fonts, and drawing helpers
    window.py                 # UI composition
artifacts/                    # Trained weights, evaluation records, and application screenshots
tests/                        # Preprocessing, inference, scheduling, and GUI tests
scripts/capture_preview.py    # Captures application screenshots using Qt mouse events
```

The UI receives results through signals instead of calling the model directly. The worker processes one frame at a time and retains only the latest canvas state while busy. Clearing or undoing prevents older results from reappearing. During continuous drawing, the UI displays the most recently completed result so frequent mouse movement does not cause every prediction to be discarded. Inference is scheduled at 65 ms intervals, and the animation timer runs at 16 ms intervals; actual performance depends on the device.

Each feature-map group displays three fixed channels of actual pooling output, selected during training. Each purple hidden node groups eight hidden units. Connection intensity represents the sum of their absolute contributions, and only the three strongest connections are shown for each output. Particles illustrate data flow rather than actual execution timing. Probability values come from Softmax, while the bars use brief smooth transitions. The main predicted digit uses a 180 ms crossfade: repeated results do not restart the animation, rapid changes continue from the current visual state, and clearing removes the digit immediately.

## Verification

```bash
python -m pytest -q
python -m pip check
```

Tests cover blank and edge-touching handwriting, translation invariance, missing or corrupt models, probability normalization, feature changes, latest-frame scheduling, stale-result rejection after clearing, continuous drawing, undo, window resizing, and thread shutdown. Tests automatically use Qt's offscreen platform. Model tests use the weights included in the repository.

To capture screenshots of the running application rather than concept images:

```bash
QT_QPA_PLATFORM=offscreen python scripts/capture_preview.py
```

The script writes `artifacts/app-preview.png` and also captures a smaller window and an empty canvas. Do not set `QT_QPA_PLATFORM=offscreen` when using the application normally.

If the model is missing or corrupt, the footer displays a recovery command and the canvas remains usable. Inference errors are displayed and logged to the terminal. Dataset downloads during training use HTTPS certificate verification.

Component documentation: [Qt for Python](https://doc.qt.io/qtforpython-6/), [PyTorch](https://docs.pytorch.org/), [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html), and [NumPy](https://numpy.org/doc/stable/).
