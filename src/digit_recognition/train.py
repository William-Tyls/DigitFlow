"""Reproducible MNIST training. Run independently from the desktop UI."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import certifi
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from .config import DEFAULT_DATA, DEFAULT_MODEL, MNIST_MEAN, MNIST_STD
from .model import DigitCNN


def augment(images: torch.Tensor) -> torch.Tensor:
    """Batched affine and occasional stroke-width augmentation."""
    count = images.shape[0]
    angles = (torch.rand(count, device=images.device) - 0.5) * 0.4
    scales = 0.9 + torch.rand(count, device=images.device) * 0.25
    matrix = torch.zeros(count, 2, 3, device=images.device)
    matrix[:, 0, 0] = angles.cos() * scales
    matrix[:, 0, 1] = -angles.sin() * scales
    matrix[:, 1, 0] = angles.sin() * scales
    matrix[:, 1, 1] = angles.cos() * scales
    matrix[:, :, 2] = (torch.rand(count, 2, device=images.device) - 0.5) * 0.16
    grid = F.affine_grid(matrix, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, align_corners=False)
    # Include thicker mouse strokes while retaining mostly original samples.
    thicker = F.max_pool2d(images, 3, stride=1, padding=1)
    mask = (torch.rand(count, 1, 1, 1, device=images.device) < 0.15)
    return torch.where(mask, images * 0.45 + thicker * 0.55, images)


@torch.inference_mode()
def accuracy(model: DigitCNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device).float()[:, None] / 255
        output = model((images - MNIST_MEAN) / MNIST_STD)
        correct += int((output.argmax(1).cpu() == labels).sum())
        total += len(labels)
    return correct / total


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    # Supply a verified CA bundle on Python distributions without system CAs.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_name = args.device
    device = torch.device(device_name)
    print(f"Loading MNIST · device={device} · seed={args.seed}", flush=True)
    dataset = MNIST(str(args.data_dir), train=True, download=True)
    order = torch.randperm(len(dataset))
    training, validation = order[5000:], order[:5000]
    train_loader = DataLoader(
        TensorDataset(dataset.data[training], dataset.targets[training]),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(dataset.data[validation], dataset.targets[validation]),
        batch_size=512, num_workers=0,
    )
    model = DigitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_accuracy = -1.0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, labels in train_loader:
            images = augment(images.to(device).float()[:, None] / 255)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model((images - MNIST_MEAN) / MNIST_STD), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = accuracy(model, val_loader, device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "validation_accuracy": score})
        print(f"Epoch {epoch + 1}/{args.epochs} · loss={np.mean(losses):.4f} "
              f"· validation={score:.2%}", flush=True)
        if score > best_accuracy:
            best_accuracy = score
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = MNIST(str(args.data_dir), train=False, download=True)
    test_loader = DataLoader(TensorDataset(test.data, test.targets), batch_size=512)
    test_accuracy = accuracy(model, test_loader, device)
    model.eval()
    with torch.inference_mode():
        samples = dataset.data[validation[:512]].to(device).float()[:, None] / 255
        _, first, second, _ = model.forward_with_features((samples - MNIST_MEAN) / MNIST_STD)
        channels = [feature.mean((0, 2, 3)).topk(3).indices.cpu().tolist()
                    for feature in (first, second)]
    checkpoint = {"architecture": "digit-cnn-v1", "state_dict": best_state,
                  "visual_channels": channels, "seed": args.seed,
                  "test_accuracy": test_accuracy, "validation_accuracy": best_accuracy}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    metrics = {"dataset": "MNIST", "train_samples": 55000, "validation_samples": 5000,
               "test_samples": 10000, "seed": args.seed, "device": str(device),
               "epochs": args.epochs, "best_validation_accuracy": best_accuracy,
               "test_accuracy": test_accuracy, "history": history,
               "training_seconds": round(time.perf_counter() - started, 2)}
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Test accuracy: {test_accuracy:.2%}\nSaved: {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        train(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
