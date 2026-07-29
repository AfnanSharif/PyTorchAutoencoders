from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import TrainConfig
from .model import ConvAutoencoder


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        target = torch.device(requested)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if target.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is not available")
        return target
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    dataset = datasets.MNIST(config.data_dir, train=True, download=True, transform=transforms.ToTensor())
    generator = torch.Generator().manual_seed(config.seed)
    train_set, validation_set = random_split(dataset, [55_000, 5_000], generator=generator)
    options = {"batch_size": config.batch_size, "num_workers": 2, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_set, shuffle=True, generator=generator, **options),
        DataLoader(validation_set, shuffle=False, **options),
    )


def _run_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    noise_std: float,
) -> float:
    model.train(optimizer is not None)
    total_loss, total_items = 0.0, 0
    context = torch.enable_grad() if optimizer else torch.inference_mode()
    with context:
        for images, _ in batches:
            images = images.to(device, non_blocking=True)
            inputs = torch.clamp(images + torch.randn_like(images) * noise_std, 0, 1) if noise_std else images
            reconstructions = model(inputs)
            loss = criterion(reconstructions, images)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_items += images.size(0)
    return total_loss / max(total_items, 1)


def train(config: TrainConfig) -> tuple[ConvAutoencoder, list[EpochMetrics]]:
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_loader, validation_loader = build_loaders(config)
    model = ConvAutoencoder(config.latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.output_dir / "best.pt"
    history: list[EpochMetrics] = []
    best_loss, stale_epochs = float("inf"), 0

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer, config.noise_std)
        validation_loss = _run_epoch(model, validation_loader, criterion, device, None, 0)
        scheduler.step(validation_loss)
        history.append(EpochMetrics(epoch, train_loss, validation_loss))
        print(f"epoch={epoch:02d} train={train_loss:.6f} validation={validation_loss:.6f}")
        if validation_loss < best_loss - 1e-6:
            best_loss, stale_epochs = validation_loss, 0
            save_checkpoint(best_path, model, config, history)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    payload = [metric.__dict__ if hasattr(metric, "__dict__") else {"epoch": metric.epoch, "train_loss": metric.train_loss, "validation_loss": metric.validation_loss} for metric in history]
    (config.output_dir / "history.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    best_payload = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_payload["state_dict"])
    model.eval()
    return model, history


def save_checkpoint(path: Path, model: ConvAutoencoder, config: TrainConfig, history: list[EpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": config.to_dict(), "history": [{"epoch": m.epoch, "train_loss": m.train_loss, "validation_loss": m.validation_loss} for m in history]},
        path,
    )


def load_checkpoint(path: str | Path, device: str = "auto") -> tuple[ConvAutoencoder, dict[str, object]]:
    target = resolve_device(device)
    # Checkpoints contain tensors plus primitive configuration values only. The
    # restricted loader avoids executing arbitrary pickle payloads from uploads.
    payload = torch.load(path, map_location=target, weights_only=True)
    model = ConvAutoencoder(int(payload["config"]["latent_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.to(target).eval()
    return model, payload
