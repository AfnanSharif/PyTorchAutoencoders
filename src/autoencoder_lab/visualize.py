from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def save_grid(images: torch.Tensor, destination: str | Path, title: str = "Generated digits") -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(images.detach().cpu(), nrow=min(8, len(images)), padding=2)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.imshow(grid.permute(1, 2, 0).squeeze(), cmap="magma")
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination

