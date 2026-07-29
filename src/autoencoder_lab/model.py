from __future__ import annotations

import math

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Compact convolutional autoencoder for 28x28 single-channel images."""

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Flatten(),
        )
        self.to_latent = nn.Linear(32 * 7 * 7, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder_features(images))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(latent))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))

    @torch.inference_mode()
    def sample(self, count: int, *, temperature: float = 1.0) -> torch.Tensor:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("count and temperature must be positive")
        device = next(self.parameters()).device
        latent = torch.randn(count, self.latent_dim, device=device) * temperature
        was_training = self.training
        self.eval()
        try:
            return self.decode(latent)
        finally:
            self.train(was_training)
