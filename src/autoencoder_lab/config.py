from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    latent_dim: int = 16
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    noise_std: float = 0.0
    patience: int = 3
    # Resolve environment-backed defaults at construction time so dotenv files
    # loaded by an entry point are honored even when this module was imported.
    seed: int = field(default_factory=lambda: int(os.getenv("AE_SEED", "42")))
    device: str = field(default_factory=lambda: os.getenv("AE_DEVICE", "auto"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("AE_DATA_DIR", "data")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("AE_OUTPUT_DIR", "artifacts")))

    def validate(self) -> "TrainConfig":
        if self.latent_dim < 2:
            raise ValueError("latent_dim must be at least 2")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
        if not 0 <= self.noise_std <= 1:
            raise ValueError("noise_std must be between 0 and 1")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not self.device.strip():
            raise ValueError("device cannot be empty")
        return self

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        return values
