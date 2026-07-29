from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST convolutional autoencoder lab")
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train", help="download MNIST and train a model")
    train_cmd.add_argument("--epochs", type=int, default=10)
    train_cmd.add_argument("--latent-dim", type=int, default=16)
    train_cmd.add_argument("--batch-size", type=int, default=128)
    train_cmd.add_argument("--noise", type=float, default=0.0, help="denoising augmentation standard deviation")
    train_cmd.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:N, or mps")
    train_cmd.add_argument("--output", type=Path, default=None)
    sample_cmd = commands.add_parser("sample", help="sample a trained decoder")
    sample_cmd.add_argument("checkpoint", type=Path)
    sample_cmd.add_argument("--count", type=int, default=32)
    sample_cmd.add_argument("--temperature", type=float, default=0.8)
    sample_cmd.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    args = build_parser().parse_args()
    # Keep parser/help startup dependency-light. The training stack imports
    # NumPy, PyTorch, torchvision, and matplotlib only for an actual command.
    from .training import load_checkpoint, train
    from .visualize import save_grid

    if args.command == "train":
        config = TrainConfig(
            epochs=args.epochs,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            noise_std=args.noise,
            device=args.device or os.getenv("AE_DEVICE", "auto"),
            output_dir=args.output or Path(os.getenv("AE_OUTPUT_DIR", "artifacts")),
        )
        model, _ = train(config)
        save_grid(model.sample(32), config.output_dir / "latest_samples.png")
    else:
        model, _ = load_checkpoint(args.checkpoint)
        output = args.output or Path(os.getenv("AE_OUTPUT_DIR", "artifacts")) / "generated.png"
        path = save_grid(model.sample(args.count, temperature=args.temperature), output)
        print(path)


if __name__ == "__main__":
    main()
