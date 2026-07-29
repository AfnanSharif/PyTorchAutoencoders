import os
import unittest
from pathlib import Path
from unittest.mock import patch

from autoencoder_lab.config import TrainConfig


class ConfigTests(unittest.TestCase):
    def test_config_serializes_paths(self) -> None:
        values = TrainConfig(data_dir=Path("input"), output_dir=Path("output")).to_dict()
        self.assertEqual(values["data_dir"], "input")
        self.assertEqual(values["output_dir"], "output")

    def test_config_rejects_invalid_noise(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig(noise_std=2).validate()

    def test_environment_defaults_are_read_at_construction_time(self) -> None:
        with patch.dict(os.environ, {"AE_OUTPUT_DIR": "custom-output", "AE_SEED": "7"}):
            config = TrainConfig()
        self.assertEqual(config.output_dir, Path("custom-output"))
        self.assertEqual(config.seed, 7)


if __name__ == "__main__":
    unittest.main()
