import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_forward_shape_and_range(self) -> None:
        import torch
        from autoencoder_lab.model import ConvAutoencoder

        model = ConvAutoencoder(latent_dim=8).eval()
        with torch.inference_mode():
            result = model(torch.rand(3, 1, 28, 28))
        self.assertEqual(tuple(result.shape), (3, 1, 28, 28))
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)


if __name__ == "__main__":
    unittest.main()
