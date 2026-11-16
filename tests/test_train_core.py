import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class TrainingCoreTests(unittest.TestCase):
    def test_polyloss_is_cross_entropy_when_epsilon_is_zero(self):
        from infant_benchmark.train_core import poly_loss
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        weights = torch.tensor([1.0, 1.0])
        expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights, label_smoothing=0.05)
        self.assertTrue(torch.allclose(poly_loss(logits, labels, weights, epsilon=0.0), expected))

    def test_weighted_class_loss_handles_missing_classes(self):
        from infant_benchmark.train_core import class_weights
        with self.assertRaises(ValueError):
            class_weights([0, 1, 1], classes=5)

    def test_majority_class_gets_smaller_weight(self):
        from infant_benchmark.train_core import class_weights
        weights = class_weights([0, 0, 0, 1], classes=2)
        self.assertLess(weights[0].item(), weights[1].item())


if __name__ == '__main__':
    unittest.main()
