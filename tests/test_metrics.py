import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np


class MetricsTests(unittest.TestCase):
    def test_perfect_five_class_predictions(self):
        from infant_benchmark.metrics import evaluate_predictions
        labels = np.arange(5)
        scores = np.eye(5)
        result = evaluate_predictions(labels, scores)
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['macro_f1'], 1.0)
        self.assertEqual(result['confusion_matrix'], np.eye(5, dtype=int).tolist())
        self.assertEqual(result['per_class']['eairh']['recall'], 1.0)
        self.assertEqual(result['per_class']['eairh']['support'], 1)

    def test_shape_mismatch_is_rejected(self):
        from infant_benchmark.metrics import evaluate_predictions
        with self.assertRaises(ValueError):
            evaluate_predictions(np.arange(5), np.ones((4, 5)))


if __name__ == '__main__':
    unittest.main()
