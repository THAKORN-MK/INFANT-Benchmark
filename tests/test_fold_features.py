import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np


class FoldFeatureTests(unittest.TestCase):
    def test_transform_is_fit_on_training_rows_only(self):
        from infant_benchmark.fold_features import fit_feature_transform
        train = np.arange(20 * 5, dtype=float).reshape(20, 5)
        labels = np.array([0, 1] * 10)
        transformer = fit_feature_transform(train, labels, k=3, seed=42)
        heldout = np.full((2, 5), 10000.0)
        result = transformer.transform(heldout)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(transformer.named_steps['scaler'].mean_, train.mean(axis=0))
        self.assertTrue(np.isfinite(result).all())

    def test_rejects_nonfinite_train_data(self):
        from infant_benchmark.fold_features import fit_feature_transform
        with self.assertRaises(ValueError):
            fit_feature_transform(np.array([[np.nan, 1], [1, 2]]), np.array([0, 1]), k=1, seed=42)


if __name__ == '__main__':
    unittest.main()
