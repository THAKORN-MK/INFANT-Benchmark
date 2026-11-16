import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import warnings


class FeatureTests(unittest.TestCase):
    def test_feature_vector_has_213_named_finite_values(self):
        from infant_benchmark.features import acoustic_features
        samples = np.arange(32000, dtype=np.float32)
        x = np.sin(2 * np.pi * 220 * samples / 16000).astype(np.float32)
        values, names = acoustic_features(x)
        self.assertEqual(values.shape, (213,))
        self.assertEqual(len(set(names)), 213)
        self.assertTrue(np.isfinite(values).all())

    def test_empty_or_wrong_length_is_rejected(self):
        from infant_benchmark.features import acoustic_features
        with self.assertRaises(ValueError):
            acoustic_features(np.ones(12))

    def test_silence_produces_finite_values(self):
        from infant_benchmark.features import acoustic_features
        values, names = acoustic_features(np.zeros(32000, dtype=np.float32))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values.shape, (213,))

    def test_silence_does_not_attempt_pitch_tuning(self):
        from infant_benchmark.features import acoustic_features
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            acoustic_features(np.zeros(32000, dtype=np.float32))
        self.assertFalse(any('empty frequency set' in str(w.message) for w in caught))


if __name__ == '__main__':
    unittest.main()
