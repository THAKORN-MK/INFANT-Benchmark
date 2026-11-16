import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np


class AudioPreprocessingTests(unittest.TestCase):
    def test_short_signal_is_right_padded_and_peak_normalized(self):
        from infant_benchmark.audio import prepare_waveform
        x = prepare_waveform(np.array([0.1, -0.2, 0.0]), 16000)
        self.assertEqual(x.shape, (32000,))
        np.testing.assert_allclose(x[:3], [0.5, -1, 0], atol=1e-6)
        self.assertTrue(np.all(x[3:] == 0))

    def test_long_signal_is_truncated_before_normalizing(self):
        from infant_benchmark.audio import prepare_waveform
        x = np.ones(32001, dtype=np.float32)
        x[-1] = 99
        result = prepare_waveform(x, 16000)
        self.assertEqual(result.shape, (32000,))
        self.assertTrue(np.all(result == 1))

    def test_silence_stays_finite(self):
        from infant_benchmark.audio import prepare_waveform
        self.assertTrue(np.isfinite(prepare_waveform(np.zeros(5), 16000)).all())

    def test_bad_audio_or_rate_rejected(self):
        from infant_benchmark.audio import prepare_waveform
        with self.assertRaises(ValueError):
            prepare_waveform(np.array([np.nan]), 16000)
        with self.assertRaises(ValueError):
            prepare_waveform(np.ones(5), 0)


if __name__ == '__main__':
    unittest.main()
