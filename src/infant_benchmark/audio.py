"""Audio preparation; crop and padding placement are documented assumptions."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

SAMPLE_RATE = 16000
SAMPLES = 32000


def prepare_waveform(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Average channels, resample, retain start, right-pad and peak-normalize."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim not in (1, 2) or x.size == 0 or sample_rate <= 0 or not np.isfinite(x).all():
        raise ValueError('Audio must be non-empty, finite mono or samples-by-channels; rate must be positive')
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        factor = gcd(sample_rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // factor, sample_rate // factor)
    x = x[:SAMPLES]
    if x.size < SAMPLES:
        x = np.pad(x, (0, SAMPLES - x.size))
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return np.asarray(x, dtype=np.float32)


def load_waveform(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype='float32', always_2d=False)
    return prepare_waveform(audio, sample_rate)
