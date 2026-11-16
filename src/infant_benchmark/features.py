"""Explicit 213-dimensional proxy; source paper's exact schema is unavailable."""
from __future__ import annotations
import numpy as np
import librosa


def acoustic_features(waveform: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract documented proxy descriptors. Must never fit scalers here."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.shape != (32000,) or not np.isfinite(x).all():
        raise ValueError('Expected finite 16000 Hz mono waveform of exactly 32000 samples')
    sr, n_fft, hop = 16000, 512, 160
    spectrum = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    mel = librosa.feature.melspectrogram(S=spectrum ** 2, sr=sr, n_mels=128)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=1.0), n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    chroma = (np.zeros((12, spectrum.shape[1]), dtype=np.float32) if np.max(np.abs(x)) == 0 else
              librosa.feature.chroma_stft(S=spectrum ** 2, sr=sr, n_fft=n_fft, hop_length=hop))
    contrast = librosa.feature.spectral_contrast(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop, n_bands=5)
    stats = [
        ('centroid', librosa.feature.spectral_centroid(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop)),
        ('bandwidth', librosa.feature.spectral_bandwidth(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop)),
        ('rolloff', librosa.feature.spectral_rolloff(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop)),
        ('flatness', librosa.feature.spectral_flatness(S=spectrum ** 2)),
        ('zcr', librosa.feature.zero_crossing_rate(x, frame_length=n_fft, hop_length=hop)),
        ('rms', librosa.feature.rms(S=spectrum, frame_length=n_fft, hop_length=hop)),
    ]
    values, names = [], []

    def collect(prefix: str, matrix: np.ndarray) -> None:
        for i, row in enumerate(matrix):
            values.extend((float(np.mean(row)), float(np.std(row))))
            names.extend((f'{prefix}_{i:02d}_mean', f'{prefix}_{i:02d}_std'))

    for prefix, array in [('mfcc', mfcc), ('mfcc_delta', delta), ('chroma', chroma), ('spectral_contrast', contrast), *stats]:
        collect(prefix, array)
    if np.max(np.abs(x)) == 0:
        source = [0.0] * 5
    else:
        f0, voiced, probability = librosa.pyin(x, fmin=65, fmax=1000, sr=sr, frame_length=2048, hop_length=hop)
        valid = f0[np.isfinite(f0)]
        prob = np.nan_to_num(probability, nan=0.0)
        source = [float(np.mean(valid)) if valid.size else 0.0,
                  float(np.std(valid)) if valid.size else 0.0,
                  float(np.mean(prob)), float(np.std(prob)), float(np.mean(voiced))]
    values.extend(source)
    names.extend(['f0_mean', 'f0_std', 'voicing_probability_mean', 'voicing_probability_std', 'voiced_fraction'])
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (213,) or not np.isfinite(result).all():
        raise ValueError('Feature extraction produced an invalid vector')
    return result, names
