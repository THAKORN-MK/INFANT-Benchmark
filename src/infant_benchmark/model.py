"""Paper-inspired architecture, not author code; Gaussian KAN is an explicit proxy."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class SincConv(nn.Module):
    def __init__(self, sample_rate=16000, filters=64, kernel=251, stride=8):
        super().__init__()
        self.sample_rate, self.stride = sample_rate, stride
        self.low_hz = nn.Parameter(torch.linspace(40, 6500, filters))
        self.band_hz = nn.Parameter(torch.full((filters,), 120.0))
        self.register_buffer('time', torch.arange(-(kernel // 2), kernel // 2 + 1, dtype=torch.float32) / sample_rate)
        self.register_buffer('window', torch.hamming_window(kernel, periodic=False))

    def forward(self, x):
        low = self.low_hz.abs().clamp(30.0, self.sample_rate / 2 - 51)
        high = (low + self.band_hz.abs() + 50).clamp(max=self.sample_rate / 2 - 1)
        t = self.time.unsqueeze(0)
        high, low = high.unsqueeze(1), low.unsqueeze(1)
        filters = (2 * high * torch.sinc(2 * high * t) - 2 * low * torch.sinc(2 * low * t))
        filters = (filters * self.window).unsqueeze(1)
        filters = filters / (2 * (high - low)).unsqueeze(1).clamp_min(1e-6)
        return F.conv1d(x, filters, stride=self.stride, padding=len(self.time) // 2)


class WaveConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = SincConv()
        self.first = nn.Sequential(nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.MaxPool1d(4))
        self.conv = nn.Sequential(
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.LeakyReLU(0.2), nn.MaxPool1d(4),
            nn.Conv1d(96, 128, 5, padding=2), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.MaxPool1d(4),
        )

    def forward(self, wave):
        return self.conv(self.first(self.sinc(wave).abs())).transpose(1, 2)


class GaussianKAN(nn.Module):
    """Gaussian radial basis proxy: paper's precise basis implementation is unknown."""
    def __init__(self, inputs, outputs, bases=8):
        super().__init__()
        self.register_buffer('centres', torch.linspace(-1, 1, bases))
        self.width = 2 / (bases - 1)
        self.weights = nn.Parameter(torch.empty(inputs, outputs, bases))
        nn.init.normal_(self.weights, std=0.02)
        self.residual = nn.Linear(inputs, outputs)

    def forward(self, x):
        phi = torch.exp(-((torch.tanh(x).unsqueeze(-1) - self.centres) / self.width) ** 2)
        return torch.einsum('bik,iok->bo', phi, self.weights) + self.residual(x)


class SincMambaKAN(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as exc:
            raise RuntimeError('mamba_ssm with a working selective scan backend is required') from exc
        self.wave = WaveConv()
        self.norm = nn.LayerNorm(128)
        self.mamba = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.features = nn.Sequential(nn.Linear(100, 64), nn.GELU(), nn.LayerNorm(64))
        self.head = nn.Sequential(GaussianKAN(192, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3), GaussianKAN(64, 5))

    def forward(self, waveform, features):
        sequence = self.wave(waveform.unsqueeze(1))
        embedding = (self.mamba(self.norm(sequence)) + sequence).mean(dim=1)
        return self.head(torch.cat((embedding, self.features(features)), dim=1))
