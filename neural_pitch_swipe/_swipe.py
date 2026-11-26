"""SWIPE′ frontend used by the released PESTO-SWIPE-Tiny checkpoint."""

import math

import librosa
import torch
from torch import Tensor

BINS_PER_SEMITONE = 3
SWIPE_BINS = 296
CUTOFF_HZ = 8000


class SwipePrime:
    def __init__(self, sample_rate: int):
        # Center each group of three pitch bins on a MIDI semitone.
        fmin = 27.5 / 2 ** ((BINS_PER_SEMITONE - 1) / (24 * BINS_PER_SEMITONE))
        fmax = fmin * 2 ** ((SWIPE_BINS - 1) / (12 * BINS_PER_SEMITONE))
        log_pitches = torch.linspace(
            math.log2(fmin), math.log2(fmax), SWIPE_BINS, dtype=torch.float32
        )

        # Blend the two power-of-two windows nearest to the ideal length (8 periods).
        largest = math.ceil(math.log2(sample_rate * 8 / fmin))
        smallest = math.floor(math.log2(sample_rate * 8 / fmax))
        self.frame_size = 2 ** largest
        self._windows = [
            torch.hann_window(2 ** power, periodic=False, dtype=torch.float32)
            for power in range(largest, smallest - 1, -1)
        ]
        offsets = log_pitches - math.log2(sample_rate * 8 / self.frame_size)
        octaves = torch.arange(len(self._windows))[:, None]
        self._window_weights = (1 - (offsets - octaves).abs()).clamp_min(0)

        # Exclude the upper endpoint so FFT interpolation can use the next bin.
        mel_grid = librosa.mel_frequencies(
            n_mels=1024, fmin=fmin * 0.25, fmax=CUTOFF_HZ, htk=False
        )
        freqs = torch.as_tensor(mel_grid[:-1], dtype=torch.float32)
        fft_positions = freqs / (sample_rate / self.frame_size)
        self._fft_bins = fft_positions.floor().long()
        self._fft_fractions = fft_positions - self._fft_bins
        self._kernels = _make_kernels(2 ** log_pitches, freqs)

    def __call__(self, frames: Tensor) -> Tensor:
        frames = frames - frames.mean(dim=-1, keepdim=True)
        spectra = []
        for window in self._windows:
            start = (self.frame_size - window.numel()) // 2
            windowed = frames[:, start:start + window.numel()] * window
            fft = torch.fft.rfft(windowed, n=self.frame_size)
            magnitude = (
                fft[:, self._fft_bins] * (1 - self._fft_fractions)
                + fft[:, self._fft_bins + 1] * self._fft_fractions
            ).abs()
            norm = magnitude.sum(dim=-1, keepdim=True) + 1e-7
            spectra.append((magnitude / norm).sqrt())

        # Sum over frequency (f) and window size (w) for each frame (t) and pitch (p).
        return torch.einsum(
            "pf,wtf,wp->tp", self._kernels, torch.stack(spectra), self._window_weights
        )


def _make_kernels(pitches: Tensor, freqs: Tensor) -> Tensor:
    ratios = freqs[None, :] / pitches[:, None]
    cosine = (2 * torch.pi * ratios).cos()
    kernels = torch.zeros_like(ratios)
    max_harmonic = math.floor((ratios[0, -1] - 0.75).item())

    # SWIPE′ retains the fundamental and prime-numbered harmonics.
    for harmonic in range(1, max_harmonic + 1):
        if any(harmonic % factor == 0 for factor in range(2, math.isqrt(harmonic) + 1)):
            continue
        distance = (ratios - harmonic).abs()
        in_band = pitches[:, None] * harmonic <= freqs[-1]
        peaks = (distance < 0.25) & in_band
        valleys = (distance > 0.25) & (distance < 0.75) & in_band
        kernels[peaks] = cosine[peaks]
        kernels[valleys] += cosine[valleys] * 0.5

    kernels[pitches[:, None] * ratios > freqs[-1]] = 0
    kernels /= freqs.sqrt()
    positive_norm = torch.linalg.vector_norm(kernels.clamp_min(0), dim=1, keepdim=True)
    return kernels / positive_norm.clamp_min(1e-8)
