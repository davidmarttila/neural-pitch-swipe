"""Run the pretrained PESTO-SWIPE-Tiny pitch estimator on mono audio."""

import math
from importlib.resources import files

import librosa
import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor, nn

from ._swipe import BINS_PER_SEMITONE, CUTOFF_HZ, SWIPE_BINS, SwipePrime

__all__ = ["PestoSwipeTiny"]

_MIN_SAMPLE_RATE = 2 * CUTOFF_HZ
_FRAMES_PER_BATCH = 128
_CROP_BINS = 16
_OUTPUT_BINS = 128 * BINS_PER_SEMITONE


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_bins = SWIPE_BINS - 2 * _CROP_BINS
        self.toeplitz = nn.Conv1d(
            1, 1, kernel_size=input_bins + _OUTPUT_BINS - 1,
            padding=_OUTPUT_BINS - 1, bias=False, dtype=torch.float32,
        )
        self.register_buffer("shift", torch.zeros((), dtype=torch.float32))
        midi_bins = torch.arange(_OUTPUT_BINS, dtype=torch.float32) / BINS_PER_SEMITONE
        peak_offsets = torch.arange(1 - BINS_PER_SEMITONE, BINS_PER_SEMITONE)
        self.register_buffer("_midi_bins", midi_bins, persistent=False)
        self.register_buffer("_peak_offsets", peak_offsets, persistent=False)

    def forward(self, scores: Tensor) -> Tensor:
        scores = scores[:, None, _CROP_BINS:-_CROP_BINS]
        probabilities = self.toeplitz(scores).squeeze(1).softmax(dim=-1)

        # Decode a fractional MIDI note by averaging the bins around the peak.
        peak = probabilities.argmax(dim=-1, keepdim=True)
        indices = (peak + self._peak_offsets).clamp(0, _OUTPUT_BINS - 1)
        weights = probabilities.gather(1, indices)
        midi = (weights * self._midi_bins[indices]).sum(dim=1) / weights.sum(dim=1)
        return midi - self.shift


class PestoSwipeTiny:
    """Load the bundled MDB-stem-synth checkpoint for a given input sample rate."""

    def __init__(self, sample_rate: int):
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        self._input_sample_rate = sample_rate
        self._sample_rate = max(sample_rate, _MIN_SAMPLE_RATE)
        self._swipe = SwipePrime(self._sample_rate)
        self._encoder = _TinyEncoder().eval()
        with files(__package__).joinpath("pesto-swipe-tiny-mdb.pt").open("rb") as checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self._encoder.load_state_dict(state)

    @torch.inference_mode()
    def predict(
        self, audio: ArrayLike, *, hop_ms: float = 10.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        """Return frame-center times in seconds and pitch estimates in Hz."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("audio must be a non-empty, finite mono waveform (shape: samples,)")
        hop_samples = hop_ms * (self._sample_rate / 1000)
        if not math.isfinite(hop_samples) or hop_samples < 0.5:
            raise ValueError("hop_ms must be finite and round to at least one sample")
        hop_length = int(hop_samples + 0.5)

        if self._input_sample_rate < self._sample_rate:
            audio = librosa.resample(
                audio, orig_sr=self._input_sample_rate, target_sr=self._sample_rate
            )
        padded = np.pad(audio, self._swipe.frame_size // 2, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, self._swipe.frame_size)
        frames = windows[::hop_length]

        pitches = []
        for start in range(0, len(frames), _FRAMES_PER_BATCH):
            batch = torch.from_numpy(frames[start:start + _FRAMES_PER_BATCH].copy())
            midi = self._encoder(self._swipe(batch))
            pitches.append(440 * torch.exp2((midi - 69) / 12))

        times = np.arange(len(frames)) * (hop_length / self._sample_rate)
        return times, torch.cat(pitches).numpy()
