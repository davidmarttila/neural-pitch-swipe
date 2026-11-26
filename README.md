# Improving Neural Pitch Estimation with SWIPE Kernels

Run **PESTO-SWIPE-Tiny**, a monophonic pitch estimator that combines a mel-spaced SWIPE′ frontend with a single 647-weight Toeplitz layer. The included checkpoint was trained on MDB-stem-synth. [Paper by David Marttila and Joshua D. Reiss, ISMIR 2025](https://doi.org/10.5281/zenodo.17811461).

## Quick start

Requires Python 3.11 or later. From the cloned repository, run:

```sh
python -m pip install .
neural-pitch-swipe input.wav > pitch.csv
```

The command averages audio channels to mono and writes `time_s,pitch_hz`. Frames are centered every 10 ms by default (`--hop-ms` to change it), starting at zero and rounding the hop to whole samples. Audio below 16 kHz is resampled to 16 kHz. The model estimates pitch on every frame, including silence.

## Python

```python
import soundfile as sf
from neural_pitch_swipe import PestoSwipeTiny

audio, sr = sf.read("input.wav", dtype="float32", always_2d=True)
model = PestoSwipeTiny(sr)
time_s, pitch_hz = model.predict(audio.mean(axis=1))
```
