"""Estimate a pitch track from an audio file and write it as CSV."""

import argparse
import csv
import sys

import soundfile as sf

from . import PestoSwipeTiny


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="input audio file (e.g. WAV or FLAC)")
    parser.add_argument(
        "--hop-ms", type=float, default=10.0,
        help="time between frames in milliseconds (default: 10)",
    )
    args = parser.parse_args()

    try:
        audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
        model = PestoSwipeTiny(sample_rate)
        times, pitch_hz = model.predict(audio.mean(axis=1), hop_ms=args.hop_ms)
    except (OSError, ValueError, sf.LibsndfileError) as error:
        parser.error(str(error))

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("time_s", "pitch_hz"))
    writer.writerows(zip(times, pitch_hz.tolist()))
