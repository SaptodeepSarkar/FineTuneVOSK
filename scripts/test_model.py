#!/usr/bin/env python3
"""Test a fine-tuned VOSK model on an audio file.

Usage:
    python scripts/test_model.py --model output/vosk-model-finetuned --audio your-audio.wav

Requires the `vosk` pip package:
    pip install vosk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Transcribe audio with a VOSK model")
    p.add_argument("--model", required=True, help="path to the (fine-tuned) VOSK model folder")
    p.add_argument("--audio", required=True, help="audio file to transcribe (any format)")
    p.add_argument("--sample-rate", type=int, default=16000)
    args = p.parse_args()

    try:
        import wave
        from vosk import KaldiRecognizer, Model, SetLogLevel
    except ImportError as e:
        sys.exit(f"missing dependency (pip install vosk): {e}")

    SetLogLevel(-1)
    model = Model(str(args.model))
    recognizer = KaldiRecognizer(model, args.sample_rate)

    # Normalize input to the model's expected sample rate with ffmpeg on the fly.
    import subprocess
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-i", args.audio,
         "-ar", str(args.sample_rate), "-ac", "1", "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
    )
    result_text = ""
    while True:
        data = proc.stdout.read(4000)
        if not data:
            break
        if recognizer.AcceptWaveform(data):
            res = json.loads(recognizer.Result())
            result_text += res.get("text", "") + " "
    final = json.loads(recognizer.FinalResult())
    result_text += final.get("text", "")

    print("TRANSCRIPTION:")
    print(result_text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())