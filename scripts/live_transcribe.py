#!/usr/bin/env python3
"""Live microphone transcription with a VOSK model.

Records your microphone, transcribes in real time, and saves — on every run —
a brand-new pair of files side by side in a `recordings/` folder:

    recordings/rec_20260812_193000.wav      # the audio you spoke
    recordings/rec_20260812_193000.txt      # the transcript (side by side)

It also appends every clip to a master side-by-side log:
    recordings/transcripts.tsv              # audio_path <TAB> transcript

Live partial results print to the console as you speak.

Usage:
    python scripts/live_transcribe.py                    # record until you press Enter
    python scripts/live_transcribe.py --duration 15      # auto-stop after 15 s
    python scripts/live_transcribe.py --list-devices     # show microphones
    python scripts/live_transcribe.py --device 0         # pick a specific mic
    python scripts/live_transcribe.py --model models/vosk-model-small-en-in-0.4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REC_DIR = ROOT / "recordings"
SAMPLE_RATE = 16000          # VOSK requirement
CHUNK_SECONDS = 0.1          # 100 ms frames for snappy live updates


def _find_model(override: str | None) -> Path:
    """Locate a VOSK model: explicit path, the Indian-English model, or any model."""
    if override:
        p = Path(override)
        if not p.is_dir():
            sys.exit(f"[error] model folder not found: {p}")
        return p
    candidates = [
        MODELS_DIR / "vosk-model-small-en-in-0.4",
        MODELS_DIR / "vosk-model-en-in-0.5",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    # Fall back to any directory containing a VOSK model signature.
    for cand in sorted(MODELS_DIR.iterdir()):
        if cand.is_dir() and ((cand / "conf" / "model.conf").is_file()
                              or (cand / "am" / "final.mdl").is_file()):
            return cand
    sys.exit(
        "[error] no VOSK model found in models/. Put one there, e.g. "
        "vosk-model-small-en-in-0.4, or pass --model PATH"
    )


def _clean_partial(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _unique_stamp() -> str:
    """Timestamp with a collision guard so every run gets a fresh file name."""
    i = 0
    while True:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if i:
            stamp += f"_{i}"
        if not (REC_DIR / f"rec_{stamp}.wav").exists():
            return stamp
        i += 1


def _save_clip(audio: np.ndarray, transcript: str) -> tuple[Path, Path]:
    """Write the audio wav + side-by-side transcript, return both paths."""
    REC_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _unique_stamp()
    wav_path = REC_DIR / f"rec_{stamp}.wav"
    txt_path = REC_DIR / f"rec_{stamp}.txt"

    # Write 16-bit PCM mono WAV.
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    # Side-by-side transcript file.
    txt_path.write_text(
        f"Audio: {wav_path.name}\nTranscript: {transcript.strip()}\n",
        encoding="utf-8",
    )

    # Append to the master side-by-side log (audio<TAB>text).
    with open(REC_DIR / "transcripts.tsv", "a", encoding="utf-8") as log:
        log.write(f"{wav_path.name}\t{transcript.strip()}\n")

    return wav_path, txt_path


def main() -> int:
    p = argparse.ArgumentParser(description="Live VOSK microphone transcription")
    p.add_argument("--model", default=None, help="path to a VOSK model folder")
    p.add_argument("--device", type=int, default=None,
                   help="input device index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true",
                   help="list microphone devices and exit")
    p.add_argument("--duration", type=float, default=None,
                   help="auto-stop after N seconds; default: wait for Enter")
    args = p.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return 0

    model_dir = _find_model(args.model)
    print(f"[model] {model_dir}")

    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(-1)
    recognizer = KaldiRecognizer(Model(str(model_dir)), SAMPLE_RATE)
    recognizer.SetWords(False)

    # Some raw ALSA/Native hw devices reject the 16 kHz stream PortAudio asks
    # for. Probe the requested device briefly and fall back to the system
    # default (pipewire/pulse resamples) if it cannot open at 16 kHz.
    device = args.device
    try:
        sd.rec(SAMPLE_RATE // 10, samplerate=SAMPLE_RATE, channels=1,
               dtype="int16", device=device)
        sd.wait()
    except Exception as e:
        if device is not None:
            print(f"[warn] device {device} can't open at {SAMPLE_RATE} Hz "
                  f"({type(e).__name__}); falling back to the default microphone")
            device = None

    chunk_frames = int(SAMPLE_RATE * CHUNK_SECONDS)
    collected: list[np.ndarray] = []
    last_partial = ""

    def show_partial(text: str):
        nonlocal last_partial
        pad = " " * max(0, len(last_partial) - len(text))
        sys.stdout.write(f"\r[partial] {text}{pad}")
        sys.stdout.flush()
        last_partial = text

    print("[start] Speak now. Press Enter to stop"
          + (f"  (auto-stop in {args.duration:.0f}s)" if args.duration else "") + "\n")
    finished = False
    try:
        while not finished:
            audio = sd.rec(chunk_frames, samplerate=SAMPLE_RATE,
                           channels=1, dtype="int16", device=device)
            sd.wait()
            collected.append(audio.copy())

            if recognizer.AcceptWaveform(audio.tobytes()):
                res = json.loads(recognizer.Result())
                txt = _clean_partial(res.get("text", ""))
                if txt:
                    print("\r[final]   " + txt + "   ")
                    last_partial = ""
            else:
                partial = _clean_partial(
                    json.loads(recognizer.PartialResult()).get("partial", ""))
                if partial:
                    show_partial(partial)

            # Stop conditions.
            if args.duration is not None:
                seconds = len(collected) * CHUNK_SECONDS
                if seconds >= args.duration:
                    finished = True
    except KeyboardInterrupt:
        finished = True
    finally:
        print("")  # newline after the partial line

    final = json.loads(recognizer.FinalResult()).get("text", "")
    transcript = _clean_partial(final)
    if last_partial and not transcript:
        transcript = last_partial

    full_audio = np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)
    wav_path, txt_path = _save_clip(full_audio, transcript)

    print("\n===== final transcript =====")
    print(transcript if transcript else "(no speech detected)")
    print(f"\nSaved:  {wav_path}")
    print(f"        {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())