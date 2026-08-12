#!/usr/bin/env python3
"""Live microphone transcription with WHISPER.

VOSK is the model being *fine-tuned* — it is NOT used here. Transcription is
done by Whisper (faster-whisper), the same engine the fine-tune pipeline uses
to generate training labels, so your live clips and transcripts feed directly
into training.

Records your microphone, transcribes incrementally as you speak, and on every
run saves a brand-new pair of files side by side in a `recordings/` folder:

    recordings/rec_20260812_193000.wav      # the audio you spoke
    recordings/rec_20260812_193000.txt      # the transcript (side by side)
    recordings/transcripts.tsv              # master log:  audio<TAB>transcript

Usage:
    python scripts/live_transcribe.py                    # record until you press Enter
    python scripts/live_transcribe.py --duration 15      # auto-stop after 15 s
    python scripts/live_transcribe.py --whisper-model small   # faster live updates
    python scripts/live_transcribe.py --list-devices     # show microphones
    python scripts/live_transcribe.py --device 0         # pick a specific mic
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "recordings"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.2        # recording granularity
UPDATE_SECONDS = 2.0       # how often to refresh the live transcript
WHISPER_MODEL = "large-v3"  # best accent coverage; use small/medium for speed
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE = "auto"
WHISPER_LANGUAGE = None


def _samples_to_float(pcm: np.ndarray) -> np.ndarray:
    """int16 PCM -> float32 in [-1, 1] (what Whisper expects)."""
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)  # ensure 1D for VAD
    return (pcm.astype(np.float32) / 32768.0)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _transcribe(model, pcm: np.ndarray, language: str | None) -> list[str]:
    """Run Whisper on a float32 buffer; return segment texts."""
    if pcm.size == 0:
        return []
    segments, _ = model.transcribe(
        _samples_to_float(pcm),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    return [_clean(s.text) for s in segments if _clean(s.text)]


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

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    txt_path.write_text(
        f"Audio: {wav_path.name}\nTranscript: {transcript.strip()}\n",
        encoding="utf-8",
    )
    with open(REC_DIR / "transcripts.tsv", "a", encoding="utf-8") as log:
        log.write(f"{wav_path.name}\t{transcript.strip()}\n")

    return wav_path, txt_path


def main() -> int:
    p = argparse.ArgumentParser(description="Live Whisper microphone transcription")
    p.add_argument("--whisper-model", default=WHISPER_MODEL,
                   help=f"Whisper model for transcription. default: {WHISPER_MODEL}")
    p.add_argument("--device", type=int, default=None,
                   help="input device index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true",
                   help="list microphone devices and exit")
    p.add_argument("--duration", type=float, default=None,
                   help="auto-stop after N seconds; default: wait for Enter")
    p.add_argument("--language", default=WHISPER_LANGUAGE,
                   help="force transcript language (e.g. en, hi); auto-detect if unset")
    args = p.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return 0

    from faster_whisper import WhisperModel

    device = args.device
    try:
        sd.rec(SAMPLE_RATE // 10, samplerate=SAMPLE_RATE, channels=1,
               dtype="int16", device=device)
        sd.wait()
    except Exception as e:
        if device is not None:
            print(f"[warn] mic {device} can't open at {SAMPLE_RATE} Hz "
                  f"({type(e).__name__}); falling back to the default microphone")
            device = None

    compute = WHISPER_COMPUTE
    dev = WHISPER_DEVICE
    if dev == "auto" or compute == "auto":
        import ctranslate2
        has_cuda = ctranslate2.get_cuda_device_count() > 0
        if dev == "auto":
            dev = "cuda" if has_cuda else "cpu"
        if compute == "auto":
            compute = "float16" if has_cuda else "int8"

    print(f"[model] loading Whisper '{args.whisper_model}' on {dev} ({compute}) ...")
    t0 = time.time()
    if dev == "cuda":
        # The driver can report CUDA while the runtime libs (libcublas) are
        # missing, which only fails at load/encode time. Probe and fall back to
        # CPU so the tool never hard-crashes on a half-configured GPU box.
        try:
            model = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")
            list(model.transcribe(np.zeros(SAMPLE_RATE // 2, np.float32), beam_size=1))
        except Exception as e:
            print(f"[warn] CUDA unavailable ({type(e).__name__}); falling back to CPU")
            model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
            dev, compute = "cpu", "int8"
    else:
        model = WhisperModel(args.whisper_model, device=dev, compute_type=compute)
    print(f"[model] ready in {time.time() - t0:.1f}s")

    chunk_frames = int(SAMPLE_RATE * CHUNK_SECONDS)
    chunks: list[np.ndarray] = []
    cursor_frames = 0
    live_text = ""
    last_shown = ""

    def refresh_live():
        nonlocal cursor_frames, live_text, last_shown
        new_pcm = np.concatenate(chunks)[cursor_frames:] if chunks else np.zeros(0, dtype=np.int16)
        if new_pcm.size:
            new_texts = _transcribe(model, new_pcm, args.language)
            if new_texts:
                live_text = _clean(live_text + " " + " ".join(new_texts))
            cursor_frames = sum(c.size for c in chunks)
        line = f"[live] {live_text}" if live_text else "[live] (speaking...)"
        pad = " " * max(0, len(last_shown) - len(line))
        sys.stdout.write(f"\r{line}{pad}")
        sys.stdout.flush()
        last_shown = line

    print("[start] Speak now. Press Enter to stop"
          + (f"  (auto-stop in {args.duration:.0f}s)" if args.duration else "") + "\n")
    finished = False
    last_update = time.time()
    try:
        while not finished:
            audio = sd.rec(chunk_frames, samplerate=SAMPLE_RATE,
                           channels=1, dtype="int16", device=device)
            sd.wait()
            chunks.append(audio.copy())

            if time.time() - last_update >= UPDATE_SECONDS:
                refresh_live()
                last_update = time.time()

            if args.duration is not None and len(chunks) * CHUNK_SECONDS >= args.duration:
                finished = True
    except KeyboardInterrupt:
        finished = True
    finally:
        print("")

    # Final accurate transcription of the whole clip.
    full_pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    transcript = _clean(" ".join(_transcribe(model, full_pcm, args.language)))
    if not transcript:
        transcript = live_text

    wav_path, txt_path = _save_clip(full_pcm, transcript)

    print("\n===== final transcript (Whisper) =====")
    print(transcript if transcript else "(no speech detected)")
    print(f"\nSaved:  {wav_path}")
    print(f"        {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())