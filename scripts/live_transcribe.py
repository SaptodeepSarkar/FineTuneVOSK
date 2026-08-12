#!/usr/bin/env python3
"""Live microphone transcription with WHISPER — now a TUI.

VOSK is the model being *fine-tuned*; it is NOT used to transcribe. Whisper
(faster-whisper) does all transcription. A live terminal UI shows:

    • recording-seconds counter
    • where the clip will be saved (recordings/…)
    • an ASCII-art audio visualizer (live amplitude meter)
    • Silero-VAD speech/silence indicator
    • the growing live transcript

On every run it saves a brand-new pair of files side by side in `recordings/`:
    recordings/rec_20260812_193000.wav      # the audio you spoke
    recordings/rec_20260812_193000.txt      # the transcript (side by side)
    recordings/transcripts.tsv              # master log: audio<TAB>transcript

Usage:
    python scripts/live_transcribe.py                    # record until Enter
    python scripts/live_transcribe.py --duration 15      # auto-stop after 15 s
    python scripts/live_transcribe.py --whisper-model small
    python scripts/live_transcribe.py --whisper-device cpu   # force CPU
    python scripts/live_transcribe.py --list-devices
"""
from __future__ import annotations

import argparse
import re
import select
import sys
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "recordings"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.2          # recording granularity (visualizer refresh)
UPDATE_SECONDS = 2.0         # how often Whisper refreshes the live transcript
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "auto"      # auto | cpu | cuda
WHISPER_COMPUTE = "auto"
WHISPER_LANGUAGE = None

_BLOCKS = "▁▂▃▄▅▆▇█"          # ASCII-art bar segments
_GREEN, _DIM, _BOLD, _RESET = "\x1b[32m", "\x1b[2m", "\x1b[1m", "\x1b[0m"


# --------------------------------------------------------------------------- #
# Backend / CUDA detection — must never hang
# --------------------------------------------------------------------------- #
def _cuda_usable() -> bool:
    """True only if the cuBLAS runtime lib ctranslate2 needs is actually loadable.

    The NVIDIA driver can report CUDA while libcublas is missing or only lives
    inside other apps' private dirs (ollama, Resolve) — loading a model on
    'cuda' then HANGS instead of erroring. Checking `ctypes.CDLL` up front is
    fast and non-blocking, so we never reach that hang.
    """
    import ctypes
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False
    for name in ("libcublas.so.12", "libcublas.so"):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _resolve_backend(device: str) -> tuple[str, str]:
    if device == "cpu":
        return "cpu", "int8"
    if device == "cuda":
        return "cuda", "float16"
    # auto
    if _cuda_usable():
        return "cuda", "float16"
    print(f"{_DIM}[info] CUDA runtime unavailable here (no usable libcublas) — using CPU "
          f"so this never hangs. Use --whisper-device cuda on a CUDA-ready box.{_RESET}")
    return "cpu", "int8"


def _model_is_cached(model_name: str) -> bool:
    hubs = Path.home() / ".cache" / "huggingface" / "hub"
    if not hubs.is_dir():
        return False
    for d in hubs.glob(f"*{model_name}*"):
        if d.is_dir() and any((d / "snapshots").glob("*/model.bin")):
            return True
    return False


# --------------------------------------------------------------------------- #
# TUI
# --------------------------------------------------------------------------- #
BAR_WIDTH = 28


class TUI:
    def __init__(self):
        self._h = 0

    def render(self, lines: list[str]):
        if self._h:
            sys.stdout.write(f"\x1b[{self._h}A")   # move to top of panel
        sys.stdout.write("\x1b[J")                 # clear panel area
        sys.stdout.write("\n".join(lines) + "\n")
        self._h = len(lines)
        sys.stdout.flush()

    def close(self):
        if self._h:
            sys.stdout.write(f"\x1b[{self._h}A\x1b[J")
            self._h = 0
        sys.stdout.write("\n")
        sys.stdout.flush()


def _amp(audio: np.ndarray) -> float:
    """Peak-normalized RMS in 0..1 for a float32 chunk."""
    if audio.size == 0:
        return 0.0
    x = np.abs(audio).astype(np.float32)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return float(np.clip(rms * 5.0, 0.0, 1.0))   # scale up quiet speech


def _meter(history: deque) -> str:
    """ASCII-art bar: current amplitude block + history strip."""
    if not history:
        return _DIM + "▁" * BAR_WIDTH + _RESET
    level = history[-1]
    filled = int(round(level * BAR_WIDTH))
    filled = max(0, min(BAR_WIDTH, filled))
    cur = _GREEN + "█" * filled + _DIM + "░" * (BAR_WIDTH - filled) + _RESET
    strip = "".join(_BLOCKS[min(7, int(round(l * 7)))] for l in list(history)[-24:])
    return f"{cur}  {strip}"


def _fmt_seconds(secs: float) -> str:
    return f"{int(secs) // 60:02d}:{int(secs) % 60:02d}"


# --------------------------------------------------------------------------- #
# VAD (Silero) — speech / silence detection
# --------------------------------------------------------------------------- #
class SileroVad:
    def __init__(self):
        from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model
        self._model = get_vad_model()   # bundled silero_vad_v6.onnx
        self._opts = VadOptions(threshold=0.5, min_silence_duration_ms=250)
        self._get = get_speech_timestamps
        self._votes: deque = deque(maxlen=10)

    def update(self, chunk_float: np.ndarray) -> bool:
        """Call per chunk; returns smoothed speak/silence (True = speaking)."""
        try:
            ts = self._get(chunk_float.astype(np.float32),
                           self._opts, sampling_rate=SAMPLE_RATE)
            speech = bool(ts)
        except Exception:
            speech = False
        self._votes.append(speech)
        return any(self._votes)   # speaking if recently heard speech


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #
def _unique_stamp() -> str:
    i = 0
    while True:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if i:
            stamp += f"_{i}"
        if not (REC_DIR / f"rec_{stamp}.wav").exists():
            return stamp
        i += 1


def _save_clip(audio: np.ndarray, transcript: str) -> tuple[Path, Path]:
    REC_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _unique_stamp()
    wav_path = REC_DIR / f"rec_{stamp}.wav"
    txt_path = REC_DIR / f"rec_{stamp}.txt"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.asarray(audio, dtype=np.int16).tobytes())
    txt_path.write_text(
        f"Audio: {wav_path.name}\nTranscript: {transcript.strip()}\n", encoding="utf-8")
    with open(REC_DIR / "transcripts.tsv", "a", encoding="utf-8") as log:
        log.write(f"{wav_path.name}\t{transcript.strip()}\n")
    return wav_path, txt_path


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Live Whisper mic transcription (TUI)")
    p.add_argument("--whisper-model", default=WHISPER_MODEL, help=f"default: {WHISPER_MODEL}")
    p.add_argument("--whisper-device", default=WHISPER_DEVICE,
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--device", type=int, default=None, help="mic index (--list-devices)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--language", default=WHISPER_LANGUAGE)
    args = p.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return 0

    # ---- mic probe / fallback ----
    device = args.device
    try:
        sd.rec(SAMPLE_RATE // 10, samplerate=SAMPLE_RATE, channels=1,
               dtype="int16", device=device)
        sd.wait()
    except Exception as e:
        if device is not None:
            print(f"[warn] mic {device} can't open at {SAMPLE_RATE} Hz "
                  f"({type(e).__name__}); using default microphone")
        device = None

    # ---- backend (never hangs) ----
    dev, compute = _resolve_backend(args.whisper_device)

    from faster_whisper import WhisperModel

    if not _model_is_cached(args.whisper_model):
        print(_DIM + "[info] model not in cache yet — downloading on first load "
                     f"(~3 GB for {args.whisper_model}). Please wait...{_RESET}")
    print(f"[model] loading Whisper '{args.whisper_model}' on {dev} ({compute}) ...")
    t0 = time.time()
    try:
        model = WhisperModel(args.whisper_model, device=dev, compute_type=compute)
        if dev == "cuda":  # probe; fall back to CPU if runtime libs still fail
            list(model.transcribe(np.zeros(SAMPLE_RATE // 2, np.float32), beam_size=1))
    except Exception as e:
        if dev == "cuda":
            print(f"[warn] CUDA failed at runtime ({type(e).__name__}); using CPU")
            model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
            dev, compute = "cpu", "int8"
        else:
            raise
    print(f"[model] ready in {time.time() - t0:.1f}s")

    # ---- Silero VAD ----
    try:
        vad = SileroVad()
        vad_ok = True
    except Exception as e:
        print(f"[warn] Silero VAD unavailable ({type(e).__name__}); no speech indicator")
        vad = None
        vad_ok = False

    chunk_frames = int(SAMPLE_RATE * CHUNK_SECONDS)
    tui = TUI()
    chunks: list[np.ndarray] = []
    hist: deque = deque(maxlen=24)
    cursor_frames = 0
    live_text = ""
    last_update = time.time()

    render = ["loading..."]
    tui.render(render)
    finished = False
    try:
        while not finished:
            fs = time.time()
            audio = sd.rec(chunk_frames, samplerate=SAMPLE_RATE, channels=1,
                           dtype="int16", device=device)
            sd.wait()
            chunks.append(audio.copy())
            elapsed = len(chunks) * CHUNK_SECONDS

            # visualizer + VAD on this chunk
            hist.append(_amp(audio))
            speaking = vad.update(audio.astype(np.float32)) if vad else True

            # refresh transcription periodically
            if time.time() - last_update >= UPDATE_SECONDS and len(chunks) > 1:
                new_pcm = np.concatenate(chunks)[cursor_frames:]
                if new_pcm.size:
                    try:
                        segs, _ = model.transcribe(
                            (new_pcm.astype(np.float32) / 32768.0).reshape(-1),
                            language=args.language, vad_filter=True, beam_size=5)
                        texts = [_clean(s.text) for s in segs if _clean(s.text)]
                        if texts:
                            live_text = _clean(live_text + " " + " ".join(texts))
                    except Exception as e:
                        live_text = _clean(live_text + f" [err:{type(e).__name__}]")
                    cursor_frames = sum(c.size for c in chunks)
                    last_update = time.time()

            # status line
            vad_str = (_GREEN + "● SPEAKING" + _RESET) if speaking else (_DIM + "○ silent" + _RESET)
            bg = _BOLD + f" Whisper {args.whisper_model} " + _RESET
            bg += f"{_BOLD}on {dev}/{compute}{_RESET}"
            live_line = (_GREEN + f" live: {live_text[:80]}" + _RESET) if live_text \
                else (_DIM + " live: (speaking...)" + _RESET)
            render = [
                f"┌─ FineTuneVOSK  ·  {bg}",
                f"{_DIM}│{_RESET} saving → {REC_DIR}",
                f"{_DIM}│{_RESET} [{_BOLD}{_fmt_seconds(elapsed)}{_RESET}]  {_meter(hist)}   VAD: {vad_str}",
                f"{_DIM}│{_RESET} {live_line}",
                f"{_DIM}└─{_RESET} press Enter to stop" + (f"  (auto in {max(0.0, (args.duration or 0) - elapsed):.0f}s)" if args.duration else ""),
            ]
            tui.render(render)

            # stop conditions
            if args.duration is not None and elapsed >= args.duration:
                finished = True
            elif _enter_pressed():
                tui.close()
                print("[stop] Enter pressed")
                finished = True
    except KeyboardInterrupt:
        finished = True
    finally:
        tui.close()

    # final accurate transcription of the whole clip
    full_pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    try:
        segs, _ = model.transcribe((full_pcm.astype(np.float32) / 32768.0).reshape(-1),
                                   language=args.language, vad_filter=True, beam_size=5)
        final_text = _clean(" ".join(_clean(s.text) for s in segs))
    except Exception as e:
        final_text = _clean(live_text)
    if not final_text:
        final_text = live_text
    if not final_text:
        final_text = "(no speech detected)"

    wav_path, txt_path = _save_clip(full_pcm, final_text)
    print("\n===== final transcript (Whisper) =====")
    print(final_text)
    print(f"\nSaved:  {wav_path}")
    print(f"        {txt_path}")
    return 0


def _enter_pressed() -> bool:
    """Non-blocking check for an Enter keypress on stdin (Linux)."""
    try:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            return line != "" and line.strip() == ""
    except Exception:
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(main())