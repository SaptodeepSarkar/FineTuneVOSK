"""Audio normalization.

Takes *any* audio file (wav / mp3 / m4a / ogg / flac / webm ... at any sample
rate / bit depth / channel count, nested in any folder structure) and converts
it to the 16 kHz, mono, 16-bit PCM WAV that Kaldi requires.

Relies on ffmpeg, which decodes virtually every container/codec in existence,
so there are no format restrictions on the input audio.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .config import CHANNELS, SAMPLE_RATE, SUB_FORMAT, WORK_DIR


class AudioError(RuntimeError):
    pass


# Extensions ffmpeg is known to decode. Anything not in this allow-list is
# still attempted (ffmpeg will tell us if it cannot read it) — this list is
# only used to decide whether a file is "plausibly audio" before processing.
AUDIO_EXTS = {
    ".wav", ".wave", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".flac", ".wma", ".aiff", ".aif", ".amr", ".mp2", ".mka", ".webm",
    ".mp4", ".mkv", ".mov", ".3gp", ".avi", ".wv", ".au", ".snd",
}


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTS


def _safe_id(path: Path) -> str:
    """Build a stable, filesystem-safe utterance id from a file path.

    Uses the file's path (excluding extension and leading root/drive) so that
    two files with the same basename in different folders never collide.
    """
    parts = list(path.with_suffix("").parts)
    # Drop the leading root "/" or Windows drive component.
    if parts and (parts[0] in ("/", "\\") or len(parts[0]) == 2 and parts[0].endswith(":")):
        parts = parts[1:]
    slug = "_".join(parts)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    slug = slug.strip("_")
    return slug or "utt"


def find_audio_files(root: Path) -> list[Path]:
    """Recursively collect every audio file under `root`."""
    if not root.is_dir():
        raise AudioError(f"audio source is not a directory: {root}")
    files = sorted(p for p in root.rglob("*") if is_audio_file(p))
    if not files:
        raise AudioError(
            f"no audio files found under {root}. Supported: "
            + ", ".join(sorted(AUDIO_EXTS))
        )
    return files


def _ffmpeg_raw_decode(src: Path, dst: Path) -> None:
    """Decode src -> dst using ffmpeg with explicit 16k/mono/pcm_s16le."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", SUB_FORMAT,
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(
            f"ffmpeg failed to convert {src}:\n{proc.stderr.strip()}"
        )
    if not dst.is_file() or dst.stat().st_size == 0:
        raise AudioError(f"ffmpeg produced no output for {src}")


def normalize_file(src: Path, out_dir: Path) -> Path:
    """Convert one source audio file to 16k/mono WAV in `out_dir`.

    Returns the path of the normalized wav.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    utt_id = _safe_id(src)
    dst = out_dir / f"{utt_id}.wav"
    if dst.is_file() and dst.stat().st_size > 0:
        return dst  # already normalized (idempotent re-runs)
    _ffmpeg_raw_decode(src, dst)
    return dst


def normalize_dataset(src_root: Path, work_dir: Path) -> list[tuple[Path, Path]]:
    """Normalize every audio file under `src_root`.

    Returns a list of (source_audio, normalized_wav) pairs.
    """
    wav_dir = work_dir / "wav"
    pairs = []
    for src in find_audio_files(src_root):
        wav = normalize_file(src, wav_dir)
        pairs.append((src, wav))
    return pairs


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False