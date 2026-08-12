"""Whisper transcription.

Transcribes normalized audio with OpenAI Whisper (via faster-whisper) to
produce the ground-truth text used to fine-tune the VOSK acoustic model.

Whisper large-v3 is used by default because it has the best coverage of
accents and non-native speech (Indian English, German English, etc.). Any
`large-v2`, `medium`, `small`, `base` or `tiny` model name also works — pass
it with `--whisper-model`.

The transcription is purely a *label generator* — it never touches the model
being fine-tuned. If you already have transcripts (a `.txt` next to each audio
file, or a `metadata.csv`), the pipeline will use those instead and skip
Whisper entirely (see `--use-existing-transcripts`).
"""
from __future__ import annotations

import time
from pathlib import Path

from .config import (
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    WORK_DIR,
)


class TranscriptionError(RuntimeError):
    pass


def _load_model(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from e
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        # The driver can report CUDA while the runtime libs (libcublas) are
        # missing, which only fails at encode time. Probe once and fall back.
        if device == "cuda":
            import numpy as np
            list(model.transcribe(np.zeros(16000, np.float32), beam_size=1))
        return model
    except Exception:
        if device != "cpu" or compute_type != "int8":
            print("[transcribe] WARNING: requested backend unavailable; "
                  "falling back to CPU/int8")
            return WhisperModel(model_name, device="cpu", compute_type="int8")
        raise


def transcribe_file(model, wav_path: Path, language=None, beam_size=5) -> str:
    """Transcribe one normalized wav and return the cleaned text."""
    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
    )
    parts = [seg.text.strip() for seg in segments]
    text = " ".join(parts).strip()
    return text


def _existing_transcript_for(wav_path: Path, src_audio: Path) -> str | None:
    """Look for a hand-written transcript next to the source audio.

    Accepted naming patterns (sibling of the source file):
      - myfile.txt            (same stem)
      - myfile.transcript.txt
      - myfile.txt inside a sidecar/ subfolder
    Returns the cleaned single-line text, or None if none found.
    """
    candidates = [
        src_audio.with_suffix(".txt"),
        src_audio.with_suffix(".transcript.txt"),
    ]
    for cand in candidates:
        if cand.is_file():
            text = cand.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return _clean_text(text)
    return None


def _clean_text(text: str) -> str:
    """Normalize a transcript to a single line of simple whitespace."""
    text = text.replace("\n", " ")
    return " ".join(text.split())


def transcribe_dataset(
    pairs,
    work_dir: Path,
    model_name: str = WHISPER_MODEL,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
    language: str | None = WHISPER_LANGUAGE,
    beam_size: int = WHISPER_BEAM_SIZE,
    use_existing: bool = True,
    force: bool = False,
) -> dict[str, str]:
    """Transcribe every normalized wav, writing one transcript per file.

    `pairs` is the list of (source_audio, normalized_wav) from audio.normalize_dataset.
    Returns {utterance_id: transcript}. Transcripts are also written to
    work_dir/transcripts/<id>.txt for transparency / later reuse.
    """
    tr_dir = work_dir / "transcripts"
    tr_dir.mkdir(parents=True, exist_ok=True)

    model = None
    results: dict[str, str] = {}

    for src_audio, wav in pairs:
        utt_id = wav.stem
        out_txt = tr_dir / f"{utt_id}.txt"

        # 1) Reuse an existing transcript if present (idempotent / manual labels).
        if use_existing and not force:
            existing = _existing_transcript_for(wav, src_audio)
            if existing is None and out_txt.is_file():
                existing = _clean_text(out_txt.read_text(encoding="utf-8", errors="ignore"))
            if existing:
                results[utt_id] = existing
                continue

        # 2) Otherwise run Whisper.
        if model is None:
            print(f"[transcribe] loading Whisper model '{model_name}' on {device} "
                  f"({compute_type}) ...")
            t0 = time.time()
            model = _load_model(model_name, device, compute_type)
            print(f"[transcribe] model ready in {time.time() - t0:.1f}s")

        text = transcribe_file(model, wav, language=language, beam_size=beam_size)
        if not text:
            # Very short / silent utterance — keep it so the utt id stays aligned.
            text = "<UNK>"
        out_txt.write_text(text + "\n", encoding="utf-8")
        results[utt_id] = text

    print(f"[transcribe] {len(results)} utterance(s) transcribed "
          f"(transcripts saved to {tr_dir})")
    return results