"""Lexicon generation.

Kaldi needs a pronunciation dictionary (`lexicon.txt`) mapping every word in
your transcripts to its phonemes. We generate it automatically with
espeak-ng, which supports ~100 languages and many accents, so no manual
dictionary work is required.

Outputs the standard Kaldi dict layout under `data/local/dict/`:
    lexicon.txt  nonsilence_phones.txt  silence_phones.txt  optional_silence.txt

If espeak-ng is not installed, fall back to a CMU-style phoneme set derived
from the word's characters (lossy but keeps the pipeline running), and warn.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .config import ROOT


class LexiconError(RuntimeError):
    pass


OOV_WORD = "<UNK>"          # out-of-vocabulary token
OOV_PHONE = "SPN"           # spoken-noise phoneme for OOV
SIL_PHONE = "SIL"
SPN_PHONE = "SPN"


def _find_espeak() -> str | None:
    for name in ("espeak-ng", "espeak"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _espeak_phonemes(exe: str, word: str, lang: str) -> str | None:
    """Return space-separated phonemes for `word` via espeak.

    Uses `--ipa=1` (undivided IPA) which matches the phone set Kaldi treats as
    plain symbols. Returns None on any failure (e.g. word not pronounceable).
    """
    try:
        proc = subprocess.run(
            [exe, "-q", "-v", lang, "--ipa=1", word],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    # Light IPA normalization: drop stress/length markers and combining marks
    # so the phone set stays a clean space-separated symbol list for Kaldi.
    out = re.sub(r"[ˈˌː\u02C8\u02CC\u02D0\u0301\u0300\u0311\u0361\u200D]", "", out)
    out = out.replace("_", " ").strip()
    out = re.sub(r"\s+", " ", out)
    return out or None


def _fallback_phonemes(word: str) -> str:
    """Lossy character-based fallback phoneme set when espeak is missing."""
    return " ".join(
        c.lower() if c.isalpha() else "SPN"
        for c in word
        if c.isalpha()
    )


def _tokenize_text(text: str) -> list[str]:
    """Split a transcript into lowercase word tokens (keeping UNK)."""
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
    return words


def build_lexicon(
    transcripts: dict[str, str],
    dict_dir: Path,
    lang: str = "en",
) -> dict[str, str]:
    """Build the full Kaldi dict dir from transcripts.

    Returns {word: phonemes}. Handles the <UNK> token and rewrites
    nonsilence/silence phone files.
    """
    espeak = _find_espeak()
    if espeak:
        print(f"[lexicon] using {espeak} (lang={lang}) for phonemization")
    else:
        print("[lexicon] WARNING: espeak-ng not found — using lossy fallback "
              "phonemizer. Install espeak-ng for accurate pronunciations.")

    # Collect the vocabulary from all transcripts.
    vocab: set[str] = set()
    for text in transcripts.values():
        vocab.update(_tokenize_text(text))

    lexicon: dict[str, str] = {}
    if espeak:
        for word in sorted(vocab):
            ph = _espeak_phonemes(espeak, word, lang)
            lexicon[word] = ph if ph else _fallback_phonemes(word)
    else:
        for word in sorted(vocab):
            lexicon[word] = _fallback_phonemes(word)

    # Always ensure the OOV token maps to a spoken-noise phone.
    lexicon.setdefault(OOV_WORD, OOV_PHONE)
    # Ensure OOV is treated as a word (Kaldi needs it in the dict).
    if "<UNK>" not in vocab:
        vocab.add("<UNK>")

    dict_dir.mkdir(parents=True, exist_ok=True)

    # lexicon.txt  (word followed by its phonemes)
    lex_lines = []
    for word, phones in sorted(lexicon.items()):
        lex_lines.append(f"{word} {phones}")
    (dict_dir / "lexicon.txt").write_text("\n".join(lex_lines) + "\n", encoding="utf-8")

    # nonsilence_phones.txt — every phone except the special silence/noise ones.
    phones = set()
    for phones_str in lexicon.values():
        if phones_str == OOV_PHONE:
            continue
        phones.update(phones_str.split())
    phones.discard(SIL_PHONE)
    phones.discard(SPN_PHONE)
    (dict_dir / "nonsilence_phones.txt").write_text(
        "\n".join(sorted(phones)) + "\n", encoding="utf-8")

    # silence_phones.txt + optional_silence.txt
    (dict_dir / "silence_phones.txt").write_text(
        "\n".join([SIL_PHONE, OOV_PHONE, SPN_PHONE]) + "\n", encoding="utf-8")
    (dict_dir / "optional_silence.txt").write_text(SIL_PHONE + "\n", encoding="utf-8")

    return lexicon