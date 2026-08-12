"""Kaldi dataset preparation.

Turns the normalized wavs + Whisper transcripts into the Kaldi `data/`
directory layout that the mini_librispeech chain recipe expects:

    data/train/{text, wav.scp, utt2spk, spk2utt}
    data/test/{text, wav.scp, utt2spk, spk2utt}
    data/local/corpus.txt

Speakers are derived automatically from the folder a source audio lives in
(files at the top of the raw folder share one default speaker). A small
held-out test split is produced for decoding/validation during training.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from .audio import normalize_dataset


class KaldiPrepError(RuntimeError):
    pass


DEFAULT_SPEAKER = "speaker1"
TEST_RATIO = 0.1


def _speaker_for(src: Path, root: Path) -> str:
    """Speaker id = the source audio's immediate parent folder name."""
    try:
        rel = src.parent.relative_to(root)
        parts = rel.parts
        if parts:
            name = parts[0]
            # Files placed directly in a top folder get that folder's name.
            return _safe_spk(name)
    except ValueError:
        pass
    return DEFAULT_SPEAKER


def _safe_spk(name: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return s or DEFAULT_SPEAKER


def _clean_text(text: str) -> str:
    import re
    text = text.replace("\n", " ")
    text = re.sub(r"<UNK>", "", text)  # drop OOV markers from train text
    # Collapse punctuation to spaces (Kaldi words are letters/digits/hyphen).
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().upper()


def build_dataset(
    pairs,
    work_dir: Path,
    root: Path,
    transcripts: dict[str, str],
    test_ratio: float = TEST_RATIO,
) -> dict[str, Path]:
    """Write the Kaldi data dirs. Returns {split: data_dir}."""
    data_dir = work_dir / "data"

    # Group utterances by speaker.
    spk_utts: dict[str, list[tuple[str, Path, Path, str]]] = defaultdict(list)
    for src, wav in pairs:
        utt_id = wav.stem
        text = transcripts.get(utt_id, "").strip()
        if not text:
            raise KaldiPrepError(
                f"no transcript for {src.name} — run the transcribe step first, "
                "or place a {name}.txt beside the audio."
            )
        spk = _speaker_for(src, root)
        spk_utts[spk].append((utt_id, wav, src, text))

    # Deterministic train/test split across all utterances.
    all_utts = [u for utts in spk_utts.values() for u in utts]
    all_utts.sort(key=lambda u: u[0])
    n_test = max(1, math.floor(len(all_utts) * test_ratio))
    test_ids = {u[0] for u in all_utts[:n_test]}
    train_ids = {u[0] for u in all_utts[n_test:]}

    splits = {}
    for split, id_set in (("train", train_ids), ("test", test_ids)):
        split_dir = data_dir / split
        _write_split(split_dir, spk_utts, id_set, split)
        splits[split] = split_dir

    # corpus.txt for the language model (train transcripts only, one per line).
    corpus_lines = []
    for utt_id, wav, src, text in all_utts:
        if utt_id in train_ids:
            corpus_lines.append(_clean_text(text))
    corpus_dir = data_dir / "local"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus.txt").write_text(
        "\n".join(corpus_lines) + "\n" if corpus_lines else "\n",
        encoding="utf-8",
    )

    print(f"[kaldi_prep] train={len(train_ids)} test={len(test_ids)} utterances "
          f"written to {data_dir}")
    return splits


def _write_split(split_dir: Path, spk_utts, id_set, split: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    text_lines, wav_lines, utt2spk_lines = [], [], []
    for spk, utts in spk_utts.items():
        for utt_id, wav, src, text in utts:
            if utt_id not in id_set:
                continue
            text_lines.append(f"{utt_id} {_clean_text(text)}")
            wav_lines.append(f"{utt_id} {wav}")
            utt2spk_lines.append(f"{utt_id} {spk}")
    # utt2spk must be sorted by utt id for validation.
    utt2spk_lines.sort()

    (split_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (split_dir / "wav.scp").write_text("\n".join(wav_lines) + "\n", encoding="utf-8")
    (split_dir / "utt2spk").write_text("\n".join(utt2spk_lines) + "\n", encoding="utf-8")