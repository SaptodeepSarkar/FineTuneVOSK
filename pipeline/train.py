"""Kaldi training orchestration.

Runs the Kaldi acoustic-model training that produces a VOSK-compatible model.
The actual work is done by the shell scripts in `scripts/` (adapted from the
proven `matteo-39/vosk-build-model` recipe):

    1. align_train.sh   — mono -> tri1 -> tri2b(LDA+MLLT) -> tri3b(SAT) alignments
    2. run_tdnn_1j.sh   — chain (LF-MMI) TDNN training seeded from the seed model
    3. copy_final_result.sh — assemble the packaged VOSK model folder

Kaldi must be built and the seed VOSK model must be present. Use the provided
`setup_kaldi.sh` or the Docker image to obtain a working Kaldi install.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import KALDI_EGS_DIR, MODELS_DIR, NUM_JOBS, ROOT, WORK_DIR


class TrainingError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[train] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise TrainingError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}")


def _patch_recipe(recipe: Path, use_gpu: str) -> None:
    """Point the recipe's chain scripts at our train/test sets and GPU mode.

    This automates the manual `sed` edits described in the vosk-build-model
    guide (train_clean_5 -> train, dev_clean_2 -> test, --use-gpu).
    """
    targets = [
        recipe / "local" / "chain" / "tuning" / "run_tdnn_1j.sh",
        recipe / "local" / "nnet3" / "run_ivector_common.sh",
    ]
    for f in targets:
        if not f.is_file():
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        before = txt
        txt = txt.replace("train_clean_5", "train")
        txt = txt.replace("dev_clean_2", "test")
        if txt != before:
            f.write_text(txt, encoding="utf-8")
            print(f"[train] patched {f.relative_to(recipe)} (train_clean_5->train, "
                  "dev_clean_2->test)")

    # GPU mode.
    if use_gpu == "auto":
        try:
            import torch
            use_gpu = "yes" if torch.cuda.is_available() else "no"
        except Exception:
            use_gpu = "no"
    chain = recipe / "local" / "chain" / "tuning" / "run_tdnn_1j.sh"
    if chain.is_file():
        txt = chain.read_text(encoding="utf-8", errors="ignore")
        txt = txt.replace("--use-gpu=yes", f"--use-gpu={use_gpu}")
        # Also handle the bare flag form seen in some recipe versions.
        if "--use-gpu" not in txt:
            txt = txt.replace("--use-gpu", f"--use-gpu={use_gpu}")
        chain.write_text(txt, encoding="utf-8")
        print(f"[train] set chain training --use-gpu={use_gpu}")


def find_seed_model(models_dir: Path = MODELS_DIR) -> Path | None:
    """Locate the seed VOSK model folder inside models/ (recursive)."""
    if not models_dir.is_dir():
        return None
    for cand in models_dir.rglob("final.mdl"):
        return cand.parent
    return None


def _seed_ivector(seed_model: Path, recipe: Path) -> None:
    """Copy the seed model's ivector extractor to seed chain adaptation."""
    src = seed_model / "ivector"
    dst = recipe / "exp" / "chain" / "tdnn1_ivector_seed"
    if not src.is_dir():
        print("[train] seed model has no ivector/ dir; skipping ivector seed")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("final.dubm", "final.ie", "final.mat", "global_cmvn.stats",
              "online_cmvn.conf", "splice_opts"):
        p = src / f
        if p.is_file():
            shutil.copy2(p, dst / f)
    print(f"[train] seeded ivector extractor from {seed_model}")


def run_training(
    work_dir: Path = WORK_DIR,
    kaldi_egs: Path = KALDI_EGS_DIR,
    models_dir: Path = MODELS_DIR,
    num_jobs: int = NUM_JOBS,
    use_gpu: str = "auto",
) -> None:
    """Run the full Kaldi fine-tuning pipeline for the prepped dataset."""
    recipe = kaldi_egs
    if not (recipe / "path.sh").is_file():
        raise TrainingError(
            f"Kaldi mini_librispeech recipe not found at {recipe}. "
            "Run scripts/setup_kaldi.sh or use the Docker image."
        )

    seed = find_seed_model(models_dir)
    if seed:
        print(f"[train] using seed VOSK model: {seed}")
        _seed_ivector(seed, recipe)
    else:
        print("[train] WARNING: no seed model found in models/. "
              "Training will build a fresh model from your data.")

    data_dir = work_dir / "data"
    if not (data_dir / "train" / "text").is_file():
        raise TrainingError("dataset not prepared — run the dataset step first.")

    # Copy prepared data into the recipe.
    for split in ("train", "test"):
        dst = recipe / "data" / split
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("text", "wav.scp", "utt2spk", "spk2utt"):
            src_f = data_dir / split / f
            if src_f.is_file():
                shutil.copy2(src_f, dst / f)
        # spk2utt is derived by Kaldi's fix/validate scripts.
    shutil.copy2(data_dir / "local" / "corpus.txt", recipe / "data" / "local" / "corpus.txt")

    # Copy the dict (lexicon) into the recipe.
    dict_src = data_dir / "local" / "dict"
    if dict_src.is_dir():
        shutil.rmtree(recipe / "data" / "local" / "dict", ignore_errors=True)
        shutil.copytree(dict_src, recipe / "data" / "local" / "dict")

    # Validate the data dirs, then build data/lang from the lexicon.
    _run(["utils/validate_data_dir.sh", "--no-feats", "data/train"], cwd=recipe)
    _run(["utils/fix_data_dir.sh", "data/train"], cwd=recipe)
    _run(["utils/prepare_lang.sh", "data/local/dict", "<UNK>",
          "data/local/lang", "data/lang"], cwd=recipe)

    # Point the recipe's chain scripts at our train/test sets + GPU mode.
    _patch_recipe(recipe, use_gpu)

    # 1) Alignments (mono -> tri).
    _run(["bash", str(ROOT / "scripts" / "align_train.sh"), "--nj", str(num_jobs)],
         cwd=recipe)

    # 2) Chain TDNN training.
    chain_script = recipe / "local" / "chain" / "tuning" / "run_tdnn_1j.sh"
    if not chain_script.is_file():
        raise TrainingError(f"chain training script not found: {chain_script}")
    _run(["bash", str(chain_script)], cwd=recipe)

    print("[train] training complete.")