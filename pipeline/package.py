"""Model packaging.

Assembles the Kaldi chain output into a ready-to-use VOSK model folder and
writes the `conf/model.conf` that VOSK requires (it is not produced by Kaldi).

The packaged folder is the final deliverable — point VOSK at it exactly as you
would at a stock model downloaded from alphacephei.com/vosk/models.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import KALDI_EGS_DIR, OUTPUT_DIR, ROOT

MODEL_CONF = """\
--min-active=200
--max-active=3000
--beam=10.0
--lattice-beam=2.0
--acoustic-scale=1.0
--frame-subsampling-factor=3
--endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10
--endpoint.rule2.min-trailing-silence=0.5
--endpoint.rule3.min-trailing-silence=1.0
--endpoint.rule4.min-trailing-silence=2.0
"""


class PackageError(RuntimeError):
    pass


def _find_online_dir(recipe: Path) -> Path | None:
    for cand in sorted(recipe.glob("exp/chain/tdnn1*_sp_online")):
        if cand.is_dir() and (cand / "final.mdl").is_file():
            return cand
    return None


def _find_graph(recipe: Path) -> Path:
    for cand in (recipe / "exp" / "chain" / "tree_sp").glob("graph_tgsmall"):
        if cand.is_dir():
            return cand
    raise PackageError("chain graph_tgsmall not found under exp/chain/tree_sp/")


def package_model(
    output_dir: Path = OUTPUT_DIR,
    kaldi_egs: Path = KALDI_EGS_DIR,
    name: str = "vosk-model-finetuned",
) -> Path:
    """Package the trained chain model into a VOSK model folder."""
    recipe = kaldi_egs
    online = _find_online_dir(recipe)
    if online is None:
        raise PackageError(
            "no trained online chain dir found under exp/chain/ "
            "(pattern tdnn1*_sp_online with final.mdl). Run training first."
        )
    graph = _find_graph(recipe)

    model_dir = output_dir / name
    iv_dir = model_dir / "ivector"
    iv_dir.mkdir(parents=True, exist_ok=True)

    # ivector extractor files.
    iv_src = online / "ivector_extractor"
    for f in ("final.dubm", "final.ie", "final.mat", "global_cmvn.stats",
              "online_cmvn.conf", "splice_opts"):
        p = iv_src / f
        if p.is_file():
            shutil.copy2(p, iv_dir / f)
    # splice.conf lives under conf/ in the online dir.
    sc = online / "conf" / "splice.conf"
    if sc.is_file():
        shutil.copy2(sc, iv_dir / "splice.conf")

    # Model + graph files.
    shutil.copy2(online / "conf" / "mfcc.conf", model_dir / "mfcc.conf")
    shutil.copy2(online / "final.mdl", model_dir / "final.mdl")
    shutil.copy2(graph / "HCLG.fst", model_dir / "HCLG.fst")
    shutil.copy2(graph / "words.txt", model_dir / "words.txt")
    wbi = graph / "phones" / "word_boundary.int"
    if wbi.is_file():
        shutil.copy2(wbi, model_dir / "word_boundary.int")

    # model.conf — required by VOSK, not produced by Kaldi.
    (model_dir / "conf").mkdir(parents=True, exist_ok=True)
    (model_dir / "conf" / "model.conf").write_text(MODEL_CONF, encoding="utf-8")

    print(f"[package] VOSK model written to {model_dir}")
    print("[package] Test with:  vosk-transcriber -m {0} --input your-audio.wav".format(model_dir))
    return model_dir