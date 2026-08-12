#!/usr/bin/env python3
"""FineTuneVOSK — fine-tune a VOSK speech model on your own audio.

Designed to be usable by anyone with zero friction:
  * Drop your audio anywhere in `data/raw/` (any format: wav/mp3/m4a/ogg/...,
    any sample rate, any folder nesting). Nothing else is required.
  * Drop a seed VOSK model into `models/` (or pass `--model`).
  * The pipeline normalizes every file, transcribes it with Whisper large-v3
    (best accent coverage), builds the Kaldi dataset + lexicon automatically,
    fine-tunes the VOSK model, and packages a ready-to-use model.

Examples
--------
    # Full pipeline, guided defaults:
    python run.py all --audio data/raw --model models/vosk-model-en-us-0.22

    # Use a smaller/faster Whisper for transcription:
    python run.py all --audio data/raw --model models/vosk-model-en-us-0.22 \\
        --whisper-model medium

    # You already have transcripts (a .txt next to each audio file):
    python run.py all --audio data/raw --model models/vosk-model-en-us-0.22 --no-whisper

    # Step-by-step:
    python run.py prepare --audio data/raw --model models/vosk-model-en-us-0.22
    python run.py train    --work work/
    python run.py package  --work work/ --out output/ --name vosk-model-finetuned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pipeline.config as cfg

PY = sys.executable


def _prepare(args) -> dict:
    """Normalize + transcribe + build dataset + lexicon."""
    from pipeline import audio, kaldi_prep, lexicon, transcribe

    src = Path(args.audio)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if not audio.check_ffmpeg():
        sys.exit("[error] ffmpeg is required. Install it (e.g. `sudo apt install ffmpeg`).")

    print(f"[1/4] normalizing audio from {src} ...")
    pairs = audio.normalize_dataset(src, work)
    print(f"      {len(pairs)} file(s) -> 16kHz mono wav in {work/'wav'}")

    print(f"[2/4] transcribing with Whisper ({args.whisper_model}) ...")
    transcripts = transcribe.transcribe_dataset(
        pairs, work,
        model_name=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        use_existing=not args.no_whisper,
    )

    print("[3/4] building Kaldi dataset ...")
    splits = kaldi_prep.build_dataset(pairs, work, src, transcripts,
                                      test_ratio=args.test_ratio)

    print("[4/4] building lexicon ...")
    dict_dir = work / "data" / "local" / "dict"
    lexicon.build_lexicon(transcripts, dict_dir, lang=args.lang)
    print(f"      lexicon -> {dict_dir}")

    return {"pairs": pairs, "transcripts": transcripts, "splits": splits}


def _cmd_prepare(args) -> int:
    _prepare(args)
    print("\nDataset ready in work/. Next: python run.py train")
    return 0


def _cmd_train(args) -> int:
    from pipeline import train
    train.run_training(
        work_dir=Path(args.work),
        kaldi_egs=Path(args.kaldi),
        models_dir=Path(args.model_dir),
        num_jobs=args.nj,
        use_gpu=args.gpu,
    )
    print("\nTraining complete. Next: python run.py package")
    return 0


def _cmd_package(args) -> int:
    from pipeline import package
    package.package_model(
        output_dir=Path(args.out),
        kaldi_egs=Path(args.kaldi),
        name=args.name,
    )
    return 0


def _cmd_all(args) -> int:
    from pipeline import package, train

    _prepare(args)

    print("\n[+] running Kaldi fine-tuning ...")
    train.run_training(
        work_dir=Path(args.work),
        kaldi_egs=Path(args.kaldi),
        models_dir=Path(args.model_dir),
        num_jobs=args.nj,
        use_gpu=args.gpu,
    )

    print("\n[+] packaging VOSK model ...")
    model_dir = package.package_model(
        output_dir=Path(args.out),
        kaldi_egs=Path(args.kaldi),
        name=args.name,
    )
    print(f"\nDone. Your fine-tuned VOSK model: {model_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Fine-tune a VOSK model on your own audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def _add_common(sp):
        sp.add_argument("--audio", default=str(cfg.DATA_DIR),
                        help=f"folder of your raw audio (any format). default: {cfg.DATA_DIR}")
        sp.add_argument("--model", dest="model_dir", default=str(cfg.MODELS_DIR),
                        help=f"folder containing the seed VOSK model. default: {cfg.MODELS_DIR}")
        sp.add_argument("--work", default=str(cfg.WORK_DIR),
                        help="working dir for intermediate files. default: work/")
        sp.add_argument("--out", default=str(cfg.OUTPUT_DIR),
                        help="output dir for the packaged model. default: output/")
        sp.add_argument("--kaldi", default=str(cfg.KALDI_EGS_DIR),
                        help="path to the Kaldi mini_librispeech s5 recipe.")
        sp.add_argument("--whisper-model", default=cfg.WHISPER_MODEL,
                        help=f"Whisper model for transcription. default: {cfg.WHISPER_MODEL}")
        sp.add_argument("--whisper-device", default=cfg.WHISPER_DEVICE,
                        choices=["auto", "cpu", "cuda"])
        sp.add_argument("--compute-type", default=cfg.WHISPER_COMPUTE_TYPE,
                        choices=["auto", "int8", "float16", "int8_float16"])
        sp.add_argument("--language", default=cfg.WHISPER_LANGUAGE,
                        help="force transcript language (e.g. en, de, hi); auto-detect if unset")
        sp.add_argument("--beam-size", type=int, default=cfg.WHISPER_BEAM_SIZE)
        sp.add_argument("--no-whisper", action="store_true",
                        help="skip Whisper; use existing .txt transcripts beside audio files")
        sp.add_argument("--lang", default="en",
                        help="espeak language code for lexicon phonemization. default: en")
        sp.add_argument("--test-ratio", type=float, default=0.1,
                        help="fraction of utterances held out for test. default: 0.10")
        sp.add_argument("--nj", type=int, default=cfg.NUM_JOBS,
                        help="Kaldi parallel jobs. default: 4")
        sp.add_argument("--gpu", default="auto", choices=["auto", "yes", "no", "wait"],
                        help="Kaldi GPU usage. auto uses GPU if detected.")
        return sp

    sp_prepare = sub.add_parser("prepare", help="normalize + transcribe + build dataset/lexicon")
    _add_common(sp_prepare)
    sp_prepare.set_defaults(func=_cmd_prepare)

    sp_train = sub.add_parser("train", help="run Kaldi fine-tuning on the prepared dataset")
    _add_common(sp_train)
    sp_train.set_defaults(func=_cmd_train)

    sp_package = sub.add_parser("package", help="package the trained model into a VOSK folder")
    _add_common(sp_package)
    sp_package.set_defaults(func=_cmd_package)

    sp_all = sub.add_parser("all", help="run the complete pipeline end-to-end")
    _add_common(sp_all)
    sp_all.set_defaults(func=_cmd_all)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())