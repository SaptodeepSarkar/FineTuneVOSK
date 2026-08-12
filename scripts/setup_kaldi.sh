#!/usr/bin/env bash
# setup_kaldi.sh — clone + build Kaldi and download a default seed VOSK model.
#
# This is the one-time host setup for running the actual fine-tune outside
# Docker. Prefer the Docker route if you can (see Dockerfile) — it is fully
# reproducible. Either way you need a working Kaldi build before `train`.
#
# Usage:
#   bash scripts/setup_kaldi.sh                 # build everything, get a seed model
#   bash scripts/setup_kaldi.sh --no-srilm      # skip SRILM (LM falls back)
#   bash scripts/setup_kaldi.sh --gpu           # build Kaldi with CUDA support
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_SRILM=1
CUDA_FLAG=""
KALDI_VERSION="master"
SEED_MODEL_URL="${SEED_MODEL_URL:-https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip}"
SEED_MODEL_DIR="models/vosk-model-small-en-us-0.15"

for arg in "$@"; do
  case "$arg" in
    --no-srilm) WITH_SRILM=0 ;;
    --gpu) CUDA_FLAG="--cudatoolkit-dir=/usr/local/cuda" ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "==> [1/4] Installing system dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  git build-essential automake autoconf libtool subversion \
  zlib1g-dev libbz2-dev liblzma-dev libomp-dev \
  ffmpeg espeak-ng python3 python3-pip python3-venv \
  sox wget unzip curl

echo "==> [2/4] Cloning Kaldi"
if [ ! -d kaldi/.git ]; then
  git clone --depth 1 https://github.com/kaldi-asr/kaldi.git
else
  echo "      kaldi/ already present — skipping clone"
fi

echo "==> [3/4] Building Kaldi (this takes a while)"
cd kaldi/tools
make -j"$(nproc)"
if [ "$WITH_SRILM" = "1" ]; then
  echo "      Installing SRILM (language modeling) ..."
  ./install_srilm.sh || echo "      WARN: SRILM install failed — LM step will need it manually."
  . ./env.sh
fi
cd ../src
./configure --shared ${CUDA_FLAG}
make depend -j"$(nproc)"
make -j"$(nproc)"
cd "$ROOT"

echo "==> [4/4] Downloading a default seed VOSK model into models/"
if [ ! -d "$SEED_MODEL_DIR" ]; then
  wget -q "$SEED_MODEL_URL" -O /tmp/seed.zip
  mkdir -p models
  unzip -q /tmp/seed.zip -d models
  rm -f /tmp/seed.zip
  echo "      Seed model -> $SEED_MODEL_DIR"
else
  echo "      Seed model already present -> $SEED_MODEL_DIR"
fi

echo ""
echo "Setup complete. You can now run:"
echo "  python run.py all --audio data/raw --model models/$(basename "$SEED_MODEL_DIR")"
echo ""
echo "Note: the mini_librispeech s5 recipe used for training lives at:"
echo "  $(pwd)/kaldi/egs/mini_librispeech/s5"