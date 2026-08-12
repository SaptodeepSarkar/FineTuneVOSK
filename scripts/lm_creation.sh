#!/bin/bash
# lm_creation.sh — build the N-gram language model (G.fst) with SRILM.
# Adapted from matteo-39/vosk-build-model (MIT).
# Run from inside the Kaldi mini_librispeech s5 recipe directory.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

lm_order=3

echo "[lm] making lm.arpa"
loc=$(which ngram-count)

if [ -z "$loc" ]; then
  if uname -a | grep 64 >/dev/null; then
    sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
  else
    sdir=$KALDI_ROOT/tools/srilm/bin/i686
  fi
  if [ -f "$sdir/ngram-count" ]; then
    echo "[lm] using SRILM from $sdir"
    export PATH=$PATH:$sdir
  else
    echo "[lm] ERROR: SRILM toolkit is not installed."
    echo "        Install it under $KALDI_ROOT/tools with tools/install_srilm.sh"
    exit 1
  fi
fi

local=data/local
lang=data/lang
mkdir -p "$local/tmp"
ngram-count -order $lm_order -write-vocab "$local/tmp/vocab-full.txt" \
    -wbdiscount -text "$local/corpus.txt" -lm "$local/tmp/lm.arpa"

echo "[lm] making G.fst"
arpa2fst --max-arpa-warnings=-1 --disambig-symbol=#0 \
    --read-symbol-table="$lang/words.txt" "$local/tmp/lm.arpa" "$lang/G.fst"
gzip -c "$local/tmp/lm.arpa" > "$local/tmp/lm.arpa.gz"

echo "[lm] done"