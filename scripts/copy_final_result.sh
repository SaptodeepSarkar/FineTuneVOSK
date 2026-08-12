#!/bin/bash
# copy_final_result.sh — assemble the trained chain model into a VOSK-compatible
# folder (ivector extractor + final.mdl + HCLG.fst + words + word_boundary).
# Adapted from matteo-39/vosk-build-model (MIT).
#
# Usage: copy_final_result.sh <output_dir> <model_name>
# Run from inside the Kaldi mini_librispeech s5 recipe directory.

dir="${1:-$HOME/model}"
name="${2:-vosk-model-finetuned}"

if [ -d "$dir/$name" ]; then
  echo "[package] $dir/$name already exists; removing to refresh"
  rm -rf "$dir/$name"
fi
mkdir -p "$dir/$name/ivector"

ONLINE=$(ls -d exp/chain/tdnn1*_sp_online 2>/dev/null | head -1)
if [ -z "$ONLINE" ]; then
  echo "[package] ERROR: no online chain dir found (exp/chain/tdnn1*_sp_online)"
  exit 1
fi
echo "[package] packaging from $ONLINE"

cp "$ONLINE/ivector_extractor/final.dubm"       "$dir/$name/ivector/"
cp "$ONLINE/ivector_extractor/final.ie"         "$dir/$name/ivector/"
cp "$ONLINE/ivector_extractor/final.mat"        "$dir/$name/ivector/"
cp "$ONLINE/ivector_extractor/global_cmvn.stats" "$dir/$name/ivector/"
cp "$ONLINE/ivector_extractor/online_cmvn.conf" "$dir/$name/ivector/"
cp "$ONLINE/ivector_extractor/splice_opts"      "$dir/$name/ivector/"
cp "$ONLINE/ivector_extractor/conf/splice.conf" "$dir/$name/ivector/" 2>/dev/null || true

cp "$ONLINE/conf/mfcc.conf" "$dir/$name/"
cp "$ONLINE/final.mdl"      "$dir/$name/"
cp exp/chain/tree_sp/graph_tgsmall/HCLG.fst "$dir/$name/"
cp exp/chain/tree_sp/graph_tgsmall/words.txt "$dir/$name/"
cp exp/chain/tree_sp/graph_tgsmall/phones/word_boundary.int "$dir/$name/"

echo "[package] done -> $dir/$name"