#!/bin/bash
# align_train.sh — build mono -> tri1 -> tri2b (LDA+MLLT) -> tri3b (SAT)
# acoustic systems and their alignments, up to the point the chain recipe needs.
#
# Adapted from matteo-39/vosk-build-model (MIT) for the FineTuneVOSK pipeline.
# Run from inside the Kaldi mini_librispeech s5 recipe directory.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

nj=4
stage=0

# ---- stage 0: monophone + alignment ----
if [ $stage -le 0 ]; then
  echo "[align] train monophone system"
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/mono
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/mono exp/mono_ali_train
fi

# ---- stage 1: delta+delta-delta triphone ----
if [ $stage -le 1 ]; then
  echo "[align] train first delta+delta-delta triphone system"
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      2000 10000 data/train data/lang exp/mono_ali_train exp/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/tri1 exp/tri1_ali_train
fi

# ---- stage 2: LDA+MLLT ----
if [ $stage -le 2 ]; then
  echo "[align] train LDA+MLLT system"
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
      data/train data/lang exp/tri1_ali_train exp/tri2b
  steps/align_si.sh --nj $nj --cmd "$train_cmd" --use-graphs true \
      data/train data/lang exp/tri2b exp/tri2b_ali_train
fi

# ---- stage 3: SAT ----
if [ $stage -le 3 ]; then
  echo "[align] train SAT system"
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
      data/train data/lang exp/tri2b_ali_train exp/tri3b
fi

# ---- stage 4: pronunciation probabilities + LM graph ----
if [ $stage -le 4 ]; then
  echo "[align] compute pron/sil probabilities and rebuild lang"
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang exp/tri3b
  mv data/local/dict/lexicon.txt data/local/dict/lexicon_old.txt
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
      data/local/dict \
      exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
      exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

  . ./lm_creation.sh
  utils/build_const_arpa_lm.sh \
      data/local/tmp/lm.arpa.gz data/lang data/lang_test_tglarge

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/tri3b exp/tri3b_ali_train
fi

# ---- stage 5: copy train data to test dir (recipe expects data/test) ----
if [ $stage -le 5 ]; then
  echo "[align] finalize lang dirs and mirror test data"
  cp -r data/lang data/lang_test_tgsmall
  cp -r data/lang data/lang_test_tgmed
  rsync -av --progress data/train/* data/test/ --exclude split* 2>/dev/null || true
fi

echo "[align] alignment stage done"