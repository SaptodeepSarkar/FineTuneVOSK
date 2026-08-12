"""FineTuneVOSK pipeline package.

Automates the process of:
  1. normalizing arbitrary audio into Kaldi-ready 16 kHz mono WAV,
  2. transcribing it with Whisper to produce ground-truth text,
  3. building a Kaldi-format training dataset + lexicon + language model,
  4. fine-tuning a VOSK (Kaldi chain) acoustic model on that data,
  5. packaging the result into a ready-to-use VOSK model folder.
"""