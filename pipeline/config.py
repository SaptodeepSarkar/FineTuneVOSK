"""Central configuration for the FineTuneVOSK pipeline.

Everything here can be overridden on the command line (see run.py).
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (relative to the project root)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent

# Where the user drops their seed VOSK model (any format, any depth).
MODELS_DIR = ROOT / "models"

# Where the user drops their raw audio (any format, any depth).
DATA_DIR = ROOT / "data" / "raw"

# Intermediate / result directories.
WORK_DIR = ROOT / "work"                  # normalized wav + transcripts + dataset
OUTPUT_DIR = ROOT / "output"              # final packaged VOSK model
KALDI_EGS_DIR = ROOT / "kaldi" / "egs" / "mini_librispeech" / "s5"

# --------------------------------------------------------------------------- #
# Whisper transcription defaults
# --------------------------------------------------------------------------- #
# "large-v3" is the most accurate for accented / non-native speech. Use a
# smaller model (tiny/base/small/medium) if you want faster, lighter runs.
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "auto"          # auto | cpu | cuda
WHISPER_COMPUTE_TYPE = "auto"    # auto | int8 | float16
WHISPER_LANGUAGE = None          # None = auto-detect per file, or e.g. "en"
WHISPER_BEAM_SIZE = 5

# --------------------------------------------------------------------------- #
# Audio normalization
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 16000              # Kaldi requirement
CHANNELS = 1                     # mono
SUB_FORMAT = "wav"               # pcm_s16le output

# --------------------------------------------------------------------------- #
# Kaldi training
# --------------------------------------------------------------------------- #
# Number of parallel jobs for feature extraction / alignment / decoding.
NUM_JOBS = 4
# Seed chain config: a small two-layer TDNN is enough for adaptation.
TDNN_LAYERS = 2
TDNN_HIDDEN = 625
# LM N-gram order (SRILM).
LM_ORDER = 3