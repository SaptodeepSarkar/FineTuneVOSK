# FineTuneVOSK

**Fine-tune a VOSK speech-recognition model on *your own* audio — with zero dataset
hassle.**

Drop your audio anywhere (any format, any length, any accent). Whisper **large-v3**
transcribes it automatically to create your training labels; the pipeline then
normalizes everything, builds a fully-formed Kaldi dataset + pronunciation lexicon
for you, fine-tunes a VOSK (Kaldi chain) acoustic model on it, and packages a
ready-to-use model you can drop straight into VOSK.

No manually-curated transcript files.
No required folder structure.
No forced dataset format.
Just point, run, and get a model that knows *your* words and *your* voice.

---

## How it works

```
  your audio (any format, any nesting)
        │
        ▼  ffmpeg — convert to 16 kHz / mono / PCM wav      [normalize]
        │
        ▼  Whisper large-v3 — transcribe each file          [transcribe]
        │  (or reuse your own .txt transcripts if provided)
        │
        ▼  build Kaldi data/ dir + lexicon + LM             [kaldi_prep + lexicon]
        │  espeak-ng generates pronunciations automatically
        │
        ▼  Kaldi chain (LF-MMI) fine-tune,
        │  seeded from your seed VOSK model                 [train]
        │
        ▼  package final.mdl + HCLG.fst + ivector           [package]
        │
        ▼
  a ready-to-use VOSK model folder
```

---

## Quick start

```bash
# 1. Install the Python side
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. Put a seed VOSK model in the models/ folder
#    (e.g. download one from https://alphacephei.com/vosk/models and unzip it there)

# 3. Put your raw audio in data/raw/  (any format: wav/mp3/m4a/ogg/... any subfolders)

# 4. Fine-tune. One command does everything:
python run.py all --audio data/raw --model models/vosk-model-small-en-us-0.15
```

When it finishes you get a fine-tuned model at `output/vosk-model-finetuned`.
Test it:

```bash
python scripts/test_model.py --model output/vosk-model-finetuned --audio some-speech.wav
```

---

## Options you'll actually use

| Flag | Default | What it does |
|------|---------|--------------|
| `--audio DIR` | `data/raw` | Your audio folder (any format, any nesting) |
| `--model DIR` | `models/` | The seed VOSK model folder to fine-tune |
| `--whisper-model` | `large-v3` | Whisper model for transcription. `large-v3` is most accurate for accents; use `medium`/`small`/`base`/`tiny` for speed |
| `--whisper-device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `--language` | *(auto-detect)* | Force a transcript language, e.g. `--language en` |
| `--no-whisper` | off | Skip Whisper; use transcript `.txt` files you already wrote beside each audio |
| `--lang` | `en` | espeak language code used for pronunciation (lexicon) generation, e.g. `de`, `hi`, `it` |
| `--nj N` | `4` | Kaldi parallel jobs |
| `--gpu` | `auto` | `auto` \| `yes` \| `no` \| `wait` |
| `--work DIR` | `work` | Working dir for intermediate files |
| `--out DIR` | `output` | Where the packaged model goes |
| `--name NAME` | `vosk-model-finetuned` | Output model folder name |

Run the pipeline step-by-step if you prefer:

```bash
python run.py prepare --audio data/raw --model models/<your-model>   # normalize+transcribe+dataset+lexicon
python run.py train                                                  # Kaldi fine-tune
python run.py package                                                # package the VOSK model
```

---

## "Any audio, in any format" — what that really means

* **Formats**: `wav`, `mp3`, `m4a`, `aac`, `ogg`, `opus`, `flac`, `wma`, `aiff`, `amr`, `webm`, `mp4`, `mkv`, `mov`, `3gp` … anything `ffmpeg` can read. If `ffmpeg` can play it, the pipeline can train on it.
* **Folder structure**: none required. Files can sit directly in `data/raw`, nested in subfolders, wherever. Speaker identity is **derived automatically** from the folder each file lives in.
* **Sample rate / channels / bit depth**: any. Everything is converted to Kaldi's 16 kHz mono.
* **No transcripts needed**: Whisper large-v3 writes them for you. If you *already have* transcripts, naming them `name.txt` next to the audio (or running with `--no-whisper`) skips Whisper entirely.
* **Accents**: Whisper large-v3 is specifically chosen for the best coverage of non-native and accented speech (Indian English, German English, etc.).

**The only hard requirement** is that your audio actually contains speech matching
what you want the model to recognize — that's the data you're teaching it.

---

## Environment / requirements

| Component | Notes |
|-----------|-------|
| **OS** | Linux recommended (macOS/Windows via WSL work with tweaks) |
| **ffmpeg** | Audio normalization. `sudo apt install ffmpeg` |
| **espeak-ng** | Auto pronunciation/lexicon. `sudo apt install espeak-ng` |
| **Python 3.9+** | `pip install -r requirements.txt` |
| **Kaldi** | The actual trainer. Build it (see below) or use Docker |
| **GPU** | *Optional but recommended.* The TDNN chain recipe runs on CPU too, just slower. 6 GB VRAM is enough |
| **Disk** | ~15 GB free for Kaldi + Whisper + models |

### Getting a working Kaldi (one-time, ~20–60 min)

You have two options:

**Option A — Docker (recommended, fully reproducible):**
```bash
docker build -t finetunevosk .
docker run --gpus all -it --rm \
    -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/output:/app/output" \
    finetunevosk all --audio /app/data/raw --model /app/models/<your-model>
```

**Option B — build on your host:**
```bash
bash scripts/setup_kaldi.sh            # clones + builds Kaldi, pulls a seed model
# re-run training as before
```

> Kaldi is built from source because VOSK's models are Kaldi chain models and the
> official VOSK project does not ship fine-tuning scripts. This recipe (mono →
> triphone → LDA+MLLT → SAT → chain TDNN) is the established open-source method for
> producing a new VOSK-compatible model and is widely used in the community.

---

## Project layout

```
FineTuneVOSK/
├── run.py                 # CLI — the whole pipeline in one file
├── requirements.txt       # Python deps (whisper + core)
├── Dockerfile             # reproducible Kaldi+Python training image
├── pipeline/              # Python automation
│   ├── config.py          #   defaults (paths, whisper, kaldi)
│   ├── audio.py           #   any-format → 16k/mono wav
│   ├── transcribe.py      #   Whisper transcription / transcript reuse
│   ├── kaldi_prep.py      #   data/train + test + corpus.txt
│   ├── lexicon.py         #   espeak pronunciation → lexicon.txt
│   ├── train.py           #   Kaldi orchestration (align + chain)
│   └── package.py         #   assemble VOSK model + conf/model.conf
├── scripts/
│   ├── setup_kaldi.sh     #   clone+build Kaldi, download seed model
│   ├── align_train.sh     #   mono→tri→SAT alignments
│   ├── lm_creation.sh     #   SRILM language model / G.fst
│   ├── copy_final_result.sh # package model files
│   └── test_model.py      #   transcribe with the fine-tuned model
├── models/                # ← put your seed VOSK model here
└── data/raw/              # ← put your audio here (any format)
```

---

## FAQ / Troubleshooting

* **"Kaldi recipe not found"** — you haven't run `setup_kaldi.sh`/Docker yet. The
  trainer needs `kaldi/egs/mini_librispeech/s5` present.
* **"SRILM toolkit is not installed"** — the language-model step needs SRILM. Install
  it under `kaldi/tools` (see `tools/install_srilm.sh`), or run
  `lm_creation.sh` once SRILM is present.
* **Whisper is slow** — use a smaller `--whisper-model`, or `--whisper-device cpu`
  with `--compute-type int8` if you have no GPU.
* **Transcription looks wrong** — Whisper `large-v3` is the default for a reason;
  smaller models transcribe worse. The transcripts are one file per audio under
  `work/transcripts/` — you can edit any of them and it will be picked up on re-run.
* **The model only knows the words in my audio** — correct. This recipe builds a
  model *from your data*. The more diverse and longer your audio, the better the
  coverage. Add speech covering words/phrases you care about.
* **Still stuck?** Open an issue with the `work/` folder contents and the error.

---

## Contributing

Issues and PRs welcome. The Python pipeline (`pipeline/`) is dependency-light and
pure-stdlib by design (only `faster-whisper` at runtime) so it's easy to read and
extend.

## License

MIT — see [LICENSE](LICENSE). The Kaldi shell scripts are adapted from
[matteo-39/vosk-build-model](https://github.com/matteo-39/vosk-build-model) (MIT).