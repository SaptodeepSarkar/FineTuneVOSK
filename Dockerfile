# FineTuneVOSK — reproducible training image.
# Builds Kaldi + the mini_librispeech chain recipe + the Python automation
# (Whisper transcription), then runs the pipeline. Everything is pinned by the
# apt/base images so builds are repeatable.

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    KALDI_ROOT=/opt/kaldi \
    PATH="/opt/kaldi/src/bin:/opt/kaldi/tools/openfst/bin:/opt/kaldi/src/fstbin:/opt/kaldi/src/gmmbin:/opt/kaldi/src/featbin:/opt/kaldi/src/ivectorbin:/opt/kaldi/src/nnetbin:/opt/kaldi/src/nnet2bin:/opt/kaldi/src/nnet3bin:/opt/kaldi/src/online2bin:/opt/kaldi/src/chainbin:/opt/kaldi/src/decoderbin:/opt/kaldi/src/latbin:/opt/kaldi/src/sgmm2bin:/opt/kaldi/src/fgmmbin:/opt/kaldi/src/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential automake autoconf libtool subversion \
    zlib1g-dev libbz2-dev liblzma-dev libomp-dev \
    python3 python3-pip python3-venv \
    ffmpeg espeak-ng sox wget unzip curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Kaldi ----
WORKDIR /opt
RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git
WORKDIR /opt/kaldi/tools
RUN make -j"$(nproc)" && ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /opt/kaldi/src
RUN ./configure --shared --use-cuda=yes --cudatoolkit-dir=/usr/local/cuda \
    && make depend -j"$(nproc)" \
    && make -j"$(nproc)"

# ---- Python automation ----
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- App ----
COPY . .
RUN chmod +x scripts/*.sh
ENV PYTHONPATH=/app

WORKDIR /app
ENTRYPOINT ["python3", "run.py"]