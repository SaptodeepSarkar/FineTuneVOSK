# FineTuneVOSK convenience targets.
PYTHON ?= python3
RUN := $(PYTHON) run.py

.PHONY: help prepare train package test-model clean

help:
	@echo "Targets:"
	@echo "  prepare      normalize + transcribe + build dataset/lexicon"
	@echo "  train        run Kaldi chain fine-tune on the prepared dataset"
	@echo "  package      package the trained model into a VOSK folder"
	@echo "  test-model   transcribe a wav with the packaged model"
	@echo "  clean        remove work/ and output/"

prepare:
	$(RUN) prepare

train:
	$(RUN) train

package:
	$(RUN) package

test-model:
	$(PYTHON) scripts/test_model.py --model output/vosk-model-finetuned --audio $(AUDIO)

clean:
	rm -rf work output