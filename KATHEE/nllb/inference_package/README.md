# English → Kashmiri Inference

Translates English text to Kashmiri (`kas_Arab`) using two LoRA adapters
fine-tuned on top of NLLB-200. This reproduces the submitted system.

## Setup

```bash
pip install -r requirements.txt
```

The adapters are included in this package under `adapters/` (a symlink to
`../ckpt`, kept in-repo alongside the rest of the training code):

```
inference_package/
├── translate.py
├── requirements.txt
└── adapters/
    ├── best_adapter_nllb_combined_1.3B/
    └── best_adapter_nllb_combined_3.3B/
```

If you copy `translate.py` elsewhere, either bring the `adapters/` folder
along with it or pass `--adapter-dir /path/to/adapters` pointing at a
directory containing both subfolders above. Each adapter folder is a
standard PEFT LoRA checkpoint (`adapter_config.json` +
`adapter_model.safetensors`); the base models (`facebook/nllb-200-1.3B`,
`facebook/nllb-200-3.3B`) are downloaded automatically from Hugging Face
on first run.

## Usage

```bash
# Reproduces the submitted system (ensemble of both adapters)
python translate.py --input dev.csv --output preds.csv

# Cheaper: 1.3B adapter only, single beam-search pass
python translate.py --input dev.csv --output preds.csv --mode single

# Force CPU (slow, but works without a GPU)
python translate.py --input dev.csv --output preds.csv --device cpu
```

**Input**: a CSV with an `ID` column and a text column (default `sentence`,
override with `--text-column`).

**Output**: a CSV with `ID` and `kashmiri_text`.

## Modes

- `ensemble` (default): each model generates an 8-candidate n-best list,
  the two lists are pooled, every candidate is rescored under *both*
  models, and the length-normalised weighted-best candidate is kept. This
  is the system that was submitted.
- `single`: 1.3B adapter only, top beam-search output. Much faster, no
  scoring pass, small quality drop.

## Hardware

Models are loaded one at a time and freed after use, so peak GPU memory is
roughly that of the larger model alone: ~7 GB in fp16 for the 3.3B model,
plus activation overhead. A 16 GB GPU runs the default settings
comfortably; on smaller GPUs lower `--gen-batch-size` and
`--score-batch-size`. On CPU, generation falls back to fp32 and is
considerably slower — expect it mainly to be useful for smoke-testing on
a handful of sentences.

## Other flags

Run `python translate.py --help` for the full list, including
`--max-length` (source/target token cap) and `--id-column` (if your input
uses something other than `ID`).
