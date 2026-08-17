# English → Kashmiri Inference

Translates English text to Kashmiri (`kas_Arab`) using two LoRA adapters
fine-tuned on top of NLLB-200. This reproduces the submitted system.

> **Get this via `git clone`, not GitHub's "Download ZIP" button.**
> `adapters/` below is a symlink into the repo's `ckpt/` folder, and GitHub's
> zip export does not reliably preserve it as a working directory link.

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

```

If you copy `translate.py` elsewhere, either bring the `adapters/`
           python -u translate.py \
            --input smoke.csv \
            --output smoke_out.csv \
            --adapter-dir /nllb/ckpt/ \
 folder
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


```

**Input**: a CSV with an `ID` column and a text column (default `sentence`,
override with `--text-column`).

**Output**: a CSV with `ID` and `kashmiri_text`.

## Modes

- `ensemble` (default): each model generates an 8-candidate n-best list,
  the two lists are pooled, every candidate is rescored under *both*
  models, and the length-normalised weighted-best candidate is kept. This
  is the system that was submitted.

## Hardware

The enmsebled models need a gpu with a vram of atleast 24gb

## Other flags

Run `python translate.py --help` for the full list, including
`--max-length` (source/target token cap) and `--id-column` (if your input
uses something other than `ID`).
