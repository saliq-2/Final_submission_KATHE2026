# English → Kashmiri MT (NLLB-200 + LoRA)
Note: Clone the repo dont use download as zip

LoRA fine-tuning of NLLB-200 for English → Kashmiri (`kas_Arab`), with a
cross-model reranking ensemble at inference.

## Requirements

Python 3.10, one CUDA GPU. Memory depends on which model you train:

| Model | GPU memory | Notes |
|---|---|---|
| `distilled-600M` | ~16 GB | |
| `1.3B` | ~24 GB | batch size 12 |
| `3.3B` | ~40 GB | batch size 6 |

```bash
conda create -n kashmiri python=3.10 -y
conda activate kashmiri

```

The `torch==2.5.1` / `peft==0.20.0` pin matters. On this pairing PEFT's DTensor
check reads `torch.distributed.tensor`, which torch 2.5 does not bind eagerly,
so `get_peft_model` raises `AttributeError`. The training script imports the
submodule explicitly to work around it. Upgrading torch to ≥2.6 or downgrading
PEFT to <0.15 also resolves it.

## Layout
KATHEE/
```
config.py                    shared paths and constants
prepare_combined.py          builds the combined training mix
metric.py                    scoring (BLEU, chrF++, geometric mean)
ground_truth_eval.py         loads the held-out reference set
nllb/
  nllb_common.py             --model alias resolution, model tags
  finetune_nllb.py           LoRA training
  eval_nllb_nodiac.py        evaluation vs the reference set
  ensemble_nllb.py           tunes the ensemble weights
  make_dev_submission_ensemble.py   writes a submission CSV
  inference_package/          final evaluation
  data/                      train_combined.parquet, val_combined.parquet
  ckpt/                      adapters 
work/                        intermediate parquets and outputs (created)
```

`config.py` derives every path from its own location, so the repo runs from
wherever it is cloned. `DATA_DIR`, `WORK_DIR`, and `CKPT_DIR` are created on
import.

## Datasets

Training data is not distributed with the repo. Three parallel English–Kashmiri
sources are combined.

### BPCC

The bulk of the data. Bharat Parallel Corpus Collection, from the Hugging Face
Hub:

- Dataset: [`ai4bharat/BPCC`](https://huggingface.co/datasets/ai4bharat/BPCC)
- File: `bpcc-seed-latest/kas_Arab.tsv`

```bash
huggingface-cli download ai4bharat/BPCC \
    bpcc-seed-latest/kas_Arab.tsv \
    --repo-type dataset --local-dir data/
```

`clean_bpcc.py` runs a repair pass over it for encoding and alignment problems
and writes `work/bpcc_clean.parquet`.

### bignew

The bignew dataset was sourced from https://huggingface.co/datasets/SMUQamar/Kashmiri-English-Parallel-Corpus/tree/main

Expected at `work/train_split_bignew.parquet` with `en` and `ks` columns.

### Register-matched subset

A smaller, cleaner corpus much closer in register to the target domain.
Oversampled in the combined mix so it is not drowned out by BPCC.

<!-- TODO: source, size, licence, and how it was cleaned and filtered -->

Expected at `work/train_split_newdata_clean.parquet` with `en` and `ks` columns.

### Building the mix

```bash
python clean_bpcc.py          # repair pass, writes work/bpcc_clean.parquet
python prepare_combined.py    # builds nllb/data/train_combined.parquet
```

`prepare_combined.py` concatenates the three sources and oversamples the
register-matched subset; the ratio and oversampling factor are set at the top of
that script.

### Evaluation data

The held-out reference set used by `eval_nllb_nodiac.py` is loaded by
`ground_truth_eval.py`.

<!-- TODO: source and size of the reference set; note whether it is redistributable -->

Other training branches use different splits — `--bpcc` reads
`work/train_split.parquet`, and the default branch reads
`work/train_split_newdata_clean.parquet`. All split paths are defined in
`config.py`.

Verify before training:

```bash
python -c "
import pandas as pd
for f in ('nllb/data/train_combined.parquet', 'nllb/data/val_combined.parquet'):
    d = pd.read_parquet(f)
    print(f, len(d), list(d.columns))
"
```

Both need `en` and `ks` columns.

## Training
cd /KATHEE/nllb/
```bash
python -u nllb/finetune_nllb.py --combined --model 1.3b --epochs 10 --batch-size 12
```

| Flag | Effect |
|---|---|
| `--combined` | the combined mix (default 6 epochs) |
| `--bpcc` | BPCC only (default 3 epochs) |
| *(neither)* | register-matched subset only (default 15 epochs) |
| `--model` | `600m`, `1.3b`, `3.3b`, or any NLLB hub id |
| `--epochs` | overrides the branch default |
| `--batch-size` | overrides the per-model default; grad accumulation adjusts to hold the effective batch near 32 |

Checkpoints land in `nllb/ckpt/combined_lora_<TAG>/`, capped at 3. The final
adapter is written to `nllb/ckpt/best_adapter_nllb_combined_<TAG>/`.

Train both sizes — the ensemble needs both:

```bash
python -u nllb/finetune_nllb.py --combined --model 1.3b --epochs 10 --batch-size 12
python -u nllb/finetune_nllb.py --combined --model 3.3b --epochs 10 --batch-size 6
```

Weights & Biases logging is on by default via `report_to="wandb"`. Run
`wandb login` first, or set `WANDB_MODE=offline` (and `wandb sync` later) if the
training node has no outbound network.


## Inference
Please use this file for running the final evaluation of the model



cd KATHEE/nllb/inference_package/

python -u translate.py \
            --input smoke.csv \ 
            --output smoke_out.csv \ 
            --adapter-dir /KATHEE/nllb/ckpt 

The adapters are at the location KATHEE/nllb/ckpt            

Then you can copy the output file to run against your own evaluation metric









## Notes

Beam search is deterministic, but fp16 reductions are not bit-reproducible
across GPU models, so candidates scoring within rounding error of each other can
swap between runs. This affects a small fraction of outputs.

`diacritizer_gated.py` imports `diacritizer_apply`, which is not in the repo.
Anything touching the diacritic restoration pipeline — including the original
`eval_nllb.py` — will fail on import. Use `eval_nllb_nodiac.py` instead.
