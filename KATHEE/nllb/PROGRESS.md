# NLLB-200 — Progress Log

Side-branch of the KATHE 2026 En→Kashmiri project (see `../PROGRESS.md` for
the main IndicTrans2 pipeline). Tests whether a genuinely different
pretraining distribution (NLLB-200, vs. IndicTrans2's Indic-specific
pretraining) can compete with or beat the current champion (IndicTrans2
LoRA on `newdata_clean`, real submission score 14.23). Kept in its own
folder/checkpoint tree (`nllb/ckpt/`, `nllb/data/`), isolated from the main
pipeline's files, per explicit user request ("create separate folder for
that"). Written 2026-08-10, last updated 2026-08-11 (Run 2 finished
training, evaluated, decisively beat the champion, submission file
generated).

---

## 1. Zero-shot baseline (pre-existing, `../nllb_zeroshot.py`)

From earlier in the project, not re-run this session. Tested
`facebook/nllb-200-distilled-600M` and `facebook/nllb-200-1.3B` with no
fine-tuning: `tok(batch, ...)`, `model.generate(forced_bos_token_id=
tok.convert_tokens_to_ids("kas_Arab"), num_beams=5, max_new_tokens=256,
no_repeat_ngram_size=4)`. **Failed badly** — far below the IndicTrans2
pipeline, with hallucination and English/Hindi-conflation failure modes.
Expected for any untuned model on a low-resource language; doesn't
establish NLLB's real ceiling on its own. Not adopted, but motivated
testing LoRA fine-tuning before writing NLLB off entirely.

## 2. Decision: LoRA, not full fine-tuning

User had already ruled out full fine-tuning for the main IndicTrans2
pipeline (see `../PROGRESS.md` §14 area). When asked whether trying NLLB
meant an exception to that rule, explicit answer: **"lets use lora with it
for fine tuning not fully fine tuning it."** So NLLB is fine-tuned via LoRA
only, same as the main project's standard recipe, never full parameter
fine-tuning.

Before committing, verified NLLB actually has the same attention module
names IndicTrans2 uses for LoRA targeting (not assumed):

```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
# named_modules() ending in q_proj/k_proj/v_proj/out_proj:
# {'q_proj', 'out_proj', 'v_proj', 'k_proj'}  <- confirmed present
# total params: 615,073,792
```

## 3. Run 1 — LoRA on `newdata_clean` (matching the champion's exact data)

`nllb/finetune_nllb.py` (no flag). Goal: apples-to-apples data comparison
against the current champion.

**Config:**
- `MODEL_NAME = "facebook/nllb-200-distilled-600M"`, `SRC_LANG="eng_Latn"`, `TGT_LANG="kas_Arab"`
- `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","out_proj"], task_type="SEQ_2_SEQ_LM")`
- Preprocessing: `tok(en, text_target=ks, truncation=True, max_length=256)` — no `IndicProcessor` needed (that's IndicTrans2-specific numeral/currency normalization; plain NLLB tokenizer with `src_lang`/`tgt_lang` set handles it)
- Data: `TRAIN_SPLIT_NEWDATA_CLEAN` / `VAL_NEWDATA_CLEAN` (3,239 train rows after 200-row val holdout, 3,439 total)
- `per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=2` (effective batch 32)
- `learning_rate=2e-4, num_train_epochs=15, warmup_ratio=0.03, weight_decay=0.01, fp16=True`
- `eval_steps=100, save_steps=100, save_total_limit=3, load_best_model_at_end=True, predict_with_generate=False`
- `report_to="wandb"`, run name `lora-nllb-newdata-clean-nllb-200-distilled-600M`

**Result:** 4,718,592 / 619,792,384 trainable params (0.7613%). 1,515 steps,
trained in **422 seconds (~7 min)** — 115.1 samples/sec, 3.589 steps/sec
(3.78 it/s), much faster than IndicTrans2-200M's full fine-tune throughput.
Final `train_loss=1.9812`, final `eval_loss=1.8418` at epoch 14.78. wandb
run: `https://wandb.ai/as-p/kathe2026-en-kas/runs/8d3oeqrq`. Saved to
`nllb/ckpt/best_adapter_nllb_newdata_clean`.

Sample outputs were fluent, on-topic Kashmiri — **LoRA fine-tuning fixed
the zero-shot hallucination/conflation problem.** E.g. "She is five feet
tall." → `سۄ چھےٚ پانٛژ فٹ بلند۔` (ref: `سۄ چھےٚ پانٛژ ھ فُٹہٕ تٔھز۔`).

### Evaluation (`nllb/eval_nllb.py`, Ground_Truth-100)

| config | bleu | chrf++ | geo_mean | length_ratio |
|---|---|---|---|---|
| raw (no restoration) | 3.3067 | 29.4202 | 9.8633 | 0.9986 |
| + hybrid restoration (τ=0.2) | 3.6077 | 25.8593 | 9.6588 | 0.9986 |
| IndicTrans2 champion (newdata_clean) | 5.6788 | 31.0827 | 13.2858 | 1.0152 |

Output density 0.1783 vs. gold density 0.1653 — NLLB's raw output is
already **above** gold density, the opposite of IndicTrans2's usual
under-diacritization pattern. That's why hybrid restoration (calibrated
for under-marking) made NLLB's chrF++ worse (29.42→25.86) even though BLEU
ticked up slightly (3.31→3.61) — a real methodological finding, not noise:
the restoration pipeline is IndicTrans2-specific, doesn't transfer.

Paired bootstrap (restored vs. champion): `mean_delta=-3.688,
CI=[-6.778,-0.947], P(nllb-lora>champion)=0.001` — decisive loss.
**Not adopted.** IndicTrans2's Indic-specific pretraining head start wins
even after LoRA adaptation to identical data.

## 4. Fairness pushback and eval script changes

User flagged two issues with Run 1's framing:
1. Restoration should be dropped for NLLB's reported score, since it
   measurably hurts it (see density finding above) — comparing NLLB's
   *raw* output to the champion's *restored* output is the fairer read of
   NLLB's real ceiling.
2. The IndicTrans2 side of the comparison wasn't representative of the
   whole IndicTrans2 effort — the 12.93-real-score BPCC-only champion
   (`best_adapter`) exists too, and NLLB had only been tested against the
   smaller `newdata_clean` branch.

`eval_nllb.py` updated to print **both** bootstrap comparisons (raw vs.
champion, and restored vs. champion) rather than only the restored one, and
to accept `--bpcc` / `--combined` flags selecting which NLLB adapter and
which IndicTrans2 comparison point to use:
- (no flag): `best_adapter_nllb_newdata_clean` vs. `best_adapter_newdata_clean`
- `--bpcc`: `best_adapter_nllb_bpcc` vs. `best_adapter`
- `--combined`: `best_adapter_nllb_combined` vs. `best_adapter_newdata_clean` (best overall IndicTrans2 result)

`finetune_nllb.py` updated with a `--bpcc` branch mirroring the IndicTrans2
v1 champion's exact recipe (`TRAIN_SPLIT`/`VAL_FILE`, full 87,899-row BPCC,
3 epochs, `eval_steps=500`, `adapter_name="best_adapter_nllb_bpcc"`). This
run was about to be launched (GPU/process check done) when the user said
"wait" and redirected to a different, larger combined-dataset approach
instead (§5) — the `--bpcc`-only run was never actually executed.

## 5. Run 2 — LoRA on combined BPCC + bignew + filtered dataset

User's explicit instruction: **"use @data/bignew_eng_kmr.csv +bpcc+
cleaned_new_eng_kmr_filtetred datasets for training."** Give NLLB every
dataset in play at once, not just one source at a time, before concluding.

### Data prep (`nllb/prepare_combined.py`)

- **BPCC (repaired)**: loads `TRAIN_SPLIT` (87,899 rows), drops rows where
  `is_wrong_script()` (majority-Devanagari transliteration instead of
  proper kas_Arab script) is true, applies `collapse_repeated_marks()` to
  fix doubled-diacritic-mark noise — both functions imported directly from
  `../clean_bpcc.py` (the BPCC audit done earlier this session: 15
  wrong-script rows dropped project-wide, 337 doubled-mark rows fixed).
  Result: **87,884 rows (61.8% of the mix)**.
- **bignew**: loaded directly from `../data/bignew_eng_kmr.csv` — already
  alignment-fixed (a stray blank line in the raw `English.txt` had silently
  shifted ~59% of rows by one before that fix) and exact-deduped from
  earlier in the session. **26,702 rows (18.8%)**.
- **filtered**: `../data/cleaned_new_eng_kmr_filtered.csv` (`English`/
  `Kashmiri` columns renamed to `en`/`ks`, stripped), the contamination-
  filtered register-matched set that produced the actual 14.23 real
  champion. Oversampled **8x → 27,512 rows (19.4%)** — plain concatenation
  would have diluted this smallest-but-most-valuable set to ~2.9% of the
  mix, the same drowning-out failure already confirmed twice for
  IndicTrans2 LoRA mixing (`../PROGRESS.md`: v3 diluted a register-matched
  set to ~6% and it didn't help). 8x chosen to land roughly at parity with
  bignew's share, not a tuned value.

Concatenated, shuffled with the project `SEED`, split 300 rows off for val.
**Total: 142,098 rows → train=141,798, val=300.** Saved to
`nllb/data/train_combined.parquet` / `val_combined.parquet`.

### Training config

`finetune_nllb.py --combined`, priority-ordered before `--bpcc` in the
flag-checking chain. `num_epochs=6` — raised from the BPCC-scale default of
3 per explicit user request ("also increase the epochs"), reasoned at the
time as still a ~2hr job given the newdata_clean run's 3.78 it/s throughput
(that estimate turned out wrong once batch size had to shrink — see below).
`eval_save_steps=500`. Output adapter name `best_adapter_nllb_combined`.

### Crash 1 — CUDA OOM during eval, bs=16

First launch (`per_device_train_batch_size=16`, same as the working
Run 1) crashed partway through, during an eval step:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB.
GPU 0 has a total capacity of 11.57 GiB of which 1.17 GiB is free.
Including non-PyTorch memory, this process has 9.87 GiB memory in use.
Of the allocated memory 8.13 GiB is allocated by PyTorch, and 1.50 GiB
is reserved by PyTorch but unallocated.
```

Raised inside `accelerate/utils/operations.py`'s `_convert_to_fp32`, during
eval loss computation. wandb run:
`https://wandb.ai/as-p/kathe2026-en-kas/runs/bkilw3vq`.

Root cause: NLLB-600M is 3x IndicTrans2-200M's params, and this branch
includes BPCC's much longer sentences (avg 19 words/sentence vs.
newdata_clean's 7.4) padded to `max_length=256` — activation memory at
bs=16 that was fine for Run 1's shorter/smaller data wasn't fine here.

**Fix attempt 1:** added `model.gradient_checkpointing_enable()` +
`model.enable_input_require_grads()` for the `USE_BPCC or USE_COMBINED`
case, placed **after** `get_peft_model()` wrapping. Reduced
`per_device_train/eval_batch_size` 16→8, `gradient_accumulation_steps`
2→4 (effective batch held at 32).

### Crash 2 — backprop broke entirely, checkpointing after LoRA wrap

Relaunch crashed immediately, on the very first training step (0 steps
completed):

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

at `self.accelerator.backward(loss, **kwargs)` inside `training_step`.
wandb run: `https://wandb.ai/as-p/kathe2026-en-kas/runs/28es7i5i`.

**Fix attempt 2:** reordered — moved `gradient_checkpointing_enable()` +
`enable_input_require_grads()` to run on the **base** model, **before**
`get_peft_model()` wraps it, matching the exact ordering the main
project's `--1b` branch uses in `../finetune.py` (comment there: "required
for grad checkpointing + frozen base + LoRA").

### Crash 3 — same error again, despite matching the proven `--1b` ordering

Relaunch crashed **identically** — same `RuntimeError: element 0 of
tensors does not require grad and does not have a grad_fn`, same point in
`training_step`, zero steps completed. wandb run:
`https://wandb.ai/as-p/kathe2026-en-kas/runs/4lbey4ox`.

This ruled out ordering as the cause. Working theory: `enable_input_
require_grads()`'s forward hook only forces `requires_grad=True` on the
*encoder*-side input embeddings' output. On an encoder-decoder model, the
decoder stack has its own separately-checkpointed segments fed by the
*decoder*'s input embeddings — if those aren't covered by the same hook
(e.g. because NLLB/M2M100's decoder embedding path isn't the exact tensor
`get_input_embeddings()` returns, or checkpointing is applied per-layer to
both stacks independently), the decoder-side checkpointed segments have no
tensor requiring grad flowing in, so `backward()` fails there regardless of
what the encoder side does. Not confirmed via deeper inspection (would need
to trace NLLB's `_set_gradient_checkpointing` internals) — the pragmatic
fix below sidesteps the question entirely rather than resolving it.

**Fix attempt 3 (final, worked):** removed gradient checkpointing
entirely. Reasoning: LoRA only trains 4,718,592 params (0.76%) — the
memory pressure here is activation memory from batch size and long
sequences, not optimizer-state memory from a large trainable-parameter
count, so checkpointing was never actually load-bearing the way it is for
the `--1b`/full-FT IndicTrans2 branches. Reduced `per_device_train/eval_
batch_size` further, 8→4, `gradient_accumulation_steps` 4→8 (effective
batch still 32), no checkpointing at all.

### Training completed — 6 epochs, ~6h10m wall-clock

Relaunch succeeded and ran to completion with no further crashes.
`26,586` total optimizer steps, final `train_loss=1.614` at epoch 6.0.
Saved to `nllb/ckpt/best_adapter_nllb_combined`. wandb run:
`https://wandb.ai/as-p/kathe2026-en-kas/runs/qddkwbht`.

### Interim checkpoint spot-checks during training

GPU was fully occupied by training, so interim checks were run on CPU
(inference only, ~10min per checkpoint including the paired-bootstrap
resampling) against intermediate `checkpoint-N` saves, using raw output
scored against a **matched-methodology weakened champion** (lexicon-only
restoration, geo_mean 12.75, not the champion's true best of 13.29 which
needs the GPU-based neural diacritizer that training had locked up):

| checkpoint | epoch | raw geo_mean | density | bootstrap vs. weakened champion (12.75) |
|---|---|---|---|---|
| 7,000 | 1.58 | 14.26 | 0.164 | P=0.833 (early CPU-only run, incomplete methodology) |
| 11,500 | 2.60 | 19.20 | 0.170 | P=0.995 |
| 13,000 | 2.95 | 18.28 | 0.165 | P=0.991 |
| 17,000 | 3.83 | 18.14 | 0.166 | P=0.991 |
| 19,000 | 4.31 | **20.32** (peak so far) | 0.172 | P=0.998 |
| 19,500 | 4.42 | 19.60 | 0.172 | P=0.997 |
| 21,000 | 4.74 | 19.27 | 0.171 | P=0.994 |

Pattern: fast initial climb (epoch 1.6→2.6), then a stable high plateau
(~18-20 geo_mean) from epoch ~2.6 onward, not still-rising and not
degrading — a reproducible result across many checkpoints, not a lucky
single spot-check. Density stayed close to gold (0.165) throughout, drifting
mildly upward (~0.17) in later checkpoints but never approaching the kind
of over-diacritization that would suggest instability.

### Final evaluation (GPU free, full pipeline, `nllb/eval_nllb.py --combined`)

| config | bleu | chrf++ | geo_mean | density |
|---|---|---|---|---|
| raw (no restoration) | 11.75 | 40.17 | **21.72** | 0.1685 (gold: 0.1653) |
| + hybrid restoration | 5.98 | 34.40 | 14.34 | — |
| IndicTrans2 champion (true best, `best_adapter_newdata_clean`) | 5.68 | 31.08 | 13.29 | — |

Final raw score (21.72) exceeds every interim checkpoint spot-check,
including the epoch-4.31 peak (20.32) — the last ~1.2 epochs kept helping.

Paired bootstrap:
- NLLB **+ restoration** vs. champion: `P=0.741` (restoration handicaps it, as expected — see the repeated raw-vs-restored delta above and Run 1's identical finding)
- **NLLB raw** vs. champion: `P=1.000`, `mean_delta=+8.461`, `CI=[4.407, 12.522]` — every one of 1000 bootstrap resamples favored NLLB. As decisive as this methodology can produce.

**Adopted.** This is now the strongest local result in the whole KATHE 2026
project, by a wide margin over the previous champion (13.29).

### Submission generated

`nllb/make_dev_submission_nllb.py` — mirrors the main project's
`make_dev_submission.py` but NLLB-specific and **deliberately skips
restoration** (confirmed harmful to NLLB in every measurement above).
Translated `englishdev.csv` (1,730 rows) with `best_adapter_nllb_combined`,
raw output, `num_beams=8`, `no_repeat_ngram_size=4`. Wrote
`work/englishdev_submission_nllb_combined.csv` — 1,730 rows, 1,730 unique
IDs, zero empty predictions. Not yet submitted to the real leaderboard as
of this writing.

## 6. Conclusion

Run 1 (LoRA on `newdata_clean` alone, matching the champion's exact data)
lost decisively (P=0.001) despite genuinely fixing the zero-shot
hallucination problem — that result stood as evidence IndicTrans2's
Indic-specific pretraining was a real, hard-to-close gap.

Run 2 (LoRA on the combined BPCC + bignew + 8x-oversampled-filtered
dataset, 3x the data of Run 1) reverses that conclusion entirely: **NLLB
beats the IndicTrans2 champion decisively (P=1.000) once given comparable
data volume and diversity.** The earlier loss wasn't really about
pretraining-distribution superiority — it was about data starvation. With
enough register-diverse data, NLLB's different pretraining distribution
turned out to be an asset, not a liability, for this specific task.

Real-world validation (an actual leaderboard score) is the natural next
step to confirm this holds outside the local proxy, the same way the
IndicTrans2 champion's local scores were validated by real submissions
throughout the main project.
