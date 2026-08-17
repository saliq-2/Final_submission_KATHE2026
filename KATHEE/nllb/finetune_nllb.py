# nllb/finetune_nllb.py
# LoRA fine-tune of NLLB-200 (a genuinely different pretraining distribution
# than IndicTrans2) on the same register-matched data as our current
# champion (newdata_clean, real score 14.23), for a fair comparison. Earlier
# zero-shot NLLB testing (../nllb_zeroshot.py) failed badly -- far below the
# IndicTrans2 pipeline, with hallucination and English/Hindi-conflation
# failure modes -- but that's expected of any untuned model on a low-
# resource language and doesn't establish NLLB's real ceiling. Kept in its
# own folder/checkpoint tree, isolated from the IndicTrans2 pipeline.
#
# Same LoRA recipe as the main project (r=16, alpha=32, dropout=0.05 on
# q/k/v/out_proj -- verified these module names exist on NLLB too, not
# assumed) at the main project's standard LoRA learning rate (2e-4); full
# fine-tuning is deliberately not used here, per explicit user decision.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import pandas as pd
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model
from config import TRAIN_SPLIT_NEWDATA_CLEAN, VAL_NEWDATA_CLEAN, TRAIN_SPLIT, VAL_FILE, SEED

from nllb_common import get_arg, resolve_model_name, model_tag
import torch
import torch.distributed.tensor  # noqa: F401  — binds submodule for peft's DTensor check
USE_BPCC = "--bpcc" in sys.argv
USE_COMBINED = "--combined" in sys.argv

# --model accepts a short alias or any full HF hub id (see nllb_common.py).
# --epochs overrides whichever branch-based default below would apply.
MODEL_NAME = resolve_model_name()
MODEL_TAG = model_tag(MODEL_NAME)
EPOCHS_OVERRIDE = get_arg("--epochs", None, cast=int)

# Defaults calibrated for the cluster's 40GB GPU, not the local 12GB 4070
# Ti these were originally tuned on (which forced per_device_bs=4 for the
# long-sequence BPCC/combined branches). Rough estimates, not profiled on
# the actual cluster hardware -- override with --batch-size if a run OOMs
# or clearly has headroom to push further. Effective batch size is held at
# TARGET_EFFECTIVE_BATCH via grad_accum unless --batch-size is given.
DEFAULT_BATCH_SIZE_BY_TAG = {
    "distilled-600M": 24,
    "1.3B": 12,
    "distilled-1.3B": 12,
    "3.3B": 6,
}
TARGET_EFFECTIVE_BATCH = 32
PER_DEVICE_BS = get_arg("--batch-size", None, cast=int) or DEFAULT_BATCH_SIZE_BY_TAG.get(MODEL_TAG, 8)
GRAD_ACCUM = max(1, TARGET_EFFECTIVE_BATCH // PER_DEVICE_BS)

SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"
NLLB_DIR = Path(__file__).resolve().parent
CKPT_DIR = NLLB_DIR / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)
COMBINED_TRAIN = NLLB_DIR / "data" / "train_combined.parquet"
COMBINED_VAL = NLLB_DIR / "data" / "val_combined.parquet"

os.environ.setdefault("WANDB_PROJECT", "kathe2026-en-kas")

tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# No gradient checkpointing anywhere in this script, on purpose --
# checkpointing + PEFT on this encoder-decoder model broke backprop
# entirely twice ("element 0 of tensors does not require grad") even with
# enable_input_require_grads() and correct pre-wrap ordering (matching
# --1b's pattern in the main finetune.py), most likely because that hook
# only covers the encoder-side input embeddings while the decoder stack's
# own checkpointed segments have no forced requires_grad path. LoRA's
# memory footprint is tiny (4.7M-ish trainable params depending on model
# size) so checkpointing was never actually load-bearing -- controlling
# activation memory via batch size (PER_DEVICE_BS above) is simpler and
# avoids the whole interaction.

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                  task_type="SEQ_2_SEQ_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()


def preprocess(batch):
    model_inputs = tok(list(batch["en"]), text_target=list(batch["ks"]),
                       truncation=True, max_length=256)
    return model_inputs


def main():
    if USE_COMBINED:
        # BPCC (repaired) + bignew + cleaned_new_eng_kmr_filtered (8x
        # oversampled, ~19% share) -- see prepare_combined.py. 141,798 train
        # rows, BPCC-scale but with the register-matched data given a real
        # (non-drowned-out) share this time. 6 epochs default -- NLLB LoRA
        # trains fast (~3.8 it/s observed on the 600M newdata-clean run),
        # override with --epochs for a bigger/slower model.
        train_file, val_file, out_name = COMBINED_TRAIN, COMBINED_VAL, f"combined_lora_{MODEL_TAG}"
        num_epochs, eval_save_steps = 6, 500
        adapter_name = f"best_adapter_nllb_combined_{MODEL_TAG}"
        run_tag = "combined"
    elif USE_BPCC:
        # mirrors the IndicTrans2 v1 champion's exact data/recipe (full
        # BPCC, 3 epochs) -- the earlier newdata_clean-only NLLB comparison
        # only matched IndicTrans2's smaller-data branch; BPCC is the other
        # half of the IndicTrans2 lineage (real score 12.93) and NLLB hadn't
        # been tested against it with matching data/volume.
        train_file, val_file, out_name = TRAIN_SPLIT, VAL_FILE, f"bpcc_lora_{MODEL_TAG}"
        num_epochs, eval_save_steps = 3, 500
        adapter_name = f"best_adapter_nllb_bpcc_{MODEL_TAG}"
        run_tag = "bpcc"
    else:
        train_file, val_file, out_name = TRAIN_SPLIT_NEWDATA_CLEAN, VAL_NEWDATA_CLEAN, f"newdata_clean_lora_{MODEL_TAG}"
        num_epochs, eval_save_steps = 15, 100
        adapter_name = f"best_adapter_nllb_newdata_clean_{MODEL_TAG}"
        run_tag = "newdata-clean"

    if EPOCHS_OVERRIDE is not None:
        num_epochs = EPOCHS_OVERRIDE

    train = pd.read_parquet(train_file)[["en", "ks"]]
    val = pd.read_parquet(val_file)[["en", "ks"]]

    ds_train = Dataset.from_pandas(train).map(preprocess, batched=True, remove_columns=train.columns.tolist())
    ds_val = Dataset.from_pandas(val).map(preprocess, batched=True, remove_columns=val.columns.tolist())

    collator = DataCollatorForSeq2Seq(tok, model=model)

    print(f"per_device_bs={PER_DEVICE_BS} grad_accum={GRAD_ACCUM} "
          f"(effective batch={PER_DEVICE_BS * GRAD_ACCUM})")

    args = Seq2SeqTrainingArguments(
        output_dir=str(CKPT_DIR / out_name),
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        num_train_epochs=num_epochs,
        warmup_ratio=0.03,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps", eval_steps=eval_save_steps,
        save_strategy="steps", save_steps=eval_save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        predict_with_generate=False,
        report_to="wandb",
        run_name=f"lora-nllb-{run_tag}-{MODEL_NAME.split('/')[-1]}",
    )

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds_train,
                             eval_dataset=ds_val, data_collator=collator, tokenizer=tok)
    trainer.train()
    out_dir = CKPT_DIR / adapter_name
    trainer.model.save_pretrained(str(out_dir))
    print("saved adapter to", out_dir)


if __name__ == "__main__":
    main()
