#!/usr/bin/env python3
"""
English -> Kashmiri (kas_Arab) inference.

Two LoRA adapters over NLLB-200 (1.3B and 3.3B). The default `ensemble` mode
reproduces the submitted system: each model produces an n-best list, the lists
are pooled, every candidate is then scored under *both* models, and the
length-normalised weighted best is selected. `single` mode uses the 1.3B
adapter alone and is much cheaper.

Usage
-----
    python translate.py --input dev.csv --output preds.csv
    python translate.py --input dev.csv --output preds.csv --mode single
    python translate.py --input dev.csv --output preds.csv --device cpu

Input CSV needs an `ID` column and a text column (default `sentence`; override
with --text-column). Output CSV has `ID` and `kashmiri_text`.
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.distributed.tensor
SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"

# Selection weights, tuned on a held-out set. W is the weight on the 1.3B
# model's score; (1 - W) goes to the 3.3B. LENNORM is the exponent used to
# divide the summed log-probability by the candidate's token count, which
# stops the selector from systematically preferring shorter candidates.
W = 0.4
LENNORM = 0.8

# Candidates generated per model. Deliberately 8 rather than something larger:
# wider pools scored better on our small tuning set but worse on held-out data,
# because the argmax over a noisy scoring function degrades as the pool grows.
NUM_BEAMS = 8

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_DIR = SCRIPT_DIR / "adapters"

MODELS = [
    ("1.3B", "facebook/nllb-200-1.3B", "best_adapter_nllb_combined_1.3B"),
    ("3.3B", "facebook/nllb-200-3.3B", "best_adapter_nllb_combined_3.3B"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="English -> Kashmiri translation with NLLB-200 LoRA adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Input CSV with an ID column and a text column.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output CSV path (ID, kashmiri_text).")
    p.add_argument("--mode", choices=["ensemble", "single"], default="ensemble",
                   help="'ensemble' reproduces the submitted system; "
                        "'single' uses the 1.3B adapter alone.")
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR,
                   help="Directory holding the adapter subfolders.")
    p.add_argument("--text-column", default="sentence",
                   help="Name of the source-text column in the input CSV.")
    p.add_argument("--id-column", default="ID",
                   help="Name of the ID column in the input CSV.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Compute device.")
    p.add_argument("--gen-batch-size", type=int, default=4,
                   help="Batch size for generation. Lower this if you hit OOM.")
    p.add_argument("--score-batch-size", type=int, default=16,
                   help="Batch size for candidate scoring (ensemble mode only).")
    p.add_argument("--max-length", type=int, default=256,
                   help="Max source and target length in tokens.")
    return p.parse_args()


def resolve_device(choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but no CUDA device is visible.")
    return choice


def load_model(base_name, adapter_path, device):
    """Load a base NLLB model and apply the LoRA adapter.

    fp16 is used on GPU for speed; CPU stays in fp32 because half precision
    is not reliably supported for these ops on CPU.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  loading {base_name} ...", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_name, torch_dtype=dtype)
    print(f"  applying adapter {adapter_path} ...", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    return model.to(device).eval()


@torch.no_grad()
def generate_nbest(sentences, tok, model, device, args, n_best):
    """Return one list of up to `n_best` candidates per input sentence.

    Index 0 of each list is the top beam, i.e. what plain beam search would
    have returned, so callers can use it directly as a single-model output.
    """
    tgt_id = tok.convert_tokens_to_ids(TGT_LANG)
    out = []
    total = len(sentences)
    for i in range(0, total, args.gen_batch_size):
        batch = sentences[i:i + args.gen_batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_length).to(device)
        gen = model.generate(
            **enc,
            forced_bos_token_id=tgt_id,
            num_beams=NUM_BEAMS,
            num_return_sequences=n_best,
            max_new_tokens=args.max_length,
            no_repeat_ngram_size=4,
        )
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch)):
            out.append(decoded[j * n_best:(j + 1) * n_best])
        print(f"  generated {min(i + args.gen_batch_size, total)}/{total}", flush=True)
    return out


@torch.no_grad()
def score_candidates(pairs, tok, model, device, args):
    """Force-decode each (source, candidate) pair and return the summed
    log-probability of the candidate's tokens plus its length in tokens.

    Both models share NLLB-200's SentencePiece vocabulary, so a candidate
    produced by one model can be scored under the other without any
    retokenisation.
    """
    logprobs, lengths = [], []
    total = len(pairs)
    for i in range(0, total, args.score_batch_size):
        chunk = pairs[i:i + args.score_batch_size]
        batch = tok([s for s, _ in chunk],
                    text_target=[c for _, c in chunk],
                    return_tensors="pt", padding=True,
                    truncation=True, max_length=args.max_length).to(device)
        labels = batch.pop("labels")
        logits = model(**batch, labels=labels).logits.float()
        token_logprobs = torch.log_softmax(logits, dim=-1)
        mask = labels.ne(tok.pad_token_id)
        gathered = token_logprobs.gather(
            -1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
        logprobs += (gathered * mask).sum(-1).tolist()
        lengths += mask.sum(-1).clamp(min=1).tolist()
        if (i // args.score_batch_size) % 20 == 0:
            print(f"  scored {min(i + args.score_batch_size, total)}/{total}", flush=True)
    return logprobs, lengths


def run_single(sentences, args, device):
    tag, base_name, adapter_name = MODELS[0]
    adapter = args.adapter_dir / adapter_name
    if not adapter.is_dir():
        sys.exit(f"adapter not found: {adapter}")
    print(f"[{tag}] single-model mode", flush=True)
    tok = AutoTokenizer.from_pretrained(base_name, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = load_model(base_name, adapter, device)
    nbest = generate_nbest(sentences, tok, model, device, args, n_best=1)
    return [c[0] for c in nbest]


def run_ensemble(sentences, args, device):
    for _, _, adapter_name in MODELS:
        adapter = args.adapter_dir / adapter_name
        if not adapter.is_dir():
            sys.exit(f"adapter not found: {adapter}")

    # The tokenizer is shared across NLLB-200 sizes, so one instance serves both.
    tok = AutoTokenizer.from_pretrained(MODELS[0][1], src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    # Phase 1: n-best candidates from each model. Models are loaded and freed
    # one at a time so peak memory is that of the larger model alone.
    nbest = {}
    for tag, base_name, adapter_name in MODELS:
        print(f"\n[{tag}] generating candidates", flush=True)
        model = load_model(base_name, args.adapter_dir / adapter_name, device)
        nbest[tag] = generate_nbest(sentences, tok, model, device, args, n_best=NUM_BEAMS)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    pooled = [
        list(dict.fromkeys(
            c for c in itertools.chain(*(nbest[t][i] for t, _, _ in MODELS)) if c.strip()
        ))
        for i in range(len(sentences))
    ]
    mean_pool = sum(map(len, pooled)) / max(len(pooled), 1)
    print(f"\npooled {mean_pool:.1f} distinct candidates per sentence on average", flush=True)

    # Phase 2: score every pooled candidate under both models.
    flat = [(sentences[i], c) for i in range(len(sentences)) for c in pooled[i]]
    owner = [i for i in range(len(sentences)) for _ in pooled[i]]
    print(f"scoring {len(flat)} candidates under each model", flush=True)

    scores = {}
    for tag, base_name, adapter_name in MODELS:
        print(f"\n[{tag}] scoring", flush=True)
        model = load_model(base_name, args.adapter_dir / adapter_name, device)
        scores[tag] = score_candidates(flat, tok, model, device, args)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Phase 3: pick the highest weighted, length-normalised score per sentence.
    picks = [None] * len(sentences)
    best_value = [float("-inf")] * len(sentences)
    for j, i in enumerate(owner):
        small = scores["1.3B"][0][j] / (scores["1.3B"][1][j] ** LENNORM)
        large = scores["3.3B"][0][j] / (scores["3.3B"][1][j] ** LENNORM)
        value = W * small + (1 - W) * large
        if value > best_value[i]:
            best_value[i], picks[i] = value, flat[j][1]

    # Any sentence with no usable candidate falls back to the 1.3B top beam.
    fallbacks = 0
    for i, p in enumerate(picks):
        if p is None or not p.strip():
            picks[i] = nbest["1.3B"][i][0]
            fallbacks += 1
    if fallbacks:
        print(f"fell back to the 1.3B top beam for {fallbacks} sentences", flush=True)
    return picks


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device: {device} | mode: {args.mode}", flush=True)
    if device == "cpu":
        print("warning: CPU inference is very slow for these model sizes.", flush=True)

    if not args.input.is_file():
        sys.exit(f"input file not found: {args.input}")
    df = pd.read_csv(args.input)
    for col in (args.id_column, args.text_column):
        if col not in df.columns:
            sys.exit(f"column {col!r} not in input; found: {list(df.columns)}")

    sentences = df[args.text_column].astype(str).tolist()
    print(f"{len(sentences)} sentences to translate", flush=True)

    if args.mode == "single":
        preds = run_single(sentences, args, device)
    else:
        preds = run_ensemble(sentences, args, device)

    # A blank prediction scores zero on both BLEU and chrF++, so emit a
    # placeholder rather than an empty field.
    preds = [p if p and p.strip() else "؟" for p in preds]
    assert len(preds) == len(df), f"{len(preds)} predictions for {len(df)} rows"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": df[args.id_column].tolist(),
                  "kashmiri_text": preds}).to_csv(args.output, index=False)
    print(f"\nwrote {args.output} ({len(preds)} rows)", flush=True)


if __name__ == "__main__":
    main()
