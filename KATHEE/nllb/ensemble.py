# nllb/ensemble_nllb.py
# Cross-scoring ensemble of the 1.3B and 3.3B combined-dataset LoRA adapters.
# Both share NLLB-200's SentencePiece vocab, so an n-best list from either
# model can be force-scored under the other without retokenization.
#   1. generate K candidates per source from each model
#   2. pool + dedupe
#   3. length-normalized logprob of every candidate under both models
#   4. pick argmax of w*lp_small + (1-w)*lp_big
# Also reports each model alone, so one job answers "is 3.3B any good" and
# "does the ensemble beat either" together.
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ground_truth_eval import load_ground_truth
from metric import kathe_score
import torch.distributed.tensor  # noqa: F401

SRC_LANG, TGT_LANG = "eng_Latn", "kas_Arab"
CKPT = Path(__file__).resolve().parent / "ckpt"
OUT_DIR = Path(__file__).resolve().parent / "eval_out"
MODELS = [
    ("1.3B", "facebook/nllb-200-1.3B", CKPT / "best_adapter_nllb_combined_1.3B"),
    ("3.3B", "facebook/nllb-200-3.3B", CKPT / "best_adapter_nllb_combined_3.3B_10_epochs_best_one"),
]
K = 8               # candidates kept per model
GEN_BS = 4          # generation batch (beam*K is memory-hungry on 3.3B)
SCORE_BS = 16


def load(base_name, adapter):
    m = AutoModelForSeq2SeqLM.from_pretrained(base_name, torch_dtype=torch.float16)
    m = PeftModel.from_pretrained(m, str(adapter)).to("cuda").eval()
    return m


@torch.no_grad()
def gen_nbest(sents, tok, model):
    tgt_id = tok.convert_tokens_to_ids(TGT_LANG)
    out = []
    for i in range(0, len(sents), GEN_BS):
        enc = tok(sents[i:i + GEN_BS], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to("cuda")
        g = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=K,
                           num_return_sequences=K, max_new_tokens=256,
                           no_repeat_ngram_size=4)
        dec = tok.batch_decode(g, skip_special_tokens=True)
        out += [dec[j:j + K] for j in range(0, len(dec), K)]
        print(f"gen {min(i + GEN_BS, len(sents))}/{len(sents)}", flush=True)
    return out


@torch.no_grad()
def score(pairs, tok, model):
    """pairs: list of (src, cand). Returns list of sum-logprob and token count."""
    lps, lens = [], []
    for i in range(0, len(pairs), SCORE_BS):
        chunk = pairs[i:i + SCORE_BS]
        b = tok([s for s, _ in chunk], text_target=[c for _, c in chunk],
                return_tensors="pt", padding=True, truncation=True,
                max_length=256).to("cuda")
        labels = b.pop("labels")
        logits = model(**b, labels=labels).logits.float()
        lp = torch.log_softmax(logits, dim=-1)
        mask = labels.ne(tok.pad_token_id)
        safe = labels.masked_fill(~mask, 0)
        tok_lp = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        lps += (tok_lp * mask).sum(-1).tolist()
        lens += mask.sum(-1).tolist()
        print(f"score {min(i + SCORE_BS, len(pairs))}/{len(pairs)}", flush=True)
    return lps, lens


def main():
    en, refs = load_ground_truth()
    tok = AutoTokenizer.from_pretrained(MODELS[0][1], src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    # --- phase 1: n-best from each model, plus each model's own top-1 ---
    nbest, solo = {}, {}
    for tag, base, adapter in MODELS:
        if not adapter.exists():
            sys.exit(f"adapter not found: {adapter}")
        print(f"\n### generating with {tag}", flush=True)
        m = load(base, adapter)
        nb = gen_nbest(en, tok, m)
        nbest[tag] = nb
        solo[tag] = [c[0] for c in nb]          # beam-1 output = the solo system
        del m; torch.cuda.empty_cache()

    pooled = [list(dict.fromkeys(itertools.chain(*(nbest[t][i] for t, _, _ in MODELS))))
              for i in range(len(en))]
    print(f"\npooled candidates: mean {sum(map(len, pooled)) / len(pooled):.1f} per sentence")

    # --- phase 2: score every pooled candidate under both models ---
    flat = [(en[i], c) for i in range(len(en)) for c in pooled[i]]
    owner = [i for i in range(len(en)) for _ in pooled[i]]
    all_lp = {}
    for tag, base, adapter in MODELS:
        print(f"\n### scoring under {tag}", flush=True)
        m = load(base, adapter)
        lp, ln = score(flat, tok, m)
        all_lp[tag] = (lp, ln)
        del m; torch.cuda.empty_cache()

    # --- phase 3: sweep weight and length-norm exponent ---
    results = {}
    for tag, _, _ in MODELS:
        s = kathe_score(solo[tag], refs)
        results[f"solo_{tag}"] = s
        print(f"\n=== {tag} alone ===\n{s}")

    best = None
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        for a in [0.6, 0.8, 1.0]:
            picks = [None] * len(en)
            bestval = [float("-inf")] * len(en)
            for j, i in enumerate(owner):
                lp_s = all_lp["1.3B"][0][j] / (all_lp["1.3B"][1][j] ** a)
                lp_b = all_lp["3.3B"][0][j] / (all_lp["3.3B"][1][j] ** a)
                v = w * lp_s + (1 - w) * lp_b
                if v > bestval[i]:
                    bestval[i], picks[i] = v, flat[j][1]
            sc = kathe_score(picks, refs)
            print(f"w={w:.1f} lennorm={a:.1f} -> geo_mean={sc['geo_mean']:.2f} "
                  f"bleu={sc['bleu']:.2f} chrf++={sc['chrf++']:.2f}", flush=True)
            if best is None or sc["geo_mean"] > best[0]["geo_mean"]:
                best = (sc, w, a, picks)

    sc, w, a, picks = best
    print(f"\n=== best ensemble: w={w} lennorm={a} ===\n{sc}")
    results["ensemble_best"] = {"score": sc, "w": w, "lennorm": a}

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "ensemble_scores_16aug.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    with open(OUT_DIR / "ensemble_hyps_16aug.json", "w", encoding="utf-8") as f:
        json.dump([{"en": e, "hyp": h, "ref": r} for e, h, r in zip(en, picks, refs)],
                  f, ensure_ascii=False, indent=2)
    print("\nwrote", OUT_DIR / "ensemble_scores_16aug.json")


if __name__ == "__main__":
    main()
