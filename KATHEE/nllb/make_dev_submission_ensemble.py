# nllb/make_dev_submission_ensemble.py
# Ensemble submission: pools n-best candidates from the 1.3B and 3.3B
# combined-dataset LoRA adapters, cross-scores every candidate under both,
# and picks argmax of W*lp_1.3B + (1-W)*lp_3.3B with length normalization.
# W/LENNORM are the values tuned on Ground_Truth-100 (geo_mean 26.12 vs
# 24.47 for 1.3B alone). Raw output, no diacritic restoration -- restoration
# is calibrated for IndicTrans2's under-diacritization and hurts NLLB.
# Generation is cached to disk so a failure in the scoring phase doesn't
# cost the (expensive) generation phase.
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import ROOT_DIR, WORK_DIR
import torch.distributed.tensor  # noqa: F401

W = 0.4           # weight on 1.3B; (1-W) on 3.3B
LENNORM = 0.8     # length-normalization exponent
K = 8             # candidates kept per model
GEN_BS = 4
SCORE_BS = 16

SRC_LANG, TGT_LANG = "eng_Latn", "kas_Arab"
NLLB_DIR = Path(__file__).resolve().parent
CKPT = NLLB_DIR / "ckpt"
CACHE = NLLB_DIR / "eval_out" / "dev_nbest_cache.json"
DEV_FILE = ROOT_DIR / "englishdev.csv"
OUT_FILE = WORK_DIR / "englishdev_submission_nllb_ensemble.csv"
MODELS = [
    ("1.3B", "facebook/nllb-200-1.3B", CKPT / "best_adapter_nllb_combined_1.3B"),
    ("3.3B", "facebook/nllb-200-3.3B", CKPT / "best_adapter_nllb_combined_3.3B"),
]


def load(base_name, adapter):
    m = AutoModelForSeq2SeqLM.from_pretrained(base_name, torch_dtype=torch.float16)
    return PeftModel.from_pretrained(m, str(adapter)).to("cuda").eval()


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
        tok_lp = lp.gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
        lps += (tok_lp * mask).sum(-1).tolist()
        lens += mask.sum(-1).clamp(min=1).tolist()
        print(f"score {min(i + SCORE_BS, len(pairs))}/{len(pairs)}", flush=True)
    return lps, lens


def main():
    dev = pd.read_csv(DEV_FILE)
    assert {"ID", "sentence"}.issubset(dev.columns), dev.columns
    en = dev["sentence"].astype(str).tolist()
    print(f"{len(en)} dev sentences", flush=True)

    for tag, _, adapter in MODELS:
        if not adapter.exists():
            sys.exit(f"adapter not found: {adapter}")

    tok = AutoTokenizer.from_pretrained(MODELS[0][1], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    CACHE.parent.mkdir(exist_ok=True)

    # --- phase 1: n-best from each model (cached) ---
    if CACHE.exists():
        nbest = json.load(open(CACHE, encoding="utf-8"))
        assert all(len(nbest[t]) == len(en) for t, _, _ in MODELS), "stale cache; delete it"
        print("loaded cached n-best lists", flush=True)
    else:
        nbest = {}
        for tag, base, adapter in MODELS:
            print(f"\n### generating with {tag}", flush=True)
            m = load(base, adapter)
            nbest[tag] = gen_nbest(en, tok, m)
            del m; torch.cuda.empty_cache()
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(nbest, f, ensure_ascii=False)
        print("cached n-best to", CACHE, flush=True)

    pooled = [list(dict.fromkeys(itertools.chain(*(nbest[t][i] for t, _, _ in MODELS))))
              for i in range(len(en))]
    print(f"pooled: mean {sum(map(len, pooled)) / len(pooled):.1f} candidates/sentence", flush=True)

    # --- phase 2: cross-score ---
    flat = [(en[i], c) for i in range(len(en)) for c in pooled[i]]
    owner = [i for i in range(len(en)) for _ in pooled[i]]
    print(f"{len(flat)} candidates to score under each model", flush=True)

    all_lp = {}
    for tag, base, adapter in MODELS:
        print(f"\n### scoring under {tag}", flush=True)
        m = load(base, adapter)
        all_lp[tag] = score(flat, tok, m)
        del m; torch.cuda.empty_cache()

    # --- phase 3: pick with the tuned weights ---
    picks = [None] * len(en)
    bestval = [float("-inf")] * len(en)
    for j, i in enumerate(owner):
        lp_s = all_lp["1.3B"][0][j] / (all_lp["1.3B"][1][j] ** LENNORM)
        lp_b = all_lp["3.3B"][0][j] / (all_lp["3.3B"][1][j] ** LENNORM)
        v = W * lp_s + (1 - W) * lp_b
        if v > bestval[i]:
            bestval[i], picks[i] = v, flat[j][1]

    # fall back to the 1.3B top beam if anything came out empty
    n_empty = 0
    for i, p in enumerate(picks):
        if p is None or not p.strip():
            picks[i] = nbest["1.3B"][i][0]
            n_empty += 1
    picks = [p if p.strip() else "؟" for p in picks]
    print(f"\nfell back to 1.3B beam-1 for {n_empty} sentences", flush=True)

    # how often did the ensemble diverge from plain 1.3B beam search?
    solo = [nbest["1.3B"][i][0] for i in range(len(en))]
    changed = sum(1 for a, b in zip(picks, solo) if a != b)
    print(f"ensemble differs from 1.3B beam-1 on {changed}/{len(en)} "
          f"({100 * changed / len(en):.1f}%)", flush=True)

    assert len(picks) == len(dev)
    out = pd.DataFrame({"ID": dev["ID"].tolist(), "kashmiri_text": picks})
    out.to_csv(OUT_FILE, index=False)
    print("wrote", OUT_FILE, "rows:", len(out))


if __name__ == "__main__":
    main()
