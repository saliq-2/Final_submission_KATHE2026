


# nllb/make_dev_submission_ensemble_v2.py
# v2 ensemble submission: K=16 with both plain and diverse beam search as
# candidate sources, cross-scored under the 1.3B and 3.3B combined adapters.
# W/LENNORM are the Ground_Truth-100 optimum for this candidate pool
# (geo_mean 26.62, vs 26.12 for the v1 K=8 beam-only pool).
# Note the pool change also lowers the solo beam-1 baseline (22.42 vs 24.47
# at K=8) -- wider beams find shorter, more generic outputs -- so nearly all
# of v2's headline number comes from reranking, not from base decoding.
# Writes a separate CSV; the v1 submission (dev 26.4) stays as fallback.
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import ROOT_DIR, WORK_DIR
import torch.distributed.tensor  # noqa: F401

W = 0.2                   # weight on 1.3B; (1-W) on 3.3B
LENNORM = 0.6
K = 16
DIVERSE_GROUPS = 4        # must divide K evenly
DIVERSITY_PENALTY = 0.5
GEN_BS = 2
SCORE_BS = 16

SRC_LANG, TGT_LANG = "eng_Latn", "kas_Arab"
NLLB_DIR = Path(__file__).resolve().parent
CKPT = NLLB_DIR / "ckpt"
OUT_DIR = NLLB_DIR / "eval_out"
CACHE = OUT_DIR / "dev_nbest_cache_v2.json"
DEV_FILE = ROOT_DIR / "englishdev.csv"
OUT_FILE = WORK_DIR / "englishdev_submission_nllb_ensemble_v2.csv"
V1_FILE = WORK_DIR / "englishdev_submission_nllb_ensemble.csv"
MODELS = [
    ("1.3B", "facebook/nllb-200-1.3B", CKPT / "best_adapter_nllb_combined_1.3B"),
    ("3.3B", "facebook/nllb-200-3.3B", CKPT / "best_adapter_nllb_combined_3.3B"),
]


def load(base_name, adapter):
    m = AutoModelForSeq2SeqLM.from_pretrained(base_name, torch_dtype=torch.float16)
    return PeftModel.from_pretrained(m, str(adapter)).to("cuda").eval()


@torch.no_grad()
def gen_nbest(sents, tok, model):
    """Variable-length candidate list per sentence; index 0 is always the
    plain beam-search winner."""
    tgt_id = tok.convert_tokens_to_ids(TGT_LANG)
    out = []
    for i in range(0, len(sents), GEN_BS):
        batch = sents[i:i + GEN_BS]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to("cuda")

        g1 = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=K,
                            num_return_sequences=K, max_new_tokens=256,
                            no_repeat_ngram_size=4)
        d1 = tok.batch_decode(g1, skip_special_tokens=True)

        g2 = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=K,
                            num_beam_groups=DIVERSE_GROUPS,
                            diversity_penalty=DIVERSITY_PENALTY,
                            num_return_sequences=K, max_new_tokens=256,
                            no_repeat_ngram_size=4, do_sample=False)
        d2 = tok.batch_decode(g2, skip_special_tokens=True)

        for j in range(len(batch)):
            cands = d1[j * K:(j + 1) * K] + d2[j * K:(j + 1) * K]
            out.append(list(dict.fromkeys(c for c in cands if c.strip())))
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
        if (i // SCORE_BS) % 20 == 0:
            print(f"score {min(i + SCORE_BS, len(pairs))}/{len(pairs)}", flush=True)
    return lps, lens


def main():
    dev = pd.read_csv(DEV_FILE)
    assert {"ID", "sentence"}.issubset(dev.columns), dev.columns
    en = dev["sentence"].astype(str).tolist()
    print(f"{len(en)} dev sentences | W={W} LENNORM={LENNORM} K={K}", flush=True)

    for tag, _, adapter in MODELS:
        if not adapter.exists():
            sys.exit(f"adapter not found: {adapter}")

    tok = AutoTokenizer.from_pretrained(MODELS[0][1], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    OUT_DIR.mkdir(exist_ok=True)

    # --- phase 1: candidates (cached; delete the cache if K or strategy changes) ---
    if CACHE.exists():
        nbest = json.load(open(CACHE, encoding="utf-8"))
        assert all(len(nbest[t]) == len(en) for t, _, _ in MODELS), "stale cache; delete it"
        print("loaded cached candidates", flush=True)
    else:
        nbest = {}
        for tag, base, adapter in MODELS:
            print(f"\n### generating with {tag} (beam + diverse beam, K={K})", flush=True)
            m = load(base, adapter)
            nbest[tag] = gen_nbest(en, tok, m)
            del m; torch.cuda.empty_cache()
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(nbest, f, ensure_ascii=False)
        print("cached candidates to", CACHE, flush=True)

    pooled = [list(dict.fromkeys(itertools.chain(*(nbest[t][i] for t, _, _ in MODELS))))
              for i in range(len(en))]
    print(f"pooled: mean {sum(map(len, pooled)) / len(pooled):.1f} candidates/sentence "
          f"(min {min(map(len, pooled))}, max {max(map(len, pooled))})", flush=True)

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
        v = (W * all_lp["1.3B"][0][j] / (all_lp["1.3B"][1][j] ** LENNORM)
             + (1 - W) * all_lp["3.3B"][0][j] / (all_lp["3.3B"][1][j] ** LENNORM))
        if v > bestval[i]:
            bestval[i], picks[i] = v, flat[j][1]

    n_fallback = 0
    for i, p in enumerate(picks):
        if p is None or not p.strip():
            picks[i] = nbest["1.3B"][i][0]
            n_fallback += 1
    picks = [p if p.strip() else "؟" for p in picks]
    print(f"\nfell back to 1.3B beam-1 for {n_fallback} sentences", flush=True)

    solo = [nbest["1.3B"][i][0] for i in range(len(en))]
    changed = sum(1 for a, b in zip(picks, solo) if a != b)
    print(f"differs from 1.3B beam-1 on {changed}/{len(en)} "
          f"({100 * changed / len(en):.1f}%)", flush=True)

    if V1_FILE.exists():
        v1 = pd.read_csv(V1_FILE)
        if len(v1) == len(picks):
            d = sum(1 for a, b in zip(picks, v1["kashmiri_text"].astype(str)) if a != b)
            print(f"differs from v1 submission on {d}/{len(en)} "
                  f"({100 * d / len(en):.1f}%)", flush=True)

    assert len(picks) == len(dev)
    pd.DataFrame({"ID": dev["ID"].tolist(), "kashmiri_text": picks}).to_csv(OUT_FILE, index=False)
    print("wrote", OUT_FILE, "rows:", len(dev))


if __name__ == "__main__":
    main()
