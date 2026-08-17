# nllb/ensemble_nllb_v2.py
# v2 of the cross-scoring ensemble: wider candidate pool (K=16) and two
# generation strategies per model (standard beam + diverse beam search),
# since diverse beam surfaces candidates plain beam search never reaches
# and the reranker can only pick from what it's given.
# v1 (K=8, beam only) scored geo_mean 26.12 on Ground_Truth-100 at
# w=0.4/lennorm=0.8, vs 24.47 for the 1.3B adapter alone.
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
NLLB_DIR = Path(__file__).resolve().parent
CKPT = NLLB_DIR / "ckpt"
OUT_DIR = NLLB_DIR / "eval_out"
CACHE = OUT_DIR / "gt100_nbest_cache_v2.json"

MODELS = [
    ("1.3B", "facebook/nllb-200-1.3B", CKPT / "best_adapter_nllb_combined_1.3B"),
    ("3.3B", "facebook/nllb-200-3.3B", CKPT / "best_adapter_nllb_combined_3.3B"),
]
K = 16                    # beams / candidates per strategy
DIVERSE_GROUPS = 4        # must divide K evenly
DIVERSITY_PENALTY = 0.5
GEN_BS = 2                # halved from v1: K doubled and two passes per batch
SCORE_BS = 16

WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
LENNORMS = [0.6, 0.8, 1.0]


def load(base_name, adapter):
    m = AutoModelForSeq2SeqLM.from_pretrained(base_name, torch_dtype=torch.float16)
    return PeftModel.from_pretrained(m, str(adapter)).to("cuda").eval()


@torch.no_grad()
def gen_nbest(sents, tok, model):
    """Returns one variable-length candidate list per sentence; index 0 is
    always the plain beam-search winner, so callers can still read solo output."""
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
    en, refs = load_ground_truth()
    for tag, _, adapter in MODELS:
        if not adapter.exists():
            sys.exit(f"adapter not found: {adapter}")
    tok = AutoTokenizer.from_pretrained(MODELS[0][1], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    OUT_DIR.mkdir(exist_ok=True)

    # --- phase 1: candidates from each model (cached) ---
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

    solo = {t: [nbest[t][i][0] for i in range(len(en))] for t, _, _ in MODELS}
    pooled = [list(dict.fromkeys(itertools.chain(*(nbest[t][i] for t, _, _ in MODELS))))
              for i in range(len(en))]
    print(f"pooled: mean {sum(map(len, pooled)) / len(pooled):.1f} candidates/sentence "
          f"(min {min(map(len, pooled))}, max {max(map(len, pooled))})", flush=True)

    # --- phase 2: cross-score every pooled candidate under both models ---
    flat = [(en[i], c) for i in range(len(en)) for c in pooled[i]]
    owner = [i for i in range(len(en)) for _ in pooled[i]]
    print(f"{len(flat)} candidates to score under each model", flush=True)

    all_lp = {}
    for tag, base, adapter in MODELS:
        print(f"\n### scoring under {tag}", flush=True)
        m = load(base, adapter)
        all_lp[tag] = score(flat, tok, m)
        del m; torch.cuda.empty_cache()

    # --- phase 3: solo baselines, then sweep ---
    results = {"config": {"K": K, "diverse_groups": DIVERSE_GROUPS,
                          "diversity_penalty": DIVERSITY_PENALTY}}
    for tag, _, _ in MODELS:
        s = kathe_score(solo[tag], refs)
        results[f"solo_{tag}"] = s
        print(f"\n=== {tag} alone (plain beam-1) ===\n{s}")

    print()
    best = None
    for w in WEIGHTS:
        for a in LENNORMS:
            picks = [None] * len(en)
            bestval = [float("-inf")] * len(en)
            for j, i in enumerate(owner):
                v = (w * all_lp["1.3B"][0][j] / (all_lp["1.3B"][1][j] ** a)
                     + (1 - w) * all_lp["3.3B"][0][j] / (all_lp["3.3B"][1][j] ** a))
                if v > bestval[i]:
                    bestval[i], picks[i] = v, flat[j][1]
            sc = kathe_score(picks, refs)
            changed = sum(1 for p, s_ in zip(picks, solo["1.3B"]) if p != s_)
            print(f"w={w:.1f} lennorm={a:.1f} -> geo_mean={sc['geo_mean']:.2f} "
                  f"bleu={sc['bleu']:.2f} chrf++={sc['chrf++']:.2f} "
                  f"changed={100 * changed / len(en):.0f}%", flush=True)
            if best is None or sc["geo_mean"] > best[0]["geo_mean"]:
                best = (sc, w, a, picks)

    sc, w, a, picks = best
    print(f"\n=== best: w={w} lennorm={a} ===\n{sc}")
    results["ensemble_best"] = {"score": sc, "w": w, "lennorm": a}

    with open(OUT_DIR / "ensemble_v2_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    with open(OUT_DIR / "ensemble_v2_hyps.json", "w", encoding="utf-8") as f:
        json.dump([{"en": e, "hyp": h, "ref": r} for e, h, r in zip(en, picks, refs)],
                  f, ensure_ascii=False, indent=2)
    print("\nwrote", OUT_DIR / "ensemble_v2_scores.json")


if __name__ == "__main__":
    main()
