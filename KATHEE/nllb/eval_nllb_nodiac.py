# nllb/eval_nllb_nodiac.py
# Same methodology as eval_nllb.py minus the diacritic-restoration path
# (diacritizer_apply.py is missing from both checkouts). Restoration is
# calibrated for IndicTrans2's under-diacritization and was already
# established to hurt NLLB, whose raw density sits near gold -- so raw is
# NLLB's reported config anyway. Champion comparison is attempted but
# non-fatal, so a missing dependency there doesn't cost the GPU run.
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ground_truth_eval import load_ground_truth
from metric import kathe_score, paired_bootstrap
from diacritic_diagnostics import density
from nllb_common import resolve_model_name, model_tag

MODEL_NAME = resolve_model_name()
MODEL_TAG = model_tag(MODEL_NAME)
SRC_LANG, TGT_LANG = "eng_Latn", "kas_Arab"
NLLB_CKPT_DIR = Path(__file__).resolve().parent / "ckpt"
OUT_DIR = Path(__file__).resolve().parent / "eval_out"
BATCH_SIZE = 16


@torch.no_grad()
def translate_nllb_lora(sentences, tok, model, num_beams=8):
    out = []
    tgt_id = tok.convert_tokens_to_ids(TGT_LANG)
    for i in range(0, len(sentences), BATCH_SIZE):
        enc = tok(sentences[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to("cuda")
        gen = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=num_beams,
                             max_new_tokens=256, no_repeat_ngram_size=4)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"{min(i + BATCH_SIZE, len(sentences))}/{len(sentences)}", flush=True)
    return out


def main():
    if "--combined" in sys.argv:
        nllb_adapter, champion_adapter, tag = f"best_adapter_nllb_combined_{MODEL_TAG}", "best_adapter_newdata_clean", "combined"
    elif "--bpcc" in sys.argv:
        nllb_adapter, champion_adapter, tag = f"best_adapter_nllb_bpcc_{MODEL_TAG}", "best_adapter", "bpcc"
    else:
        nllb_adapter, champion_adapter, tag = f"best_adapter_nllb_newdata_clean_{MODEL_TAG}", "best_adapter_newdata_clean", "newdata-clean"

    adapter_path = NLLB_CKPT_DIR / nllb_adapter
    if not adapter_path.exists():
        sys.exit(f"adapter not found: {adapter_path}")
    print(f"base={MODEL_NAME}  adapter={nllb_adapter}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda").eval()
    model = PeftModel.from_pretrained(base, str(adapter_path)).to("cuda").eval()

    en, refs = load_ground_truth()
    raw = translate_nllb_lora(en, tok, model)

    score = kathe_score(raw, refs)
    print("\n=== raw NLLB-LoRA output (no restoration) ===")
    print(score)
    print("output density:", density(raw))
    print("gold density:", density(refs))

    result = {"base": MODEL_NAME, "adapter": nllb_adapter, "branch": tag,
              "raw_score": score, "raw_density": density(raw), "gold_density": density(refs)}

    try:
        from ground_truth_eval import get_hyps
        champion = get_hyps(champion_adapter, restore=False)
        c_score = kathe_score(champion, refs)
        print(f"\n=== IndicTrans2 champion ({champion_adapter}), raw ===")
        print(c_score)
        bs = paired_bootstrap(raw, champion, refs)
        print(f"\npaired bootstrap (NLLB raw vs champion raw): mean_delta={bs['mean_delta']:.3f} "
              f"CI=[{bs['ci_2.5']:.3f},{bs['ci_97.5']:.3f}] P(nllb>champion)={bs['P(A>B)']:.3f}")
        result["champion_score"] = c_score
        result["bootstrap"] = {k: float(v) for k, v in bs.items()}
    except Exception as e:
        print(f"\n[champion comparison skipped: {type(e).__name__}: {e}]")
        champion = None

    OUT_DIR.mkdir(exist_ok=True)
    stem = f"{tag}_{MODEL_TAG}"
    with open(OUT_DIR / f"eval_{stem}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / f"hyps_{stem}.json", "w", encoding="utf-8") as f:
        json.dump([{"en": e, "hyp": h, "ref": r} for e, h, r in zip(en, raw, refs)],
                  f, ensure_ascii=False, indent=2)
    print("\nwrote", OUT_DIR / f"eval_{stem}.json")

    print("\n=== sample outputs (raw) ===")
    for e, h, r in list(zip(en, raw, refs))[:8]:
        print("EN :", e); print("OUT:", h); print("REF:", r); print()


if __name__ == "__main__":
    main()
