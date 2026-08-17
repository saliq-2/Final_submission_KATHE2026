# nllb/eval_nllb.py
# Evaluate the LoRA-fine-tuned NLLB adapter on Ground_Truth-100, same
# restoration pipeline and comparison methodology as the IndicTrans2
# champion (best_adapter_newdata_clean, real score 14.23).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import CKPT_DIR as MAIN_CKPT_DIR
from diacritic_restore import load_dictionary
from diacritizer_gated import hybrid_restore_gated
from ground_truth_eval import load_ground_truth, get_hyps
from metric import kathe_score, paired_bootstrap
from diacritic_diagnostics import density
from nllb_common import resolve_model_name, model_tag
import torch.distributed.tensor
MODEL_NAME = resolve_model_name()
MODEL_TAG = model_tag(MODEL_NAME)
SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"
NLLB_CKPT_DIR = Path(__file__).resolve().parent / "ckpt"
BATCH_SIZE = 16


@torch.no_grad()
def translate_nllb_lora(sentences, tok, model, num_beams=8):
    out = []
    tgt_id = tok.convert_tokens_to_ids(TGT_LANG)
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        gen = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=num_beams,
                             max_new_tokens=256, no_repeat_ngram_size=4)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def main():
    USE_BPCC = "--bpcc" in sys.argv
    USE_COMBINED = "--combined" in sys.argv
    if USE_COMBINED:
        # best overall IndicTrans2 result is still the right bar to clear
        nllb_adapter, champion_adapter = f"best_adapter_nllb_combined_{MODEL_TAG}", "best_adapter_newdata_clean"
    elif USE_BPCC:
        nllb_adapter, champion_adapter = f"best_adapter_nllb_bpcc_{MODEL_TAG}", "best_adapter"
    else:
        nllb_adapter, champion_adapter = f"best_adapter_nllb_newdata_clean_{MODEL_TAG}", "best_adapter_newdata_clean"
    print(f"comparing NLLB-LoRA ({nllb_adapter}) vs IndicTrans2 champion ({champion_adapter})")

    adapter_path = str(NLLB_CKPT_DIR / nllb_adapter)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda").eval()
    model = PeftModel.from_pretrained(base, adapter_path).to("cuda").eval()

    en, refs = load_ground_truth()
    raw = translate_nllb_lora(en, tok, model)

    print("=== raw NLLB-LoRA output (no restoration) ===")
    print(kathe_score(raw, refs))
    print("output density:", density(raw))
    print("gold density:", density(refs))

    d = load_dictionary()
    restored = [hybrid_restore_gated(h, d, 0.2) for h in raw]
    print("\n=== + confidence-gated hybrid restoration (champion's pipeline) ===")
    print(kathe_score(restored, refs))
    print("(restoration is calibrated for IndicTrans2's under-diacritization pattern --")
    print(" NLLB's raw density is already above gold, so compare both configs below)")

    champion_raw = get_hyps(champion_adapter, restore=False)
    champion = [hybrid_restore_gated(h, d, 0.2) for h in champion_raw]
    print(f"\n=== IndicTrans2 champion ({champion_adapter}) for comparison ===")
    print(kathe_score(champion, refs))

    bs_restored = paired_bootstrap(restored, champion, refs)
    print(f"\npaired bootstrap (NLLB+restoration vs champion): mean_delta={bs_restored['mean_delta']:.3f} "
          f"CI=[{bs_restored['ci_2.5']:.3f},{bs_restored['ci_97.5']:.3f}] "
          f"P(nllb-lora>champion)={bs_restored['P(A>B)']:.3f}")

    bs_raw = paired_bootstrap(raw, champion, refs)
    print(f"paired bootstrap (NLLB raw, no restoration, vs champion): mean_delta={bs_raw['mean_delta']:.3f} "
          f"CI=[{bs_raw['ci_2.5']:.3f},{bs_raw['ci_97.5']:.3f}] "
          f"P(nllb-lora>champion)={bs_raw['P(A>B)']:.3f}")

    print("\n=== sample outputs (raw, no restoration -- NLLB's best config) ===")
    for e, h, r in list(zip(en, raw, refs))[:8]:
        print("EN :", e)
        print("OUT:", h)
        print("REF:", r)
        print()


if __name__ == "__main__":
    main()
