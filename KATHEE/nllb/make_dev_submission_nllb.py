# nllb/make_dev_submission_nllb.py
# Translates englishdev.csv with the NLLB-LoRA combined-dataset adapter
# (best_adapter_nllb_combined) -- the new best local result, geo_mean 21.72
# on Ground_Truth-100 vs. the IndicTrans2 champion's 13.29, paired bootstrap
# P=1.000. Deliberately RAW output, no restoration -- confirmed repeatedly
# (every checkpoint spot-check during training, and the final evaluation)
# that our hybrid restoration pipeline hurts NLLB rather than helping it:
# it's calibrated for IndicTrans2's under-diacritization pattern, but NLLB's
# raw output density (~0.168) is already close to gold (~0.165), so applying
# it over-corrects.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import ROOT_DIR, WORK_DIR
from nllb_common import resolve_model_name, model_tag
import torch.distributed.tensor
MODEL_NAME = resolve_model_name()
MODEL_TAG = model_tag(MODEL_NAME)
SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"
DEV_FILE = ROOT_DIR / "englishdev.csv"
ADAPTER_PATH = Path(__file__).resolve().parent / "ckpt" / f"best_adapter_nllb_combined_{MODEL_TAG}"
OUT_FILE = WORK_DIR / f"englishdev_submission_nllb_combined_{MODEL_TAG}.csv"
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
        print(f"{min(i + BATCH_SIZE, len(sentences))}/{len(sentences)} done", flush=True)
    return out


def main():
    dev = pd.read_csv(DEV_FILE)
    assert {"ID", "sentence"}.issubset(dev.columns), dev.columns

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda").eval()
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH)).to("cuda").eval()

    preds = translate_nllb_lora(dev["sentence"].astype(str).tolist(), tok, model)

    # Safety: no empty predictions (empty output tanks both metrics)
    preds = [p if p.strip() else "؟" for p in preds]
    assert len(preds) == len(dev)

    out = pd.DataFrame({"ID": dev["ID"].tolist(), "kashmiri_text": preds})
    out.to_csv(OUT_FILE, index=False)

    print("wrote", OUT_FILE, "rows:", len(dev))


if __name__ == "__main__":
    main()
