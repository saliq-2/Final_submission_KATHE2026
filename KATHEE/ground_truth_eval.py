# ground_truth_eval.py
# Reusable harness for the 100-row Ground_Truth proxy set: loads it, and
# generates+caches (adapter, restore) hypothesis sets so repeated
# champion/challenger comparisons (paired bootstrap) don't re-run inference.
# NEVER trains on this set or builds a lexicon from it (R4) -- read-only.
import json
import pandas as pd
from config import *
from translate import load_model, translate
from diacritic_restore import load_dictionary, restore_diacritics

GT_FILE = DATA_DIR / "Evaluation_Set_With_ChatGPT_Translations.xlsx"
HYPS_CACHE_DIR = WORK_DIR / "gt_hyps_cache"
HYPS_CACHE_DIR.mkdir(exist_ok=True)

def load_ground_truth():
    df = pd.read_excel(GT_FILE)
    en = df.English.astype(str).str.strip().tolist()
    refs = df.Ground_Truth.astype(str).str.strip().tolist()
    return en, refs

def get_hyps(adapter_name=None, restore=True, use_cache=True):
    """adapter_name: e.g. 'best_adapter', 'best_adapter_v3', or None for zero-shot baseline."""
    key = f"{adapter_name or 'baseline'}_{'restored' if restore else 'raw'}"
    cache_path = HYPS_CACHE_DIR / f"{key}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())

    en, _ = load_ground_truth()
    adapter_path = str(CKPT_DIR / adapter_name) if adapter_name else None
    tok, model = load_model(BASELINE_MODEL, adapter_path=adapter_path)
    hyps = translate(en, tok, model)

    if restore:
        d = load_dictionary()
        hyps = [restore_diacritics(h, d) for h in hyps]

    cache_path.write_text(json.dumps(hyps, ensure_ascii=False))
    return hyps
