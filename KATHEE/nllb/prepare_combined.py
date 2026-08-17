# nllb/prepare_combined.py
# Combines all three datasets in play for NLLB training: BPCC (repaired,
# 87,884 rows), bignew (data/bignew_eng_kmr.csv, already alignment-fixed and
# deduped, 26,702 rows), and cleaned_new_eng_kmr_filtered.csv (the
# contamination-filtered register-matched set that produced the actual
# 14.23 champion, 3,439 rows).
#
# Without oversampling, the smallest and (historically) most valuable set
# would be diluted to ~2.9% of the mix -- the same drowning-out failure
# already confirmed twice for IndicTrans2 LoRA (v3: daily register diluted
# to ~6%, didn't help). Oversampled 8x here (~23% of the mix, roughly
# parity with bignew's share) rather than repeating that known failure mode
# unmodified.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import TRAIN_SPLIT, DATA_DIR, SEED
from clean_bpcc import collapse_repeated_marks, is_wrong_script

BIGNEW_CSV = DATA_DIR / "bignew_eng_kmr.csv"
FILTERED_CSV = DATA_DIR / "cleaned_new_eng_kmr_filtered.csv"
OVERSAMPLE_FILTERED = 8
VAL_SIZE = 300

NLLB_DATA_DIR = Path(__file__).resolve().parent / "data"
NLLB_DATA_DIR.mkdir(exist_ok=True)
OUT_TRAIN = NLLB_DATA_DIR / "train_combined.parquet"
OUT_VAL = NLLB_DATA_DIR / "val_combined.parquet"


def main():
    bpcc = pd.read_parquet(TRAIN_SPLIT)[["en", "ks"]].copy()
    wrong_script = bpcc["ks"].astype(str).map(is_wrong_script)
    bpcc = bpcc[~wrong_script].reset_index(drop=True)
    bpcc["ks"] = bpcc["ks"].astype(str).map(collapse_repeated_marks)

    bignew = pd.read_csv(BIGNEW_CSV)[["en", "ks"]]

    filtered = pd.read_csv(FILTERED_CSV).rename(columns={"English": "en", "Kashmiri": "ks"})[["en", "ks"]]
    filtered["en"] = filtered["en"].astype(str).str.strip()
    filtered["ks"] = filtered["ks"].astype(str).str.strip()

    mix = pd.concat([bpcc, bignew] + [filtered] * OVERSAMPLE_FILTERED, ignore_index=True)
    mix = mix.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    val = mix.iloc[:VAL_SIZE]
    train = mix.iloc[VAL_SIZE:]

    train.to_parquet(OUT_TRAIN)
    val.to_parquet(OUT_VAL)

    filtered_n = len(filtered) * OVERSAMPLE_FILTERED
    total = len(bpcc) + len(bignew) + filtered_n
    print(f"bpcc (repaired) = {len(bpcc)} ({100*len(bpcc)/total:.1f}%)")
    print(f"bignew = {len(bignew)} ({100*len(bignew)/total:.1f}%)")
    print(f"filtered x{OVERSAMPLE_FILTERED} = {filtered_n} ({100*filtered_n/total:.1f}%)")
    print(f"combined = {total} rows -> train={len(train)}, val={len(val)}")
    print(f"-> {OUT_TRAIN}")
    print(f"-> {OUT_VAL}")


if __name__ == "__main__":
    main()
