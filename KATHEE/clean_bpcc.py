# clean_bpcc.py
# Ad-hoc audit of work/bpcc_clean.parquet (93,899 rows) prompted by a
# suspicion that BPCC itself might be dirty, not just register-mismatched.
# Verdict: structurally sound at bulk scale (0 exact-dup pairs, 0 blanks,
# en/ks length correlation 0.924) -- the known problems (9.95% diacritic
# density vs gold's 16.5%, narrow long-sentence register) are systematic
# label-distribution issues, not corruption. But two small, genuine, easily
# fixed defects were found:
#   1. 337 rows (0.36%) have a doubled/tripled identical combining mark back
#      to back (e.g. "شُُرین", "چُُھ") -- an OCR/data-entry typo pattern, not
#      valid Kashmiri orthography (confirmed by spot-checking run==2 cases,
#      not just the run>=3 extreme cases). Fixed by collapsing runs to 1.
#   2. 15 rows are majority-Devanagari script -- actual Kashmiri content
#      mistranscribed in the wrong script (should be kas_Arab/Perso-Arabic).
#      Distinct from ~6 other rows with a single embedded Devanagari/Sanskrit
#      loanword inside an otherwise-Arabic-script sentence, which are
#      legitimate and left alone. Dropped.
# Writes a repaired copy; does NOT touch bpcc_clean.parquet/train_split.parquet
# so every earlier result in this project stays reproducible from the
# original files.
import re
import unicodedata
import pandas as pd
from config import WORK_DIR, CLEAN_PARQUET

OUT_FILE = WORK_DIR / "bpcc_clean_repaired.parquet"

DEVA_RE = re.compile(r"[ऀ-ॿ]")
ARABIC_BLOCK_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def collapse_repeated_marks(s):
    out = []
    prev = None
    for c in s:
        if unicodedata.combining(c) != 0 and c == prev:
            continue  # drop the repeat
        out.append(c)
        prev = c if unicodedata.combining(c) != 0 else None
    return "".join(out)


def is_wrong_script(s):
    d = len(DEVA_RE.findall(s))
    a = len(ARABIC_BLOCK_RE.findall(s))
    return d > 0 and d > a


def main():
    df = pd.read_parquet(CLEAN_PARQUET)
    before = len(df)

    wrong_script = df["ks"].astype(str).map(is_wrong_script)
    print(f"dropping {wrong_script.sum()} wrong-script (majority-Devanagari) rows")
    df = df[~wrong_script].reset_index(drop=True)

    n_touched = (df["ks"].astype(str) != df["ks"].astype(str).map(collapse_repeated_marks)).sum()
    print(f"collapsing repeated combining marks in {n_touched} rows")
    df["ks"] = df["ks"].astype(str).map(collapse_repeated_marks)

    df.to_parquet(OUT_FILE)
    print(f"{before} -> {len(df)} rows -> {OUT_FILE}")


if __name__ == "__main__":
    main()
