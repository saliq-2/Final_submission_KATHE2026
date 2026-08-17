# config.py
from pathlib import Path

# --- Paths ---
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
WORK_DIR   = ROOT_DIR / "work"
CKPT_DIR   = ROOT_DIR / "ckpt"
for d in (DATA_DIR, WORK_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Raw BPCC file (downloaded from HF: ai4bharat/BPCC, bpcc-seed-latest/kas_Arab.tsv)
RAW_TSV = DATA_DIR / "bpcc_seed_latest_kas_Arab.tsv"

# Cleaned + split outputs (all carved locally out of BPCC; there is no
# competition test.csv yet, so "test" here means our own held-out local test set)
CLEAN_PARQUET = WORK_DIR / "bpcc_clean.parquet"
TRAIN_SPLIT   = WORK_DIR / "train_split.parquet"
VAL_FILE      = WORK_DIR / "val.parquet"
TEST_SPLIT    = WORK_DIR / "test_split.parquet"   # local, labeled held-out set

SUBMISSION = WORK_DIR / "submission.csv"

# FLORES-200 eng_Latn-kas_Arab (independent of BPCC, professionally
# translated). dev is mixed into training to broaden register beyond BPCC's
# narrow Wikipedia style; devtest is kept fully held out as a second,
# independent local validation set (BPCC-derived val/test turned out to be
# optimistic since it shares BPCC's distribution with training).
FLORES_DEV_FILE      = WORK_DIR / "flores_dev.parquet"
FLORES_DEVTEST_FILE  = WORK_DIR / "flores_devtest.parquet"
TRAIN_SPLIT_AUGMENTED = WORK_DIR / "train_split_augmented.parquet"

# BPCC "daily" subset: colloquial/voice-assistant register, shorter avg
# sentence length (~9.9 words) and higher diacritic density (~12.1%) than
# bpcc-seed-latest (~9.5%) -- closer match to the real competition test's
# short-sentence register than FLORES-dev was. Used whole (all 4,311 rows)
# in training since it's not a held-out eval set.
DAILY_FILE = WORK_DIR / "bpcc_daily.parquet"
TRAIN_SPLIT_V3 = WORK_DIR / "train_split_v3.parquet"

# v4: FLORES-dev + daily oversampled (repeated) instead of added once --
# at natural proportions (~6% of the v3 mix) the short-sentence register
# got drowned out by BPCC's dominant long-sentence signal (v3 didn't help).
OVERSAMPLE_FACTOR = 10
TRAIN_SPLIT_V4 = WORK_DIR / "train_split_v4.parquet"

# cleaned_new_eng_kmr.csv: genuinely register-matched (avg 7.44 words) and
# correctly-diacritized (17.6% density, exceeds gold's 16.5%) new dataset,
# confirmed via exact-match to share provenance with Ground_Truth-100.
TRAIN_SPLIT_NEWDATA = WORK_DIR / "train_split_newdata.parquet"
# Task Brief 5 §1: filtered to exclude rows overlapping englishdev.csv /
# Ground_Truth-100 (639 + 44, plus 2 near-duplicates caught by normalized
# matching) -- 3,439 of the original 4,104 rows survive.
TRAIN_SPLIT_NEWDATA_CLEAN = WORK_DIR / "train_split_newdata_clean.parquet"
VAL_NEWDATA_CLEAN = WORK_DIR / "val_newdata_clean.parquet"
VAL_NEWDATA = WORK_DIR / "val_newdata.parquet"

# data/English.txt + data/Kashmiri.txt: new 30k-pair dataset, fixed for a
# stray blank-line misalignment (see prepare_bignew.py), exact-deduped, no
# contamination with englishdev.csv/Ground_Truth-100. Mixed register (long
# low-density "formal" half + shorter higher-density "conversational" half),
# used in full per explicit user decision.
TRAIN_SPLIT_BIGNEW = WORK_DIR / "train_split_bignew.parquet"
VAL_BIGNEW = WORK_DIR / "val_bignew.parquet"

# BPCC (repaired, see clean_bpcc.py) + bignew oversampled 3x (~47% share) --
# intended for full fine-tuning, see prepare_bignew_mix.py header.
TRAIN_SPLIT_BIGNEW_MIX = WORK_DIR / "train_split_bignew_mix.parquet"

# --- Column names (raw tsv columns are src=English, tgt=Kashmiri) ---
RAW_SRC_COL = "src"
RAW_TGT_COL = "tgt"

# --- Language tags for IndicTrans2 ---
SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"

# --- Models ---
BASELINE_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"  # fast, baseline
BIG_MODEL      = "ai4bharat/indictrans2-en-indic-1B"         # higher ceiling

# --- Split sizes ---
VAL_SIZE  = 3000
TEST_SIZE = 3000   # local held-out test, distinct from val (used for model selection)
SEED      = 42

# --- Decoding (tuned on local val via tune_decode.py; the sweep was flat --
# all (beams, lp) combos landed within ~0.1 geo_mean of each other, so this
# is the nominal best, not a strong signal) ---
NUM_BEAMS      = 8
LENGTH_PENALTY = 1.0
MAX_NEW_TOKENS = 256
BATCH_SIZE     = 32
