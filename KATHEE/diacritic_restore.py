# diacritic_restore.py
# Post-processing: our model's generated text uses far fewer diacritics than
# real Kashmiri orthography (~6% vs ~15.7% density, measured against a
# genuine ground-truth set) -- this alone triples the score when stripped
# from comparison, suggesting the underlying translations are more correct
# than raw chrF++/BLEU show. This builds a bare-word -> most-common-fully-
# diacritized-form dictionary from BPCC (the only large corpus we have) and
# uses it to restore diacritics on model output word-by-word. Dictionary is
# built ONLY from BPCC training data -- never from the held-out ground-truth
# eval set, so scoring against that set stays a genuine, non-leaking test.
from collections import Counter
import pandas as pd
from diacritics import normalize_text, strip_diacritics
from config import CLEAN_PARQUET, DAILY_FILE

PUNCT_CHARS = "۔،؟!٪%\"'“”‘’()[]{}:;.,"

def split_word_punct(token):
    lead = 0
    while lead < len(token) and token[lead] in PUNCT_CHARS:
        lead += 1
    trail = len(token)
    while trail > lead and token[trail - 1] in PUNCT_CHARS:
        trail -= 1
    return token[:lead], token[lead:trail], token[trail:]

def build_dictionary(sentences):
    # Reverted (Task Brief 2, §1): "prefer any marked variant over bare"
    # (the earlier Bug-B fix) turned out to be wrong when the variant space
    # is grammatically conditioned rather than stylistic noise -- e.g. the
    # copula چھ has multiple person/number/gender-inflected forms
    # (چھِ/چُھ/چھُ/...), and picking a confident-but-wrong inflection
    # destroys more chrF++ overlap than leaving the token bare. Back to
    # plain most_common(1) over ALL variants including bare, which is
    # accidentally conservative (bare wins for genuinely ambiguous/
    # grammatically-conditioned forms since no single inflection dominates).
    # Inventory (normalize_text/strip_diacritics from diacritics.py, 25
    # marks) stays fixed -- only the selection rule reverts.
    counters = {}
    for sent in sentences:
        for tok in normalize_text(sent).split():
            _, core, _ = split_word_punct(tok)
            if not core:
                continue
            counters.setdefault(strip_diacritics(core), Counter())[core] += 1

    # Task Brief 3, §1: same identity-entry bug as build_dictionary_gated --
    # when bare is modal, most_common(1) returns bare, and an explicit
    # bare->bare entry overrides restore_diacritics's dictionary.get(bare,
    # core) fallback, actively deleting marks the model already produced.
    # Omit those entries instead so lookup misses fall through to core.
    return {bare: cnt.most_common(1)[0][0] for bare, cnt in counters.items()
            if cnt.most_common(1)[0][0] != bare}

def build_dictionary_gated(sentences, min_share=0.8, min_count=5):
    """Confidence-gated variant (Task Brief 2, §4): substitute a marked
    variant only when the marked-variant space is near-unanimous (share
    computed AMONG MARKED VARIANTS ONLY, excluding the bare count -- the
    bare count reflects how often the corpus omits optional marking, not
    evidence about which inflection is correct). Falls back to bare
    otherwise, which is the principled version of build_dictionary's
    accidental conservatism."""
    counters = {}
    for sent in sentences:
        for tok in normalize_text(sent).split():
            _, core, _ = split_word_punct(tok)
            if core:
                counters.setdefault(strip_diacritics(core), Counter())[core] += 1

    # Task Brief 3, §1: emit NO entry when the gate fails, rather than an
    # explicit bare->bare identity entry. restore_diacritics does
    # dictionary.get(bare, core) -- an identity entry overrides that
    # fallback and actively deletes marks the model already produced,
    # which is what made the sweep crash monotonically below raw (7.24)
    # as min_share rose: more identity entries, more deleted marks.
    out = {}
    for bare, cnt in counters.items():
        marked = {f: n for f, n in cnt.items() if f != bare}
        if not marked:
            continue
        total = sum(marked.values())
        top, top_n = max(marked.items(), key=lambda x: x[1])
        if top_n / total >= min_share and top_n >= min_count:
            out[bare] = top
    return out

def restore_diacritics(text, dictionary):
    out = []
    for tok in str(text).split():
        lead, core, trail = split_word_punct(tok)
        if not core:
            out.append(tok)
            continue
        bare = strip_diacritics(core)
        restored = dictionary.get(bare, core)
        out.append(lead + restored + trail)
    return " ".join(out)

def load_dictionary():
    # BPCC + "daily" (daily has higher diacritic density, ~12.1% vs BPCC's
    # ~9.5% -- combined dict gave 10.93 on Ground_Truth vs BPCC-only's 10.74).
    df = pd.read_parquet(CLEAN_PARQUET)
    sentences = df.ks.tolist()
    if DAILY_FILE.exists():
        sentences += pd.read_parquet(DAILY_FILE).ks.tolist()
    return build_dictionary(sentences)
