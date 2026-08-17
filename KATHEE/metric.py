# metric.py
#
# DELIBERATE: this module uses KashmiriNormalizer.constants.KASHMIRI_DIACRITICS
# (11 marks) even though the corpus actually uses 25 -- see diacritics.py for
# the complete, data-derived inventory. Do NOT "fix" this here. This module
# exists to mirror the organizers' real scoring script exactly, incomplete
# constant and all; changing it would make local scores stop predicting the
# leaderboard and cost us our only calibration point (proxy 7.24 -> real
# 9.34). The complete inventory is used only in diacritic_restore.py,
# density/coverage measurement code, and Phase 2 diacritizer code.
import sacrebleu, numpy as np
from KashmiriNormalizer import KashmiriNormalizer

# The organizers' local score checker normalizes both hyp and ref through
# KashmiriNormalizer before scoring (canonicalizes character-variant
# encodings, digits, punctuation spacing). chrF++ is character-level, so
# skipping this step makes local scores non-comparable to the real one.
_normalizer = KashmiriNormalizer()

def _normalize(text):
    return _normalizer.normalize("" if text is None else str(text))

def bleu(hyps, refs):
    return sacrebleu.corpus_bleu(hyps, [refs]).score

def chrfpp(hyps, refs):
    return sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score

def length_ratio(hyps, refs):
    # hyp/ref word-count ratio, summed over the corpus (matches how
    # sacrebleu's brevity penalty compares tokenized lengths). <0.95 means
    # the brevity penalty is likely eating BLEU -- a decoding/length-control
    # problem, distinct from an orthography (chrF++) problem.
    hyp_len = sum(len(str(h).split()) for h in hyps)
    ref_len = sum(len(str(r).split()) for r in refs)
    return hyp_len / max(ref_len, 1)

def kathe_score(hyps, refs):
    hyps_n = [_normalize(h) for h in hyps]
    refs_n = [_normalize(r) for r in refs]
    b = bleu(hyps_n, refs_n)
    c = chrfpp(hyps_n, refs_n)
    g = 0.0 if b <= 0 or c <= 0 else float(np.sqrt(b * c))
    lr = length_ratio(hyps_n, refs_n)
    return {"bleu": b, "chrf++": c, "geo_mean": g, "length_ratio": lr}

def _geo_mean(hyps_n, refs_n):
    b = bleu(hyps_n, refs_n)
    c = chrfpp(hyps_n, refs_n)
    return 0.0 if b <= 0 or c <= 0 else float(np.sqrt(b * c))

def paired_bootstrap(hyps_a, hyps_b, refs, n=1000, seed=0):
    """
    Paired bootstrap significance test between system A and system B against
    the same references. Resamples sentence INDICES with replacement (same
    indices for both systems each draw) so sentence-difficulty variance
    cancels -- much more sensitive than comparing two independent CIs.
    Returns mean(delta), the 95% CI on delta, and P(A > B), where
    delta = geo_mean(A) - geo_mean(B) on each resample.
    """
    hyps_a_n = [_normalize(h) for h in hyps_a]
    hyps_b_n = [_normalize(h) for h in hyps_b]
    refs_n = [_normalize(r) for r in refs]
    n_sent = len(refs_n)
    rng = np.random.default_rng(seed)

    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_sent, size=n_sent)
        a_sub = [hyps_a_n[j] for j in idx]
        b_sub = [hyps_b_n[j] for j in idx]
        r_sub = [refs_n[j] for j in idx]
        deltas[i] = _geo_mean(a_sub, r_sub) - _geo_mean(b_sub, r_sub)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean_delta": float(deltas.mean()),
        "ci_2.5": float(lo),
        "ci_97.5": float(hi),
        "P(A>B)": float((deltas > 0).mean()),
    }
