# diacritic_diagnostics.py
# Re-measures density/coverage with the corrected 25-mark inventory
# (diacritics.py), plus new diagnostics: effective coverage, attainable
# density, variant ambiguity. Uses the FULL mark set everywhere here --
# never used for scoring (metric.py stays on the incomplete constant).
import math
import difflib
from collections import Counter
import pandas as pd
from diacritics import normalize_text, is_mark, strip_diacritics
from diacritic_restore import split_word_punct, build_dictionary

def density(texts):
    total_chars = 0
    total_marks = 0
    for t in texts:
        t = normalize_text(t)
        total_chars += len(t)
        total_marks += sum(1 for c in t if is_mark(c))
    return total_marks / max(total_chars, 1)

def raw_counters(sentences):
    """bare -> Counter of surface variants (needed for ambiguity/attainable-density)."""
    counters = {}
    for sent in sentences:
        for tok in normalize_text(sent).split():
            _, core, _ = split_word_punct(tok)
            if not core:
                continue
            counters.setdefault(strip_diacritics(core), Counter())[core] += 1
    return counters

def effective_coverage(hyps, dictionary):
    """Fraction of lookups that actually CHANGE the token (not just hit)."""
    total, changed, hit = 0, 0, 0
    for h in hyps:
        for tok in normalize_text(h).split():
            _, core, _ = split_word_punct(tok)
            if not core:
                continue
            total += 1
            bare = strip_diacritics(core)
            if bare in dictionary:
                hit += 1
                if dictionary[bare] != core:
                    changed += 1
    return {"total": total, "hit": hit, "changed": changed,
            "hit_rate": hit / max(total, 1), "effective_rate": changed / max(total, 1)}

def attainable_density(hyps, counters):
    """For each word type in hyps: does ANY marked variant exist in the
    corpus? Report the fraction that do, and the density reachable by always
    taking the single most-marked (highest mark-count) variant available."""
    types = set()
    for h in hyps:
        for tok in normalize_text(h).split():
            _, core, _ = split_word_punct(tok)
            if core:
                types.add(strip_diacritics(core))

    has_marked = 0
    # Build the "always take the most-diacritized available variant" text
    # for density estimation, word-by-word over the actual hyps.
    best_marked_dict = {}
    for bare in types:
        cnt = counters.get(bare)
        if not cnt:
            continue
        variants = list(cnt.keys())
        marked_variants = [v for v in variants if v != bare]
        if marked_variants:
            has_marked += 1
            # "most marked" = highest count of combining marks in the variant
            best = max(marked_variants, key=lambda v: sum(1 for c in v if is_mark(c)))
            best_marked_dict[bare] = best

    max_marked_hyps = []
    for h in hyps:
        out = []
        for tok in normalize_text(h).split():
            lead, core, trail = split_word_punct(tok)
            if not core:
                out.append(tok)
                continue
            bare = strip_diacritics(core)
            out.append(lead + best_marked_dict.get(bare, core) + trail)
        max_marked_hyps.append(" ".join(out))

    return {
        "types_total": len(types),
        "types_with_marked_variant": has_marked,
        "fraction_with_marked_variant": has_marked / max(len(types), 1),
        "ceiling_density": density(max_marked_hyps),
    }

def variant_ambiguity(counters):
    """Fraction of bare types with >=2 distinct marked variants, and mean
    entropy of the variant distribution over those types."""
    ambiguous = []
    for bare, cnt in counters.items():
        marked = {v: n for v, n in cnt.items() if v != bare}
        if len(marked) >= 2:
            total = sum(marked.values())
            entropy = -sum((n / total) * math.log2(n / total) for n in marked.values())
            ambiguous.append(entropy)
    n_bare_types = len(counters)
    return {
        "bare_types_total": n_bare_types,
        "ambiguous_types": len(ambiguous),
        "fraction_ambiguous": len(ambiguous) / max(n_bare_types, 1),
        "mean_entropy_over_ambiguous": (sum(ambiguous) / len(ambiguous)) if ambiguous else 0.0,
    }

def token_weighted_ambiguity(hyps, counters):
    """Same idea as variant_ambiguity, but weighted by how often each bare
    form actually occurs as a RESTORATION DECISION in model output -- the
    copula alone is one type but a huge fraction of decisions."""
    decisions = 0
    ambiguous_decisions = 0
    entropies = []
    for h in hyps:
        for tok in normalize_text(h).split():
            _, core, _ = split_word_punct(tok)
            if not core:
                continue
            bare = strip_diacritics(core)
            cnt = counters.get(bare)
            if not cnt:
                continue
            decisions += 1
            marked = {v: n for v, n in cnt.items() if v != bare}
            if len(marked) >= 2:
                ambiguous_decisions += 1
                total = sum(marked.values())
                entropy = -sum((n / total) * math.log2(n / total) for n in marked.values())
                entropies.append(entropy)
    return {
        "decisions_total": decisions,
        "ambiguous_decisions": ambiguous_decisions,
        "fraction_ambiguous_decisions": ambiguous_decisions / max(decisions, 1),
        "mean_entropy_over_ambiguous_decisions": (sum(entropies) / len(entropies)) if entropies else 0.0,
    }

def top_ambiguous_by_output_frequency(hyps, corpus_counters, refs, n=20):
    """Top-N ambiguous bare types by how often they occur in model output,
    with their variant distribution in the training corpus vs in
    Ground_Truth -- shows where corpus preference and gold preference
    diverge (grammatical conditioning, not spelling noise)."""
    freq = Counter()
    for h in hyps:
        for tok in normalize_text(h).split():
            _, core, _ = split_word_punct(tok)
            if core:
                freq[strip_diacritics(core)] += 1

    gt_counters = raw_counters(refs)

    rows = []
    for bare, n_occ in freq.most_common():
        cnt = corpus_counters.get(bare)
        if not cnt:
            continue
        marked = {v: c for v, c in cnt.items() if v != bare}
        if len(marked) < 2:
            continue
        rows.append({
            "bare": bare,
            "output_freq": n_occ,
            "corpus_variants": Counter(cnt).most_common(5),
            "gt_variants": Counter(gt_counters.get(bare, {})).most_common(5),
        })
        if len(rows) >= n:
            break
    return rows

def per_occurrence_oracle(hyps, refs, max_edit_distance=0):
    """Deliberately leaky, throwaway oracle: align each hyp's bare-token
    sequence to its ref's bare-token sequence (difflib matching blocks, so
    it tolerates length/order differences between hyp and ref), and for
    ALIGNED positions substitute the ref's exact token at that occurrence --
    not a corpus-wide winner. Unaligned tokens stay as raw hyp output.
    Answers: what could a perfectly context-aware diacritizer achieve, given
    our current translation's word choices? Never a submission.

    max_edit_distance=0: exact bare-form match only (difflib LCS blocks).
    max_edit_distance=1: after exact matching, greedily pair any remaining
    unaligned tokens whose bare forms are within Levenshtein distance 1 --
    catches inflectional/orthographic near-misses that exact matching drops
    (e.g. one-character stem differences), which the LCS blocks would skip
    entirely if surrounded by non-matching context.
    """
    import Levenshtein

    restored = []
    total_tokens = 0
    aligned_tokens = 0
    for h, r in zip(hyps, refs):
        h_toks = normalize_text(h).split()
        r_toks = normalize_text(r).split()
        h_bare = [strip_diacritics(split_word_punct(t)[1]) for t in h_toks]
        r_bare = [strip_diacritics(split_word_punct(t)[1]) for t in r_toks]

        sm = difflib.SequenceMatcher(a=h_bare, b=r_bare, autojunk=False)
        out_toks = list(h_toks)
        matched_h, matched_r = set(), set()
        for block in sm.get_matching_blocks():
            for k in range(block.size):
                hi, ri = block.a + k, block.b + k
                lead, _, trail = split_word_punct(h_toks[hi])
                _, r_core, _ = split_word_punct(r_toks[ri])
                out_toks[hi] = lead + r_core + trail
                matched_h.add(hi)
                matched_r.add(ri)

        if max_edit_distance > 0:
            unmatched_r = [ri for ri in range(len(r_toks)) if ri not in matched_r]
            for hi in range(len(h_toks)):
                if hi in matched_h or not h_bare[hi]:
                    continue
                best_ri, best_dist = None, max_edit_distance + 1
                for ri in unmatched_r:
                    if not r_bare[ri]:
                        continue
                    dist = Levenshtein.distance(h_bare[hi], r_bare[ri])
                    if dist < best_dist:
                        best_ri, best_dist = ri, dist
                if best_ri is not None:
                    lead, _, trail = split_word_punct(h_toks[hi])
                    _, r_core, _ = split_word_punct(r_toks[best_ri])
                    out_toks[hi] = lead + r_core + trail
                    matched_h.add(hi)
                    unmatched_r.remove(best_ri)

        aligned_tokens += len(matched_h)
        total_tokens += len(h_toks)
        restored.append(" ".join(out_toks))

    return restored, aligned_tokens / max(total_tokens, 1)
