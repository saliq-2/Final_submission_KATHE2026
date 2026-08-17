# diacritizer_gated.py
# Task Brief 4 §2: confidence-gated diacritizer (pure and hybrid).
import torch
from diacritizer import sentence_to_base_tags, apply_tags
from diacritizer_apply import _get_model
from diacritic_restore import split_word_punct, strip_diacritics

@torch.no_grad()
def _sentence_predictions(text, variant="weighted"):
    """Returns (bases, pred_tags, confidences) for one sentence -- confidence
    per base-character position, so callers can gate at whatever granularity
    they need (character or token, via min-over-span)."""
    model, char_vocab, tag_vocab, device = _get_model(variant)
    bases, _ = sentence_to_base_tags(text)
    if not bases:
        return [], [], []
    char_ids = torch.tensor([[char_vocab.stoi.get(c, char_vocab.stoi["<unk>"]) for c in bases]], device=device)
    pad_mask = torch.zeros(1, len(bases), dtype=torch.bool, device=device)
    logits = model(char_ids, pad_mask)
    probs = torch.softmax(logits, dim=-1)[0]
    conf, pred_ids = probs.max(dim=-1)
    tags = []
    for i in pred_ids.tolist():
        t = tag_vocab.itos[i]
        tags.append(t if isinstance(t, tuple) else tuple())
    return bases, tags, conf.tolist()

def diacritizer_restore_thresholded(text, tau, variant="weighted"):
    """§2a: pure thresholded diacritizer -- emit a mark only where confident,
    else leave that base char bare."""
    bases, tags, confs = _sentence_predictions(text, variant)
    if not bases:
        return text
    gated_tags = [t if c >= tau else tuple() for t, c in zip(tags, confs)]
    return apply_tags(bases, gated_tags)

def hybrid_restore_gated(text, freq_dict, tau, variant="weighted"):
    """§2b: proper 3-branch hybrid --
    1. lexicon has a confident entry -> use it
    2. lexicon abstains AND diacritizer confidence (min over the token's
       base-char positions) > tau -> use the diacritizer
    3. otherwise -> leave the token as the model produced it (both abstain)
    """
    bases, tags, confs = _sentence_predictions(text, variant)
    if not bases:
        return text
    diac_full = apply_tags(bases, tags)
    diac_words = diac_full.split()

    # per-token min confidence: need base-char index ranges per token.
    # Recompute token boundaries over the (space-joined) base sequence by
    # re-splitting on whitespace positions within `bases`.
    token_conf = []
    cur = []
    for b, c in zip(bases, confs):
        if b == " ":
            if cur:
                token_conf.append(min(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
        token_conf.append(min(cur))

    out_words = []
    raw_words = text.split()
    for i, tok in enumerate(raw_words):
        lead, core, trail = split_word_punct(tok)
        if not core:
            out_words.append(tok)
            continue
        bare = strip_diacritics(core)
        if bare in freq_dict:
            out_words.append(lead + freq_dict[bare] + trail)
        elif i < len(diac_words) and i < len(token_conf) and token_conf[i] >= tau:
            _, d_core, _ = split_word_punct(diac_words[i])
            out_words.append(lead + d_core + trail)
        else:
            out_words.append(tok)
    return " ".join(out_words)
