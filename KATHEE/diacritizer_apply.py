# diacritizer_apply.py
# Mode A (Task Brief 3 §3.5): apply the trained "weighted" diacritizer
# directly to MT output text, as a drop-in replacement for the frequency
# lexicon in diacritic_restore.py.
import torch
from diacritizer import sentence_to_base_tags, apply_tags
from eval_diacritizer import load_model_and_vocabs, predict_tags

_cache = {}

def _get_model(variant="weighted", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if variant not in _cache:
        _cache[variant] = load_model_and_vocabs(variant, device) + (device,)
    return _cache[variant]

def diacritizer_restore(text, variant="weighted"):
    model, char_vocab, tag_vocab, device = _get_model(variant)
    bases, _ = sentence_to_base_tags(text)
    if not bases:
        return text
    pred_tags = predict_tags(model, char_vocab, tag_vocab, bases, device)
    return apply_tags(bases, pred_tags)
