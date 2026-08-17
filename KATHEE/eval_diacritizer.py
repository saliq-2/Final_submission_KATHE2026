# eval_diacritizer.py
# Standalone evaluation of the diacritizer AS a diacritizer, before it ever
# touches the MT pipeline (Task Brief 3 §3.4). Reports mark-level F1,
# word-level exact match, and -- the load-bearing metric -- F1 restricted
# to the top-20 ambiguous types from Brief 2 §3, against a frequency
# baseline. If the model doesn't beat the baseline there specifically, it
# has learned the majority class and nothing else.
import json
import torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from config import CLEAN_PARQUET, DAILY_FILE
from diacritizer import (sentence_to_base_tags, build_vocabs, DiacritizerModel,
                          Vocab, PAD, UNK, RARE_TAG, apply_tags)
from train_diacritizer import DIACRITIZER_DIR, load_corpus
from diacritic_diagnostics import raw_counters, top_ambiguous_by_output_frequency
from diacritics import strip_diacritics
from ground_truth_eval import load_ground_truth, get_hyps

def load_model_and_vocabs(variant, device):
    char_itos = json.loads((DIACRITIZER_DIR / "char_vocab.json").read_text())
    tag_itos_raw = json.loads((DIACRITIZER_DIR / "tag_vocab.json").read_text())
    # JSON round-trips tuples as lists; normalize back to tuples (except the
    # RARE_TAG string special) so they're hashable dict keys again.
    tag_itos = [tuple(t) if isinstance(t, list) else t for t in tag_itos_raw]

    char_vocab = Vocab.__new__(Vocab)
    char_vocab.itos, char_vocab.stoi = char_itos, {s: i for i, s in enumerate(char_itos)}
    tag_vocab = Vocab.__new__(Vocab)
    tag_vocab.itos, tag_vocab.stoi = tag_itos, {s: i for i, s in enumerate(tag_itos)}
    tag_vocab.itos_tuples = tag_itos

    model = DiacritizerModel(len(char_vocab), len(tag_vocab)).to(device)
    model.load_state_dict(torch.load(DIACRITIZER_DIR / f"model_{variant}.pt", map_location=device))
    model.eval()
    return model, char_vocab, tag_vocab

@torch.no_grad()
def predict_tags(model, char_vocab, tag_vocab, bases, device):
    if not bases:
        return []
    char_ids = torch.tensor([[char_vocab.stoi.get(c, char_vocab.stoi[UNK]) for c in bases]], device=device)
    pad_mask = torch.zeros(1, len(bases), dtype=torch.bool, device=device)
    logits = model(char_ids, pad_mask)
    pred_ids = logits.argmax(-1)[0].tolist()
    # RARE_TAG (the catch-all for tags below min_tag_count) is a string
    # special, not a real mark tuple -- map it to "no marks" rather than
    # let apply_tags iterate over its individual characters.
    return [t if isinstance(t, tuple) else tuple() for t in (tag_vocab.itos[i] for i in pred_ids)]

@torch.no_grad()
def predict_tags_thresholded(model, char_vocab, tag_vocab, bases, device, tau):
    """Task Brief 4 §2a: emit a mark only where softmax probability of the
    argmax tag exceeds tau; otherwise leave the position bare (empty tag).
    Gives the diacritizer an abstention mechanism, mirroring the frequency
    lexicon's ability to decline rather than guess."""
    if not bases:
        return []
    char_ids = torch.tensor([[char_vocab.stoi.get(c, char_vocab.stoi[UNK]) for c in bases]], device=device)
    pad_mask = torch.zeros(1, len(bases), dtype=torch.bool, device=device)
    logits = model(char_ids, pad_mask)
    probs = torch.softmax(logits, dim=-1)[0]  # (T, n_tags)
    conf, pred_ids = probs.max(dim=-1)
    out = []
    for c, i in zip(conf.tolist(), pred_ids.tolist()):
        if c < tau:
            out.append(tuple())
            continue
        t = tag_vocab.itos[i]
        out.append(t if isinstance(t, tuple) else tuple())
    return out

def diacritize(model, char_vocab, tag_vocab, sentence, device):
    bases, _ = sentence_to_base_tags(sentence)
    pred_tags = predict_tags(model, char_vocab, tag_vocab, bases, device)
    return apply_tags(bases, pred_tags)

def mark_level_f1(model, char_vocab, tag_vocab, dev_sentences, device):
    y_true, y_pred = [], []
    for s in dev_sentences:
        bases, gold_tags = sentence_to_base_tags(s)
        if not bases:
            continue
        pred_tags = predict_tags(model, char_vocab, tag_vocab, bases, device)
        y_true.extend(gold_tags)
        y_pred.extend(pred_tags)

    labels = sorted(set(y_true) | set(y_pred))
    label_to_id = {l: i for i, l in enumerate(labels)}
    y_true_ids = [label_to_id[t] for t in y_true]
    y_pred_ids = [label_to_id[t] for t in y_pred]

    _, _, macro_f1, _ = precision_recall_fscore_support(y_true_ids, y_pred_ids, labels=list(range(len(labels))), average="macro", zero_division=0)
    _, _, micro_f1, _ = precision_recall_fscore_support(y_true_ids, y_pred_ids, average="micro", zero_division=0)
    word_exact = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)
    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "char_position_exact_match": word_exact}

def word_level_exact_match(model, char_vocab, tag_vocab, dev_sentences, device):
    total, exact = 0, 0
    for s in dev_sentences:
        bases, gold_tags = sentence_to_base_tags(s)
        if not bases:
            continue
        pred_tags = predict_tags(model, char_vocab, tag_vocab, bases, device)
        gold_words = apply_tags(bases, gold_tags).split()
        pred_words = apply_tags(bases, pred_tags).split()
        for g, p in zip(gold_words, pred_words):
            total += 1
            if g == p:
                exact += 1
    return exact / max(total, 1)

def ambiguous_types_f1(model, char_vocab, tag_vocab, dev_sentences, freq_dict, top20_bares, device):
    """F1 (word exact match, restricted to top-20 ambiguous bare forms) for
    the model vs the frequency baseline (build_dictionary's lookup)."""
    from diacritic_restore import split_word_punct

    model_correct, freq_correct, total = 0, 0, 0
    for s in dev_sentences:
        bases, gold_tags = sentence_to_base_tags(s)
        if not bases:
            continue
        pred_tags = predict_tags(model, char_vocab, tag_vocab, bases, device)
        gold_full = apply_tags(bases, gold_tags)
        pred_full = apply_tags(bases, pred_tags)
        for g_tok, p_tok in zip(gold_full.split(), pred_full.split()):
            _, g_core, _ = split_word_punct(g_tok)
            _, p_core, _ = split_word_punct(p_tok)
            bare = strip_diacritics(g_core)
            if bare not in top20_bares:
                continue
            total += 1
            if p_core == g_core:
                model_correct += 1
            freq_guess = freq_dict.get(bare, bare)
            if freq_guess == g_core:
                freq_correct += 1
    return {
        "n": total,
        "model_exact_match": model_correct / max(total, 1),
        "freq_baseline_exact_match": freq_correct / max(total, 1),
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_sentences = json.loads((DIACRITIZER_DIR / "dev_sentences.json").read_text())

    bpcc = pd.read_parquet(CLEAN_PARQUET)
    daily = pd.read_parquet(DAILY_FILE)
    corpus_sentences = bpcc.ks.tolist() + daily.ks.tolist()
    counters = raw_counters(corpus_sentences)

    en, refs = load_ground_truth()
    hyps_raw = get_hyps("best_adapter", restore=False)
    top20 = top_ambiguous_by_output_frequency(hyps_raw, counters, refs, n=20)
    top20_bares = {row["bare"] for row in top20}
    print("top-20 ambiguous bares:", top20_bares)

    from diacritic_restore import build_dictionary
    freq_dict = build_dictionary(corpus_sentences)

    for variant in ["filtered", "weighted"]:
        model, char_vocab, tag_vocab = load_model_and_vocabs(variant, device)
        print(f"\n=== {variant} ===")
        print("mark-level F1:", mark_level_f1(model, char_vocab, tag_vocab, dev_sentences, device))
        print("word-level exact match:", word_level_exact_match(model, char_vocab, tag_vocab, dev_sentences, device))
        print("ambiguous-types (model vs freq baseline):",
              ambiguous_types_f1(model, char_vocab, tag_vocab, dev_sentences, freq_dict, top20_bares, device))

if __name__ == "__main__":
    main()
