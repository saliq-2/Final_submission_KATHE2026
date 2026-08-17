# diacritizer.py
# Phase 2 (NEXT_STEPS.md P2.1): per-character tagging diacritizer, not
# seq2seq. For each base character, predict the combining marks that follow
# it. Output length == input length by construction; alignment is
# monotonic; the model cannot drop or hallucinate words -- all real risks
# for seq2seq on 7-word sentences.
import torch
import torch.nn as nn
from diacritics import normalize_text, is_mark, HONORIFICS

PAD, UNK = "<pad>", "<unk>"
RARE_TAG = "<rare>"

def sentence_to_base_tags(sentence):
    """Split a sentence into (base_chars, tags) where tags[i] is the tuple
    of marks that followed base_chars[i]. Honorifics are excluded from tags
    entirely (mapped to nothing) -- they're named-entity content, never to
    be predicted, per Task Brief 3 §3.1."""
    s = normalize_text(sentence)
    bases, tags = [], []
    for c in s:
        if is_mark(c):
            if c in HONORIFICS:
                continue
            if tags:
                tags[-1] = tags[-1] + (c,)
        else:
            bases.append(c)
            tags.append(tuple())
    return bases, tags

def sentence_density(sentence):
    s = normalize_text(sentence)
    if not s:
        return 0.0
    return sum(1 for c in s if is_mark(c)) / len(s)

class Vocab:
    def __init__(self, items, specials):
        self.itos = list(specials) + list(items)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, item):
        return self.stoi.get(item, self.stoi[UNK if UNK in self.stoi else RARE_TAG])

def build_vocabs(sentences, min_tag_count=20):
    from collections import Counter
    char_counter, tag_counter = Counter(), Counter()
    for s in sentences:
        bases, tags = sentence_to_base_tags(s)
        char_counter.update(bases)
        tag_counter.update(tags)

    chars = [c for c, _ in char_counter.most_common()]
    tags = [t for t, n in tag_counter.most_common() if n >= min_tag_count and t != tuple()]

    char_vocab = Vocab(chars, [PAD, UNK])
    # empty tuple ("no marks") is always index right after specials, guaranteed present
    tag_vocab = Vocab([tuple()] + tags, [RARE_TAG])
    return char_vocab, tag_vocab

class DiacritizerModel(nn.Module):
    def __init__(self, n_chars, n_tags, d_model=256, n_layers=6, n_heads=8, dim_ff=1024, max_len=600):
        super().__init__()
        self.embed = nn.Embedding(n_chars, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_tags)

    def forward(self, char_ids, pad_mask):
        # char_ids: (B, T), pad_mask: (B, T) True where PAD
        pos = torch.arange(char_ids.size(1), device=char_ids.device).unsqueeze(0)
        x = self.embed(char_ids) + self.pos_embed(pos)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.classifier(x)  # (B, T, n_tags)

def collate(batch, char_vocab, tag_vocab):
    max_len = max(len(b[0]) for b in batch)
    B = len(batch)
    char_ids = torch.zeros(B, max_len, dtype=torch.long)
    tag_ids = torch.full((B, max_len), -100, dtype=torch.long)  # -100 = ignore in loss
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    weights = torch.zeros(B, dtype=torch.float32)
    for i, (bases, tags, weight) in enumerate(batch):
        L = len(bases)
        char_ids[i, :L] = torch.tensor([char_vocab.encode(c) for c in bases], dtype=torch.long)
        tag_ids[i, :L] = torch.tensor([tag_vocab.encode(t) for t in tags], dtype=torch.long)
        pad_mask[i, :L] = False
        weights[i] = weight
    return char_ids, tag_ids, pad_mask, weights

def apply_tags(bases, tags):
    out = []
    for b, t in zip(bases, tags):
        out.append(b)
        out.extend(t)
    return "".join(out)
