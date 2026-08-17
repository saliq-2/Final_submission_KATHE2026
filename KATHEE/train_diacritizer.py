# train_diacritizer.py
import sys
import json
import random
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import CLEAN_PARQUET, DAILY_FILE, WORK_DIR, SEED
from diacritizer import (sentence_to_base_tags, sentence_density, build_vocabs,
                          DiacritizerModel, collate, PAD)

DIACRITIZER_DIR = WORK_DIR / "diacritizer"
DIACRITIZER_DIR.mkdir(exist_ok=True)

class SentDataset(Dataset):
    def __init__(self, sentences, weights=None):
        self.items = []
        for i, s in enumerate(sentences):
            bases, tags = sentence_to_base_tags(s)
            if not bases:
                continue
            w = weights[i] if weights is not None else 1.0
            self.items.append((bases, tags, w))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def load_corpus():
    bpcc = pd.read_parquet(CLEAN_PARQUET)
    daily = pd.read_parquet(DAILY_FILE)
    return bpcc.ks.tolist() + daily.ks.tolist()

def split_train_dev(sentences, dev_frac=0.05, seed=SEED):
    rng = random.Random(seed)
    idx = list(range(len(sentences)))
    rng.shuffle(idx)
    n_dev = int(len(sentences) * dev_frac)
    dev_idx = set(idx[:n_dev])
    train = [sentences[i] for i in idx if i not in dev_idx]
    dev = [sentences[i] for i in idx if i in dev_idx]
    return train, dev

def train_one(variant, char_vocab, tag_vocab, train_sentences, dev_sentences,
              device, epochs=6, batch_size=128, lr=3e-4):
    if variant == "filtered":
        weights = None
        train_sentences = [s for s in train_sentences if sentence_density(s) >= 0.12]
    elif variant == "weighted":
        weights = [sentence_density(s) for s in train_sentences]
    else:
        raise ValueError(variant)

    print(f"[{variant}] train sentences: {len(train_sentences)}")
    ds = SentDataset(train_sentences, weights)
    dev_ds = SentDataset(dev_sentences)

    def coll(batch):
        return collate(batch, char_vocab, tag_vocab)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model = DiacritizerModel(len(char_vocab), len(tag_vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for char_ids, tag_ids, pad_mask, w in loader:
            char_ids, tag_ids, pad_mask, w = char_ids.to(device), tag_ids.to(device), pad_mask.to(device), w.to(device)
            logits = model(char_ids, pad_mask)  # (B, T, n_tags)
            loss_per_tok = ce(logits.transpose(1, 2), tag_ids)  # (B, T)
            valid = (tag_ids != -100).float()
            loss_per_seq = (loss_per_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            loss = (loss_per_seq * w).sum() / w.sum().clamp(min=1e-8)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        dev_loss, dev_correct, dev_total = 0.0, 0, 0
        with torch.no_grad():
            for char_ids, tag_ids, pad_mask, w in dev_loader:
                char_ids, tag_ids, pad_mask = char_ids.to(device), tag_ids.to(device), pad_mask.to(device)
                logits = model(char_ids, pad_mask)
                loss_per_tok = ce(logits.transpose(1, 2), tag_ids)
                valid = (tag_ids != -100)
                dev_loss += (loss_per_tok * valid.float()).sum().item()
                pred = logits.argmax(-1)
                dev_correct += ((pred == tag_ids) & valid).sum().item()
                dev_total += valid.sum().item()
        print(f"[{variant}] epoch {epoch+1}/{epochs}  train_loss={total_loss/n_batches:.4f}  "
              f"dev_loss={dev_loss/dev_total:.4f}  dev_tag_acc={dev_correct/dev_total:.4f}")

    torch.save(model.state_dict(), DIACRITIZER_DIR / f"model_{variant}.pt")
    return model

def main():
    sentences = load_corpus()
    train_sentences, dev_sentences = split_train_dev(sentences)
    print(f"total={len(sentences)} train_pool={len(train_sentences)} dev={len(dev_sentences)}")

    char_vocab, tag_vocab = build_vocabs(train_sentences)
    print(f"char vocab: {len(char_vocab)}  tag vocab: {len(tag_vocab)}")
    (DIACRITIZER_DIR / "char_vocab.json").write_text(json.dumps(char_vocab.itos, ensure_ascii=False))
    (DIACRITIZER_DIR / "tag_vocab.json").write_text(json.dumps(tag_vocab.itos, ensure_ascii=False))
    (DIACRITIZER_DIR / "dev_sentences.json").write_text(json.dumps(dev_sentences, ensure_ascii=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    variants = sys.argv[1:] if len(sys.argv) > 1 else ["filtered", "weighted"]
    for variant in variants:
        train_one(variant, char_vocab, tag_vocab, train_sentences, dev_sentences, device)

if __name__ == "__main__":
    main()
