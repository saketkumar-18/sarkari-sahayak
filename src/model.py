"""Sarkari Sahayak — BiGRU + attention intent classifier (train/eval/ONNX export).

Architecture: Emb(64) -> BiGRU(64, bidirectional) -> scaled-dot attention pooling
-> FC(64) -> logits over 10 service intents. Small enough to run in-browser via
onnxruntime-web WASM on a $50 Android phone; trained on CPU in minutes.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import LABELS, load_split, tokenize

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MAX_LEN = 14
EMB_DIM = 64
HID_DIM = 64


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int = EMB_DIM,
                 hid_dim: int = HID_DIM, pad_idx: int = 0, n_layers: int = 1):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, batch_first=True,
                          bidirectional=True)
        self.attn = nn.Linear(2 * hid_dim, 1, bias=False)
        self.fc1 = nn.Linear(2 * hid_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, lengths=None):
        emb = self.embedding(x)                                   # (B, T, E)
        out, _ = self.gru(emb)                                    # (B, T, 2H)
        scores = self.attn(out).squeeze(-1)                       # (B, T)
        mask = (x == self.pad_idx)
        scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=1)                        # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), out).squeeze(1)  # (B, 2H)
        h = F.relu(self.fc1(pooled))
        return self.fc2(h)                                        # (B, C)


def build_vocab(split_rows: list[dict], min_count: int = 1) -> dict:
    counts: dict[str, int] = {}
    for r in split_rows:
        for tok in tokenize(r["text"]):
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= min_count and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int = MAX_LEN) -> list[int]:
    ids = [vocab.get(t, 1) for t in tokenize(text)][:max_len]
    return ids + [0] * (max_len - len(ids))


def make_batches(rows: list[dict], vocab: dict, batch_size: int, shuffle: bool,
                 label_index: dict, seed: int = 0):
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for s in range(0, len(idx), batch_size):
        batch = [rows[i] for i in idx[s:s + batch_size]]
        x = torch.tensor([encode(r["text"], vocab) for r in batch], dtype=torch.long)
        y = torch.tensor([label_index[r["label"]] for r in batch], dtype=torch.long)
        yield x, y


def train(epochs: int = 60, batch_size: int = 64, lr: float = 5e-3, seed: int = 42,
          patience: int = 12, verbose: bool = True):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_rows = load_split("train")
    val_rows = load_split("val")
    vocab = build_vocab(train_rows)
    label_index = {l: i for i, l in enumerate(LABELS)}
    n_classes = len(LABELS)

    model = BiGRUClassifier(len(vocab), n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # mild class-balanced weighting from inverse frequency
    counts = np.zeros(n_classes)
    for r in train_rows:
        counts[label_index[r["label"]]] += 1
    w = torch.tensor(np.max(counts) / (counts + 1e-9), dtype=torch.float32)

    best_val, best_state, bad = 0.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, tloss = 0, 0, 0.0
        for x, y in make_batches(train_rows, vocab, batch_size, shuffle=True,
                                 label_index=label_index, seed=epoch):
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=w)
            loss.backward()
            opt.step()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tloss += loss.item() * y.size(0)
        # eval
        val_acc, val_f1, _ = evaluate(model, val_rows, vocab, label_index)
        history.append({"epoch": epoch, "train_loss": tloss / total,
                        "train_acc": correct / total, "val_acc": val_acc, "val_f1": val_f1})
        if verbose:
            print(f"epoch {epoch:02d}  loss {tloss/total:.4f}  train_acc {correct/total:.4f}"
                  f"  val_acc {val_acc:.4f}  val_f1 {val_f1:.4f}")
        if val_f1 > best_val:
            best_val, bad = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best val_f1 {best_val:.4f})")
                break
    if best_state:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(best_state or model.state_dict(), MODELS_DIR / "intent_bigru.pt")
    with (MODELS_DIR / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    with (MODELS_DIR / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=1)
    with (MODELS_DIR / "history.json").open("w", encoding="utf-8") as b:
        json.dump(history, b, indent=1)
    return model, vocab, label_index, best_val, history


def evaluate(model, rows, vocab, label_index, batch_size: int = 128):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in make_batches(rows, vocab, batch_size, shuffle=False,
                                 label_index=label_index):
            logits = model(x)
            y_pred.extend(logits.argmax(1).tolist())
            y_true.extend(y.tolist())
    return accuracy_f1(y_true, y_pred, len(label_index))


def accuracy_f1(y_true, y_pred, n_classes):
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    acc = correct / max(1, len(y_true))
    f1s = []
    per_class = {}
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    macro_f1 = sum(f1s) / n_classes
    return acc, macro_f1, per_class


def export_onnx(model: BiGRUClassifier, vocab: dict, path: Path, max_len: int = MAX_LEN):
    """Export a single self-contained ONNX file (weights inlined, no .data sidecar)."""
    import onnx
    model.eval()
    dummy = torch.tensor([encode("kisan bill kaise bharein", vocab)], dtype=torch.long)
    tmp = path.with_suffix(".tmp.onnx")
    torch.onnx.export(
        model, dummy, tmp,
        input_names=["tokens"], output_names=["logits"],
        dynamic_axes={"tokens": {0: "batch"}},
        opset_version=15,
    )
    m = onnx.load(str(tmp))
    onnx.save_model(m, str(path), save_as_external_data=False)
    tmp.unlink(missing_ok=True)
    path.with_name(path.name.replace(".onnx", ".tmp.onnx.data")).unlink(missing_ok=True)
    tmp_data = tmp.with_name(tmp.name + ".data")
    tmp_data.unlink(missing_ok=True)
    return path


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    model, vocab, label_index, best_val, history = train()
    test_rows = load_split("test")
    acc, f1, per_class = evaluate(model, test_rows, vocab, label_index)
    print(f"TEST  acc={acc:.4f}  macro_f1={f1:.4f}")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in make_batches(test_rows, vocab, 128, False, label_index):
            preds = model(x).argmax(1).tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())
    cm = confusion_matrix(y_true, y_pred, len(LABELS))
    report = {
        "test_accuracy": round(acc, 4), "test_macro_f1": round(f1, 4),
        "best_val_f1": round(best_val, 4),
        "per_class": {LABELS[c]: v for c, v in per_class.items()},
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "history_tail": history[-5:],
        "params": count_params(model),
    }
    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    export_onnx(model, vocab, MODELS_DIR / "intent_bigru.onnx")
    print("metrics + onnx exported -> models/")
    print("params:", count_params(model))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
