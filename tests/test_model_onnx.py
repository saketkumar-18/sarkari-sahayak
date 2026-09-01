"""ONNX + E2E intent tests: transliterated Hindi voice queries through the real graph."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from data import LABELS, tokenize
from translit import HINDI_E2E, normalize

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WEB_MODEL = ROOT / "web" / "model"
MAX_LEN = 14


def _encode(text, vocab):
    ids = [vocab.get(t, 1) for t in tokenize(normalize(text))][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))


def _session(path: Path):
    assert path.exists(), f"missing model artifact: {path}"
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _predict(sess, vocab, labels, text):
    x = np.array([_encode(text, vocab)], dtype=np.int64)
    logits = sess.run(None, {"tokens": x})[0][0]
    return labels[int(np.argmax(logits))], float(np.max(logits))


@pytest.fixture(scope="module")
def artifacts():
    vocab = json.loads((MODELS / "vocab.json").read_text(encoding="utf-8"))
    labels = json.loads((MODELS / "labels.json").read_text(encoding="utf-8"))
    return vocab, labels


def test_model_files_present():
    for f in ("intent_bigru.pt", "intent_bigru.onnx", "vocab.json", "labels.json", "metrics.json"):
        assert (MODELS / f).exists(), f
    for f in ("intent_bigru.onnx", "vocab.json", "labels.json"):
        assert (WEB_MODEL / f).exists(), f"web asset missing: {f}"
    assert (ROOT / "models" / "intent_bigru.onnx.data").exists() or \
        (MODELS / "intent_bigru.onnx").stat().st_size > 100_000


def test_repo_and_web_vocab_identical():
    a = (MODELS / "vocab.json").read_bytes()
    b = (WEB_MODEL / "vocab.json").read_bytes()
    assert a == b


def test_labels_match_training_classes(artifacts):
    _, labels = artifacts
    assert labels == LABELS


def test_metrics_threshold(artifacts):
    m = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))
    assert m["test_accuracy"] >= 0.95, "regression: test accuracy below 95%"
    assert m["test_macro_f1"] >= 0.95
    assert m["params"] < 500_000, "model too heavy for low-end phones"
    assert len(m["confusion_matrix"]) == len(LABELS)


def _hindi_e2e_accuracy(sess, vocab, labels):
    preds = [_predict(sess, vocab, labels, t)[0] for t, _ in HINDI_E2E]
    correct = sum(p == lab for p, (_, lab) in zip(preds, HINDI_E2E))
    return correct / len(HINDI_E2E)


def test_web_model_hindi_voice_e2e(artifacts):
    """THE core promise: Devanagari STT text -> translit -> in-browser model -> right service."""
    vocab, labels = artifacts
    sess = _session(WEB_MODEL / "intent_bigru.onnx")
    acc = _hindi_e2e_accuracy(sess, vocab, labels)
    assert acc >= 0.85, f"Hindi voice E2E accuracy too low: {acc:.2f}"


def test_repo_model_matches_web_model(artifacts):
    """Both artifacts must agree on every E2E sample (same graph, single source of truth)."""
    vocab, labels = artifacts
    try:
        repo_sess = _session(MODELS / "intent_bigru.onnx")
    except Exception:
        pytest.skip("repo onnx external-data pair not loadable here")
    web_sess = _session(WEB_MODEL / "intent_bigru.onnx")
    for text, _ in HINDI_E2E:
        a = _predict(repo_sess, vocab, labels, text)[0]
        b = _predict(web_sess, vocab, labels, text)[0]
        assert a == b, f"repo/web disagree on {text!r}: {a} vs {b}"


def test_roman_queries_direct(artifacts):
    vocab, labels = artifacts
    sess = _session(WEB_MODEL / "intent_bigru.onnx")
    cases = [
        ("kisan bill kaise bharein", "pm_kisan"),
        ("bhaiya mujhe sarkari kaam me madad chahiye", "general"),
        ("gas subsidy ka paisa nahi aaya", "ujjwala"),
        ("ration card me naam jodna hai", "ration"),
        ("pf ka paisa nikalna hai", "epfo"),
        ("mera aadhar card kho gaya", "aadhaar"),
    ]
    for text, want in cases:
        got, _ = _predict(sess, vocab, labels, text)
        assert got == want, f"{text!r} -> {got}, want {want}"
