"""Dataset integrity tests: splits, balance, hygiene."""
from collections import Counter

from data import LABELS, build, load_split, tokenize

EXPECTED_CLASSES = 10
PER_CLASS_TARGET = 320


def test_all_splits_exist_and_sized():
    counts = {name: len(load_split(name)) for name in ("train", "val", "test")}
    total = sum(counts.values())
    assert total >= EXPECTED_CLASSES * PER_CLASS_TARGET * 0.85
    assert counts["train"] > counts["val"] >= 200
    assert counts["test"] >= 200


def test_label_distribution_approximately_balanced():
    for name in ("train", "val", "test"):
        c = Counter(r["label"] for r in load_split(name))
        assert set(c) == set(LABELS)
        lo, hi = min(c.values()), max(c.values())
        assert hi / lo < 2.0, f"{name} imbalanced: {c}"


def test_splits_are_disjoint():
    seen = {}
    for name in ("train", "val", "test"):
        for r in load_split(name):
            key = r["text"]
            assert key not in seen, f"text {key!r} appears in {seen.get(key)} and {name}"
            seen[key] = name


def test_rows_are_wellformed():
    for name in ("train", "val", "test"):
        for r in load_split(name):
            assert r["label"] in LABELS
            toks = tokenize(r["text"])
            assert 3 <= len(toks) <= 18
            assert all(t == t.lower() for t in toks)
            assert all(t and t.strip() for t in toks)


def test_one_text_never_two_labels():
    m = {}
    for name in ("train", "val", "test"):
        for r in load_split(name):
            prev = m.setdefault(r["text"], r["label"])
            assert prev == r["label"]


def test_rebuild_is_deterministic():
    """Same seed must give identical splits (reproducibility for the report)."""
    before = [r["text"] for r in load_split("test")][:50]
    build(seed=42)
    after = [r["text"] for r in load_split("test")][:50]
    assert before == after
