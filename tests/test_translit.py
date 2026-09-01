"""Transliteration tests — the STT-to-model bridge."""
from translit import HINDI_E2E, deva_to_roman, is_deva, normalize


def test_is_deva():
    assert is_deva("किसान बिल")
    assert not is_deva("kisan bill")
    assert not is_deva("123")


def test_basic_mappings():
    # inherent-vowel rule: bare consonants get trailing 'a' (kisana, bila, adhara)
    assert deva_to_roman("किसान") == "kisana"
    assert deva_to_roman("नाम") == "nama"
    assert "अ" not in deva_to_roman("आधार")
    assert is_deva("आधार") and normalize("आधार") == "adhara"


def test_virama_suppressed():
    # बिल = ब(ba) ि(i) ल(la) -> "bila" with inherent a; virama case:
    assert "्" not in deva_to_roman("किस्त")


def test_normalize_strips_punct_and_spaces():
    assert normalize("किसान, बिल!") == "kisana bila"
    assert normalize("  PF   ka paisa  ") == "pf ka paisa"
    assert normalize("") == ""


def test_no_devanagari_survives_normalization():
    for text, _ in HINDI_E2E:
        out = normalize(text)
        assert out, text
        assert not is_deva(out), f"Devanagari leaked: {text} -> {out}"
        assert out == out.lower()


def test_e2e_set_wellformed():
    labels = {lab for _, lab in HINDI_E2E}
    assert len(HINDI_E2E) >= 20
    assert len(labels) == 10
