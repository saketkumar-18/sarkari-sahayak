"""Devanagari -> roman (Hinglish) transliterator.

Web Speech API (hi-IN) returns Devanagari text; the intent model was trained
on romanized queries. This module converts Devanagari speech transcripts into
the model's roman token space. The JS port in web/app.js mirrors this logic
exactly (same tables, same inherent-vowel rule) — tests assert parity on the
same sentence set.
"""
from __future__ import annotations

CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "ळ": "l", "ऱ": "r", "ज़": "z", "फ़": "f",
}
VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri",
    "ऍ": "e", "ऑ": "o",
}
MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
SIGNS = {"ं": "n", "ँ": "n", "ः": "h", "ऽ": "a"}
VIRAMA = "्"
NUKTA = "़"
DIGITS = {"०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
          "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"}


def is_deva(text: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x097F for c in text)


def deva_to_roman(s: str) -> str:
    out = []
    chars = list(s)
    i = 0
    n = len(chars)
    while i < n:
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < n else ""
        if ch == VIRAMA or ch == NUKTA:
            i += 1
            continue
        if ch in CONS:
            base = CONS[ch]
            # inherent 'a' unless a matra/virama follows
            if nxt in MATRAS or nxt == VIRAMA:
                out.append(base)
            else:
                out.append(base + "a")
        elif ch in VOWELS:
            out.append(VOWELS[ch])
        elif ch in MATRAS:
            out.append(MATRAS[ch])
        elif ch in SIGNS:
            out.append(SIGNS[ch])
        elif ch in DIGITS:
            out.append(DIGITS[ch])
        elif ch == "।" or ch == "॥":
            out.append(" ")
        elif 0x0900 <= ord(ch) <= 0x097F:
            pass  # stray marks dropped
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def normalize(text: str) -> str:
    import re
    text = deva_to_roman(text) if is_deva(text) else text
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# Hand-crafted real-world Hindi voice queries (Devanagari) used as an
# end-to-end test set: STT-style text -> translit -> ONNX model.
HINDI_E2E = [
    ("किसान बिल कैसे भरें", "pm_kisan"),
    ("पीएम किसान का पैसा नहीं आया", "pm_kisan"),
    ("किसान की किस्त कैसे चेक करूं", "pm_kisan"),
    ("आयुष्मान कार्ड कैसे बनवाएं", "ayushman"),
    ("आयुष्मान कार्ड से इलाज कैसे कराएं", "ayushman"),
    ("ई-श्रम कार्ड कैसे बनवाएं", "eshram"),
    ("श्रमिक कार्ड कहां से बनेगा", "eshram"),
    ("पीएफ का पैसा कैसे निकलवाएं", "epfo"),
    ("यूएएन नंबर कैसे निकालें", "epfo"),
    ("आधार में नाम कैसे बदलें", "aadhaar"),
    ("आधार कार्ड खो गया क्या करें", "aadhaar"),
    ("गैस कनेक्शन कैसे बनवाएं", "ujjwala"),
    ("गैस की सब्सिडी नहीं मिली", "ujjwala"),
    ("मनरेगा में काम कैसे कराएं", "nrega"),
    ("जॉब कार्ड कैसे बनवाएं", "nrega"),
    ("बुढ़ापा पेंशन कैसे बनवाएं", "pension"),
    ("विधवा पेंशन कैसे मिलेगी", "pension"),
    ("राशन कार्ड कैसे बनवाएं", "ration"),
    ("राशन में नहीं मिल रहा क्या करें", "ration"),
    ("सरकारी योजनाओं के बारे में जानना है", "general"),
]

if __name__ == "__main__":
    for text, _label in HINDI_E2E:
        print(f"{text}  ->  {normalize(text)}")
