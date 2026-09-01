# MODEL CARD — Sarkari Sahayak Intent Classifier

## Model details
- **Architecture:** Embedding(64) → BiGRU (hidden 64, 1 layer, bidirectional) →
  scaled-dot attention pooling → FC(64) → 10 logits
- **Params:** 103,242 · **ONNX fp32:** 410 KB · opset 15, dynamic batch axis
- **Runtime:** onnxruntime-web 1.17.3 (WASM) in browser; CPU/ort-node in CI tests

## Intended use
Map a spoken/written user query (Hindi Devanagari via STT, or romanized Hinglish)
to one of 10 govt-service intents: pm_kisan, ayushman, eshram, epfo, aadhaar,
ujjwala, nrega, pension, ration, general. Output drives a curated bilingual
playbook. **Not** a form-filler, not a benefits-eligibility decider.

## Training data
- 3,200 synthetic rows (seed=42, deterministic): 240/class roman Hinglish composites
  (fillers, districts, question suffixes, romanization typos) + 80/class
  transliterated Devanagari seeds (the exact STT distribution at inference).
- Splits 70/15/15, class-balanced (±10%), globally deduped, no split leakage
  (tested).

## Performance
- Test accuracy **99.58%**, macro-F1 **99.61%** (models/metrics.json, full
  per-class P/R/F1 + confusion matrix).
- Held-out Hindi-voice E2E (20 Devanagari queries unseen in training):
  **100%** correct intent (translit → ONNX).
- Browser E2E: 37/37 assertions (30 queries across both scripts, KB wiring,
  JS↔Python translit parity, <1MB size budget).

## Limitations & bias
- Trained on authored Hindi/Hinglish; dialectal Marathi/Bhojpuri-STT outputs outside
  Devanagari fall to `general` (safe fallback with UMANG route), never a wrong
  service playbook.
- Synthetic data reflects the authors' phrasing priors; real-world phrasing may
  differ. The confidence badge is always shown; <80% confidence prefixes a caution.
- No adversarial hardening; input is normalized to [a-z0-9 ], so prompt-injection
  via speech cannot reach the optional LLM (which itself only re-words curated
  text).

## Upgrade path (docs/upgrades.md)
IndicWhisper/Indic-Conformer (AI4Bharat) for wider dialect coverage; same intent
model consumed the roman text unchanged.
