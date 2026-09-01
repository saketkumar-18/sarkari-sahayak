# Sarkari Sahayak 🏛️🎤

**Voice-First Govt Services Assistant for low-literacy users** — *bolkar sarkari kaam kaise karein.*

> "kisan bill kaise bharein?" → bolke poochho → guided steps aapki bhasha me, phone par, offline.

**Live:** https://sarkari-sahayak-delta.vercel.app

## Why

India me crore log aise hain jo padhna likhna nahi jante, lekin sarkari services (PM-Kisan, Ayushman, e-Shram, PF, Aadhaar, Ujjwala, MGNREGA, Pension, Ration) ke liye online forms aur portals navigate karna padta hai. Yeh gap low-literacy users ke liye systematic exclusion hai.

Sarkari Sahayak ek **purely voice interface** hai:

- 🎤 **Bolen** — mic tap karo, apni bhasha me poochho (Hindi + 12 Indic locales, romanized Hinglish bhi chalega)
- 🧠 **Samjhein** — trained BiGRU+attention intent model, **100% on-device** (ONNX, WASM), internet zaroori nahi
- 📢 **Sunein** — curated bilingual playbook step-by-step bolke sunata hai + one-tap helpline call
- 🔒 **Privacy** — koi signup nahi, koi server nahi, sab kuch phone me

## Architecture

```
mic (Web Speech API, hi-IN + 12 locales)
  → Devanagari→roman transliteration (JS, mirrors src/translit.py)
  → BiGRU+attention intent classifier (ONNX, onnxruntime-web WASM)
    10 intents · 103K params · 410KB model · <50ms inference
  → curated bilingual playbook (data/services.json, verified helplines)
  → speechSynthesis Hindi voice reply
  → optional: Puter.js free LLM re-words steps in simpler Hindi (graceful fallback)
```

| Layer | Tech | Why |
|---|---|---|
| STT/TTS | Web Speech API | Free, on-device, 12+ Indic languages, Android Chrome native |
| Intent | BiGRU + scaled-dot attention → ONNX | 103K params — 410KB, loads on ₹5,000 phones, offline after first load |
| Knowledge | `services.json` (hand-verified, cited) | LLM-free grounded answers; zero hallucination for helpline numbers |
| LLM (optional) | Puter.js (free tier) | Only simplifies curated text; never invents facts; playbook fallback |
| ASR (upgradable) | AI4Bharat IndicWhisper/Indic-Conformer | Pipeline slot documented in `docs/upgrades.md` — swap Web Speech for whisper-web when needed |
| Delivery | Static PWA + service worker | Vercel free tier, offline-first, installable APK-able TWA |

## Results (trained model)

| Metric | Value |
|---|---|
| Test accuracy | **99.58%** |
| Test macro-F1 | **99.61%** |
| Params | 103,242 |
| Model size (ONNX, fp32) | 410 KB |
| Hindi-voice E2E accuracy (20 held-out Devanagari queries, translit → model) | **100%** |
| Browser E2E (37 assertions: 30 queries + KB + translit parity + size budget) | **37/37** |

Per-class metrics + confusion matrix: `models/metrics.json`.

## Repo layout

```
data/
  services.json         # bilingual playbooks, helplines (sources cited), fraud warnings
  queries_*.jsonl       # train/val/test splits (3200 rows, seeded)
src/
  data.py               # dataset builder (roman + Devanagari-seed translit augmentation)
  hindi_seeds.py        # spoken-Hindi seeds per service
  translit.py           # Devanagari→roman (STT→model bridge)
  model.py              # BiGRU+attention, training, ONNX export
models/
  intent_bigru.onnx     # single-file inlined model (browser artifact)
  metrics.json          # accuracy/F1/confusion/params
web/
  index.html styles.css app.js sw.js manifest.json
  model/ kb/            # browser assets
tests/                  # 31 pytest tests (data, KB safety, translit, ONNX E2E, web assets)
e2e/                    # Node E2E — real app.js + real model + KB
docs/                   # ethics, model card, upgrade notes
```

## Run locally

```bash
pip install -r requirements.txt
python src/data.py      # rebuild dataset (seeded, deterministic)
python src/model.py     # train + export ONNX + metrics
pytest tests/ -q        # 31 tests
cd web && python -m http.server 8765
cd e2e && node e2e.mjs  # 37-assertion E2E (needs: npm i onnxruntime-node)
```

## Helplines (verified 2026-09-01)

| Service | Helpline | Source |
|---|---|---|
| Ayushman Bharat PM-JAY | 14555 | nha.gov.in/contact-us |
| PM-Kisan | 155261 / 011-24300606 | pmkisan.gov.in, PIB |
| e-Shram | 14434 / 18008896811 | eshram.gov.in/helpdesk |
| EPFO | 1800118005 / 14470 | epfindia.gov.in citizen charter |
| UIDAI Aadhaar | 1947 | uidai.gov.in, PIB PRID 1952094 |
| PMUY LPG | 1906 | pmuy.gov.in |
| PDS/ration | 1967 | DoCAF/nfsa.nic.in |
| Cyber fraud | 1930 | cybercrime.gov.in |

MGNREGA aur NSAP pension helplines state-specific hain — playbook me zila karyalay route document hai.

## Ethics & safety

- **No fabrication:** helpline numbers curl-verified ya official charter se cite; LLM sirf curated text re-word karta hai, facts nahi bana sakta
- **Fraud-first:** har response me OTP/advance-fee scam warnings + 1930 cyber helpline
- **Privacy:** zero server calls after load (optional LLM explicitly user-triggered), no accounts, no analytics
- See `docs/ETHICS.md`, `docs/MODEL_CARD.md`

## Author

Saket Kumar — B.Sc. Data Science & AI, IIT Guwahati (capstone #19)

License: MIT (code); playbooks CC-BY-4.0 (translatable, shareable for NGO/CSC use).
