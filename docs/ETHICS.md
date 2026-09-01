# ETHICS & RESPONSIBLE DEPLOYMENT — Sarkari Sahayak

## 1. Who this is for
Low-literacy and semi-literate users (aanganwadi, kisan, mazdoor, buzurg) who cannot
navigate text-heavy government portals. Voice-first by design: every action has a
spoken output, reading is never required.

## 2. Data ethics
- Training data is **synthetic, authored in-house** (src/data.py + hindi_seeds.py).
  No real user recordings are collected — the app never uploads audio anywhere.
- No PII is stored. No accounts. No analytics. No server logs (static hosting only).
- At inference, speech stays on-device: Web Speech API processes locally on Android
  Chrome; the intent model runs in WASM locally; only optional AI-simplify sends
  curated text (never the user's raw voice or name) to Puter.js, and only when the
  user taps that button.

## 3. Truthfulness
- Playbooks are hand-authored from official portals and cite sources per service.
- Helpline numbers were curl-verified against official pages on 2026-09-01
  (NHA 14555, PMUY 1906) or cited from official documents (EPFO citizen charter,
  UIDAI/PIB, e-Shram helpdesk, PM-Kisan portal/PIB).
- The optional LLM is **constrained**: it may only re-word the given playbook text
  and is forbidden from inventing numbers/URLs/rules in the prompt. If it fails or
  times out, the curated playbook is spoken verbatim. This is the anti-hallucination
  design: the model of record for facts is the JSON, not the LLM.

## 4. Fraud protection (a core feature, not a disclaimer)
Low-literacy users are the primary targets of OTP/advance-fee/lottery scams. Every
response therefore carries:
- "card banwana muft hai — paise maangnewa thag hai"
- "OTP kisi ko nahi batayen"
- 1930 cyber-fraud helpline front and centre.

## 5. Limitations honestly stated
- **12 locales via Web Speech STT**, but the intent model itself understands
  Hindi/Hinglish (Devanagari + roman). Other-language transcripts (Marathi, Bangla…)
  are transliterated only if Devanagari; otherwise the typed-box fallback and general
  playbook apply. Full multilingual coverage needs the IndicWhisper upgrade path
  (docs/upgrades.md).
- **9 services curated**, general route for the rest. Not a substitute for
  government grievance systems (CPGRAMS/1912 pointers included in playbooks).
- **MGNREGA/NSAP helplines are state-specific** — playbook routes to zila karyalay
  and never invents a number.
- Web Speech API quality varies by device/Chrome version; mic permission is a
  user-gated browser feature (we explain the denial in plain Hindi).

## 6. Accessibility
- WCAG-AA contrast palette, 128px touch target, no reading required for core flow.
- Speech rate 0.95, Hindi voice preferred; every spoken step is also on screen
  (for semi-literate users and helpers).
- Works offline after first load (service worker) — patchy-connectivity villages.

## 7. If deployed for real (NGO/CSC guidance)
- Pair with a human helper (CSC/VLE) for form-filling — this app advises, it does
  not submit forms or handle money. Ever.
- No paid tier, no ads, no data sale — ever. MIT + CC-BY-4.0.
