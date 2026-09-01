# ASR Upgrade Path — IndicWhisper / Indic-Conformer

The brief calls for an IndicWhisper-based pipeline. v1 ships with the Web Speech
API for STT (free, on-device, 12 Indic locales, zero download) because the core
deliverable — intent understanding + voice guidance — must run on ₹5,000 Android
phones offline. This document is the drop-in upgrade plan when wider dialect
coverage is required.

## Why not IndicWhisper in-browser today
- whisper-tiny Indic fine-tunes are ~40–75M params → 40–150MB downloads; WASM
  decoding of 30s audio takes 10–60s on low-end phones. Usable, but not for the
  "instant tap-and-talk" UX of a village kiosk.
- The intent model already accepts STT text; the swap is upstream-only.

## Option A — Indic-Conformer (recommended)
`ai4bharat/indic-conformer-600m-multilingual` (11 Indic languages, CTC + RNNT):
1. Quantize to INT8 → ONNX (~300MB→~150MB) or run server-side.
2. For kiosk/NGO deployment: a single Raspberry-Pi/mini-PC local server with
   sherpa-onnx (see parismitaglobalsolutions/indicconformer-sherpa-onnx, 198MB
   Hindi build) serving LAN devices — no internet at all.
3. Feed the CTC transcript into the same `normalize()` → intent ONNX pipeline —
   zero changes downstream (this repo's tests validate that contract).

## Option B — whisper-small Indic fine-tune in worker
1. `@xenova/transformers` (transformers.js) runs whisper-tiny/small GGUF/WASM in a
   Web Worker; load lazily after first visit (service-worker precache optional).
2. Use `language="hi"`, `task="transcribe"`; output Devanagari → existing
   `devaToRoman()` handles the rest.
3. Fallback ladder in app.js: try worker ASR → Web Speech API → typing box.

## Contract to preserve (tested in tests/test_model_onnx.py)
- Any ASR upgrade must output either Devanagari (goes through translit) or roman
  text; the encoder + model are unchanged.
- models/metrics.json thresholds (acc ≥ 0.95) gate any retrained model.

## Verified references
- HF: ai4bharat/indic-conformer-600m-multilingual (checked 2026-09-01)
- sherpa-onnx Indic build: indicconformer-sherpa-onnx (Hindi, 198MB, CTC)
