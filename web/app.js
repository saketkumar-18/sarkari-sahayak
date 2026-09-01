/* Sarkari Sahayak — voice-first govt services assistant.
 * Pipeline: mic (Web Speech API, any Indic locale)
 *   -> Devanagari->roman translit (for hi input)
 *   -> ONNX BiGRU+attention intent classifier (onnxruntime-web WASM, on-device)
 *   -> curated bilingual playbook (kb/services.json, verified helplines)
 *   -> speechSynthesis voice reply
 * Optional: Puter.js free AI to re-word the playbook in simpler spoken Hindi.
 */
"use strict";

const ort = window.ort;

/* ---------------- config ---------------- */
const MODEL_URL = "model/intent_bigru.onnx";
const VOCAB_URL = "model/vocab.json";
const LABELS_URL = "model/labels.json";
const KB_URL = "kb/services.json";
const MAX_LEN = 14;
const LLM_TIMEOUT_MS = 12000;

/* ---------------- transliteration (mirror of src/translit.py) ---------------- */
const T_CONS = {
  "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
  "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
  "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
  "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
  "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m",
  "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh",
  "स": "s", "ह": "h", "ळ": "l", "ऱ": "r", "ज़": "z", "फ़": "f",
};
const T_VOWELS = {
  "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
  "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri",
  "ऍ": "e", "ऑ": "o",
};
const T_MATRAS = {
  "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
  "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
};
const T_SIGNS = { "ं": "n", "ँ": "n", "ः": "h", "ऽ": "a" };
const T_DIGITS = { "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
  "५": "5", "६": "6", "७": "7", "८": "8", "९": "9" };

function isDeva(s) {
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (c >= 0x0900 && c <= 0x097f) return true;
  }
  return false;
}

function devaToRoman(s) {
  const out = [];
  const chars = [...s];
  for (let i = 0; i < chars.length; i++) {
    const ch = chars[i];
    const nxt = chars[i + 1] || "";
    if (ch === "्" || ch === "़") continue;               // virama / nukta
    if (T_CONS[ch] !== undefined) {
      const base = T_CONS[ch];
      out.push((T_MATRAS[nxt] !== undefined || nxt === "्") ? base : base + "a");
    } else if (T_VOWELS[ch] !== undefined) out.push(T_VOWELS[ch]);
    else if (T_MATRAS[ch] !== undefined) out.push(T_MATRAS[ch]);
    else if (T_SIGNS[ch] !== undefined) out.push(T_SIGNS[ch]);
    else if (T_DIGITS[ch] !== undefined) out.push(T_DIGITS[ch]);
    else if (ch === "।" || ch === "॥") out.push(" ");
    else {
      const c = ch.codePointAt(0);
      if (c >= 0x0900 && c <= 0x097f) continue;            // stray marks
      out.push(ch);
    }
  }
  return out.join("");
}

function normalize(text) {
  let t = isDeva(text) ? devaToRoman(text) : text;
  t = t.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
  return t;
}

/* ---------------- state ---------------- */
const S = {
  session: null, vocab: null, labels: null, kb: null,
  recognizing: false, recognition: null, lastResult: null, voice: null,
};

/* ---------------- dom ---------------- */
const $ = (id) => document.getElementById(id);
const micBtn = $("micBtn"), statusEl = $("status"), transcriptEl = $("transcript");
const resultEl = $("result"), svcName = $("svcName"), stepsEl = $("steps");
const helplineNum = $("helplineNum"), callBtn = $("callBtn"), portalBtn = $("portalBtn");
const docsList = $("docsList"), faqList = $("faqList"), fraudList = $("fraudList");
const chips = $("chips"), typeBox = $("typeBox"), typeGo = $("typeGo");
const speakAllBtn = $("speakAll"), aiAnswerEl = $("aiAnswer");

function toast(msg) {
  const t = $("toast") || (() => {
    const d = document.createElement("div"); d.id = "toast"; document.body.appendChild(d); return d;
  })();
  t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 2600);
}

function setStatus(text, cls = "ready") {
  statusEl.textContent = text;
  statusEl.className = "status " + cls;
}

/* ---------------- model ---------------- */
async function loadAssets() {
  setStatus("Taiyari ho rahi hai…");
  const [mb, vb, lb, kb] = await Promise.all([
    fetch(MODEL_URL).then((r) => r.arrayBuffer()),
    fetch(VOCAB_URL).then((r) => r.json()),
    fetch(LABELS_URL).then((r) => r.json()),
    fetch(KB_URL).then((r) => r.json()),
  ]);
  S.vocab = vb; S.labels = lb; S.kb = kb;
  S.session = await ort.InferenceSession.create(mb, { executionProviders: ["wasm"] });
  setStatus("Taiyar hain! Tap karein aur bolen");
}

function encode(text) {
  const ids = normalize(text).split(" ").slice(0, MAX_LEN)
    .map((t) => (t in S.vocab ? S.vocab[t] : 1));
  while (ids.length < MAX_LEN) ids.push(0);
  return ids;
}

function softmax(xs) {
  const m = Math.max(...xs);
  const es = xs.map((x) => Math.exp(x - m));
  const ssum = es.reduce((a, b) => a + b, 0);
  return es.map((e) => e / ssum);
}

async function classify(text) {
  const ids = encode(text);
  const input = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, MAX_LEN]);
  const out = await S.session.run({ tokens: input });
  const logits = Array.from(out.logits.data);
  const probs = softmax(logits);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { label: S.labels[best], confidence: probs[best], probs };
}

/* ---------------- speech synthesis ---------------- */
function pickVoice(locale) {
  const prefix = (locale || "hi-IN").split("-")[0];
  const voices = speechSynthesis.getVoices();
  return voices.find((v) => v.lang && v.lang.toLowerCase().startsWith(prefix))
    || voices.find((v) => v.lang && v.lang.toLowerCase().startsWith("hi"))
    || voices.find((v) => v.default) || null;
}

function speak(text, locale) {
  return new Promise((resolve) => {
    if (!("speechSynthesis" in window) || !text) return resolve();
    const u = new SpeechSynthesisUtterance(text);
    const v = pickVoice(locale);
    if (v) u.voice = v;
    u.lang = locale || "hi-IN";
    u.rate = 0.95; u.pitch = 1;
    u.onend = () => resolve();
    u.onerror = () => resolve();
    speechSynthesis.speak(u);
  });
}

async function speakSequence(texts, locale) {
  speechSynthesis.cancel();
  for (const t of texts) await speak(t, locale);
}

/* ---------------- speech recognition ---------------- */
function makeRecognizer(locale) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const r = new SR();
  r.lang = locale || "hi-IN";
  r.continuous = false;
  r.interimResults = true;
  r.maxAlternatives = 1;
  return r;
}

function startListening() {
  if (S.recognizing) { try { S.recognition.stop(); } catch (e) {} return; }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    setStatus("Is browser me mic nahi hai — neeche likh kar bhejein", "error");
    toast("Mic support nahi — typing box use karein");
    return;
  }
  const locale = $("locale").value;
  const r = makeRecognizer(locale);
  S.recognition = r; S.recognizing = true;
  micBtn.classList.add("listening");
  setStatus("Sun raha hoon… boliye", "listening");
  transcriptEl.hidden = true;
  let finalText = "";
  r.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += t + " ";
      else interim += t;
    }
    transcriptEl.hidden = false;
    transcriptEl.textContent = (finalText || interim).trim() || "…";
  };
  r.onerror = (e) => {
    S.recognizing = false;
    micBtn.classList.remove("listening");
    if (e.error === "no-speech") setStatus("Kuch sunayi nahi diya — dobara tap karein", "error");
    else if (e.error === "not-allowed") setStatus("Mic ki permission dein tab hi bol sakte hain", "error");
    else setStatus("Mic dikkat: " + e.error, "error");
  };
  r.onend = () => {
    S.recognizing = false;
    micBtn.classList.remove("listening");
    if (finalText.trim()) handleQuery(finalText.trim());
    else setStatus("Tap karein aur bolen");
  };
  try { r.start(); } catch (e) { S.recognizing = false; micBtn.classList.remove("listening"); }
}

/* ---------------- playbook render + voice ---------------- */
function romanOfKey(key) {
  const svc = S.kb.services[key];
  return svc ? svc.name_roman : "";
}

function renderService(label, confidence) {
  const svc = (S.kb.services && S.kb.services[label]) || S.kb.general;
  S.lastResult = { label, confidence, svc };

  resultEl.hidden = false;
  svcName.textContent = svc.name_hi + " — " + svc.name_roman;
  $("intentBadge").textContent = "समझा: " + svc.name_roman;
  $("confBadge").textContent = (confidence * 100).toFixed(1) + "%";
  aiAnswerEl.hidden = true; aiAnswerEl.textContent = "";

  stepsEl.innerHTML = "";
  (svc.steps_hi || []).forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    stepsEl.appendChild(li);
  });

  const hasHelpline = svc.helpline && svc.helpline !== "state-specific";
  if (hasHelpline) {
    callBtn.style.display = "";
    callBtn.href = "tel:" + svc.helpline.replace(/[^0-9+]/g, "");
    helplineNum.textContent = svc.helpline + (svc.helpline_alt ? " / " + svc.helpline_alt : "");
  } else {
    callBtn.style.display = "none";
  }
  portalBtn.href = "https://" + svc.portal.split(" ")[0];

  docsList.innerHTML = "";
  (svc.docs_hi || []).forEach((d) => {
    const li = document.createElement("li"); li.textContent = d; docsList.appendChild(li);
  });

  faqList.innerHTML = "";
  (($.faqBox).hidden = !(svc.faq && svc.faq.length));
  (svc.faq || []).forEach((f) => {
    const qa = document.createElement("div"); qa.className = "qa";
    const q = document.createElement("p"); q.innerHTML = "<b>प्रश्न:</b> " + f.q_hi;
    const a = document.createElement("p"); a.innerHTML = "<b>जवाब:</b> " + f.a_hi;
    qa.append(q, a); faqList.appendChild(qa);
  });

  resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function speakPlaybook(label, confidence) {
  const locale = $("locale").value;
  const svc = (S.kb.services && S.kb.services[label]) || S.kb.general;
  const confHi = confidence >= 0.8 ? "" : "Aap thoda alag poochh rahe hain, phir bhi suniye. ";
  const texts = [
    confHi + "Aap ne " + svc.name_hi + " ke baare me poochha hai.",
    ...(svc.steps_hi || []),
    svc.helpline && svc.helpline !== "state-specific"
      ? "Koi dikkat ho toh helpline " + svc.helpline + " par phone karein. Yeh number muft hai."
      : "Koi dikkat ho toh apne zile ka karyalaya se sampark karein.",
  ];
  await speakSequence(texts, locale);
}

/* ---------------- optional AI re-wording (Puter.js, free) ---------------- */
async function aiSimplify() {
  if (!S.lastResult) return;
  const svc = S.lastResult.svc;
  toast("AI thoda aur saral kar raha hai…");
  try {
    if (!window.puter) {
      await new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = "https://js.puter.com/v2/";
        s.onload = res; s.onerror = rej;
        document.head.appendChild(s);
        setTimeout(rej, LLM_TIMEOUT_MS);
      });
    }
    const steps = (svc.steps_hi || []).join(" ");
    const prompt =
      "You are Sarkari Sahayak, a voice assistant for low-literacy village users in India. " +
      "Rewrite the following official guidance as 3-5 very short, warm spoken-Hindi (Devanagari) sentences. " +
      "Use only the facts given. Do NOT invent any number, website or rule. End with the helpline " +
      (svc.helpline || "") + " if present.\n\n" + steps;
    const reply = await Promise.race([
      puter.ai.chat(prompt),
      new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), LLM_TIMEOUT_MS)),
    ]);
    const text = (reply && (reply.message?.content ?? reply)) || "";
    const clean = String(text).trim().replace(/\*\*|__/g, "");
    if (clean && clean.length > 30) {
      aiAnswerEl.hidden = false;
      aiAnswerEl.textContent = "🤖 AI saral jawab: " + clean;
      await speakSequence([clean], $("locale").value);
    } else {
      toast("AI jawab nahi bana saka — sarkari steps hi sahi hain");
    }
  } catch (e) {
    toast("AI abhi uplabdh nahi — sarkari steps hi suniye");
  }
}

/* ---------------- main flow ---------------- */
async function handleQuery(text) {
  try {
    setStatus("Samajh raha hoon…");
    transcriptEl.hidden = false;
    transcriptEl.textContent = "🎤 " + text;
    const { label, confidence } = await classify(text);
    renderService(label, confidence);
    setStatus(label === "general" ? "Kaun se kaam me madad chahiye? Neeche dikhaiye" : "Jawab sunayi diya? Dobara bolen ya neeche dekhein");
    await speakPlaybook(label, confidence);
  } catch (e) {
    console.error(e);
    setStatus("Kuch gadbad ho gayi — dobara bolen", "error");
    toast("Error: " + (e.message || e));
  }
}

/* ---------------- boot ---------------- */
function buildChips() {
  const items = [
    ["kisan bill kaise bharein", "किसान", "PM-Kisan"],
    ["ayushman card banwana hai", "आयुष्मान", "Card"],
    ["shramik card banana hai", "श्रमिक", "e-Shram"],
    ["pf ka paisa nikalna hai", "पीएफ", "PF"],
    ["gas connection chahiye", "गैस", "Ujjwala"],
    ["manrega me kaam chahiye", "मनरेगा", "Job"],
    ["budhapa pension banana hai", "पेंशन", "Old age"],
    ["ration card banwana hai", "राशन", "PDS"],
  ];
  chips.innerHTML = "";
  items.forEach(([q, hi, en]) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.innerHTML = "<b>" + hi + "</b><span>" + en + "</span>";
    b.setAttribute("aria-label", q);
    b.onclick = () => handleQuery(q);
    chips.appendChild(b);
  });
}

function renderFraud() {
  fraudList.innerHTML = "";
  (S.kb.safety.fraud_warnings_hi || []).forEach((w) => {
    const li = document.createElement("li"); li.textContent = w; fraudList.appendChild(li);
  });
}

async function boot() {
  micBtn.addEventListener("click", startListening);
  typeGo.addEventListener("click", () => {
    const t = typeBox.value.trim();
    if (t) { typeBox.value = ""; handleQuery(t); }
  });
  typeBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter") typeGo.click();
  });
  speakAllBtn.addEventListener("click", () => S.lastResult && speakPlaybook(S.lastResult.label, S.lastResult.confidence));
  const aiBtn = document.createElement("button");
  aiBtn.className = "btn"; aiBtn.textContent = "🤖 AI se aur saral";
  aiBtn.onclick = aiSimplify;
  document.querySelector(".actions").appendChild(aiBtn);

  if ("speechSynthesis" in window) {
    speechSynthesis.onvoiceschanged = () => { S.voice = pickVoice($("locale").value); };
  }
  buildChips();
  try {
    await loadAssets();
    renderFraud();
    if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
      navigator.serviceWorker.register("sw.js").catch(() => {});
    }
  } catch (e) {
    console.error(e);
    setStatus("Model load nahi hua — internet check karein aur page reload karein", "error");
  }
}

/* E2E hooks (used by the browser verification suite) */
window.__sahayak = { classify, handleQuery, speak: (t) => speak(t, $("locale").value), S };

boot();
