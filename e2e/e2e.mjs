/* E2E: runs the REAL web pipeline (app.js functions) + REAL model in Node.
 * Mirrors exactly what the browser does: fetch assets -> translit (deva input)
 * -> encode -> ONNX session -> softmax -> label -> KB playbook lookup.
 * Exit 0 = all assertions passed.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import ort from "onnxruntime-node";

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, "..");
const WEB = join(ROOT, "web");

/* ---- extract translit + encode logic verbatim from app.js ---- */
const src = readFileSync(join(WEB, "app.js"), "utf-8");
const start = src.indexOf("const T_CONS");
const end = src.indexOf("/* ---------------- state ---------------- */");
const MAX_LEN = 14;

const translitCode = src.slice(start, end);
const ctx = {};
new Function("window", translitCode + `
  return { isDeva, devaToRoman, normalize };
`)(undefined);
const { isDeva, devaToRoman, normalize } = new Function(translitCode + `
  return { isDeva, devaToRoman, normalize };
`)();

const vocab = JSON.parse(readFileSync(join(WEB, "model", "vocab.json"), "utf-8"));
const labels = JSON.parse(readFileSync(join(WEB, "model", "labels.json"), "utf-8"));
const kb = JSON.parse(readFileSync(join(WEB, "kb", "services.json"), "utf-8"));

/* app.js encode(): normalize -> tokens -> vocab ids -> pad to MAX_LEN */
function encode(text) {
  const ids = normalize(text).split(" ").slice(0, MAX_LEN)
    .map((t) => (t in vocab ? vocab[t] : 1));
  while (ids.length < MAX_LEN) ids.push(0);
  return ids;
}

function softmax(xs) {
  const m = Math.max(...xs);
  const es = xs.map((x) => Math.exp(x - m));
  const s = es.reduce((a, b) => a + b, 0);
  return es.map((e) => e / s);
}

/* ---- real model, real inputs ---- */
const modelPath = join(WEB, "model", "intent_bigru.onnx");
const sess = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });

async function classify(text) {
  const ids = encode(text);
  const input = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, MAX_LEN]);
  const out = await sess.run({ tokens: input });
  const logits = Array.from(out.logits.data).map(Number);
  const probs = softmax(logits);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { label: labels[best], confidence: probs[best], probs };
}

/* ---- test battery ---- */
let pass = 0, fail = 0;
function check(name, cond, extra = "") {
  if (cond) { pass++; console.log("PASS", name, extra); }
  else { fail++; console.log("FAIL", name, extra); }
}

/* Devanagari voice queries (STT output) — the product's core promise */
const devaCases = [
  ["किसान बिल कैसे भरें", "pm_kisan"],
  ["पीएम किसान का पैसा नहीं आया", "pm_kisan"],
  ["आयुष्मान कार्ड कैसे बनवाएं", "ayushman"],
  ["आयुष्मान से इलाज कैसे कराएं", "ayushman"],
  ["ई श्रम कार्ड कैसे बनेगा", "eshram"],
  ["मजदूर कार्ड बनवाना है", "eshram"],
  ["पीएफ का पैसा कैसे मिलेगा", "epfo"],
  ["पीएफ निकालना है", "epfo"],
  ["आधार कार्ड खो गया क्या करें", "aadhaar"],
  ["आधार में नाम सुधारना है", "aadhaar"],
  ["गैस कनेक्शन बनवाना है", "ujjwala"],
  ["गैस सब्सिडी रुक गई है", "ujjwala"],
  ["मनरेगा में काम कैसे मिलेगा", "nrega"],
  ["जॉब कार्ड बनवाना है", "nrega"],
  ["बुढ़ापा पेंशन बनवाना है", "pension"],
  ["विधवा पेंशन कैसे मिलेगी", "pension"],
  ["राशन कार्ड बनवाना है", "ration"],
  ["राशन नहीं मिल रहा है", "ration"],
  ["सरकारी योजना के बारे में बताओ", "general"],
  ["सरकारी काम में मदद चाहिए", "general"],
];
for (const [q, want] of devaCases) {
  const r = await classify(q);
  check(`deva: ${q}`, r.label === want, `-> ${r.label} (${(r.confidence * 100).toFixed(1)}%)`);
}

/* Roman queries */
const romanCases = [
  ["kisan bill kaise bharein", "pm_kisan"],
  ["ayushman card banwana hai", "ayushman"],
  ["shramik card banana hai", "eshram"],
  ["pf ka paisa nikalna hai", "epfo"],
  ["mera aadhar card kho gaya", "aadhaar"],
  ["gas connection chahiye", "ujjwala"],
  ["manrega me kaam chahiye", "nrega"],
  ["budhapa pension banana hai", "pension"],
  ["ration card banwana hai", "ration"],
  ["bhaiya mujhe sarkari kaam me madad chahiye", "general"],
];
for (const [q, want] of romanCases) {
  const r = await classify(q);
  check(`roman: ${q}`, r.label === want, `-> ${r.label} (${(r.confidence * 100).toFixed(1)}%)`);
}

/* KB wiring: every predicted label must resolve to a full playbook */
for (const q of ["किसान बिल कैसे भरें", "ration card banwana hai", "गैस कनेक्शन बनवाना है"]) {
  const r = await classify(q);
  const svc = kb.services[r.label] || kb.general;
  check(`kb: ${r.label} has playbook`, svc.steps_hi && svc.steps_hi.length >= 5,
    `${svc.steps_hi.length} steps, helpline=${svc.helpline}`);
}

/* translit sanity via extracted app.js code */
check("translit: kisana", devaToRoman("किसान") === "kisana");
check("translit: adhara", normalize("आधार") === "adhara");
check("isDeva ok", isDeva("किसान") && !isDeva("kisan"));

/* model size budget (low-end phone) */
import { statSync } from "node:fs";
const mb = statSync(modelPath).size / 1024;
check("model under 1MB", mb < 1024, `${mb.toFixed(0)} KB`);

console.log(`\n==== E2E RESULT: ${pass} passed, ${fail} failed ====`);
process.exit(fail ? 1 : 0);
