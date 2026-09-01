/* Standalone JS translit parity test — mirrors src/translit.py.
 * Expected pairs frozen from the Python implementation (see tests/test_js_parity.py).
 * Run: node web/test_translit.mjs   (exit 1 on any mismatch)
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "app.js"), "utf-8");

// Extract the translit section from app.js and evaluate it in this module scope.
const start = src.indexOf("const T_CONS");
const end = src.indexOf("function normalize");
if (start < 0 || end < 0) throw new Error("translit section not found in app.js");
const code = src.slice(start, end) + "\nreturn { isDeva, devaToRoman };\n";
const { isDeva, devaToRoman } = new Function(code)();

const CASES = [
  ["किसान बिल कैसे भरें", "kisana bila kaise bharen"],
  ["पीएम किसान का पैसा नहीं आया", "piema kisana ka paisa nahin aya"],
  ["आयुष्मान कार्ड कैसे बनवाएं", "ayushmana karda kaise banavaen"],
  ["ई-श्रम कार्ड कैसे बनवाएं", "i-shrama karda kaise banavaen"],
  ["पीएफ का पैसा कैसे निकलवाएं", "piefa ka paisa kaise nikalavaen"],
  ["आधार में नाम कैसे बदलें", "adhara men nama kaise badalen"],
  ["गैस कनेक्शन कैसे बनवाएं", "gaisa kanekshana kaise banavaen"],
  ["मनरेगा में काम कैसे कराएं", "manarega men kama kaise karaen"],
  ["बुढ़ापा पेंशन कैसे बनवाएं", "budhaapa penshana kaise banavaen"],
  ["राशन कार्ड कैसे बनवाएं", "rashana karda kaise banavaen"],
];

let fail = 0;
for (const [input, expected] of CASES) {
  const got = devaToRoman(input).replace(/\s+/g, " ").trim();
  const ok = got === expected;
  if (!ok) {
    fail++;
    console.log("MISMATCH:", input, "\n  expected:", expected, "\n  got:     ", got);
  }
}
if (!isDeva("किसान")) { fail++; console.log("isDeva failed for Devanagari"); }
if (isDeva("kisan bill")) { fail++; console.log("isDeva false-positive on roman"); }
if (fail) { console.error(fail + " parity failure(s)"); process.exit(1); }
console.log("JS translit parity: all", CASES.length, "cases match; isDeva ok");
