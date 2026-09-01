"""Web asset hygiene + optional JS/py translit parity via node."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def test_index_references_resolve():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8")
    combined = html + js
    for asset in ["styles.css", "app.js", "manifest.json", "model/intent_bigru.onnx",
                  "kb/services.json", "model/vocab.json", "model/labels.json"]:
        assert asset in combined, f"web assets should reference {asset}"
    for f in ("styles.css", "app.js", "manifest.json", "sw.js"):
        assert (WEB / f).exists()
    assert (WEB / "model" / "intent_bigru.onnx").stat().st_size > 100_000


def test_manifest_icons_exist():
    m = json.loads((WEB / "manifest.json").read_text(encoding="utf-8"))
    for icon in m["icons"]:
        p = WEB / icon["src"]
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 100


def test_sw_caches_only_existing_files():
    sw = (WEB / "sw.js").read_text(encoding="utf-8")
    import re
    for m in re.finditer(r'"(\./[^"]+)"', sw):
        rel = m.group(1)[2:]
        assert (WEB / rel).exists(), f"sw.js caches missing file: {rel}"


def test_app_js_has_translit_and_model_pipeline():
    js = (WEB / "app.js").read_text(encoding="utf-8")
    for needle in ["devaToRoman", "ort.InferenceSession", "MAX_LEN = 14",
                   "speechSynthesis", "SpeechRecognition", "services["]:
        assert needle in js, f"app.js missing {needle}"


def test_js_py_translit_parity():
    node = shutil.which("node")
    if not node:
        print("node not found — skipping JS parity run")
        return
    r = subprocess.run([node, str(WEB / "test_translit.mjs")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"JS parity failed:\n{r.stdout}\n{r.stderr}"
    assert "parity" in r.stdout
