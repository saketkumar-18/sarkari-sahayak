"""Knowledge-base schema and content-safety tests."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = json.loads((ROOT / "data" / "services.json").read_text(encoding="utf-8"))

EXPECTED_SERVICES = {
    "pm_kisan", "ayushman", "eshram", "epfo", "aadhaar",
    "ujjwala", "nrega", "pension", "ration",
}


def test_nine_services_plus_general_and_safety():
    assert set(KB["services"]) == EXPECTED_SERVICES
    assert "general" in KB and "steps_hi" in KB["general"]
    assert "safety" in KB and KB["safety"]["fraud_warnings_hi"]


def test_service_schema():
    for key, svc in KB["services"].items():
        assert svc["name_hi"] and svc["name_roman"], key
        assert svc["portal"], key
        assert isinstance(svc["steps_hi"], list) and len(svc["steps_hi"]) >= 5, key
        assert isinstance(svc["steps_roman"], list), key
        assert len(svc["steps_hi"]) == len(svc["steps_roman"]), f"{key}: hi/roman step count"
        assert svc["docs_hi"] and svc["docs_roman"], key
        assert len(svc["keywords"]) >= 8, key
        assert svc["source"], f"{key} must cite a source"
        for faq in svc.get("faq", []):
            assert faq["q_hi"] and faq["a_hi"]


def test_helplines_format():
    for key, svc in KB["services"].items():
        h = svc["helpline"]
        assert h, f"{key}: helpline required (use 'state-specific' if none)"
        if h != "state-specific":
            assert re.fullmatch(r"[0-9\-]+", h), f"{key}: bad helpline {h}"
        alt = svc.get("helpline_alt", "")
        if alt:
            assert re.fullmatch(r"[0-9\-+/ ()a-zA-Z]*", alt), f"{key}: bad alt {alt}"


def test_keywords_do_not_collide_across_services():
    owner = {}
    clashes = []
    for key, svc in KB["services"].items():
        for kw in svc["keywords"]:
            kw = kw.lower()
            owner.setdefault(kw, set()).add(key)
    for kw, owners in owner.items():
        if len(owners) > 2:
            clashes.append((kw, sorted(owners)))
    assert not clashes, f"keywords shared by >2 services: {clashes}"


def test_no_money_demand_language():
    """Playbooks must never tell users to pay an agent."""
    blob = json.dumps(KB, ensure_ascii=False).lower()
    for bad in ["fee dena hai", "agent ko paise", "cash dein"]:
        assert bad not in blob


def test_fraud_warnings_present_and_actionable():
    warns = KB["safety"]["fraud_warnings_hi"]
    assert len(warns) >= 4
    joined = " ".join(warns)
    assert "ओटीपी" in joined          # OTP warning
    assert "1930" in joined          # cyber helpline


def test_web_kb_copy_is_current():
    web_kb = (ROOT / "web" / "kb" / "services.json").read_text(encoding="utf-8")
    assert web_kb == (ROOT / "data" / "services.json").read_text(encoding="utf-8")
