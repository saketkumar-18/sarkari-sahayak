"""Sarkari Sahayak — Hinglish voice-query dataset builder.

Authors a multi-dialect (romanized Hinglish) intent dataset for low-literacy
govt-service queries, expands it with templated fillers/typos, and emits
train/val/test JSONL splits with a fixed seed for reproducibility.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

LABELS = [
    "pm_kisan", "ayushman", "eshram", "epfo", "aadhaar",
    "ujjwala", "nrega", "pension", "ration", "general",
]

# ---------------------------------------------------------------------------
# Base seed queries per intent. Written in romanised Hinglish the way
# semi-literate users actually speak to a voice assistant.
# ---------------------------------------------------------------------------
BASE: dict[str, list[str]] = {
    "pm_kisan": [
        "kisan bill kaise bharein", "pm kisan ka paisa nahi aaya", "pm kisan me registration kaise karun",
        "kisan ki kist kaise check karun", "pm kisan ka paisa kitna milta hai", "kisan ka paisa kab aayega",
        "pm kisan samman nidhi me naam kaise jodein", "kisan ekyk kaise karun", "pm kisan ka status kaise dekhein",
        "kheti ke liye sarkari paisa kaise milega", "kisan ko 6000 rupaye wali yojana ka paisa nahi aaya",
        "pm kisan me mobile number kaise badlein", "kisan ka paisa ruk gaya kya karein",
        "kishan samman nidhi me kist nahi aayi", "kisan registration ke liye kya kagaz chahiye",
        "pm kisan app se paisa kaise check karein", "fasal ke saath kisan yojana ka paisa kaise milega",
        "kisan bank me paisa nahi mila", "pm kisan helpline number kya hai",
        "kisan ka paisa aadhaar se link ho gaya hai kya", "pm kisan me ekyc kaise hota hai",
        "gaon me kisan paisa wali yojana ka form kahan bharein", "kisan ko har mahine paisa milta hai kya",
        "pm kisan beneficiary status kaise jaanein", "kisan ka paisa kaat diya gaya",
        "khet ke dastavez ke bina kisan paisa milega", "pm kisan 2000 rupaye kab aayenge",
        "kisan seva kendra kahan hai", "kisan bima ka paisa kaise milega",
    ],
    "ayushman": [
        "ayushman card kaise banwayein", "ayushman card nahi bana raha", "ayushman bharat me naam kaise jodein",
        "ayushman card se ilaaj kaise karwayen", "ayushman me 5 lakh ka ilaaj kaise milega",
        "ospital me ayushman card chalega kya", "ayushman card banwane me kitna kharch aayega",
        "ayushman card list me naam nahi hai", "ayushman card kahan se milega",
        "hospital ka paisa ayushman se kaise milega", "ayushman card se operation hoga kya",
        "nha ayushman card kaise banega", "ayushman card ka helpline number",
        "ayushman bharat yojana ke liye kya kagaz chahiye", "mujhe ayushman card nahi mila",
        "ayushman card pe photo galat hai", "ayushman card se dawai muft milti hai kya",
        "ayushman se private hospital me ilaaj hoga", "ayushman card banane ke liye kahan jaana padega",
        "ayushman card se kharch ka paisa", "vay vandana card kaise banega",
        "buzurg ke liye ayushman card", "ayushman card ghar baithe banega kya",
        "ayushman card ka paisa kaise aata hai", "ayushman me kitna paisa milta hai ilaaj ka",
        "ayushman card ki jaanch kaise hoti hai", "ayushman arogya mandir me card banega kya",
        "ayushman card bana diya par kaam nahi kar raha", "mujhe muft ilaaj chahiye",
        "ayushman card me address change kaise karein",
    ],
    "eshram": [
        "e shram card kaise banwayein", "shramik card kaise banega", "eshram card ke liye kya chahiye",
        "eshram card banane me paisa lagta hai kya", "shram card kahan se banega",
        "eshram card ka uan number kaise milega", "mazdoor card kaise banwayen",
        "eshram card download kaise karein", "dihadi mazdoor ke liye sarkari card",
        "eshram card me naam galat hai", "shramik card ka paisa milta hai kya",
        "eshram card se kya fayda hai", "eshram card csc se banega kya",
        "thelewale ke liye shram card", "eshram card ka helpline number kya hai",
        "eshram card banwane ke liye mobile chahiye", "riksha chalane wale ka shramik card",
        "eshram card me vyavsay kaise badlein", "eshram card ke liye bank account zaroori hai kya",
        "shram card banane kitne din lagte", "eshram card online kaise banega",
        "eshram card ka paisa kab aayega", "naukri chhoot gayi shram card kaam aayega",
        "eshram card expire hota hai kya", "eshram card se bima milega kya",
        "eshram card me date of birth galat", "shramik card ka updation kaise karein",
        "eshram card apply kaise karein", "mazdoor ke liye sarkari yojana ka card",
        "eshram card me photo nahi lagi",
    ],
    "epfo": [
        "pf ka paisa kaise nikalwayein", "pf nahi mil raha kya karein", "epf claim kaise bharein",
        "pf ka balance kaise check karein", "uan number kaise nikalein", "pf account ka paisa nikalna hai",
        "naukri chhootne ke baad pf kaise nikalein", "pension ka paisa kaise milega",
        "pf ka paisa kitna hai mera", "epfo office kahan hai", "pf ka helpline number kya hai",
        "pf me naam galat likha hai", "old company ka pf kaise nikalein", "pf passbook kaise nikalein",
        "pf ka claim bhara tha paisa nahi aaya", "aadhaar se pf kaise link karein",
        "pf withdrawal ke liye form kaunsa hai", "maalik ne pf ka paisa nahi bhara",
        "pf ka paisa online kaise nikalta hai", "uan password bhool gaya kya karein",
        "pf kya hota hai", "pension scheme ka paisa", "epfo me grievance kaise karein",
        "pf claim status kaise dekhein", "bank me pf ka paisa nahi aaya",
        "pf ka paisa kab tak aayega", "provident fund ka hisaab kaise pata karein",
        "pf ki website kya hai", "nujkkri ke dauran pf nikal sakta hain kya",
        "pf ka paisa galti se kisi aur ko gaya",
    ],
    "aadhaar": [
        "aadhaar me naam kaise badlein", "aadhar card kho gaya kya karein", "aadhaar me mobile number kaise jodein",
        "aadhaar card me address change", "aadhar card ka duplicate kaise nikalein",
        "aadhaar update kaise karein", "aadhaar office kahan hai", "aadhar card me date of birth galat",
        "aadhaar enrolment ka paisa kitna", "aadhar card ka correction form kahan milega",
        "aadhaar me biometric kaise karwayen", "aadhaar card download kaise karein",
        "aadhaar pehchan patra kaise banega", "naya aadhar card banwana hai",
        "aadhaar helpline number", "aadhaar bank se link kaise karein",
        "bachhe ka aadhar card kaise banega", "aadhaar card xerox ke bina kaam hoga kya",
        "aadhar update ke liye kya document chahiye", "aadhaar me photo kaise badlein",
        "aadhar card ka mobile number badalna hai", "aadhaar se wahi kaise hota hai",
        "aadhaar card online copy kaise nikalein", "aadhaar card me fingerprint nahi lag raha",
        "aadhar card reissue kaise hoga", "aadhaar card free me banega kya",
        "uidai ka number kya hai", "aadhaar correction kitne din me hota hai",
        "aadhar card bana do", "aadhaar card ke bina ration milega kya",
    ],
    "ujjwala": [
        "gas connection kaise banwayein", "ujjwala me gas kaise milega", "gas ka silender subsidy nahi mila",
        "lpg connection ke liye kya chahiye", "gas agency me naam kaise jodein",
        "chulha gas ka connection muft wala", "ujjwala yojana ka gas connection kaise hoga",
        "gas subsidy ka paisa nahi aaya", "gas booking kaise karein", "gas chulha ka paisa kitna lagta hai",
        "indane gas ka connection banega kya", "bharat gas agency kahan hai",
        "gas cylinder ka subsidy kaise check karein", "ujjwala me naya connection form",
        "gas ka chulha muft me milta hai kya", "hp gas ka agent kaise dhundhein",
        "gas connection transfer kaise hota hai", "subsidy band ho gayi kya karein",
        "gas ki gandh aa rahi hai kya karein", "gas leak hone par kaunsa number",
        "ujjwala 2.0 me apply kaise karein", "gas connection ke liye bpl card chahiye kya",
        "cylinder ka rate kitna hai", "gas agency wale connection nahi de rahe",
        "ujjwala helpline number kya hai", "gas refill ka paisa online kaise de",
        "domestic gas connection banwana hai", "gaon me gas connection kaise milega",
        "gas chulha ka connection suspend ho gaya", "lpg id kaise banega",
    ],
    "nrega": [
        "manrega me kaam kaise karwayein", "nrega ka kaam nahi mila", "job card kaise banwayen",
        "manrega me kaam ki maang kaise bharein", "mgnrega ka paisa nahi aaya",
        "job card me naam kaise jodein", "panchayat me kaam ka form kahan milega",
        "manrega ki mazdoori ka hisaab kaise dekhein", "100 din ka kaam kaise milega",
        "job card kho gaya kya karein", "nrega ka helpline number",
        "manrega me kaam nahi de rahe kya karein", "job card banane ke liye kya chahiye",
        "mgnrega me unemployment bhatta kaise milega", "gaon ka kaam maangna hai panchayat se",
        "manrega ka paisa bank me kab aayega", "job card me photo nahi hai",
        "sarpanch kaam nahi de raha manrega me", "nrega rasid kaise milti hai",
        "manrega kaam 15 din me kab denge", "job card online check kaise karein",
        "manrega me aurat ka kaam alag hota hai kya", "nrega ka website kya hai",
        "zila se kaam ki shikayat kaise karein", "job card transfer dusre gaon me",
        "manrega ka paisa ruk gaya", "majdoori ka paisa kitna din ka",
        "mgnrega yojana ke liye form bharna hai", "panchayat me job card wale ka naam nahi",
        "rozgar ka kaam kaise maangein",
    ],
    "pension": [
        "budhapa pension kaise banwayein", "vridha pension ka paisa nahi aaya",
        "old age pension ke liye kaun sa form", "pension ke liye umar kitni chahiye",
        "vidhwa pension kaise milegi", "pension ka form kahan bharein",
        "budhapa pension me naam kaise jodein", "samaj kalyan ka pension ka paisa",
        "pension ki rashi kitni milti hai", "pension bank me nahi aayi",
        "zila karyalay me pension form", "vridhavastha pension online apply",
        "pension ka jeevan pramaan kaise bharein", "pension ruk gayi kya karein",
        "60 saal ke baad sarkari pension", "pension verification kaise hoti hai",
        "buzurg ke liye pension yojana", "pension me aadhaar link kaise karein",
        "pension ka helpline number kya hai", "pension ki jaanch ke liye kaun aata hai",
        "divyang pension kaise milegi", "pension form me galti ho gayi",
        "pension ka paisa mahine kitna aata hai", "pension ke liye bank account zaroori hai",
        "vidhwa pension ka paisa kitna", "pension certificate kahan se milega",
        "bpl pension ka form kaise bharein", "pension ka card kaise banega",
        "lif certificate pension ke liye", "pension officer se milkar baat karni hai",
    ],
    "ration": [
        "ration card kaise banwayein", "ration card me naam kaise jodein",
        "ration me nahi mil raha kya karein", "naya ration card ka form kahan milega",
        "ration card ka duplicate kaise banega", "ration ki dukaan par paisa zyada maang raha",
        "ration card ke liye kya kagaz chahiye", "ration card me pata badalna hai",
        "sasti anaj ki dukaan ka number", "ration card me wife ka naam jodna hai",
        "ration dukan wala bill nahi deta", "ration card download kaise karein",
        "bpl card banwana hai", "ration me aata nahi mila is mahine",
        "epds ka ration card check karna hai", "ration card helpline number",
        "dusre state me ration lena hai", "ek rashtriya ration card kaise chalega",
        "ration card me 1 sutri se 2 sutri kaise badlein", "ration ki dukaan ka samay kya hai",
        "ration card me galti hai naam me", "khadya vibhag me shikayat kaise karein",
        "ration ka paisa machine se kaatna hai", "ration card online kaise banega",
        "chawal ka rate kitna hai ration par", "ration card type kaunsa hai mera",
        "gareebi ke kagaz ke bina ration card", "ration me kisan ka paisa juda hai kya",
        "ration card renewal kab hota hai", "nfsa ka card banega kya",
    ],
    "general": [
        "sarkari yojanaon ke baare me jaanna hai", "umang app kaise chalayen",
        "sarkari helpline number", "sarkari kaam ke liye kaunsa app hai",
        "csc kahan hai mere gaon me", "sarkari dastavez ka kaam kaise hota hai",
        "sarkari website se form kaise bharein", "grievance kaise lodge karein",
        "sarkari adhikari se baat karni hai", "muft me sarkari kaam karwana hai",
        "sarkar ka paisa wala scheme kaunsa hai", "download kaise karein sarkari app",
        "otp kaise aata hai sarkari kaam me", "sarkari certificate banwana hai",
        "income certificate kaise banwayein", "domicile certificate ke liye form",
        "caste certificate kahan se banega", "sarkari scheme me shikayat kahan karein",
        "kisan helpline ka number batayein", "1947 par kaun si sewa milti hai",
        "rti kaise bharein", "sarkari form ka paisa lagta hai kya",
        "aadhaar se sarkari kaam kaise hota hai", "sarkar ki nayi yojana kya hai",
        "passing certificate banwana hai", "police clearance kaise banega",
        "online sarkari kaam ke liye mobile se kaam", "sarkari portal ka password bhool gaya",
        "bhaiya mujhe sarkari kaam me madad chahiye", "yojana ke liye kahan jaana padta hai",
    ],
}

# ---------------------------------------------------------------------------
# Voice-style fillers users prepend/append in real speech.
# ---------------------------------------------------------------------------
PREFIX = [
    "", "", "", "bhaiya ", "sunno ", "suno ", "arre ", "dekho ", "namaste ",
    "mujhe ", "mera ", "hamare ", "hum ", "main ",
    "kripya ", "plz ", "please ", "sarkar ", "sarkar ",
    "accha ", "aap ", "zara ",
]
PREFIX_MID = [
    "mujhe ", "meri ", "humein ", "mera ", "apna ", "apni ", "ghar me ",
    "gaon me ", "zile me ", "hum log ke ",
]
QUESTION = [
    "", "", "kaise", "kaise bharein", "kaise karun", "kaise karein", "kaise hota hai",
    "kaise milega", "kaise banega", "kya karun", "kya karna padega", "kahan jaana padega",
    "kya karne ka", "kaun batayega", "kaise pata karein", "kya process hai",
    "kaise banwayein", "kaise nikalwayein", "kab tak hoga", "kya jaldi hoga",
    "kahan se milega", "kaise check karun", "batao jaldi", "bataiye",
]
DISTRICTS = [
    "patna", "gaya", "muzaffarpur", "darbhanga", "siwan", "gorakhpur", "varanasi",
    "lucknow", "kanpur", "jaunpur", "azamgarh", "sultanpur", "pratapgarh",
    "motihari", "betiah", "chhapra", "samastipur", "begusarai", "bhagalpur",
    "katihar", "purnia", "kishanganj", "araria", "supaul", "madhepura",
    "saharsa", "khagaria", "munger", "lakhisarai", "shekhpura", "navada",
    "gopalganj", "deoria", "ballia", "ghazipur", "mirzapur", "sitapur",
    "barabanki", "raebareli", "sultanpur", "amethi", "pratapgarh", "jaipur",
    "bhopal", "indore", "nagpur", "ranchi", "gumla", "dumka", "deoghar",
    "hazipur", "buxar", "bhabhua", "rohtas", "kaimur", "bhojpur", "arra",
    "jahanabad", "aurangabad", "arwal", "nalanda", "sheohar", "sitamarhi",
]
FILLER_MID = [
    "", "", "jaldi ", "turant ", "abhi ", "aaj hi ", "kal subah ", "pichle hafte ",
    "agle hafte ", "is mahine ", "pichle mahine ", "saal bhar se ",
]

TYPO_MAP = [
    ("sh", "s"), ("ee", "i"), ("oo", "u"), ("kh", "k"), ("ph", "f"), ("th", "t"),
    ("aa", "a"), ("ou", "o"), ("ai", "ae"), ("y", "i"), ("v", "w"), ("z", "j"),
]


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _typo(text: str, rng: random.Random) -> str:
    """Inject one natural romanization variant."""
    src, dst = rng.choice(TYPO_MAP)
    if src in text and rng.random() < 0.5:
        text = text.replace(src, dst, 1)
    elif len(text) > 8 and rng.random() < 0.3:
        i = rng.randrange(2, len(text) - 2)
        text = text[:i] + text[i + 1]  # drop a letter
    return text


def _compose(base: str, rng: random.Random) -> str:
    """Expand one base query into a spoken-style utterance."""
    q = base
    if rng.random() < 0.5:
        q = rng.choice(PREFIX) + q
    if rng.random() < 0.35:
        q = rng.choice(PREFIX_MID) + q
    if rng.random() < 0.3:
        q = q + " " + rng.choice(DISTRICTS) + " me"
    if rng.random() < 0.4:
        q = q + " " + rng.choice(QUESTION)
    if rng.random() < 0.15:
        q = q + " " + rng.choice(FILLER_MID) + "jaldi bataye"
    q = _norm(q)
    if rng.random() < 0.12:
        q = _typo(q, rng)
    return q


def build(per_class_target: int = 320, seed: int = 42):
    from hindi_seeds import HINDI_SEEDS
    from translit import deva_to_roman

    rng = random.Random(seed)
    rows, seen = [], set()

    def add(label: str, text: str) -> bool:
        if not text or len(text.split()) < 3 or len(text.split()) > 18:
            return False
        key = (label, text)
        if key in seen:
            return False
        seen.add(key)
        rows.append({"text": text, "label": label})
        return True

    roman_per_class = per_class_target - 80      # 80 translit rows per class
    for label in LABELS:
        bases = BASE[label]
        made, tries = 0, 0
        while made < roman_per_class and tries < roman_per_class * 40:
            tries += 1
            text = _compose(rng.choice(bases), rng)
            if add(label, text):
                made += 1
        if made < roman_per_class:
            raise RuntimeError(f"could not reach roman target for {label}: {made}")
        # translit-augmented rows: Devanagari seeds through the SAME pipeline
        # the browser uses at inference (deva -> roman -> compose light variants)
        deva_bases = [deva_to_roman(s) for s in HINDI_SEEDS[label]]
        made = 0
        for base in deva_bases:
            if add(label, base):
                made += 1
        tries = 0
        while made < 80 and tries < 80 * 30:
            tries += 1
            if add(label, _compose(rng.choice(deva_bases), rng)):
                made += 1
        if made < 60:
            raise RuntimeError(f"translit augmentation too thin for {label}: {made}")
    rng.shuffle(rows)

    # dedupe globally: identical text must not map to two labels
    by_text = {}
    for r in rows:
        by_text.setdefault(r["text"], r)
    rows = list(by_text.values())
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    splits = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }
    DATA_DIR.mkdir(exist_ok=True)
    counts = {}
    for name, split in splits.items():
        path = DATA_DIR / f"queries_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(split)
    with (DATA_DIR / "queries_all.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return counts


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"queries_{name}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> list[str]:
    return _norm(text).split() or ["<pad>"]


if __name__ == "__main__":
    counts = build()
    print("dataset built:", counts)
    from collections import Counter
    c = Counter(r["label"] for r in load_split("train"))
    print("train label counts:", dict(c))
    print("sample:", load_split("train")[:3])
