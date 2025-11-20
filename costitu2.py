"""
constitution_aware_pipeline.py

Enhanced End-to-End Demo:
- Multilingual KB: Telugu, Hindi, Tamil, Bengali, English (extended)
- Latin->Native transliteration mapping
- Fuzzy/phonetic matching using rapidfuzz
- Bias, toxicity, fluency scoring
- Constitution-aware decoding
- Neutralized responses bilingual
- CSV output with readable formatting
- Justification for "violated" flag included
"""

from rapidfuzz import fuzz
import random
import pandas as pd
from datetime import datetime

# ---------------------------
# Extended Knowledge Base
# ---------------------------
KNOWLEDGE_BASE = {
    "hindi": ["दलित", "ब्राह्मण", "मुसलमान", "हिन्दू", "अछूत", "क्षत्रिय"],
    "telugu": ["మాదిగ", "కాపు", "ముస్లిం", "హిందూ", "బ్రాహ్మణ", "రాజు"],
    "tamil": ["தலித்", "பிராமணர்", "முஸ்லீம்", "இந்துக்", "சூப்பிரமணியர்"],
    "bengali": ["দলিত", "ব্রাহ্মণ", "মুসলিম", "হিন্দু", "কৃষ্ণ"],
    "english": ["dalit", "brahmin", "brahman", "muslim", "hindu", "kapu", "raju", "kshatriya"]
}

# ---------------------------
# Complete Latin-to-Native mapping
# ---------------------------
LATIN_TO_NATIVE = {
    "dalit": {"hindi": "दलित", "tamil": "தலித்", "bengali": "দলিত", "telugu": "మాదిగ"},
    "brahmin": {"hindi": "ब्राह्मण", "tamil": "பிராமணர்", "bengali": "ব্রাহ্মণ", "telugu": "బ్రాహ్మణ"},
    "brahman": {"hindi": "ब्राह्मण", "tamil": "பிராமணர்", "bengali": "ব্রাহ্মণ"},
    "muslim": {"hindi": "मुसलमान", "telugu": "ముస్లిం", "tamil": "முஸ்லீம்", "bengali": "মুসলিম"},
    "hindu": {"hindi": "हिन्दू", "telugu": "హిందూ", "tamil": "இந்துக்", "bengali": "হিন্দু"},
    "kapu": {"telugu": "కాపు"},
    "raju": {"telugu": "రాజు"},
    "kshatriya": {"hindi": "क्षत्रिय"}
}

# ---------------------------
# Bias detection
# ---------------------------
FUZZ_THRESHOLD = 80

def detect_bias_terms(text, kb=KNOWLEDGE_BASE, latin_to_native=LATIN_TO_NATIVE, threshold=FUZZ_THRESHOLD):
    text_low = text.lower()
    matches = set()
    # Check Latin-key presence and fuzzy match
    for eng_key, native_map in latin_to_native.items():
        if fuzz.partial_ratio(eng_key, text_low) >= threshold or eng_key in text_low:
            for lang, native_term in native_map.items():
                for kb_term in kb.get(lang, []):
                    if fuzz.partial_ratio(native_term, kb_term) >= threshold:
                        matches.add((kb_term, lang))
            for kb_eng in kb.get("english", []):
                if fuzz.partial_ratio(text_low, kb_eng) >= threshold or kb_eng in text_low:
                    matches.add((kb_eng, "english"))
    # Direct English KB match
    for kb_eng in kb.get("english", []):
        if fuzz.partial_ratio(text_low, kb_eng) >= threshold or kb_eng in text_low:
            matches.add((kb_eng, "english"))
    return matches

# ---------------------------
# Scoring heuristics
# ---------------------------
TOXIC_KEYWORDS = {"stupid", "inferior", "dirty", "backward", "lazy", "illiterate", "unclean", "hate", "fight"}

def bias_score(text):
    matches = detect_bias_terms(text)
    return len(matches), matches

def toxicity_score(text):
    t = sum(1 for kw in TOXIC_KEYWORDS if kw in text.lower())
    return min(1.0, t / 3.0)

def fluency_score(text):
    words = text.strip().split()
    if not words:
        return 0.0
    avg_len = len(words)
    if 8 <= avg_len <= 20:
        return 1.0
    if avg_len < 8:
        return max(0.0, 1.0 - (8 - avg_len) * 0.1)
    else:
        return max(0.0, 1.0 - (avg_len - 20) * 0.02)

# ---------------------------
# Simulated generator
# ---------------------------
def simulated_generator(prompt, n_candidates=4):
    templates = [
        f"The {prompt} are hardworking and intelligent.",
        f"The {prompt} are lazy and stupid.",
        f"Many {prompt} traditions vary widely.",
        f"{prompt} communities celebrate festivals joyfully."
    ]
    return random.sample(templates, k=min(n_candidates, len(templates)))

# ---------------------------
# Neutralized message (multilingual)
# ---------------------------
def neutralize_response(prompt, kb_matches=None):
    extra = ""
    if kb_matches:
        langs = sorted({lang for (_, lang) in kb_matches})
        extra = f" (Detected languages: {', '.join(langs)})"
    return ("Constitutional Neutrality Applied: As guaranteed by the Constitution of India, every individual must be treated with equality and dignity. "
            "People should be evaluated on merit and actions, not community identity." + extra)

# ---------------------------
# Constitution-aware decode
# ---------------------------
def constitution_aware_decode(prompt, n_candidates=4,
                             bias_threshold=1, toxicity_threshold=0.2,
                             fluency_weight=1.0):
    candidates = simulated_generator(prompt, n_candidates)
    scored = []
    for cand in candidates:
        bcount, bmatches = bias_score(cand)
        tox = toxicity_score(cand)
        flu = fluency_score(cand)
        combined = (fluency_weight * flu) - (0.5 * bcount) - (0.8 * tox)
        scored.append({
            "candidate": cand,
            "bias_count": bcount,
            "bias_matches": bmatches,
            "toxicity": tox,
            "fluency": flu,
            "combined": combined
        })

    best = max(scored, key=lambda x: x["combined"])
    violated = best["bias_count"] > bias_threshold or best["toxicity"] > toxicity_threshold
    final_selected = neutralize_response(prompt, best.get("bias_matches")) if violated else best["candidate"]

    return {
        "prompt": prompt,
        "candidates": scored,
        "selected": final_selected,
        "violated": violated,
        "justification": (
            "TRUE: Candidate exceeded bias/toxicity threshold; neutralized." if violated else
            "FALSE: Candidate within safe limits."
        )
    }

# ---------------------------
# Run demo and export CSV
# ---------------------------
def run_demo(prompts, out_csv=None):
    results = []
    for p in prompts:
        r = constitution_aware_decode(p)
        for cinfo in r["candidates"]:
            results.append({
                "prompt": r["prompt"],
                "candidate": cinfo["candidate"],
                "bias_count": cinfo["bias_count"],
                "bias_matches": ", ".join([f"{m[0]}({m[1]})" for m in cinfo["bias_matches"]]) if cinfo["bias_matches"] else "",
                "toxicity": round(cinfo["toxicity"], 3),
                "fluency": round(cinfo["fluency"], 3),
                "combined": round(cinfo["combined"], 3),
                "final_selected": r["selected"],
                "violated": r["violated"],
                "justification": r["justification"]
            })

    df = pd.DataFrame(results)

    # Bilingual headers for readability
    headers = {
        "prompt": "ప్రాంప్ట్ / प्रश्न / Prompt",
        "candidate": "అభ్యర్థి వాక్యం / प्रत्याशी वाक्य / Candidate",
        "bias_count": "పాక్షిక పదాల సంఖ్య / पक्षपात शब्द गिनती / Bias Count",
        "bias_matches": "గుర్తించిన పదాలు / पहचाने गए शब्द / Bias Matches",
        "toxicity": "దూషణ స్థాయి / विषाक्तता स्तर / Toxicity",
        "fluency": "ప్రవాహం / प्रवाह / Fluency",
        "combined": "మొత్తం స్కోరు / कुल स्कोर / Combined",
        "final_selected": "చివరి ఎంపిక / अंतिम चयन / Final Selected",
        "violated": "వైషమ్యం ఉందా? / पक्षपात पाया गया? / Violated?",
        "justification": "న్యాయ సమీక్ష / औचित्य / Justification"
    }

    if out_csv:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{out_csv.rstrip('.csv')}_{ts}.csv"
        df.rename(columns=headers).to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"✅ Results saved to: {filename}")

    return df

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    print("📥 Enter prompts (కామా వేరు చేసి / कौमा से अलग करें):")
    user_input = input().strip()
    DEMO_PROMPTS = [p.strip() for p in user_input.split(",")] if user_input else [
        "ముస్లిం సంప్రదాయాలు", "హిందూ పండుగలు", "दलित समुदाय", "ब्राह्मण परिवार", "Muslim traditions", "Hindu festivals", "కాపులు స్థానిక రాజకీయాల్లో", "Raju families"
    ]

    df = run_demo(DEMO_PROMPTS, out_csv="demo_results.csv")
    print(df.to_string(index=False))

    print("\n📊 Summary by Prompt:")
    print(df.groupby("prompt")[["bias_count", "toxicity", "fluency"]].mean())
