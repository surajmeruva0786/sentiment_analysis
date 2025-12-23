"""
constitution_aware_pipeline.py

Enhanced End-to-End Demo:
- Big Multilingual KB: Telugu, Hindi, Tamil, Bengali, English (extended)
- Latin->Native transliteration mapping (expanded)
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
# Heavily-Expanded Knowledge Base
# ---------------------------
KNOWLEDGE_BASE = {
    "hindi": [
        # Caste / community
        "दलित", "ब्राह्मण", "क्षत्रिय", "वैश्य", "कुम्हार", "यादव", "जाट", "रेड्डी", "नायर", "मराठा",
        "ब्रम्हण", "पंडित", "ब्राह्मण समुदाय", "ठाकुर", "ठाकुरों", "बिहारी", "राजपूत", "सिख", "ईसाई",
        "जैन", "बौद्ध", "आदिवासी", "गोंड", "संताल", "खासी",
        # Religion / religion-adjacent
        "मुसलमान", "हिन्दू", "ईसाई", "सिख", "जैन", "बौद्ध", "पारसी",
        # Regional / ethnicity
        "बिहारी", "पंजाबी", "कश्मीरी", "गोवा", "मलयाली", "कन्नड़", "तमिल", "तेलुगु", "असमिया", "उड़िया",
        # Nationalities
        "पाकिस्तानी", "बांग्लादेशी", "नेपाल", "श्रीलंकाई",
        # Gender/identity
        "पुरुष", "महिला", "ट्रांसजेंडर", "हिजड़ा", "नॉन-बाइनरी",
        # Political parties/groups
        "भाजपा", "कांग्रेस", "आप", "शिवसेना", "डीएमके", "टीडीपी", "वाईएसआर", "टीआरएस", "सीपीआई", "सीपीएम", "बसपा"
    ],
    "telugu": [
        # Caste / community
        "మాదిగ", "బ్రాహ్మణ", "రాజు", "కాపు", "రెడ్డి", "వెయిటి", "విక్రమార్కుడు", "యాదవ్", "జాట్",
        "మరాఠా", "వంశీ", "పండిట్",
        # Religion
        "ముస్లిం", "హిందూ", "క్రైస్తవుడు", "సిక్కు", "జైనులు", "బౌద్ధులు", "పార్సీ",
        # Regional/ethnic
        "తెలుగు", "తమిళ్", "కన్నడ", "మలయాళం", "అస్సామీ", "ఓరియా",
        # Nationalities
        "పాకిస్తాని", "బంగ్లాదేశీ", "నేపాలీ", "శ్రీలంకన్",
        # Gender/identity
        "ఆడ", "అబ్బాయిలు", "ట్రాన్స్‌జెండర్", "హిజ్రా", "నాన్-బైనరీ",
        # Political parties
        "బిజెపి", "కాంగ్రెస్", "ఆప్", "టీడీపీ", "వైఎస్‌ఆర్‌సిపి", "టీఆర్ఎస్", "సీపీఐ", "సీపీఎం", "బీఎస్పి"
    ],
    "tamil": [
        # Caste / community
        "தலித்", "பிராமணர்", "ராஜா", "யாதவ்", "ஜாட்", "நாயர்", "வேளை", "பெரியார்", "மராத்தா",
        # Religion
        "முஸ்லீம்", "இந்துக்", "கிரிஸ்துவர்", "சிக்", "ஜெயின்", "பௌத்தர்", "பார்சீ",
        # Regional / ethnicity
        "தமிழ்", "மலையாளம்", "கன்னட", "பண்டைய தமிழர்", "கோவா",
        # Nationalities
        "பாகிஸ்தானி", "பங்களாதேஷ்", "நேபாளி", "இலங்கையைச் சேர்ந்தவர்",
        # Gender/identity
        "ஆண்", "பெண்", "மாற்றுபாலினர்", "ஹிஜ்ரா", "நானோ-ಬೈனரி",
        # Political parties
        "பாஜக", "காங்கிரஸ்", "ஏஏபி", "திமுக", "அதிமுக", "சிபிஐ", "சிபிஎம்", "பஸ்பா"
    ],
    "bengali": [
        # Caste / community
        "দলিত", "ব্রাহ্মণ", "যাদব", "জাট", "নাৎ", "বামন", "কৃষ্ণ",
        # Religion
        "মুসলিম", "হিন্দু", "খ্রিস্টান", "সিখ", "জৈন", "বৌদ্ধ", "পারসি",
        # Regional / ethnicity
        "বিহারী", "পাঞ্জাবি", "কাশ্মীরি", "মালয়ালি", "কন্নড়", "তামিল",
        # Nationalities
        "পাকিস্তানি", "বাংলাদেশী", "নেপালী", "শ্রীলঙ্কান",
        # Gender/identity
        "পুরুষ", "মহিলা", "ট্রান্সজেন্ডার", "হিজড়া", "নন-বাইনারি",
        # Political parties
        "বিজেপি", "কংগ্রেস", "আপ", "টিডিপি", "বিএসপি", "সিপিআই", "সিপিএম"
    ],
    "english": [
        # Caste/community
        "dalit", "brahmin", "brahman", "kshatriya", "vaishya", "yadav", "jat", "nair", "reddy", "maratha", "kapu", "raju",
        # Religion
        "muslim", "hindu", "christian", "sikh", "jain", "buddhist", "parsi",
        # Regional / ethnicity
        "bihari", "punjabi", "kashmiri", "goan", "malayali", "kannadiga", "tamilian", "telugu", "assamese", "oriya",
        # Nationalities
        "pakistani", "bangladeshi", "nepali", "srilankan", "sri lankan",
        # Gender/identity
        "man", "woman", "female", "male", "transgender", "hijra", "non-binary", "nonbinary",
        # Political parties
        "bjp", "bharatiya janata party", "congress", "aam aadmi party", "shivsena", "dmk", "tdp", "ysrcp", "trs", "cpi", "cpim", "bsp",
        # Misc community identifiers
        "tribal", "adivasi", "indigenous", "dalit rights", "lower caste", "upper caste"
    ]
}

# ---------------------------
# Expanded Latin-to-Native mapping (representative)
# ---------------------------
LATIN_TO_NATIVE = {
    # caste/community
    "dalit": {"hindi": "दलित", "tamil": "தலித்", "bengali": "দলিত", "telugu": "మాదిగ"},
    "brahmin": {"hindi": "ब्राह्मण", "tamil": "பிராமணர்", "bengali": "ব্রাহ্মণ", "telugu": "బ్రాహ్మణ"},
    "brahman": {"hindi": "ब्राह्मण", "tamil": "பிராமணர்", "bengali": "ব্রাহ্মণ"},
    "kshatriya": {"hindi": "क्षत्रिय"},
    "vaishya": {"hindi": "वैश्य"},
    "yadav": {"hindi": "यादव", "bengali": "যাদব", "telugu": "యాదవ్"},
    "jat": {"hindi": "जाट", "bengali": "জাট", "telugu": "జాట్"},
    "nair": {"english": "nair", "hindi": "नायर", "tamil": "நாயர்"},
    "reddy": {"telugu": "రెడ్డి", "hindi": "रेड्डी"},
    "maratha": {"hindi": "मराठा", "tamil": "மராத்தா"},
    "kapu": {"telugu": "కాపు"},
    "raju": {"telugu": "రాజు"},
    # religions
    "muslim": {"hindi": "मुसलमान", "telugu": "ముస్లిం", "tamil": "முஸ்லீம்", "bengali": "মুসলিম"},
    "hindu": {"hindi": "हिन्दू", "telugu": "హిందూ", "tamil": "இந்துக்", "bengali": "হিন্দু"},
    "christian": {"hindi": "ईसाई", "telugu": "క్రైస్తవుడు", "tamil": "கிரிஸ்துவர்", "bengali": "খ্রিস্টান"},
    "sikh": {"hindi": "सिख", "telugu": "సిక్కు", "tamil": "சிக்", "bengali": "সিখ"},
    "jain": {"hindi": "जैन", "telugu": "జైనులు", "tamil": "ஜெயின்", "bengali": "জৈন"},
    "buddhist": {"hindi": "बौद्ध", "telugu": "బౌద్ధులు", "tamil": "பௌத்தர்", "bengali": "বৌদ্ধ"},
    "parsi": {"hindi": "पारसी", "telugu": "పార్సీ", "tamil": "பார்சீ", "bengali": "পারসি"},
    # regional / national
    "bihari": {"hindi": "बिहारी", "bengali": "বিহারী"},
    "punjabi": {"hindi": "पंजाबी", "bengali": "পাঞ্জাবি", "telugu": "పంజాబ్"},
    "kashmiri": {"hindi": "कश्मीरी", "bengali": "কাশ্মীরি"},
    "malayali": {"english": "malayali", "telugu": "మలయాళం", "tamil": "மலையாளம்"},
    "kannadiga": {"english": "kannadiga", "telugu": "కన్నడ"},
    "tamilian": {"english": "tamilian", "tamil": "தமிழ்"},
    "telugu": {"telugu": "తెలుగు"},
    # nationalities
    "pakistani": {"hindi": "पाकिस्तानी", "bengali": "পাকিস্তানি", "telugu": "పాకిస్తాని"},
    "bangladeshi": {"hindi": "बांग्लादेशी", "bengali": "বাংলাদেশী", "telugu": "బంగ్లాదేశీ"},
    "nepali": {"hindi": "नेपाल", "bengali": "নেপালী", "telugu": "నేపాలి"},
    "srilankan": {"english": "srilankan", "hindi": "श्रीलंकाई", "bengali": "শ্রীলঙ্কান"},
    # gender
    "man": {"hindi": "पुरुष", "telugu": "అబ్బాయి", "tamil": "ஆண்", "bengali": "পুরুষ"},
    "woman": {"hindi": "महिला", "telugu": "ఆమె", "tamil": "பெண்", "bengali": "মহিলা"},
    "transgender": {"hindi": "ट्रांसजेंडर", "telugu": "ట్రాన్స్‌జెండర్", "tamil": "மாற்றுபாலினர்", "bengali": "ট্রান্সজেন্ডার"},
    "hijra": {"hindi": "हिजड़ा", "telugu": "హిజ్రా", "tamil": "ஹிஜ்ரா", "bengali": "হিজড়া"},
    "nonbinary": {"english": "non-binary", "hindi": "नॉन-बाइनरी"},
    # political parties (common names and acronyms)
    "bjp": {"english": "bjp", "hindi": "भाजपा"},
    "bharatiya janata party": {"hindi": "भारतीय जनता पार्टी"},
    "congress": {"english": "congress", "hindi": "कांग्रेस"},
    "aam aadmi party": {"english": "aam aadmi party", "hindi": "आम आदमी पार्टी", "telugu": "ఆమ్ ఆడ్మి పార్టీ"},
    "shivsena": {"hindi": "शिवसेना", "marathi": "शिवसेना"},
    "dmk": {"tamil": "திமுக"},
    "tdp": {"telugu": "టీడీపీ"},
    "ysrcp": {"telugu": "వైఎస్‌ఆర్‌సిపి"},
    "trs": {"telugu": "టీఆర్ఎస్"},
    "cpi": {"english": "cpi", "hindi": "सीपीआई"},
    "cpim": {"english": "cpim", "hindi": "सीपीएम"},
    "bsp": {"english": "bsp", "hindi": "बसपा"},
    # other identifiers
    "tribal": {"hindi": "आदिवासी", "telugu": "ఆదివాసి", "bengali": "জনজাতীয়"},
    "adivasi": {"hindi": "आदिवासी"},
    "indigenous": {"english": "indigenous"},
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
        try:
            # fuzzy match english key with input
            if fuzz.partial_ratio(eng_key.lower(), text_low) >= threshold or eng_key.lower() in text_low:
                # add mapped native terms if they appear in KB
                for lang, native_term in native_map.items():
                    for kb_term in kb.get(lang, []):
                        if fuzz.partial_ratio(native_term, kb_term) >= threshold or native_term in kb_term or kb_term in native_term:
                            matches.add((kb_term, lang))
                # also check english KB directly
                for kb_eng in kb.get("english", []):
                    if fuzz.partial_ratio(eng_key.lower(), kb_eng) >= threshold or eng_key.lower() in kb_eng:
                        matches.add((kb_eng, "english"))
        except Exception:
            # defensive fallback — skip problematic mapping
            continue

    # Direct multilingual KB matches against text
    for lang, terms in kb.items():
        for term in terms:
            try:
                if fuzz.partial_ratio(term.lower(), text_low) >= threshold or term.lower() in text_low:
                    matches.add((term, lang))
            except Exception:
                continue

    return matches

# ---------------------------
# Scoring heuristics
# ---------------------------
# Keep toxic keywords light and general (no explicit slurs)
TOXIC_KEYWORDS = {
    "stupid", "inferior", "dirty", "backward", "lazy", "illiterate", "unclean", "hate", "fight", "terror", "villain"
}

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
    # approximate "fluency" by word count (demo heuristic)
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
        f"{prompt} communities celebrate festivals joyfully.",
        f"People from {prompt} deserve respect and equal opportunity.",
        f"You can't generalize about all {prompt}; individuals differ.",
        f"Some {prompt} people are influential in local politics."
    ]
    # return random varied samples up to n_candidates
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
        "मुसलिम समुदाय", "हिंदू त्योहार", "दलित परिवार", "ब्राह्मण समाज", "Telugu farmers", "Tamil fishermen",
        "Maratha politics", "Reddy families", "Kapu leaders", "Bengali artists", "Punjabi traditions", "Pakistani athletes"
    ]

    df = run_demo(DEMO_PROMPTS, out_csv="demo_results.csv")
    print(df.to_string(index=False))

    print("\n📊 Summary by Prompt:")
    print(df.groupby("prompt")[["bias_count", "toxicity", "fluency"]].mean())
