# Constitution-Aware Pipeline (costitu2.py)

Short description
- `costitu2.py` is a small demo pipeline that scans short prompts/text for potential biased or hateful language using a multilingual knowledge base and simple heuristics, then applies a constitution-aware neutralization when a candidate text is judged to exceed configured bias/toxicity thresholds.

Key features
- Multilingual knowledge base (Hindi, Telugu, Tamil, Bengali, English).
- Latin->native mapping for terms (helps detect English transliterations).
- Fuzzy matching using `rapidfuzz` to detect near-matches.
- Simple toxicity and fluency heuristics.
- A simulated candidate generator and a constitution-aware decoder that selects or neutralizes outputs.
- Exports results to a timestamped CSV (UTF-8 with BOM) for readability.

Dependencies
- Python 3.8+ (or compatible)
- `pandas`
- `rapidfuzz`

Install (PowerShell)

```powershell
python -m pip install --upgrade pip; 
python -m pip install pandas rapidfuzz
```

Run

- Interactive run (prompts are comma-separated):

```powershell
python .\costitu2.py
# When prompted, enter prompts separated by commas, or press Enter to use defaults.
```

- Programmatic run: import `run_demo` from `costitu2.py` in another script and pass a list of prompts.

Outputs
- A DataFrame is returned by `run_demo` and, if `out_csv` is provided, a CSV file is created named like `demo_results_YYYYMMDD_HHMMSS.csv`.
- CSV columns are bilingual headers (Telugu / Hindi / English) for readability.

High-level flow
1. Input prompts (provided interactively or via a list).
2. For each prompt, `constitution_aware_decode` generates simulated candidate sentences.
3. Each candidate is scored for bias, toxicity, and fluency.
4. If the best candidate exceeds configured thresholds, a neutralized response is produced along with a justification; otherwise the candidate is selected.
5. Results are aggregated and optionally saved to CSV.

Major functions (summary)
- `detect_bias_terms(text, kb=KNOWLEDGE_BASE, latin_to_native=LATIN_TO_NATIVE, threshold=FUZZ_THRESHOLD)`
  - Uses `rapidfuzz.fuzz.partial_ratio` and exact substring checks to find matches between the input text and terms in the multilingual knowledge base and latin->native mappings.
  - Returns a set of `(term, language)` matches.

- `bias_score(text)`
  - Wrapper around `detect_bias_terms`; returns `(count, matches)`.

- `toxicity_score(text)`
  - Simple heuristic: checks for presence of toxic keywords (`TOXIC_KEYWORDS`) and maps counts into a 0.0–1.0 score (capped at 1.0).

- `fluency_score(text)`
  - Very lightweight fluency heuristic based on the number of words; returns a 0.0–1.0 score with simple piecewise adjustments.

- `simulated_generator(prompt, n_candidates=4)`
  - Produces a small set of template candidate sentences (for demo only). In real use, replace with a language model/generator.

- `neutralize_response(prompt, kb_matches=None)`
  - Produces a constitutionally-framed neutral response explaining equal treatment and referencing detected languages if available.

- `constitution_aware_decode(prompt, n_candidates=4, bias_threshold=1, toxicity_threshold=0.2, fluency_weight=1.0)`
  - Generates candidates, scores them (bias, toxicity, fluency), computes a combined score and selects the best candidate.
  - If selected candidate exceeds `bias_threshold` or `toxicity_threshold`, returns a neutralized `selected` message and sets `violated=True` with a justification string.
  - Returns a dict with `prompt`, `candidates` (detailed scores), `selected`, `violated`, and `justification`.

- `run_demo(prompts, out_csv=None)`
  - Runs `constitution_aware_decode` across prompts and builds a `pandas.DataFrame` with per-candidate rows.
  - If `out_csv` is provided, writes a timestamped CSV with bilingual headers and UTF-8 BOM (`utf-8-sig`) encoding.

Configuration points
- `KNOWLEDGE_BASE` and `LATIN_TO_NATIVE` contain the multilingual terms and mappings — edit these to add or remove monitored words.
- `FUZZ_THRESHOLD` controls fuzzy-match sensitivity (default: 80).
- `TOXIC_KEYWORDS` is the small set of English toxic words used by `toxicity_score`.
- `constitution_aware_decode` accepts `bias_threshold`, `toxicity_threshold`, and `fluency_weight` to tune selection behavior.

Notes & suggestions
- The candidate generator is a demo stub. Replace `simulated_generator` with a real NLG or LM call to evaluate live outputs.
- The scoring heuristics are intentionally simple for demonstration; consider training or integrating more robust classifiers for production.
- The KB currently mixes transliterated and native terms — keep it curated to avoid false positives.
- CSV encoding uses `utf-8-sig` so that Excel on Windows displays Unicode correctly.

Possible improvements
- Add unit tests for `detect_bias_terms` and scoring functions.
- Provide a CLI argument parser (e.g., `argparse`) to pass prompts and thresholds non-interactively.
- Add language detection for mixed-language inputs to improve matching accuracy.

License & attribution
- This file documents the included `costitu2.py` demo. The detection heuristics and KB are for demonstration only and should not be used as definitive moderation tools without further validation.

