# Fire-Margin — NASA Space Apps 2026: "Flame in Freefall"

Evidence-graded fire safety for Moon-base atmospheres. An SI-powered (AI) web dashboard that
summarizes, ranks and interprets NASA microgravity combustion investigations (PSI repository).

**Deadline: submit before 11:59 p.m. IST, Sunday 15 Nov 2026.** Target submit time: 10 p.m.

## Headline insight (the pitch)
NASA plans lunar/Martian habitats at ~34% O2 / 8.2 psia (JSC 29353D: min 33.7% O2 at 8.2 psia;
challenge page: >=30% O2 at <=10.2 psi). In PSI's public data, solid materials stop at 22.2% O2.
Only 3 solid samples (Saffire-VI, 2024) have burned near Moon-base conditions (29-31% O2, ~0.54 atm):
not in PSI, one flight, no partial gravity. Gas flames (CFI-G) were tested at 33-39% O2 / 0.50 atm;
FLEX droplets reach 34% O2 but only at >= 0.69 atm, and at O2 >= 25% 80% of FLEX tests ended in
Disruption. Show exactly where evidence exists, where it is missing, and which experiment NASA should
run next.

## Stack
Python 3.12+ (tested 3.13; pinned numpy/scipy need >= 3.12), pandas, numpy, scikit-learn, scipy, plotly, streamlit, joblib, pypdf, openai (SDK only).
Deploy: Streamlit Community Cloud from this GitHub repo. Entry point: `app/app.py`.
LLM: Meta API (primary, paid credits) via the OpenAI Python SDK; NVIDIA NIM free API as optional fallback.
All settings come from env vars / Streamlit secrets — never commit keys, never hard-code URLs or models:
  `META_API_KEY`, `META_BASE_URL` (Llama API: `https://api.llama.com/compat/v1/`), `META_MODEL`
  (e.g. `Llama-4-Maverick-17B-128E-Instruct-FP8`; confirm via `GET {META_BASE_URL}/models`),
  optional `NVIDIA_API_KEY` (base `https://integrate.api.nvidia.com/v1`, model e.g. `meta/llama-3.3-70b-instruct`).
- One provider chain in `src/llm_config.py`: Meta first, NVIDIA second (skipped if no key).
- Llama API preview limits can be low (~10 req/min): retry with backoff, fall back, cache answers.
- Insight cards are pre-generated offline and saved to `data/processed/insight_cards.json`, so the
  catalog works even if the API is down. Chatbot shows a clear "AI temporarily unavailable" message
  plus the retrieved source passages if every model fails — never a crash.

## Layout
```
data/raw/          NASA files exactly as downloaded. NEVER edit. NEVER read whole large files.
data/processed/    cleaned CSVs produced by src/ scripts
data/knowledge/    text chunks for the RAG layer (from Researcher's Guide PDF + investigation pages)
notebooks/         exploration only
src/               ingest.py, features.py, models.py, ranking.py, rag.py, coverage.py
models/            saved .joblib models + metrics.json
app/app.py         Streamlit app (tabs below)
docs/              README assets, AI disclosure, slides, video script
tests/             pytest
```

## Data rules (non-negotiable — Validity is a judging criterion)
- No synthetic, interpolated or formula-generated rows presented as NASA data. Ever.
- FLEX (PSI-69, DOI 10.60555/mbq8-0451): file `data/raw/PSI-69_Experimental table_FLEX.csv`,
  cp1252, 274 rows x 15 cols (verified). Subscripts were lost in headers: "O initial ambient composition;
  mole fraction" = O2, "N ..." = N2, "CO ..." = CO2 (NOT carbon monoxide), "He ..." = He.
  Rename to snake_case: test_no, flex_id, test_date, test_gmt, fuel, p_mmhg, x_o2, x_n2, x_co2, x_he,
  d0_mm, d_ext_mm, K_mm2_s ("Burning rate; mm" is really mm^2/s), burn_time_s, test_end.
- Known issues (verified): '–' (en dash) and blanks = missing -> NaN; burn time '~6', '~9' -> 6, 9 with
  flag `burn_time_approx=True`; flex_id C11M201 has pressure 0 -> NaN; flex_id 193F001 appears twice ->
  keep both, flag `dup_id=True`. 80 distinct test dates (use as CV groups).
- Outcomes: Extinction 186 / Disruption 61 / Completion 27. Fuels: Methanol 157 / Heptane 117.
  O2 range 0.12-0.34, pressure up to ~3 atm. DON'T drop Disruption: almost every test at O2 >= 0.25
  ends in Disruption (droplet bursting) — that is a key safety finding, not noise.
- Extra PSI tables in data/raw/ (all utf-8, verified). Strip whitespace from headers. Drop rows that are
  not tests (e.g., "Pre-test flow visulation data...", all-NaN rows) and log how many were dropped.
  * PSI-25 BASS-II (129 rows, solids, 51 material labels): O2 = "Calibrated  initial O2 % by vol " (%, 13.6-22.2,
    divide by 100). NO pressure column -> p_atm NaN plus `p_assumed_atm=1.0, p_assumed=True` (ISS cabin) only for
    plotting, clearly labelled. Flow is "Fan display"/"Air display" SETTINGS (free text like "5 down to ext"),
    NOT cm/s — never convert to cm/s; keep the raw text. CO/CO2 ppm/% columns are gas readings.
  * PSI-26 BASS (122 rows): only date, test #, sample #, material. No O2/pressure/flow -> catalog only, not coverage.
  * PSI-10 "Flame Design_Inverse Flame" (120 rows, ethylene inverse gas flames): XO2 0.21-0.85,
    pmean 0.43-3.02 atm. Gas-jet flames, not cabin materials.
  * PSI-159 "CFI-G_Normal Flames" (179 rows, ethane/propane/butane blends): " XO2 before test" 0.18-0.40,
    "approx p (bar)" 0.51-3.04 (convert bar -> atm). Cool flame? 7 yes; Hot ignition Yes 132/No 47;
    end: SE (self-extinguished) 102 / FT (flow terminated) 31 / '-' 46.
  * PSI-98/99/100 Saffire-I/II/III (tiny): values are strings with units/tildes ("~ 21.5", "20 cm/s",
    "22.1 to 22.0", "0.27 mm") -> parse numbers, keep raw text, flag approx. "?-g" in headers = micro-g.
    Saffire-II micro-g spread for SIBAL 2.1 and 2.6 mm/s. Saffire-III has no O2 column (flow 30 cm/s).
- Verified coverage insight: SOLID materials in PSI (BASS-II, Saffire) span only 13.6-22.2% O2 at cabin
  pressure. Gas flames reach 0.40 (normal) / 0.85 (inverse) O2 and down to 0.43 atm; FLEX droplets 0.34 O2,
  min 0.69 atm. So: gas and droplet data touch Moon-base O2; cabin solids do not.
- BASS/BASS-II tables have NO spread-rate column. Don't invent one; use them for the coverage map only.
- Literature values are plotted as clearly labelled "literature points", with citation + URL, never mixed into
  training data. Team-verified (2026-11-15): Saffire IV-1/V-2/VI-2/VI-3/VI-4 test matrix (Urban et al.,
  ICES-2024-365; NTRS 20240007488) and PMMA rods 17% (micro-g) vs 18% (1-g) O2 (Fernandez-Pello et al.,
  Sci. Rep. 2017, doi 10.1038/s41598-017-18398-4). The BASS-II "13.6% vs 16%" number is unverifiable: removed.
  Recommender evidence weights: PSI measured 1.0; PSI with assumed 1 atm (BASS-II, Saffire I-II) 1.0, flagged;
  verified literature 0.5. Ranking: planned / facility-capability ranges are not evidence (score 0).
- Every number shown in the app traces to a PSI ID / DOI / paper.

## Modelling rules
- Validation: GroupKFold grouped by test date (no row-level random splits). Report balanced accuracy + ROC-AUC
  for extinction; R^2/MAE for K. Baseline first (logistic / ridge), then gradient boosting.
- Bar to match honestly: ~0.68 balanced acc / ~0.79 AUC (logistic), ~0.70 R^2 for K on unseen days.
- Physics features: d0, d0^2, X_O2, P (atm), X_O2*P, diluent fractions (N2/CO2/He), fuel one-hot,
  O2 margin = X_O2 - LOI(fuel) with microgravity LOI 0.12 methanol / 0.15 heptane.
- d^2-law consistency: burn_time ~ d0^2 / K. Monotonic constraint: higher O2 -> higher K.
- Uncertainty: quantile or conformal intervals; show "distance to nearest real test" for every prediction.
- P1 decisions (from honest GroupKFold metrics, see models/metrics.json):
  * The app uses the logistic models (outcome / Disruption / Extinction) and the Ridge K model; they beat
    gradient boosting on unseen days. The O2-monotonic HGB K model is only a physics sanity check.
  * Completion is shown as LOW CONFIDENCE (only 27 tests; out-of-fold recall ~0.26-0.37).
  * d0^2/K is the full-burnout time: it matches measured burn time only for Completion (r = 0.92). For tests
    that ended early (Extinction/Disruption) show it as an upper limit, never as a prediction.
  * K intervals = split-conformal-corrected 10-90% quantiles (raw quantiles under-covered: 55% vs 80%).
  * 50 FLEX dates are written M/D/YY; parsed explicitly (2009-2011 guard).

## App tabs
1. Catalog & Ranking — ~20-24 PSI combustion investigations; AI insight cards (objective, what varied,
   key result, fire-safety implication, citation); transparent weighted score with sliders
   (habitat relevance, O2/pressure overlap with exploration atmosphere, flow overlap 0-30 cm/s,
   data availability, suppression/detection relevance).
2. Flame Predictor — sliders (fuel, O2, P, diluent, d0) -> extinction probability, K, burn time,
   d^2 curve, uncertainty, nearest-real-test distance.
3. Oxygen-Margin Map — all tested conditions in O2 x pressure (x flow) space; overlay 34%/8.2 psia and
   30%/10.2 psi targets; BASS-II -2.4% oxygen-margin callout; coverage-gap shading.
4. Ask Fire-Margin (SI layer) — RAG chatbot grounded ONLY on data/knowledge/; every answer cites
   investigation/DOI; refuses when not grounded.
5. Lessons & Earth Impact — Apollo 1, Apollo 13, Mir 1997, NASA spinoffs (air tanks, fire detector, fire attire).

Stretch (all planned): next-experiment recommender (max uncertainty x distance to existing tests);
0g-1g partial-gravity bounding band for lunar 0.16 g (Grashof/pressure scaling, labelled heuristic);
Saffire-III telemetry replay; cool-flame risk flag from ACME CFI-G.

## Phases (tick as done)
- [x] P1 Data ingest + cleaning + FLEX models + metrics
- [x] P2 Investigation catalog + ranking engine + coverage analysis
- [x] P3 Knowledge base + RAG chatbot + insight cards
- [x] P4 Streamlit app (all tabs) + deploy-ready (Streamlit Cloud deploy itself is a team step: Python 3.12+, add secrets)
- [x] P5 Stretch goals: next-experiment recommender (P2); cool-flame flag (src/coolflame.py: 7 yes / 1 no /
  171 unknown, PSI-159); partial-gravity Grashof heuristic (src/partial_gravity.py, labelled, 6th "ground-test
  analogue" 34% O2 / ~0.22 atm); Saffire-III telemetry replay (src/telemetry.py -> data/processed/
  saffire3_telemetry_small.csv, 1.4 MB): CygnusData channels with published units only (psi, degC, V);
  multiplexed placeholders (0 in CygnusData, the start-of-file value in SaffireData) treated as missing;
  radiometers/anemometers shown as change from baseline in raw counts (unitless), baseline = median of the
  first 60 s of REAL samples (the file's first 60 s are placeholders only); igniter-channel pulses (Sample*IgniterI > 0.5; unit not labelled)
  marked at ~5759 s (igniter 2) and ~5954 s (igniter 1). PSI-100 DOI 10.60555/040q-4h36.
- [ ] P6 README, data dictionary, AI disclosure, 30-s video, <=7 slides, submission

## Context & memory tools
- graphify: if `graphify-out/GRAPH_REPORT.md` exists, read it before exploring files; query the graph
  instead of grepping. Rebuild with `/graphify .` after each phase (incremental, cached).
- claude-mem: at session start, use the injected memory; search it (mem-search) before re-deriving
  past decisions. Don't re-read files just to recall what an earlier session did.

## Working rules for Claude Code
- Don't read `.venv/`, `data/raw/*` beyond `head`/schema, or large telemetry files in full.
- Keep functions small; scripts runnable as `python -m src.<module>`.
- Run code and tests before claiming anything works.
- Log AI assistance in `docs/AI_DISCLOSURE.md` (tool + what it did) as you go.
