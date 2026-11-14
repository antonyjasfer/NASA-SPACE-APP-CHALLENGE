# Fire-Margin — NASA Space Apps 2026: "Flame in Freefall"

Evidence-graded fire safety for Moon-base atmospheres. An SI-powered (AI) web dashboard that
summarizes, ranks and interprets NASA microgravity combustion investigations (PSI repository).

**Deadline: submit before 11:59 p.m. IST, Sunday 15 Nov 2026.** Target submit time: 10 p.m.

## Headline insight (the pitch)
NASA plans lunar/Martian habitats at ~34% O2 / 8.2 psia (JSC 29353D: min 33.7% O2 at 8.2 psia;
challenge page: >=30% O2 at <=10.2 psi). Most microgravity combustion data in PSI sits at <=21-22% O2.
Show exactly where evidence exists, where it is missing, and which experiment NASA should run next.

## Stack
Python 3.11, pandas, numpy, scikit-learn, scipy, plotly, streamlit, joblib, pypdf, openai (SDK only).
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
- FLEX (PSI-69, DOI 10.60555/mbq8-0451): CSV encoding cp1252. Rename "Burning rate; mm" -> `K_mm2_s`
  (units are mm^2/s). FLEX-114 pressure 0 -> NaN. Flag duplicate IDs FLEX-070/073, don't drop silently.
  Keep missing values as NaN. Test end classes: Extinction / Disruption / Completion.
- BASS/BASS-II tables have NO spread-rate column. Don't invent one; use them for the coverage map only.
- Literature values (e.g., Saffire-VI SIBAL at 30% O2, 0.54 atm; BASS-II steady spread at 13.6% O2 vs
  1-g blowoff 16%) are plotted as clearly labelled "literature points", with citation, never mixed into training data.
- Every number shown in the app traces to a PSI ID / DOI / paper.

## Modelling rules
- Validation: GroupKFold grouped by test date (no row-level random splits). Report balanced accuracy + ROC-AUC
  for extinction; R^2/MAE for K. Baseline first (logistic / ridge), then gradient boosting.
- Bar to match honestly: ~0.68 balanced acc / ~0.79 AUC (logistic), ~0.70 R^2 for K on unseen days.
- Physics features: d0, d0^2, X_O2, P (atm), X_O2*P, diluent fractions (N2/CO2/He), fuel one-hot,
  O2 margin = X_O2 - LOI(fuel) with microgravity LOI 0.12 methanol / 0.15 heptane.
- d^2-law consistency: burn_time ~ d0^2 / K. Monotonic constraint: higher O2 -> higher K.
- Uncertainty: quantile or conformal intervals; show "distance to nearest real test" for every prediction.

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
- [ ] P1 Data ingest + cleaning + FLEX models + metrics
- [ ] P2 Investigation catalog + ranking engine + coverage analysis
- [ ] P3 Knowledge base + RAG chatbot + insight cards
- [ ] P4 Streamlit app (all tabs) + deploy
- [ ] P5 Stretch goals
- [ ] P6 README, data dictionary, AI disclosure, 30-s video, <=7 slides, submission

## Working rules for Claude Code
- Don't read `.venv/`, `data/raw/*` beyond `head`/schema, or large telemetry files in full.
- Keep functions small; scripts runnable as `python -m src.<module>`.
- Run code and tests before claiming anything works.
- Log AI assistance in `docs/AI_DISCLOSURE.md` (tool + what it did) as you go.
