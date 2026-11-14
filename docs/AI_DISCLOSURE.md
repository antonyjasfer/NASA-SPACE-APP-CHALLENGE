# AI assistance disclosure

Fire-Margin was built with AI coding assistance. This log records which tool did what, so judges
can separate human decisions from AI-generated work. All numbers in the app come from NASA data
(PSI IDs / DOIs), never from an LLM.

## 2026-11-14 — Phase P1 (Claude Code, Anthropic)
Tool: Claude Code (agentic coding assistant) in a cloud session, with a Python-review sub-agent.
Human: set the plan, data rules and modelling rules in CLAUDE.md; downloaded the NASA files.

What the AI did:
- Checked the FLEX CSV schema against CLAUDE.md before coding (274 rows x 15 cols, cp1252, quirks).
  Found one undocumented detail: 50 rows store dates as M/D/YY; verified unambiguous (2009-2011,
  80 distinct dates either way) and parsed both formats explicitly.
- Wrote the scaffold: folders, requirements.txt, .gitignore, `.streamlit/secrets.toml.example`,
  `app/app.py` (5 placeholder tabs), `src/llm_config.py` (Meta -> NVIDIA provider chain, env/secrets
  only, retry with backoff, never raises).
- Wrote `src/ingest.py` (cleaning rules, 274 -> 274 rows, data dictionary), `src/features.py`
  (physics features, d^2-law burn time), `src/models.py` (GroupKFold-by-test-date models, conformalized
  quantile intervals for K, nearest-real-test distance), `src/coverage.py` (O2 x pressure outcome table).
- Wrote 35 pytest tests (cleaning rules, features, fold/date separation, K monotonicity in O2,
  interval sanity, LLM fallback behaviour).
- Ran an automated code review; fixed its findings (two paths where the LLM wrapper could raise,
  calibration split made seeded-random by date, interval clipping, year-range guard), and added an
  Extinction-vs-rest model so results are comparable with the CLAUDE.md benchmark.
- Flagged weak or surprising results instead of hiding them: raw quantile bands under-covered
  (55% vs 80% nominal) -> added split-conformal correction; the monotonic HGB K model scores below
  the Ridge baseline; d0^2/K only matches measured burn time for Completion tests.

No model output, synthetic row or interpolated value was added to the NASA data.

## 2026-11-14 — Phase P2 (Claude Code, Anthropic)
Tool: Claude Code with a Python-review sub-agent and a graphify extraction sub-agent; graphify (pip
`graphifyy`, installed with the team's approval) built the code/doc knowledge graph in `graphify-out/`.
Human: chose the investigations to cover, the ranking factors, the targets and the literature points.

What the AI did:
- Merged the verified extra-PSI data rules from CLAUDE_new.md into CLAUDE.md; recorded P1 decisions.
- Read the full ISS Researcher's Guide (56 pages) and wrote `src/catalog_data.py`: 24 investigations
  where every text field and typed range cites a page (or a PSI ID). Fields the sources do not state
  are left blank; ranges are labelled tested / planned / facility capability.
- Wrote `src/psi_tables.py` to load PSI-25/26/10/159/98/99/100 (non-test rows dropped and counted;
  raw text kept next to parsed numbers; 11 PSI-10 rows with XO2 'RAMP'/'LOW O2' kept with O2 = NaN).
- Wrote `src/catalog.py` (investigations.csv, literature_points.csv, insight_cards.json: 24
  investigation cards + 4 analysis cards), `src/ranking.py` (transparent 0-100 score with a per-factor
  breakdown and "data missing" flags) and extended `src/coverage.py` (coverage_points.csv and a
  next-experiment recommender saved to next_experiments.csv).
- Literature values (BASS-II 13.6% vs 16%, Saffire-VI 30% O2 / 0.54 atm) could not be checked against
  the primary papers from this session (NTRS and the TTU repository were blocked by the network
  policy); they are stored with `value_verified = False` and their candidate sources.
- Wrote 19 new pytest tests (provenance of every range, parser edge cases, ranking determinism and
  slider effects, recommender behaviour) and fixed issues raised by the automated review.

Insight cards are written by the AI from the PDF text and PSI tables only (`generated_by` field);
no LLM API was called and no number was produced by a model.
