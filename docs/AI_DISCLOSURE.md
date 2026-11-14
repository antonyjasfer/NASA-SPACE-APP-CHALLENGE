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
