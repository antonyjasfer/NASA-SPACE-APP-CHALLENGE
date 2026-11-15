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

## 2026-11-15 — P2 fixes, Phase P3 and P4 (Claude Code, Anthropic)
Tool: Claude Code with a Python-review sub-agent; graphify for the code map.
Human: verified the literature values against the primary sources (Saffire IV-VI test matrix,
PMMA-rod Scientific Reports paper, BASS-II ICES 2019) and chose the app content.

What the AI did:
- Replaced unverified literature rows with the team-verified ones (value_verified + URL); removed the
  unverifiable BASS-II 13.6%/16% row; made planned/capability ranges count as "not evidence" in the
  ranking; reworked the recommender to use weighted evidence (PSI measured 1.0, PSI with flagged
  assumed 1 atm 1.0, verified literature 0.5); updated the headline from computed numbers.
- `src/knowledge.py`: chunks the Researcher's Guide (~800 chars, page kept), insight cards, literature,
  lessons and recommendations into `data/knowledge/chunks.jsonl`; TF-IDF retrieval with a similarity
  threshold and a term-overlap gate, tuned so off-topic questions are refused.
- `src/rag.py`: grounded answers through the Meta -> NVIDIA chain with mandatory [source, page]
  citations; refusal below threshold; retrieved passages shown when every provider fails.
- `src/lessons.py`: Apollo 1, Apollo 13, Mir 1997 and three NASA spinoffs, each with a NASA source
  found by web search (links listed in the file).
- `app/app.py`: the five-tab Streamlit app (dark theme), plus README, pinned requirements and
  `.streamlit/config.toml`. Found and fixed a pickling bug (nearest-test index saved under `__main__`).
- 21 new tests (79 total). The chatbot could not be exercised against a live LLM here: no API key is
  available in the build environment, so only the refusal and fallback paths were run end to end.
- Fixed the automated review's findings: a Python-3.12-only f-string, 24-hour caching of "AI
  unavailable" fallbacks, unverified model citations (answers must now cite only retrieved passages,
  otherwise the passages are shown), a looser short-question gate (now needs 70% of question words in
  the corpus), unbounded input/history, unlabelled assumed pressure for the PMMA-rod points, fragile
  plotting, unescaped HTML. A browser click-through also caught markdown links not rendering inside
  HTML; fixed with real anchors. 85 tests pass.

## 2026-11-15 — Phase P5 stretch goals (Claude Code, Anthropic)
- `src/coolflame.py`: cool-flame flag from PSI-159 (7 yes, 1 no; 171 blank cells kept as "unknown",
  never counted as "no"); "Invisible cool flames" card citing PSI-159 and the Researcher's Guide
  (p. 33-34, 36, 41-42); heptane note in the Flame Predictor; cool-flame markers on the map.
- `src/partial_gravity.py`: Grashof-scaling ground-test analogue (p_1g = p * sqrt(g/g_earth); Moon
  factor 0.40, Mars 0.62), labelled as a heuristic everywhere; map toggle with caption; a separate
  6th "ground-test analogue" recommendation (34% O2, ~0.22 atm). The evidence-gap top 5 are unchanged.
- Saffire-III telemetry: inspected the workbook's sheets and headers only; stopped because several
  channels (radiometers, anemometers) have no stated units. Nothing was converted or plotted.
- 6 new tests (91 total); browser click-through of the map toggle and the heptane note.
