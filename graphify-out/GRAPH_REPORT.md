# Graph Report - NASA-SPACE-APP-CHALLENGE  (2026-11-15)

## Corpus Check
- 33 files · ~42,288 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 30 file(s) not represented in the graph (top: .csv 14, .joblib 10, (none) 3)
- WARNING: 30 of 65 files seen (46%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 503 nodes · 1180 edges · 23 communities (18 shown, 5 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 70 edges (avg confidence: 0.87)
- Token cost: 186,347 input · 0 output

## Graph Freshness
- Built from commit: `35797dd4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Model Training Pipeline
- A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)
- flex_clean.csv (274 rows, no rows added/dropped/imputed)
- FLEX Ingest & Tests
- LLM Provider Chain
- test_p3.py
- test_p2.py
- coolflame.py
- coverage.py
- psi_tables.py
- Solid Fuel Ignition and Extinction (SoFIE): 10-40% O2, 0.2-1.5 atm
- Coverage insight: gas/droplet data touch Moon-base O2, cabin solids do not
- Cool flames after hot-flame extinction of heptane droplets
- LLM provider chain (Meta Llama API -> NVIDIA NIM) in src/llm_config.py
- Advanced Combustion via Microgravity Experiments (ACME)
- app.py
- Flammability map with negative oxygen margin of safety (Fig. 8)
- flex_clean.csv (274 rows, no rows added/dropped/imputed)

## God Nodes (most connected - your core abstractions)
1. `A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)` - 27 edges
2. `answer()` - 22 edges
3. `recommend_experiments()` - 16 edges
4. `load_clean()` - 15 edges
5. `rank_investigations()` - 15 edges
6. `coverage_points()` - 14 edges
7. `build_features()` - 14 edges
8. `Retriever` - 13 edges
9. `LLMResult` - 13 edges
10. `clean()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `BASS table PSI-26 (catalog only)` --semantically_similar_to--> `Burning and Suppression of Solids (BASS)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `BASS-II table PSI-25 (solids, 13.6-22.2% O2)` --semantically_similar_to--> `Burning and Suppression of Solids (BASS)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `Exploration atmosphere target (34% O2 / 8.2 psia; >=30% O2 at <=10.2 psi)` --semantically_similar_to--> `Space Exploration Atmospheres (SEA), airflow 0-30 cm/s`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `FLEX dataset PSI-69 (DOI 10.60555/mbq8-0451)` --semantically_similar_to--> `Flame Extinguishment Experiment (FLEX)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `Tab 3: Oxygen-Margin Map` --semantically_similar_to--> `Flammability map with negative oxygen margin of safety (Fig. 8)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ACME experiment suite on CIR** — data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_acme, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_bre, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_cld_flame, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_e_field_flames, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_flame_design, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_s_flame, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_cir [EXTRACTED 1.00]
- **PSI tables feeding the O2 x pressure coverage map** — claude_flex_psi_69, claude_bass_ii_psi_25, claude_flame_design_psi_10, claude_cfi_g_psi_159, claude_saffire_psi_98_99_100, claude_coverage_insight, claude_tab_oxygen_margin_map [EXTRACTED 1.00]
- **FLEX cleaning quality flags (no rows dropped)** — data_processed_flex_data_dictionary_flex_clean, data_processed_flex_data_dictionary_burn_time_approx, data_processed_flex_data_dictionary_dup_id, data_processed_flex_data_dictionary_p_was_zero, claude_data_rules [INFERRED 0.85]

## Communities (23 total, 5 thin omitted)

### Community 0 - "Model Training Pipeline"
Cohesion: 0.08
Nodes (31): _assert_no_shared_dates(), cv_classifier(), cv_quantile_interval(), cv_regressor(), CVResult, d2law_check(), fit_cqr(), _fold_dates() (+23 more)

### Community 1 - "A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)"
Cohesion: 0.11
Nodes (23): BASS table PSI-26 (catalog only), Tab 5: Lessons & Earth Impact (Apollo 1, Apollo 13, Mir 1997, spinoffs), Apollo 1 (1967) and Apollo 13 (1970) accidents, Burning and Suppression of Solids (BASS), Candle flames in microgravity (Mir), flame existence, Dietrich et al. 2000, Candle flames in non-buoyant atmospheres, Particulate+species fire detector (beat aircraft detectors in FAA tests), Friedman & Ross 2001, Microgravity Combustion: Fire in Free Fall (+15 more)

### Community 2 - "flex_clean.csv (274 rows, no rows added/dropped/imputed)"
Cohesion: 0.19
Nodes (10): Split-conformal-corrected K quantile intervals, d^2-law burn time ~ d0^2/K, Distance to nearest real test, Next-experiment recommender (stretch), Tab 2: Flame Predictor, K_mm2_s burning-rate constant (mm^2/s), test_date (80 dates, CV group; M/D/YY parsed), Undocumented M/D/YY date format in 50 FLEX rows (+2 more)

### Community 3 - "FLEX Ingest & Tests"
Cohesion: 0.07
Nodes (25): build_features(), d2law_burn_time(), main(), clean(), load_clean(), load_raw(), main(), _parse_burn_time() (+17 more)

### Community 4 - "LLM Provider Chain"
Cohesion: 0.10
Nodes (18): _cache_key(), chat(), _chat(), _default_client_factory(), get_setting(), _is_retryable(), load_providers(), Provider (+10 more)

### Community 5 - "test_p3.py"
Cohesion: 0.07
Nodes (38): build_chunks(), card_chunks(), Chunk, clean_text(), Hit, lesson_chunks(), literature_chunks(), load_chunks() (+30 more)

### Community 6 - "test_p2.py"
Cohesion: 0.06
Nodes (38): build_investigations(), build_literature(), _cite(), investigation_cards(), main(), _psi_key_results(), psi_overrides(), table_range() (+30 more)

### Community 7 - "coolflame.py"
Cohesion: 0.35
Nodes (6): card(), cool_flame_tests(), counts(), main(), observed(), test_cool_flame_counts_keep_unknown_separate()

### Community 8 - "coverage.py"
Cohesion: 0.08
Nodes (24): add_ground_test_analogue(), coverage_points(), coverage_table(), disruption_contrast(), evidence_points(), _explain(), main(), recommend_experiments() (+16 more)

### Community 10 - "psi_tables.py"
Cohesion: 0.17
Nodes (17): _check_fraction(), _drop_non_tests(), load_all(), load_bass(), load_bass2(), load_cfig(), load_inverse_flame(), _load_saffire() (+9 more)

### Community 11 - "Solid Fuel Ignition and Extinction (SoFIE): 10-40% O2, 0.2-1.5 atm"
Cohesion: 0.32
Nodes (8): Correlation of Flammability between Normal and Microgravity, External Heat Flux Effects on Ignitability and Suppression, Fundamentals of Flame Growth and Extinction in Microgravity, Narrow Channel as a 1-g Flammability Test Method, NASA-STD-6001 Test 1 (may not be conservative), Space Exploration Atmospheres (SEA), airflow 0-30 cm/s, Solid Fuel Ignition and Extinction (SoFIE): 10-40% O2, 0.2-1.5 atm, Thermal and Species Field Effects on Material Flammability

### Community 12 - "Coverage insight: gas/droplet data touch Moon-base O2, cabin solids do not"
Cohesion: 0.20
Nodes (12): BASS-II table PSI-25 (solids, 13.6-22.2% O2), Exploration atmosphere target (34% O2 / 8.2 psia; >=30% O2 at <=10.2 psi), Fire-Margin (evidence-graded fire safety dashboard), FLEX dataset PSI-69 (DOI 10.60555/mbq8-0451), JSC 29353D (min 33.7% O2 at 8.2 psia), Literature points (Saffire-VI SIBAL 30% O2; BASS-II 13.6% vs 16% blowoff), Saffire-I/II/III tables PSI-98/99/100, Stack: Python 3.11, streamlit, scikit-learn, Streamlit Community Cloud (+4 more)

### Community 13 - "Cool flames after hot-flame extinction of heptane droplets"
Cohesion: 0.20
Nodes (11): CFI-G normal flames table PSI-159, Cool-flame risk flag from ACME CFI-G (stretch), O2 margin = X_O2 - LOI(fuel) (0.12 methanol / 0.15 heptane), Physics features (d0, d0^2, X_O2, P, X_O2*P, diluents, fuel), x_o2 initial ambient O2 mole fraction, Cool flames after hot-flame extinction of heptane droplets, Flame Extinguishment Experiment (FLEX), FLEX-2 (fuel mixtures, surrogates, sooting, droplet arrays) (+3 more)

### Community 14 - "LLM provider chain (Meta Llama API -> NVIDIA NIM) in src/llm_config.py"
Cohesion: 0.40
Nodes (5): Pre-generated insight cards (insight_cards.json), data/knowledge RAG text chunks, LLM provider chain (Meta Llama API -> NVIDIA NIM) in src/llm_config.py, Tab 4: Ask Fire-Margin RAG chatbot, Tab 1: Catalog & Ranking (weighted score sliders)

### Community 15 - "Advanced Combustion via Microgravity Experiments (ACME)"
Cohesion: 0.22
Nodes (9): Flame Design inverse flame table PSI-10, Advanced Combustion via Microgravity Experiments (ACME), Burning Rate Emulator (BRE), Combustion Integrated Rack (CIR, 100-liter chamber), Coflow Laminar Diffusion Flame (CLD Flame), Electric-Field Effects on Laminar Diffusion Flames (E-FIELD), Flame Design experiment (soot inception, inverse flames), Multi-user Droplet Combustion Apparatus (MDCA): 1-5 mm droplets, 0-3 atm, up to 40% O2 (+1 more)

### Community 16 - "app.py"
Cohesion: 0.12
Nodes (20): a_link(), _cached_answer(), completion_note(), d2_curve(), fmt_range(), footer(), get_answer(), header() (+12 more)

### Community 17 - "Flammability map with negative oxygen margin of safety (Fig. 8)"
Cohesion: 0.40
Nodes (5): 0g-1g partial-gravity bounding band for lunar 0.16 g (stretch), Flammability map with negative oxygen margin of safety (Fig. 8), Low-stretch (low-velocity) flames, Olson, Ferkul, T'ien 1988, Near-limit flame spread over thin solid fuel, Upstream flame spread at low flow in microgravity (Fig. 9)

### Community 21 - "flex_clean.csv (274 rows, no rows added/dropped/imputed)"
Cohesion: 0.29
Nodes (6): burn_time_approx flag, dup_id flag (193F001), flex_clean.csv (274 rows, no rows added/dropped/imputed), p_was_zero flag (C11M201), test_end outcome (Extinction/Disruption/Completion), x_co2 (CO2, not carbon monoxide)

## Ambiguous Edges - Review These
- `Microgravity Science Glovebox (MSG)` → `Super-Critical Water Oxidation (SCWO) and SCWM`  [AMBIGUOUS]
  data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf · relation: conceptually_related_to

## Knowledge Gaps
- **31 isolated node(s):** `BASS table PSI-26 (catalog only)`, `Early gas-jet flame experiments (LSP, TGDF, ELF)`, `Local N2 suppressant can strengthen flame via O2 entrainment`, `Smoke Aerosol Measurement Experiment (SAME) / Dust Aerosol test`, `SAMS / MAMS acceleration measurement` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 153 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `answer()` connect `test_p3.py` to `app.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `answer()` (e.g. with `Retriever` and `.search()`) actually correct?**
  _`answer()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `BASS table PSI-26 (catalog only)`, `Early gas-jet flame experiments (LSP, TGDF, ELF)`, `Local N2 suppressant can strengthen flame via O2 entrainment` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Model Training Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.08418079096045197 - nodes in this community are weakly interconnected._
- **Why does `load_providers()` connect `LLM Provider Chain` to `app.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Should `A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)` be split into smaller, more focused modules?**
  _Cohesion score 0.10869565217391304 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Microgravity Science Glovebox (MSG)` and `Super-Critical Water Oxidation (SCWO) and SCWM`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._