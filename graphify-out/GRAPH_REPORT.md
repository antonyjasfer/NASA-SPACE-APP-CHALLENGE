# Graph Report - NASA-SPACE-APP-CHALLENGE  (2026-11-14)

## Corpus Check
- 25 files · ~33,350 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 28 file(s) not represented in the graph (top: .csv 14, .joblib 10, (none) 3)
- WARNING: 28 of 55 files seen (51%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 374 nodes · 806 edges · 21 communities (16 shown, 5 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 53 edges (avg confidence: 0.86)
- Token cost: 186,347 input · 0 output

## Graph Freshness
- Built from commit: `ef0383a3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Model Training Pipeline
- A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)
- flex_clean.csv (274 rows, no rows added/dropped/imputed)
- FLEX Ingest & Tests
- LLM Provider Chain
- Physics Features
- test_p2.py
- ranking.py
- psi_tables.py
- Solid Fuel Ignition and Extinction (SoFIE): 10-40% O2, 0.2-1.5 atm
- Coverage insight: gas/droplet data touch Moon-base O2, cabin solids do not
- Cool flames after hot-flame extinction of heptane droplets
- Fire-Margin (evidence-graded fire safety dashboard)
- Advanced Combustion via Microgravity Experiments (ACME)
- Spacecraft fire safety misconceptions (quiescent, ventilation-off, inherently safer)
- Flammability map with negative oxygen margin of safety (Fig. 8)

## God Nodes (most connected - your core abstractions)
1. `A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)` - 27 edges
2. `load_clean()` - 15 edges
3. `recommend_experiments()` - 13 edges
4. `run_k()` - 13 edges
5. `clean()` - 13 edges
6. `coverage_points()` - 12 edges
7. `rank_investigations()` - 12 edges
8. `build_features()` - 12 edges
9. `chat()` - 11 edges
10. `_r()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `BASS table PSI-26 (catalog only)` --semantically_similar_to--> `Burning and Suppression of Solids (BASS)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `BASS-II table PSI-25 (solids, 13.6-22.2% O2)` --semantically_similar_to--> `Burning and Suppression of Solids (BASS)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `Tab 3: Oxygen-Margin Map` --semantically_similar_to--> `Flammability map with negative oxygen margin of safety (Fig. 8)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `Exploration atmosphere target (34% O2 / 8.2 psia; >=30% O2 at <=10.2 psi)` --semantically_similar_to--> `Space Exploration Atmospheres (SEA), airflow 0-30 cm/s`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `FLEX dataset PSI-69 (DOI 10.60555/mbq8-0451)` --semantically_similar_to--> `Flame Extinguishment Experiment (FLEX)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ACME experiment suite on CIR** — data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_acme, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_bre, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_cld_flame, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_e_field_flames, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_flame_design, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_s_flame, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_cir [EXTRACTED 1.00]
- **PSI tables feeding the O2 x pressure coverage map** — claude_flex_psi_69, claude_bass_ii_psi_25, claude_flame_design_psi_10, claude_cfi_g_psi_159, claude_saffire_psi_98_99_100, claude_coverage_insight, claude_tab_oxygen_margin_map [EXTRACTED 1.00]
- **FLEX cleaning quality flags (no rows dropped)** — data_processed_flex_data_dictionary_flex_clean, data_processed_flex_data_dictionary_burn_time_approx, data_processed_flex_data_dictionary_dup_id, data_processed_flex_data_dictionary_p_was_zero, claude_data_rules [INFERRED 0.85]

## Communities (21 total, 5 thin omitted)

### Community 0 - "Model Training Pipeline"
Cohesion: 0.08
Nodes (32): _assert_no_shared_dates(), cv_classifier(), cv_quantile_interval(), cv_regressor(), CVResult, d2law_check(), fit_cqr(), _fold_dates() (+24 more)

### Community 1 - "A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)"
Cohesion: 0.16
Nodes (15): Candle flames in microgravity (Mir), flame existence, Dietrich et al. 2000, Candle flames in non-buoyant atmospheres, Early gas-jet flame experiments (LSP, TGDF, ELF), Proposed Microgravity Wind Tunnel for fire research, Microgravity Science Glovebox (MSG), A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC), Ronney et al. 1998, Flame Balls in Space (STS-83), Smoke Aerosol Measurement Experiment (SAME) / Dust Aerosol test (+7 more)

### Community 2 - "flex_clean.csv (274 rows, no rows added/dropped/imputed)"
Cohesion: 0.11
Nodes (19): Split-conformal-corrected K quantile intervals, d^2-law burn time ~ d0^2/K, Distance to nearest real test, Next-experiment recommender (stretch), O2 margin = X_O2 - LOI(fuel) (0.12 methanol / 0.15 heptane), Physics features (d0, d0^2, X_O2, P, X_O2*P, diluents, fuel), Tab 2: Flame Predictor, burn_time_approx flag (+11 more)

### Community 3 - "FLEX Ingest & Tests"
Cohesion: 0.07
Nodes (23): main(), clean(), load_clean(), load_raw(), main(), _parse_burn_time(), parse_test_date(), validate_schema() (+15 more)

### Community 4 - "LLM Provider Chain"
Cohesion: 0.10
Nodes (19): _cache_key(), chat(), _chat(), _default_client_factory(), get_setting(), _is_retryable(), LLMResult, load_providers() (+11 more)

### Community 5 - "Physics Features"
Cohesion: 0.18
Nodes (6): build_features(), d2law_burn_time(), main(), test_build_features_does_not_mutate_or_add_rows(), test_d2law_burn_time(), test_unknown_fuel_raises()

### Community 6 - "test_p2.py"
Cohesion: 0.09
Nodes (27): build_investigations(), build_literature(), _cite(), investigation_cards(), main(), _psi_key_results(), psi_overrides(), table_range() (+19 more)

### Community 7 - "ranking.py"
Cohesion: 0.11
Nodes (16): _box_distance(), flow_score(), main(), measured_by_psi(), normalise_weights(), _num(), o2_pressure_score(), rank_investigations() (+8 more)

### Community 10 - "psi_tables.py"
Cohesion: 0.17
Nodes (16): _check_fraction(), _drop_non_tests(), load_bass(), load_bass2(), load_cfig(), load_inverse_flame(), _load_saffire(), load_saffire1() (+8 more)

### Community 11 - "Solid Fuel Ignition and Extinction (SoFIE): 10-40% O2, 0.2-1.5 atm"
Cohesion: 0.24
Nodes (11): BASS table PSI-26 (catalog only), Burning and Suppression of Solids (BASS), Correlation of Flammability between Normal and Microgravity, External Heat Flux Effects on Ignitability and Suppression, Fundamentals of Flame Growth and Extinction in Microgravity, Narrow Channel as a 1-g Flammability Test Method, NASA-STD-6001 Test 1 (may not be conservative), Local N2 suppressant can strengthen flame via O2 entrainment (+3 more)

### Community 12 - "Coverage insight: gas/droplet data touch Moon-base O2, cabin solids do not"
Cohesion: 0.31
Nodes (7): BASS-II table PSI-25 (solids, 13.6-22.2% O2), Exploration atmosphere target (34% O2 / 8.2 psia; >=30% O2 at <=10.2 psi), FLEX dataset PSI-69 (DOI 10.60555/mbq8-0451), JSC 29353D (min 33.7% O2 at 8.2 psia), Literature points (Saffire-VI SIBAL 30% O2; BASS-II 13.6% vs 16% blowoff), Saffire-I/II/III tables PSI-98/99/100, Tab 3: Oxygen-Margin Map

### Community 13 - "Cool flames after hot-flame extinction of heptane droplets"
Cohesion: 0.22
Nodes (10): CFI-G normal flames table PSI-159, Cool-flame risk flag from ACME CFI-G (stretch), Combustion Integrated Rack (CIR, 100-liter chamber), Cool flames after hot-flame extinction of heptane droplets, Flame Extinguishment Experiment (FLEX), FLEX-2 (fuel mixtures, surrogates, sooting, droplet arrays), LOI significantly lower in microgravity than normal gravity, Multi-user Droplet Combustion Apparatus (MDCA): 1-5 mm droplets, 0-3 atm, up to 40% O2 (+2 more)

### Community 14 - "Fire-Margin (evidence-graded fire safety dashboard)"
Cohesion: 0.22
Nodes (10): Fire-Margin (evidence-graded fire safety dashboard), Pre-generated insight cards (insight_cards.json), data/knowledge RAG text chunks, LLM provider chain (Meta Llama API -> NVIDIA NIM) in src/llm_config.py, Stack: Python 3.11, streamlit, scikit-learn, Streamlit Community Cloud, Tab 4: Ask Fire-Margin RAG chatbot, Tab 1: Catalog & Ranking (weighted score sliders), NASA-SPACE-APP-CHALLENGE (README title) (+2 more)

### Community 15 - "Advanced Combustion via Microgravity Experiments (ACME)"
Cohesion: 0.29
Nodes (7): Flame Design inverse flame table PSI-10, Advanced Combustion via Microgravity Experiments (ACME), Burning Rate Emulator (BRE), Coflow Laminar Diffusion Flame (CLD Flame), Electric-Field Effects on Laminar Diffusion Flames (E-FIELD), Flame Design experiment (soot inception, inverse flames), Structure and Response of Spherical Diffusion Flames (s-Flame)

### Community 16 - "Spacecraft fire safety misconceptions (quiescent, ventilation-off, inherently safer)"
Cohesion: 0.33
Nodes (5): Tab 5: Lessons & Earth Impact (Apollo 1, Apollo 13, Mir 1997, spinoffs), Apollo 1 (1967) and Apollo 13 (1970) accidents, Particulate+species fire detector (beat aircraft detectors in FAA tests), Friedman & Ross 2001, Microgravity Combustion: Fire in Free Fall, Ruff et al. 2009, Safety Design for Space Systems ch.27 Fire Safety

### Community 17 - "Flammability map with negative oxygen margin of safety (Fig. 8)"
Cohesion: 0.40
Nodes (5): 0g-1g partial-gravity bounding band for lunar 0.16 g (stretch), Flammability map with negative oxygen margin of safety (Fig. 8), Low-stretch (low-velocity) flames, Olson, Ferkul, T'ien 1988, Near-limit flame spread over thin solid fuel, Upstream flame spread at low flow in microgravity (Fig. 9)

## Ambiguous Edges - Review These
- `Microgravity Science Glovebox (MSG)` → `Super-Critical Water Oxidation (SCWO) and SCWM`  [AMBIGUOUS]
  data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf · relation: conceptually_related_to

## Knowledge Gaps
- **31 isolated node(s):** `BASS table PSI-26 (catalog only)`, `0g-1g partial-gravity bounding band for lunar 0.16 g (stretch)`, `Burning Rate Emulator (BRE)`, `Coflow Laminar Diffusion Flame (CLD Flame)`, `Electric-Field Effects on Laminar Diffusion Flames (E-FIELD)` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 122 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)` connect `A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)` to `Solid Fuel Ignition and Extinction (SoFIE): 10-40% O2, 0.2-1.5 atm`, `Cool flames after hot-flame extinction of heptane droplets`, `Fire-Margin (evidence-graded fire safety dashboard)`, `Advanced Combustion via Microgravity Experiments (ACME)`, `Spacecraft fire safety misconceptions (quiescent, ventilation-off, inherently safer)`, `Flammability map with negative oxygen margin of safety (Fig. 8)`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **What connects `BASS table PSI-26 (catalog only)`, `0g-1g partial-gravity bounding band for lunar 0.16 g (stretch)`, `Burning Rate Emulator (BRE)` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Model Training Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.0825136612021858 - nodes in this community are weakly interconnected._
- **Why does `load_clean()` connect `FLEX Ingest & Tests` to `Model Training Pipeline`, `Physics Features`, `test_p2.py`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Should `flex_clean.csv (274 rows, no rows added/dropped/imputed)` be split into smaller, more focused modules?**
  _Cohesion score 0.1067193675889328 - nodes in this community are weakly interconnected._
- **Why does `build_features()` connect `Physics Features` to `Model Training Pipeline`, `FLEX Ingest & Tests`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Microgravity Science Glovebox (MSG)` and `Super-Critical Water Oxidation (SCWO) and SCWM`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._