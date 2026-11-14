# Graph Report - NASA-SPACE-APP-CHALLENGE  (2026-11-14)

## Corpus Check
- Corpus is ~23,873 words - fits in a single context window. You may not need a graph.

## Summary
- 269 nodes · 531 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 41 edges (avg confidence: 0.86)
- Token cost: 186,347 input · 0 output

## Community Hubs (Navigation)
- Model Training Pipeline
- NASA Investigations & Guide
- Project Rules & Insights
- FLEX Ingest & Tests
- LLM Provider Chain
- Physics Features
- Coverage Analysis
- Streamlit App Shell

## God Nodes (most connected - your core abstractions)
1. `A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)` - 27 edges
2. `clean()` - 13 edges
3. `run_k()` - 13 edges
4. `build_features()` - 12 edges
5. `load_clean()` - 11 edges
6. `chat()` - 11 edges
7. `_r()` - 11 edges
8. `fit_cqr()` - 11 edges
9. `run_outcome()` - 11 edges
10. `run_binary()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Exploration atmosphere target (34% O2 / 8.2 psia; >=30% O2 at <=10.2 psi)` --semantically_similar_to--> `Space Exploration Atmospheres (SEA), airflow 0-30 cm/s`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `BASS-II table PSI-25 (solids, 13.6-22.2% O2)` --semantically_similar_to--> `Burning and Suppression of Solids (BASS)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `BASS table PSI-26 (catalog only)` --semantically_similar_to--> `Burning and Suppression of Solids (BASS)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `Tab 3: Oxygen-Margin Map` --semantically_similar_to--> `Flammability map with negative oxygen margin of safety (Fig. 8)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf
- `FLEX dataset PSI-69 (DOI 10.60555/mbq8-0451)` --semantically_similar_to--> `Flame Extinguishment Experiment (FLEX)`  [INFERRED] [semantically similar]
  CLAUDE.md → data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ACME experiment suite on CIR** — data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_acme, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_bre, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_cld_flame, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_e_field_flames, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_flame_design, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_s_flame, data_raw_np_2015_10_034_jsc_combustion_in_reduced_gravity_iss_mini_book_011116_508_cir [EXTRACTED 1.00]
- **PSI tables feeding the O2 x pressure coverage map** — claude_flex_psi_69, claude_bass_ii_psi_25, claude_flame_design_psi_10, claude_cfi_g_psi_159, claude_saffire_psi_98_99_100, claude_coverage_insight, claude_tab_oxygen_margin_map [EXTRACTED 1.00]
- **FLEX cleaning quality flags (no rows dropped)** — data_processed_flex_data_dictionary_flex_clean, data_processed_flex_data_dictionary_burn_time_approx, data_processed_flex_data_dictionary_dup_id, data_processed_flex_data_dictionary_p_was_zero, claude_data_rules [INFERRED 0.85]

## Communities (10 total, 3 thin omitted)

### Community 0 - "Model Training Pipeline"
Cohesion: 0.08
Nodes (31): _assert_no_shared_dates(), cv_classifier(), cv_quantile_interval(), cv_regressor(), CVResult, d2law_check(), fit_cqr(), _fold_dates() (+23 more)

### Community 1 - "NASA Investigations & Guide"
Cohesion: 0.05
Nodes (50): BASS table PSI-26 (catalog only), CFI-G normal flames table PSI-159, Cool-flame risk flag from ACME CFI-G (stretch), Flame Design inverse flame table PSI-10, 0g-1g partial-gravity bounding band for lunar 0.16 g (stretch), Tab 5: Lessons & Earth Impact (Apollo 1, Apollo 13, Mir 1997, spinoffs), Advanced Combustion via Microgravity Experiments (ACME), Apollo 1 (1967) and Apollo 13 (1970) accidents (+42 more)

### Community 2 - "Project Rules & Insights"
Cohesion: 0.06
Nodes (39): BASS-II table PSI-25 (solids, 13.6-22.2% O2), Split-conformal-corrected K quantile intervals, d^2-law burn time ~ d0^2/K, Exploration atmosphere target (34% O2 / 8.2 psia; >=30% O2 at <=10.2 psi), Fire-Margin (evidence-graded fire safety dashboard), FLEX dataset PSI-69 (DOI 10.60555/mbq8-0451), Pre-generated insight cards (insight_cards.json), JSC 29353D (min 33.7% O2 at 8.2 psia) (+31 more)

### Community 3 - "FLEX Ingest & Tests"
Cohesion: 0.10
Nodes (18): clean(), load_clean(), load_raw(), main(), _parse_burn_time(), parse_test_date(), validate_schema(), write_data_dictionary() (+10 more)

### Community 4 - "LLM Provider Chain"
Cohesion: 0.10
Nodes (19): _cache_key(), chat(), _chat(), _default_client_factory(), get_setting(), _is_retryable(), LLMResult, load_providers() (+11 more)

### Community 5 - "Physics Features"
Cohesion: 0.19
Nodes (5): build_features(), d2law_burn_time(), main(), test_build_features_does_not_mutate_or_add_rows(), test_d2law_burn_time()

### Community 6 - "Coverage Analysis"
Cohesion: 0.39
Nodes (4): coverage_table(), disruption_contrast(), main(), test_coverage_counts_every_real_test()

## Ambiguous Edges - Review These
- `Microgravity Science Glovebox (MSG)` → `Super-Critical Water Oxidation (SCWO) and SCWM`  [AMBIGUOUS]
  data/raw/np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf · relation: conceptually_related_to

## Knowledge Gaps
- **31 isolated node(s):** `JSC 29353D (min 33.7% O2 at 8.2 psia)`, `BASS table PSI-26 (catalog only)`, `Saffire-I/II/III tables PSI-98/99/100`, `Next-experiment recommender (stretch)`, `0g-1g partial-gravity bounding band for lunar 0.16 g (stretch)` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 92 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A Researcher's Guide to ISS Combustion Science (NP-2015-10-034-JSC)` connect `NASA Investigations & Guide` to `Project Rules & Insights`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **What connects `JSC 29353D (min 33.7% O2 at 8.2 psia)`, `BASS table PSI-26 (catalog only)`, `Saffire-I/II/III tables PSI-98/99/100` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Model Training Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07936507936507936 - nodes in this community are weakly interconnected._
- **Why does `build_features()` connect `Physics Features` to `Model Training Pipeline`, `FLEX Ingest & Tests`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Should `NASA Investigations & Guide` be split into smaller, more focused modules?**
  _Cohesion score 0.05333333333333334 - nodes in this community are weakly interconnected._
- **Should `Project Rules & Insights` be split into smaller, more focused modules?**
  _Cohesion score 0.06183574879227053 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Microgravity Science Glovebox (MSG)` and `Super-Critical Water Oxidation (SCWO) and SCWM`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._