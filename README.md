# Fire-Margin

Evidence-graded fire safety for Moon-base atmospheres. NASA Space Apps 2026, challenge "Flame in Freefall".

NASA plans lunar and Mars habitats at about **34% oxygen and 8.2 psia** (0.558 atm), far richer in oxygen
than the ISS. Fire-Margin maps every NASA microgravity combustion test we could trace in the public PSI
repository, shows where the evidence stops, and recommends the experiment that would close the gap.

## The finding

In PSI's public data, solid materials stop at **22.2% O2**. Only 3 solid samples (Saffire-VI, 2024) have
burned near Moon-base conditions (29-31% O2, ~0.54 atm): not in PSI, one flight, no partial gravity. Gas
flames (CFI-G) were tested at 33-39% O2 / 0.50 atm; FLEX droplets reach 34% O2 but only at >= 0.69 atm,
and at O2 >= 25% **80% of FLEX tests (24 of 30) ended in Disruption** (the droplet burst instead of
burning out quietly).

The top recommended experiment is **solid cabin materials at 31-35% O2 and 0.56-0.72 atm**. The only
evidence there is one published flight. Full top-5 list: `data/processed/next_experiments.csv`.

## What the app does

| Tab | What you get |
|---|---|
| Catalog & Ranking | 24 NASA combustion investigations, an insight card for each, and a transparent 0-100 score whose weights you set with sliders. |
| Flame Predictor | Choose fuel, O2, pressure, CO2/He and droplet size to see the outcome probabilities, the burning rate K with a conformal interval, the d0²/K burn-time upper limit, and the distance to the nearest real test. |
| Oxygen-Margin Map | Every tested condition in O2 × pressure, with the two exploration targets, the verified literature points, the top-5 next experiments and the 80% Disruption callout. |
| Ask Fire-Margin | A chatbot that answers only from NASA text (Researcher's Guide, insight cards, verified literature), cites [source, page], and declines anything else. |
| Lessons & Earth Impact | Apollo 1, Apollo 13 and the Mir fire, plus the NASA spinoffs that came from them, each with its NASA source. |

## Run it

```bash
pip install -r requirements.txt
```

```bash
streamlit run app/app.py
```

The app starts without any secrets. The chatbot then shows the matching NASA passages instead of a
written answer. To turn on written answers, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml`, or set the same variables in Streamlit Cloud:

| Variable | Meaning |
|---|---|
| `MODEL_API_KEY` (or `META_API_KEY`) | Meta key (required to enable the chatbot) |
| `META_BASE_URL` | default `https://api.meta.ai/v1` |
| `META_MODEL` | default `muse-wonder-1.0` |
| `META_API_STYLE` | `responses` (default, OpenAI-compatible Responses API) or `chat` (old Llama API path) |
| `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, `NVIDIA_MODEL` | optional fallback (chat completions) |

Never commit real keys.

**Streamlit Community Cloud**: the entry point is `app/app.py`. Choose **Python 3.12 or newer** in Advanced
settings, because the pinned numpy and scipy require it (tested on 3.13).

Rebuild every data product from the raw NASA files:

```bash
python -m src.ingest && python -m src.models && python -m src.catalog && python -m src.coverage && python -m src.knowledge
```

```bash
python -m pytest -q
```

## Data sources

| Source | Used for |
|---|---|
| PSI-69 FLEX droplet table, DOI [10.60555/mbq8-0451](https://doi.org/10.60555/mbq8-0451) | models, coverage, Disruption finding |
| PSI-25 BASS-II, PSI-26 BASS, PSI-10 Flame Design (inverse flames), PSI-159 CFI-G, PSI-98/99/100 Saffire I-III | catalog ranges, coverage map, recommender |
| NASA ISS Researcher's Guide: Combustion Science (NP-2015-10-034-JSC) | catalog text (page-cited), chatbot knowledge |
| Urban et al., Preliminary Results from the Saffire VI Experiment, ICES-2024-365; test matrix in [NTRS 20240007488](https://ntrs.nasa.gov/api/citations/20240007488/downloads/Saffire%20ICES%202024%20Saffire%20presentation072124small.pdf) | literature points (not PSI) |
| Fernandez-Pello et al., Scientific Reports 2017, [doi:10.1038/s41598-017-18398-4](https://doi.org/10.1038/s41598-017-18398-4) | PMMA-rod literature point |
| Olson et al., ICES 2019, [BASS-II flammability limits](https://ttu-ir.tdl.org/items/1922ad9f-ef4c-4f52-96a4-980c23debe5e) | qualitative card |
| NASA/TM-2013-217377; challenge page | exploration-atmosphere targets |

Data rules:
- No row is synthetic, interpolated or formula-generated.
- Missing values stay missing.
- Literature values are labelled as literature and never enter training data.
- BASS-II and Saffire I-II have no pressure column. They are drawn at an assumed 1 atm (ISS cabin), which is flagged everywhere it is used.

## Model card

Training data: 274 FLEX droplet tests (heptane and methanol, O2 0.12-0.34, 0.69-3.05 atm). Validation is
GroupKFold with 5 folds grouped by test date (80 dates), so the model is always scored on days it has
never seen. All numbers are means over the folds; the full per-fold results are in `models/metrics.json`.

| Model (used in app) | Metric | Result |
|---|---|---|
| Outcome, 3-class logistic | balanced accuracy / macro-F1 | 0.58 / 0.53 |
| Extinction vs rest, logistic | balanced accuracy / ROC-AUC | 0.78 / 0.88 |
| Disruption vs rest, logistic | balanced accuracy / ROC-AUC | 0.80 / 0.88 |
| Burning rate K, ridge | R² / MAE | 0.71 / 0.049 mm²/s |
| K 10-90% interval, conformalized quantile | coverage (nominal 80%) | 79.8% |

Limitations:
- **Completion is low confidence.** There are only 27 such tests, and out-of-fold recall is 0.26-0.37.
- **Gradient boosting was not better.** It scored lower than these baselines on unseen days. The O2-monotonic boosted model is kept only as a physics sanity check.
- **d0²/K is a full-burnout time.** It matches the measured burn time only for Completion tests (r = 0.92). For droplets that extinguish or burst, it is an upper limit.
- **Predictions have a limited range.** The app warns when an input is outside the tested envelope. Nothing here covers partial gravity (lunar 0.16 g).
- **The recommender is a heuristic.** It scores the gap to the nearest evidence × closeness to a target. Verified literature counts at half weight, and the distance scale is 5 O2 points = 0.2 atm.

## AI disclosure

Claude Code (Anthropic) wrote the code, the tests and the insight cards from NASA text and PSI tables,
under the team's direction. The team chose the scope and verified the literature values. The chatbot uses
Meta Llama, with NVIDIA NIM as a fallback, and may only use retrieved passages. No number in the app was
produced by a language model. Full log: [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md).
