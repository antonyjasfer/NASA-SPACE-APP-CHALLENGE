"""Build the investigation catalog, literature points and insight cards.

Run: python -m src.catalog
Writes data/processed/investigations.csv, literature_points.csv and insight_cards.json.
Ranges for investigations with a PSI table are computed from that table (source = PSI ID);
all other values come from src/catalog_data.py with a page citation. Nothing is guessed.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.catalog_data import GUIDE, INVESTIGATIONS, LITERATURE_POINTS, MISCONCEPTIONS
from src.ingest import PROCESSED_DIR, load_clean
from src.psi_tables import load_all, read_raw

INVESTIGATIONS_PATH = PROCESSED_DIR / "investigations.csv"
LITERATURE_PATH = PROCESSED_DIR / "literature_points.csv"
CARDS_PATH = PROCESSED_DIR / "insight_cards.json"
METRICS_PATH = PROCESSED_DIR.parents[1] / "models" / "metrics.json"
GENERATED_BY = "Claude Code from NASA PDF"
NOT_STATED = "Not stated in the available sources."
MAX_CARD_WORDS = 80

COLUMNS = ["key", "psi_id", "name", "fuel_type", "fuels", "objective", "what_varied",
           "o2_min", "o2_max", "p_min_atm", "p_max_atm", "range_basis", "range_source",
           "flow_min_cms", "flow_max_cms", "flow_basis", "flow_source", "n_tests",
           "sensors", "data_available", "key_result", "implication",
           "suppression_detection", "supp_source", "source", "source_page"]
NUMERIC = ["o2_min", "o2_max", "p_min_atm", "p_max_atm", "flow_min_cms", "flow_max_cms"]


def _rng(s: pd.Series, nd: int = 3) -> tuple[float | None, float | None]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return (round(float(s.min()), nd), round(float(s.max()), nd)) if len(s) else (None, None)


def psi_overrides() -> dict[str, dict[str, Any]]:
    """Ranges and facts computed from the PSI tables, keyed by investigation key."""
    t = load_all()
    flex = load_clean()
    out: dict[str, dict[str, Any]] = {}

    def table_range(key, psi, df, o2="x_o2", p="p_atm", flow=None, **extra):
        o2_min, o2_max = _rng(df[o2]) if o2 else (None, None)
        p_min, p_max = _rng(df[p]) if p else (None, None)
        d = dict(o2_min=o2_min, o2_max=o2_max, p_min_atm=p_min, p_max_atm=p_max, n_tests=len(df),
                 range_basis="tested", range_source=f"{psi} table (computed)")
        if flow:
            d.update(flow_min_cms=_rng(df[flow], 1)[0], flow_max_cms=_rng(df[flow], 1)[1],
                     flow_basis="tested", flow_source=f"{psi} table (computed)")
        out[key] = {**d, **extra}

    table_range("flex", "PSI-69", flex)
    table_range("flame_design", "PSI-10", t["PSI-10"].data,
                range_source="PSI-10 table (computed; 11 rows with XO2 'RAMP'/'LOW O2' excluded)")
    b2 = t["PSI-25"].data
    table_range("bass2", "PSI-25", b2, p=None,
                fuels=f"{b2['material'].nunique()} material labels incl. PMMA films (PSI-25)",
                what_varied="Initial O2, fan/air flow settings (not cm/s), material and thickness (PSI-25 columns).",
                range_source="PSI-25 table (computed); no pressure column; flow is device settings, not cm/s")
    cf = t["PSI-159"].data
    table_range("cfig", "PSI-159", cf, fuels=", ".join(sorted(cf["fuel"].unique())),
                what_varied="O2, pressure, fuel blend, fuel flow (PSI-159 columns).",
                range_source="PSI-159 table (computed; 'XO2 before test', 'approx p (bar)' -> atm)")
    for key, psi in (("saffire1", "PSI-98"), ("saffire2", "PSI-99"), ("saffire3", "PSI-100")):
        d = t[psi].data
        o2_min, _ = _rng(d["x_o2_min"])
        _, o2_max = _rng(d["x_o2_max"])
        table_range(key, psi, d, o2=None, p=None, flow="flow_cms",
                    fuels=", ".join(sorted(d["material"].dropna().unique())))
        out[key].update(o2_min=o2_min, o2_max=o2_max,
                        range_source=f"{psi} table (computed; O2 parsed from text, '~' = approximate)"
                        if o2_min is not None else f"{psi} table has no O2 column")
    out["bass"] = {"n_tests": len(t["PSI-26"].data)}
    return out


def build_investigations() -> pd.DataFrame:
    over = psi_overrides()
    rows = [{**{c: None for c in COLUMNS}, **inv, **over.get(inv["key"], {})} for inv in INVESTIGATIONS]
    df = pd.DataFrame(rows)[COLUMNS]
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col])
    return df


def build_literature() -> pd.DataFrame:
    return pd.DataFrame(LITERATURE_POINTS)


# --------------------------------------------------------------------------- cards
def _txt(value: Any) -> str | None:
    """Text cell or None. NaN/blank cells are falsy here (plain `or` treats NaN as true)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _cite(row: pd.Series) -> str:
    page = f", p. {row['source_page']}" if _txt(row["source_page"]) else ""
    return f"{row['source']}{page}"


def _psi_key_results() -> dict[str, str]:
    t = load_all()
    b2, cf = t["PSI-25"].data, t["PSI-159"].data
    s1 = read_raw("PSI-98")
    s1 = s1[s1["Sample Number"].isin(["S1", "S2"])]
    s2 = t["PSI-99"].data
    return {
        "bass2": (f"{len(b2)} tests from {b2['date'].min()} to {b2['date'].max()} on "
                  f"{b2['material'].nunique()} material labels, initial O2 {b2['x_o2'].min():.1%} to "
                  f"{b2['x_o2'].max():.1%}. The table has no pressure or spread-rate column."),
        "cfig": (f"{len(cf)} tests: hot ignition in {int(cf['hot_ignition'].sum())}, "
                 f"{int((cf['outcome'] == 'Self-extinguished').sum())} self-extinguished, "
                 f"{int(cf['cool_flame'].sum())} flagged as cool flames, at O2 {cf['x_o2'].min():.3f} to "
                 f"{cf['x_o2'].max():.3f} and {cf['p_atm'].min():.2f} to {cf['p_atm'].max():.2f} atm."),
        "saffire1": (f"SIBAL fabric at {t['PSI-98'].data['flow_cms'].iloc[0]:g} cm/s: concurrent-flow sample "
                     f"burned {s1.iloc[0]['Burn Time (s)']} s, opposed-flow sample {s1.iloc[1]['Burn Time (s)']} s."),
        "saffire2": (f"Micro-g spread {s2.loc[s2['test_id'] == '2-5', 'ug_spread_raw'].iloc[0]} (sample 2-5, SIBAL) "
                     f"and {s2.loc[s2['test_id'] == '2-6', 'ug_spread_raw'].iloc[0]} (sample 2-6, material cell "
                     "blank); silicone samples burned insignificantly."),
        "saffire3": (f"{len(t['PSI-100'].data)} SIBAL fabric samples at "
                     f"{t['PSI-100'].data['flow_cms'].iloc[0]:g} cm/s air flow (concurrent and opposed); no O2 column."),
        "flame_design": (f"{len(t['PSI-10'].data)} inverse ethylene flames at O2 up to "
                         f"{t['PSI-10'].data['x_o2'].max():.2f} and {t['PSI-10'].data['p_atm'].min():.2f} to "
                         f"{t['PSI-10'].data['p_atm'].max():.2f} atm (PSI-10)."),
    }


def investigation_cards(inv: pd.DataFrame) -> list[dict]:
    psi_results = _psi_key_results()
    cards = []
    for _, r in inv.iterrows():
        key_result = _txt(r["key_result"]) or psi_results.get(r["key"])
        cards.append({
            "card_id": f"inv_{r['key']}", "investigation_key": r["key"], "title": r["name"],
            "objective": _txt(r["objective"]) or NOT_STATED,
            "what_varied": _txt(r["what_varied"]) or NOT_STATED,
            "key_result": key_result or NOT_STATED,
            "fire_safety_implication": _txt(r["implication"]) or NOT_STATED,
            "citation": _cite(r), "generated_by": GENERATED_BY,
        })
    return cards


def special_cards() -> list[dict]:
    from src.coverage import disruption_contrast

    flex = load_clean()
    dc = disruption_contrast(flex)
    hi, lo = dc["high_o2"], dc["air_or_below"]
    metrics = json.loads(METRICS_PATH.read_text())["d2law_consistency"]
    comp = metrics["completion_only"]
    allt = metrics["all_tests"]
    from src.catalog_data import O2_SCALE, P_SCALE

    tables = load_all()
    cf, b2 = tables["PSI-159"].data, tables["PSI-25"].data
    moon = next(p for p in LITERATURE_POINTS if p["point_id"] == "target_moon_34_8_2")
    t_o2, t_p = moon["x_o2"], moon["p_atm"]
    d = np.hypot((cf["x_o2"] - t_o2) / O2_SCALE, (cf["p_atm"] - t_p) / P_SCALE)
    near = cf[d <= 1.0]   # within one distance unit of the Moon-base point
    flex_cite = "Fire-Margin analysis of PSI-69 FLEX table (DOI 10.60555/mbq8-0451)"
    return [
        {"card_id": "flex_disruption_by_o2", "title": "High oxygen turns droplet burns into bursts",
         "objective": "Compare how FLEX droplet tests ended at high vs air-level oxygen.",
         "what_varied": "Ambient O2 (all fuels, diluents and pressures pooled).",
         "key_result": (f"At O2 >= 0.25, {hi['n_disruption']} of {hi['n_tests']} tests ended in Disruption "
                        f"({hi['disruption_share']:.0%}); at O2 <= 0.21, {lo['n_disruption']} of {lo['n_tests']} "
                        f"({lo['disruption_share']:.1%})."),
         "fire_safety_implication": ("In an oxygen-enriched habitat a burning droplet is more likely to burst "
                                     "and scatter fuel than to burn out quietly. Cells are small and pressure "
                                     "coverage is uneven, so treat this as a strong signal, not a law."),
         "citation": flex_cite, "generated_by": "Claude Code from PSI-69 data"},
        {"card_id": "pressure_gap", "title": "No droplet or solid test at Moon-base pressure",
         "objective": (f"Check whether real tests reach the {t_o2:.0%} O2 / 8.2 psia ({t_p:.3f} atm) "
                       "habitat atmosphere."),
         "what_varied": "Pressure coverage per fuel type.",
         "key_result": (f"Lowest FLEX pressure is {flex['p_atm'].min():.2f} atm, above the {t_p:.3f} atm target. "
                        f"PSI solid-material tables give no measured pressure and stop at {b2['x_o2'].max():.1%} O2. "
                        f"Gas flames do reach it: CFI-G ran {len(near)} tests within one distance unit, at O2 "
                        f"{near['x_o2'].min():.3f}-{near['x_o2'].max():.3f} and {near['p_atm'].min():.2f}-"
                        f"{near['p_atm'].max():.2f} atm."),
         "fire_safety_implication": ("Droplet and cabin-solid behaviour at the planned habitat atmosphere is "
                                     "untested in PSI data: the next experiments should target that corner."),
         "citation": f"{flex_cite}; PSI-25; PSI-159", "generated_by": "Claude Code from PSI data"},
        {"card_id": "d2law_validity", "title": "The d-squared law only fits droplets that burn out",
         "objective": "Check burn time against the d-squared law t = d0^2 / K.",
         "what_varied": "Test outcome (Completion vs early end).",
         "key_result": (f"For Completion tests, d0^2/K tracks measured burn time (r = {comp['pearson_r']:.2f}, "
                        f"n = {comp['n']}); across all tests r = {allt['pearson_r']:.2f} because extinction or "
                        "disruption ends the burn early."),
         "fire_safety_implication": ("Use d0^2/K as an upper limit on burn time, never as a prediction for "
                                     "droplets that may extinguish or burst."),
         "citation": f"{flex_cite}; models/metrics.json", "generated_by": "Claude Code from PSI-69 data"},
        {"card_id": "three_misconceptions", "title": "Three misconceptions about spacecraft fire",
         "objective": "Misconceptions that shaped early spacecraft fire safety.",
         "what_varied": NOT_STATED,
         "key_result": " ".join(f"({i}) {m}" for i, m in enumerate(MISCONCEPTIONS, 1)),
         "fire_safety_implication": "Research at NASA GRC and in microgravity showed all three are false.",
         "citation": f"{GUIDE}, p. 35", "generated_by": GENERATED_BY},
    ]


def word_count(text: str) -> int:
    return len(str(text).split())


def main() -> dict[str, Any]:
    inv = build_investigations()
    inv.to_csv(INVESTIGATIONS_PATH, index=False)
    lit = build_literature()
    lit.to_csv(LITERATURE_PATH, index=False)
    cards = investigation_cards(inv) + special_cards()
    too_long = [(c["card_id"], k) for c in cards for k, v in c.items()
                if k not in ("citation", "card_id") and word_count(v) > MAX_CARD_WORDS]
    if too_long:
        raise ValueError(f"Card fields over {MAX_CARD_WORDS} words: {too_long}")
    CARDS_PATH.write_text(json.dumps(cards, indent=2, ensure_ascii=False))
    print(inv[["name", "fuel_type", "o2_min", "o2_max", "p_min_atm", "p_max_atm",
               "flow_min_cms", "flow_max_cms", "range_basis"]].to_string(index=False))
    print(f"\n{len(inv)} investigations, {len(lit)} literature/target points, {len(cards)} cards")
    return {"investigations": inv, "literature": lit, "cards": cards}


if __name__ == "__main__":
    main()
