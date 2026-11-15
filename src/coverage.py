"""Evidence coverage: FLEX outcome bands, all tested conditions, and next-experiment picks.

Run: python -m src.coverage -> data/processed/
  flex_coverage.csv, flex_coverage_summary.json   FLEX outcomes by O2 band x pressure band
  coverage_points.csv                              every real tested condition (PSI tables only)
  next_experiments.csv                             top-5 recommended (fuel type, O2, pressure) points
Counts only real tests; empty band combinations are listed with n=0 so gaps are visible.
Literature points live in literature_points.csv and are never mixed in here.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.catalog_data import O2_SCALE, P_SCALE  # one unit = 5 O2 points or 0.2 atm
from src.ingest import PROCESSED_DIR, load_clean

COVERAGE_PATH = PROCESSED_DIR / "flex_coverage.csv"
SUMMARY_PATH = PROCESSED_DIR / "flex_coverage_summary.json"
OUTCOMES = ["Extinction", "Disruption", "Completion"]

# Tested O2 levels are 0.12-0.21, 0.23-0.25, 0.30, 0.34 (none at 0.22 or 0.26-0.29).
O2_BINS = [0.0, 0.155, 0.215, 0.245, 0.295, 1.0]
O2_LABELS = ["0.12-0.15", "0.16-0.21", "0.23-0.24", "0.25", "0.30-0.34"]
P_BINS = [0.0, 0.5, 0.9, 1.1, 10.0]
P_LABELS = ["<0.5 atm", "0.5-0.9 atm", "0.9-1.1 atm", ">1.1 atm"]
HIGH_O2, AIR_O2 = 0.25, 0.21


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    banded = df.assign(
        o2_band=pd.cut(df["x_o2"], O2_BINS, labels=O2_LABELS),
        p_band=pd.cut(df["p_atm"], P_BINS, labels=P_LABELS),
    )
    banded["p_band"] = banded["p_band"].cat.add_categories("pressure unknown").fillna("pressure unknown")
    counts = (banded.groupby(["o2_band", "p_band"], observed=False)["test_end"]
              .value_counts().unstack(fill_value=0).reindex(columns=OUTCOMES, fill_value=0))
    counts = counts.reset_index()
    counts["n"] = counts[OUTCOMES].sum(axis=1)
    counts["disruption_share"] = (counts["Disruption"] / counts["n"]).where(counts["n"] > 0).round(3)
    # keep the pressure-unknown row only if it holds data
    return counts[(counts["p_band"] != "pressure unknown") | (counts["n"] > 0)].reset_index(drop=True)


def disruption_contrast(df: pd.DataFrame) -> dict:
    def block(mask: pd.Series, label: str) -> dict:
        sub = df[mask]
        n, k = len(sub), int((sub["test_end"] == "Disruption").sum())
        return {"group": label, "n_tests": n, "n_disruption": k,
                "disruption_share": round(k / n, 3) if n else None,
                "outcomes": sub["test_end"].value_counts().to_dict(),
                "p_atm_range": [round(sub["p_atm"].min(), 3), round(sub["p_atm"].max(), 3)]}

    return {"source": "PSI-69 FLEX, DOI 10.60555/mbq8-0451",
            "high_o2": block(df["x_o2"] >= HIGH_O2, f"x_o2 >= {HIGH_O2}"),
            "air_or_below": block(df["x_o2"] <= AIR_O2, f"x_o2 <= {AIR_O2}"),
            "note": "Tests at 0.21 < x_o2 < 0.25 belong to neither group."}


# --------------------------------------------------------------------------- coverage points
POINTS_PATH = PROCESSED_DIR / "coverage_points.csv"
NEXT_PATH = PROCESSED_DIR / "next_experiments.csv"
POINT_COLUMNS = ["source", "investigation", "test_id", "fuel_type", "fuel", "x_o2", "p_atm",
                 "p_assumed_atm", "p_assumed", "flow_cms", "flow_raw", "outcome", "approx_note"]


def coverage_points() -> pd.DataFrame:
    """Every real tested condition we hold (FLEX + extra PSI tables). PSI-26 has no conditions."""
    from src.psi_tables import load_all

    t = load_all()
    flex = load_clean()
    frames = [pd.DataFrame({
        "source": "PSI-69", "investigation": "FLEX", "test_id": flex["flex_id"], "fuel_type": "droplet",
        "fuel": flex["fuel"], "x_o2": flex["x_o2"], "p_atm": flex["p_atm"], "flow_cms": np.nan,
        "flow_raw": "quiescent (Guide p.33)", "outcome": flex["test_end"],
        "approx_note": np.where(flex["burn_time_approx"], "burn time approximate", "")})]
    b2 = t["PSI-25"].data
    frames.append(pd.DataFrame({
        "source": "PSI-25", "investigation": "BASS-II", "test_id": b2["test_id"], "fuel_type": "solid",
        "fuel": b2["material"], "x_o2": b2["x_o2"], "p_atm": np.nan, "p_assumed_atm": 1.0, "p_assumed": True,
        "flow_cms": np.nan, "flow_raw": "fan " + b2["fan_display_raw"] + " | air " + b2["air_display_raw"],
        "approx_note": "pressure not measured; 1 atm assumed for plotting only"}))
    inv = t["PSI-10"].data
    frames.append(pd.DataFrame({
        "source": "PSI-10", "investigation": "Flame Design (inverse)", "test_id": inv["test_id"],
        "fuel_type": "gas", "fuel": inv["fuel"], "x_o2": inv["x_o2"], "p_atm": inv["p_atm"],
        "approx_note": np.where(inv["x_o2_not_constant"], "XO2 = " + inv["x_o2_raw"], "")}))
    cf = t["PSI-159"].data
    frames.append(pd.DataFrame({
        "source": "PSI-159", "investigation": "CFI-G", "test_id": cf["test_id"], "fuel_type": "gas",
        "fuel": cf["fuel"], "x_o2": cf["x_o2"], "p_atm": cf["p_atm"], "outcome": cf["outcome"],
        "approx_note": "pressure approximate (bar column)"}))
    for psi, name in (("PSI-98", "Saffire-I"), ("PSI-99", "Saffire-II"), ("PSI-100", "Saffire-III")):
        d = t[psi].data
        frames.append(pd.DataFrame({
            "source": psi, "investigation": name, "test_id": d["test_id"], "fuel_type": "solid",
            "fuel": d["material"], "x_o2": d["x_o2"], "p_atm": np.nan, "p_assumed_atm": 1.0,
            "p_assumed": True, "flow_cms": d["flow_cms"],
            "flow_raw": d["flow_raw"], "approx_note": np.where(d["x_o2_approx"], "O2 approximate (~); pressure assumed ISS cabin 1 atm",
                                   "pressure assumed ISS cabin 1 atm")}))
    out = pd.concat(frames, ignore_index=True).reindex(columns=POINT_COLUMNS)
    out["p_assumed"] = out["p_assumed"].fillna(False).astype(bool)
    return out


# --------------------------------------------------------------------------- recommender
GAP_CAP = 3.0                      # gaps beyond 3 units count the same (nothing nearby)
O2_GRID = np.round(np.arange(0.15, 0.4001, 0.01), 3)
P_GRID = np.round(np.arange(0.30, 1.2001, 0.02), 3)
MIN_SEPARATION = 1.0               # picks of the same fuel type must be >= 1 unit apart
MIN_SCORE = 0.1                    # a fuel type gets a guaranteed slot only above this score
FUEL_TYPES = ("solid", "droplet", "gas")


def _targets() -> list[tuple[str, float, float]]:
    from src.catalog_data import LITERATURE_POINTS

    return [(p["label"], p["x_o2"], p["p_atm"]) for p in LITERATURE_POINTS if p["kind"] == "exploration_target"]


LITERATURE_WEIGHT = 0.5            # published but not in PSI: counts half as much as a PSI test
EVIDENCE_COLUMNS = ["fuel_type", "x_o2", "p_atm", "weight", "kind", "label"]


def evidence_points(points: pd.DataFrame) -> pd.DataFrame:
    """Evidence used by the recommender: PSI tests with measured O2 + pressure (weight 1), PSI tests
    whose pressure is the flagged ISS-cabin assumption (weight 1, kind 'assumed pressure'), and
    verified literature points (weight 0.5). Targets are never evidence."""
    from src.catalog_data import LITERATURE_POINTS

    measured = points[points["x_o2"].notna() & points["p_atm"].notna()]
    assumed = points[points["x_o2"].notna() & points["p_atm"].isna() & points["p_assumed"]]
    lit = pd.DataFrame([p for p in LITERATURE_POINTS
                        if p["kind"] == "literature" and p["value_verified"] and p["p_atm"] is not None])
    frames = [
        pd.DataFrame({"fuel_type": measured["fuel_type"], "x_o2": measured["x_o2"], "p_atm": measured["p_atm"],
                      "weight": 1.0, "kind": "PSI measured",
                      "label": measured["source"] + " " + measured["test_id"].astype(str)}),
        pd.DataFrame({"fuel_type": assumed["fuel_type"], "x_o2": assumed["x_o2"], "p_atm": assumed["p_assumed_atm"],
                      "weight": 1.0, "kind": "PSI, pressure assumed 1 atm (ISS cabin)",
                      "label": assumed["source"] + " " + assumed["test_id"].astype(str)}),
    ]
    if len(lit):
        frames.append(pd.DataFrame({"fuel_type": lit["fuel_type"], "x_o2": lit["x_o2"], "p_atm": lit["p_atm"],
                                    "weight": LITERATURE_WEIGHT, "kind": "verified literature (not PSI)",
                                    "label": lit["label"]}))
    return pd.concat(frames, ignore_index=True)[EVIDENCE_COLUMNS]


def recommend_experiments(points: pd.DataFrame, top_n: int = 5,
                          fuel_types: tuple[str, ...] = FUEL_TYPES) -> pd.DataFrame:
    """Rank candidate (fuel type, O2, pressure) points: score = (1 - coverage) x closeness, where
    coverage = max over evidence of weight x (1 - min(d, GAP_CAP) / GAP_CAP) and closeness is the
    Gaussian proximity to the nearest exploration target."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    ev = evidence_points(points)
    o2, p = np.meshgrid(O2_GRID, P_GRID, indexing="ij")
    o2, p = o2.ravel(), p.ravel()
    closeness = np.zeros_like(o2)
    nearest_target = np.empty(o2.shape, dtype=object)
    for label, to2, tp in _targets():
        c = np.exp(-0.5 * (((o2 - to2) / O2_SCALE) ** 2 + ((p - tp) / P_SCALE) ** 2))
        nearest_target = np.where(c > closeness, label, nearest_target)
        closeness = np.maximum(closeness, c)
    rows = []
    for ft in fuel_types:
        e = ev[ev["fuel_type"] == ft]
        if len(e):
            d = np.sqrt(((o2[:, None] - e["x_o2"].to_numpy()) / O2_SCALE) ** 2
                        + ((p[:, None] - e["p_atm"].to_numpy()) / P_SCALE) ** 2)
            cov_each = e["weight"].to_numpy() * (1 - np.minimum(d, GAP_CAP) / GAP_CAP)
            idx = cov_each.argmax(axis=1)
            coverage = cov_each[np.arange(len(o2)), idx]
            gap = d[np.arange(len(o2)), idx]
            best = e.iloc[idx]
            desc = (best["label"] + " [" + best["kind"] + "] (O2 " + best["x_o2"].round(3).astype(str)
                    + ", " + best["p_atm"].round(2).astype(str) + " atm)").to_numpy()
        else:
            coverage, gap = np.zeros(len(o2)), np.full(len(o2), np.inf)
            desc = np.full(len(o2), "none: no evidence of this fuel type")
        rows.append(pd.DataFrame({"fuel_type": ft, "x_o2": o2, "p_atm": p, "gap_units": gap,
                                  "coverage": coverage, "closeness": closeness,
                                  "nearest_target": nearest_target, "nearest_evidence": desc,
                                  "score": (1 - coverage) * closeness}))
    cand = pd.concat(rows, ignore_index=True).sort_values(
        ["score", "fuel_type", "x_o2", "p_atm"], ascending=[False, True, False, True], kind="mergesort")
    # Diversity: first the best pick of each fuel type (if it clears MIN_SCORE), then fill by score;
    # picks of the same fuel type must be MIN_SEPARATION apart so neighbours do not crowd the list.
    firsts = cand.groupby("fuel_type", sort=False).head(1)
    firsts = firsts[firsts["score"] >= MIN_SCORE]
    picks: list[pd.Series] = []
    for _, r in pd.concat([firsts, cand]).iterrows():
        if len(picks) == top_n:
            break
        if all(q["fuel_type"] != r["fuel_type"] or
               np.hypot((q["x_o2"] - r["x_o2"]) / O2_SCALE, (q["p_atm"] - r["p_atm"]) / P_SCALE) >= MIN_SEPARATION
               for q in picks):
            picks.append(r)
    out = pd.DataFrame(picks).sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out["gap_units"] = out["gap_units"].replace(np.inf, np.nan).round(2)
    out[["coverage", "closeness", "score"]] = out[["coverage", "closeness", "score"]].round(3)
    out["explanation"] = [_explain(r) for _, r in out.iterrows()]
    return out


def _explain(r: pd.Series) -> str:
    ev = ("no evidence of this fuel type exists" if pd.isna(r["gap_units"])
          else f"best evidence {r['nearest_evidence']} is {r['gap_units']:.2f} units away "
               f"(coverage {r['coverage']:.2f})")
    return (f"Test {r['fuel_type']} fuels at {r['x_o2']:.0%} O2 and {r['p_atm']:.2f} atm: {ev}; "
            f"closeness {r['closeness']:.2f} to '{r['nearest_target']}'.")

def add_ground_test_analogue(picks: pd.DataFrame) -> pd.DataFrame:
    """Append the heuristic 1-g ground-test analogue as a separate 6th row (top 5 unchanged)."""
    from src.partial_gravity import ground_test_row

    return pd.concat([picks, pd.DataFrame([ground_test_row()])], ignore_index=True)


def main() -> pd.DataFrame:
    df = load_clean()
    table = coverage_table(df)
    assert table["n"].sum() == len(df), "coverage table must account for every real test"
    table.to_csv(COVERAGE_PATH, index=False)
    summary = disruption_contrast(df)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(table[table["n"] > 0].to_string(index=False))
    print("\nEmpty cells (no FLEX test):",
          [f"{r.o2_band} @ {r.p_band}" for r in table[table["n"] == 0].itertuples()])
    print(json.dumps(summary, indent=2))
    points = coverage_points()
    points.to_csv(POINTS_PATH, index=False)
    print("\nCoverage points:", points.groupby(["source", "fuel_type"]).size().to_dict())
    picks = recommend_experiments(points)
    picks.assign(kind="recommended (evidence gap)").pipe(add_ground_test_analogue).to_csv(NEXT_PATH, index=False)
    print(picks[["rank", "fuel_type", "x_o2", "p_atm", "gap_units", "coverage", "closeness", "score"]].to_string(index=False))
    for e in picks["explanation"]:
        print(" -", e)
    return table


if __name__ == "__main__":
    main()
