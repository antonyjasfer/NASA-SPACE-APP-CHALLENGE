"""FLEX evidence coverage: outcome counts by O2 band x pressure band.

Run: python -m src.coverage -> data/processed/flex_coverage.csv (+ flex_coverage_summary.json)
Feeds the headline insight: where FLEX tested near the Moon-base atmosphere, what happened.
Counts only real tests; empty band combinations are listed with n=0 so gaps are visible.
"""
from __future__ import annotations

import json

import pandas as pd

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
    return table


if __name__ == "__main__":
    main()
