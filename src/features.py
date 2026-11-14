"""Physics features for FLEX models (CLAUDE.md "Modelling rules").

Run: python -m src.features  (prints a summary of the feature table)
Only inputs known *before* ignition are model features; d_ext, K and burn time are outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Microgravity limiting oxygen index (mole fraction) per CLAUDE.md.
LOI: dict[str, float] = {"Methanol": 0.12, "Heptane": 0.15}
FUELS: tuple[str, ...] = tuple(LOI)

MODEL_FEATURES: list[str] = [
    "d0_mm", "d0_mm2", "x_o2", "p_atm", "po2_atm",
    "x_n2", "x_co2", "x_he", "o2_margin", "fuel_methanol", "fuel_heptane",
]
# Features on which K must be non-decreasing (higher O2 -> faster burning). o2_margin moves
# one-for-one with x_o2, so it must be constrained too or the O2 constraint leaks.
MONOTONIC_INCREASING: tuple[str, ...] = ("x_o2", "po2_atm", "o2_margin")


def d2law_burn_time(d0_mm: pd.Series, k_mm2_s: pd.Series) -> pd.Series:
    """d^2-law burn time t = d0^2 / K; NaN where K is missing or non-positive."""
    k = k_mm2_s.where(k_mm2_s > 0)
    return d0_mm**2 / k


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the clean FLEX frame with physics features added."""
    unknown = set(df["fuel"]) - set(FUELS)
    if unknown:
        raise ValueError(f"No LOI for fuel(s) {unknown}")
    out = df.copy()
    out["d0_mm2"] = out["d0_mm"] ** 2
    out["po2_atm"] = out["x_o2"] * out["p_atm"]
    out["o2_margin"] = out["x_o2"] - out["fuel"].map(LOI)
    for fuel in FUELS:
        out[f"fuel_{fuel.lower()}"] = (out["fuel"] == fuel).astype(int)
    out["d2law_burn_time_s"] = d2law_burn_time(out["d0_mm"], out["K_mm2_s"])
    return out


def main() -> pd.DataFrame:
    from src.ingest import load_clean

    feats = build_features(load_clean())
    print(feats[MODEL_FEATURES + ["d2law_burn_time_s", "burn_time_s"]].describe().T.round(3))
    both = feats[["d2law_burn_time_s", "burn_time_s"]].dropna()
    print(f"\nd^2-law vs measured burn time: n={len(both)}, "
          f"Pearson r={both.corr().iloc[0, 1]:.3f}, "
          f"median ratio={np.median(both.iloc[:, 0] / both.iloc[:, 1]):.3f}")
    return feats


if __name__ == "__main__":
    main()
