"""Partial-gravity analogue (HEURISTIC, not a validated prediction).

Run: python -m src.partial_gravity
Grashof scaling: buoyant flow ~ g * rho^2 and gas density rho ~ p, so Gr ~ g * p^2. A 1-g ground test
matches the buoyancy (Grashof number) of a habitat at gravity g_target and pressure p_target when
    p_1g = p_target * sqrt(g_target / g_earth).
Ignores radiation, residence time and flame-size effects. It bounds lunar/Mars behaviour between the
microgravity and 1-g results; PSI holds no partial-gravity data.
"""
from __future__ import annotations

import math

import pandas as pd

from src.catalog_data import LITERATURE_POINTS

G_RATIO = {"Moon": 0.16, "Mars": 0.38}
CAPTION = ("Heuristic: matches buoyancy (Grashof) only; ignores radiation, residence time and flame-size "
           "effects. Bounds lunar behaviour between the microgravity and 1-g results; not a validated "
           "prediction. PSI contains no partial-gravity combustion data.")
GROUND_TEST_TEXT = ("1-g ground test at {o2:.0%} O2 and ~{p:.2f} atm to approximate lunar buoyancy for solid "
                    "materials (Grashof scaling p_1g = p x sqrt(0.16); heuristic, not a validated prediction).")


def pressure_factor(g_ratio: float) -> float:
    """Factor on pressure that keeps the Grashof number when moving from g_ratio*g to 1 g."""
    if g_ratio <= 0:
        raise ValueError("g_ratio must be positive")
    return math.sqrt(g_ratio)


def equivalent_pressure(p_target_atm: float, g_ratio: float) -> float:
    return p_target_atm * pressure_factor(g_ratio)


def equivalent_points() -> pd.DataFrame:
    """Earth-equivalent (1-g) pressures for each exploration target under Moon and Mars gravity."""
    rows = []
    for t in (p for p in LITERATURE_POINTS if p["kind"] == "exploration_target"):
        for body, g in G_RATIO.items():
            rows.append({"target": t["label"], "body": body, "g_ratio": g, "x_o2": t["x_o2"],
                         "p_target_atm": t["p_atm"], "factor": round(pressure_factor(g), 3),
                         "p_1g_equiv_atm": round(equivalent_pressure(t["p_atm"], g), 3),
                         "label": f"{body} ({g:g} g) analogue of {t['label']}", "heuristic": True})
    return pd.DataFrame(rows)


def ground_test_row() -> dict:
    moon = equivalent_points().query("body == 'Moon' and x_o2 == 0.34").iloc[0]
    return {"rank": 6, "kind": "ground-test analogue (heuristic)", "fuel_type": "solid",
            "x_o2": float(moon["x_o2"]), "p_atm": float(moon["p_1g_equiv_atm"]),
            "explanation": GROUND_TEST_TEXT.format(o2=moon["x_o2"], p=moon["p_1g_equiv_atm"])}


def card() -> dict:
    eq = equivalent_points()
    m34 = eq.query("body == 'Moon' and x_o2 == 0.34").iloc[0]
    m30 = eq.query("body == 'Moon' and x_o2 == 0.30").iloc[0]
    return {
        "card_id": "partial_gravity_analogue", "title": "Approximating lunar gravity on the ground (heuristic)",
        "objective": "Find a 1-g test that mimics lunar buoyancy, since no partial-gravity data exist in PSI.",
        "what_varied": "Pressure, scaled by sqrt(g/g_earth): factor 0.40 for the Moon, 0.62 for Mars.",
        "key_result": (f"Buoyancy (Grashof) at 34% O2 / {m34['p_target_atm']:.3f} atm on the Moon equals a 1-g "
                       f"test at ~{m34['p_1g_equiv_atm']:.2f} atm; 30% O2 / {m30['p_target_atm']:.3f} atm maps "
                       f"to ~{m30['p_1g_equiv_atm']:.2f} atm."),
        "fire_safety_implication": ("A 1-g ground test of solid cabin materials at 34% O2 and ~0.22 atm would "
                                    "bracket lunar fire behaviour cheaply. " + CAPTION),
        "citation": "Fire-Margin heuristic (Grashof scaling Gr ~ g p^2); targets: NASA/TM-2013-217377",
        "generated_by": "Claude Code (heuristic, labelled)",
    }


def main() -> None:
    print(equivalent_points().to_string(index=False))
    print(ground_test_row())


if __name__ == "__main__":
    main()
