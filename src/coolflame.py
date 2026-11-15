"""Cool-flame flag from PSI-159 (ACME CFI-G normal flames) + the Researcher's Guide account.

Run: python -m src.coolflame
The "Cool flame?" column is mostly blank: blank = NOT RECORDED (unknown), never counted as "no".
"""
from __future__ import annotations

import pandas as pd

from src.catalog_data import GUIDE
from src.psi_tables import load_cfig

PSI_ID = "PSI-159"
# Guide p.33-34 (FLEX alkane droplets: cool flame after hot-flame extinction) and p.36 (heptane).
GUIDE_COOL_FLAME = ("After the visible (hot) flame of a heptane droplet extinguishes by radiative loss, a "
                    "low-temperature cool flame can keep the droplet vaporizing and burning for an extended "
                    "time, invisible to the eye.")
GUIDE_COOL_FLAME_CITE = f"{GUIDE}, p. 33-34, 36"
GUIDE_REIGNITION = ("Reduced gravity favours very weak, persistent flames invisible to the unaided eye that can "
                    "spontaneously re-ignite long after the visible flame is gone.")
GUIDE_REIGNITION_CITE = f"{GUIDE}, p. 41-42"


def cool_flame_tests() -> pd.DataFrame:
    """Every CFI-G test with a status: 'yes', 'no' or 'unknown' (blank cell)."""
    d = load_cfig().data
    status = d["cool_flame"].map({True: "yes", False: "no"}).fillna("unknown")
    return pd.DataFrame({"source": PSI_ID, "test_id": d["test_id"], "fuel": d["fuel"], "x_o2": d["x_o2"],
                         "p_atm": d["p_atm"], "cool_flame": status, "cool_flame_raw": d["cool_flame_raw"],
                         "hot_ignition": d["hot_ignition"], "outcome": d["outcome"]})


def counts(df: pd.DataFrame | None = None) -> dict[str, int]:
    df = cool_flame_tests() if df is None else df
    return {k: int((df["cool_flame"] == k).sum()) for k in ("yes", "no", "unknown")}


def observed(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tests with a recorded answer (yes or no); unknowns excluded."""
    df = cool_flame_tests() if df is None else df
    return df[df["cool_flame"] != "unknown"].reset_index(drop=True)


def card() -> dict:
    obs = observed()
    yes = obs[obs["cool_flame"] == "yes"]
    c = counts()
    fuels = ", ".join(sorted(yes["fuel"].unique()))
    return {
        "card_id": "invisible_cool_flames", "title": "Invisible cool flames",
        "objective": "Flag where a flame may keep burning after it looks extinguished.",
        "what_varied": "Fuel blend, O2 and pressure (PSI-159 CFI-G gas flames); heptane droplets (FLEX).",
        "key_result": (f"PSI-159 records a cool flame in {c['yes']} tests and none in {c['no']}; the other "
                       f"{c['unknown']} tests have no entry (unknown, not 'no'). All {c['yes']} were {fuels} at "
                       f"{yes['x_o2'].min():.0%}-{yes['x_o2'].max():.0%} O2 and {yes['p_atm'].min():.1f} atm. "
                       f"{GUIDE_COOL_FLAME}"),
        "fire_safety_implication": (f"{GUIDE_REIGNITION} A fire that looks out may not be out: detectors and "
                                    "crew procedures must not rely on the visible flame."),
        "citation": f"{PSI_ID} table; {GUIDE_COOL_FLAME_CITE}; {GUIDE_REIGNITION_CITE}",
        "generated_by": "Claude Code from PSI-159 data and NASA PDF",
    }


def main() -> None:
    print(counts())
    print(observed()[["test_id", "fuel", "x_o2", "p_atm", "cool_flame", "cool_flame_raw"]].to_string(index=False))


if __name__ == "__main__":
    main()
