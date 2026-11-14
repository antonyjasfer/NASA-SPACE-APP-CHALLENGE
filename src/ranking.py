"""Transparent 0-100 relevance score for the investigation catalog (app tab 1 sliders).

Run: python -m src.ranking
score = 100 * sum(w_f * s_f) / sum(w_f), each factor s_f in [0, 1]:
  habitat       solid 1.0 > droplet 0.6 > gas 0.3 (cabin materials matter most)
  o2_pressure   closeness of the tested O2 x pressure box to the exploration targets,
                exp(-d^2/2) with d in units of 5 O2 points and 0.2 atm; x0.5 if the range is only
                planned or a facility capability (no evidence yet)
  flow          share of the 0-30 cm/s ventilation range covered by the tested flow range
                (each end widened by 2.5 cm/s so a single tested speed counts)
  data          table 1.0, telemetry 0.8, video 0.5, none 0.0
  suppression   1 if a source states suppression or detection relevance
A factor whose inputs are missing scores 0 and is listed in `missing`; nothing is guessed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.catalog_data import LITERATURE_POINTS

FACTORS = ("habitat", "o2_pressure", "flow", "data", "suppression")
DEFAULT_WEIGHTS: dict[str, float] = {"habitat": 0.25, "o2_pressure": 0.30, "flow": 0.15,
                                     "data": 0.15, "suppression": 0.15}
HABITAT = {"solid": 1.0, "droplet": 0.6, "gas": 0.3}
DATA = {"table": 1.0, "telemetry": 0.8, "video": 0.5, "none": 0.0}
O2_SCALE, P_SCALE = 0.05, 0.2
UNTESTED_FACTOR = 0.5
VENT_MIN, VENT_MAX, FLOW_PAD = 0.0, 30.0, 2.5
TARGETS: tuple[tuple[float, float], ...] = tuple(
    (p["x_o2"], p["p_atm"]) for p in LITERATURE_POINTS if p["kind"] == "exploration_target")


def _num(v: Any) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _box_distance(lo: float, hi: float, x: float) -> float:
    return max(lo - x, 0.0, x - hi)


def habitat_score(row: Mapping) -> tuple[float, bool]:
    s = HABITAT.get(str(row.get("fuel_type")))
    return (s, False) if s is not None else (0.0, True)


def o2_pressure_score(row: Mapping) -> tuple[float, bool]:
    vals = [_num(row.get(k)) for k in ("o2_min", "o2_max", "p_min_atm", "p_max_atm")]
    o2_lo, o2_hi, p_lo, p_hi = vals
    if o2_lo is None or o2_hi is None or p_lo is None or p_hi is None:
        return 0.0, True
    best = 0.0
    for o2_t, p_t in TARGETS:
        d = math.hypot(_box_distance(o2_lo, o2_hi, o2_t) / O2_SCALE, _box_distance(p_lo, p_hi, p_t) / P_SCALE)
        best = max(best, math.exp(-0.5 * d * d))
    basis = str(row.get("range_basis") or "")
    return (best if basis == "tested" else best * UNTESTED_FACTOR), False


def flow_score(row: Mapping) -> tuple[float, bool]:
    lo, hi = _num(row.get("flow_min_cms")), _num(row.get("flow_max_cms"))
    if lo is None or hi is None:
        return 0.0, True
    overlap = max(0.0, min(hi + FLOW_PAD, VENT_MAX) - max(lo - FLOW_PAD, VENT_MIN))
    s = overlap / (VENT_MAX - VENT_MIN)
    return (s if row.get("flow_basis") == "tested" else s * UNTESTED_FACTOR), False


def data_score(row: Mapping) -> tuple[float, bool]:
    s = DATA.get(str(row.get("data_available")))
    return (s, False) if s is not None else (0.0, True)


def suppression_score(row: Mapping) -> tuple[float, bool]:
    v = row.get("suppression_detection")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0, True
    return (1.0 if bool(v) else 0.0), False


SCORERS = {"habitat": habitat_score, "o2_pressure": o2_pressure_score, "flow": flow_score,
           "data": data_score, "suppression": suppression_score}


def normalise_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    unknown = set(w) - set(FACTORS)
    if unknown:
        raise ValueError(f"Unknown ranking factors: {unknown}")
    if any(v < 0 for v in w.values()):
        raise ValueError("Weights must be non-negative")
    total = sum(w.values())
    return {k: (v / total if total else 0.0) for k, v in w.items()}


def score_row(row: Mapping, weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Score one investigation; returns per-factor score, contribution and missing flags."""
    w = normalise_weights(weights)
    out: dict[str, Any] = {"missing": []}
    total = 0.0
    for f in FACTORS:
        s, missing = SCORERS[f](row)
        out[f"{f}_score"] = round(s, 4)
        out[f"{f}_points"] = round(100 * w[f] * s, 2)
        total += 100 * w[f] * s
        if missing:
            out["missing"].append(f)
    out["score"] = round(total, 2)
    return out


def rank_investigations(inv: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Deterministic ranking: score desc, then name asc. Never mutates `inv`."""
    rows = []
    for rec in inv.to_dict("records"):
        s = score_row(rec, weights)
        rows.append({"key": rec["key"], "name": rec["name"], "fuel_type": rec["fuel_type"], **s,
                     "data_missing": ", ".join(s.pop("missing")) or ""})
    out = pd.DataFrame(rows).sort_values(["score", "name"], ascending=[False, True], kind="mergesort")
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def main() -> pd.DataFrame:
    from src.catalog import INVESTIGATIONS_PATH, build_investigations

    inv = build_investigations() if not INVESTIGATIONS_PATH.exists() else pd.read_csv(INVESTIGATIONS_PATH)
    ranked = rank_investigations(inv)
    cols = ["rank", "name", "score"] + [f"{f}_points" for f in FACTORS] + ["data_missing"]
    print(ranked[cols].to_string(index=False))
    return ranked


if __name__ == "__main__":
    main()
