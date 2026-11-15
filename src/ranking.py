"""Transparent 0-100 relevance score for the investigation catalog (app tab 1 sliders).

Run: python -m src.ranking
score = 100 * sum(w_f * s_f) / sum(w_f), each factor s_f in [0, 1]:
  habitat       solid 1.0 > droplet 0.6 > gas 0.3 (cabin materials matter most)
  o2_pressure   closeness of real evidence to the exploration targets, exp(-d^2/2), d in units of
                5 O2 points and 0.2 atm. If the investigation has a PSI table with measured O2 AND
                pressure, d = distance to the NEAREST MEASURED TEST (a min-max box would claim
                coverage where no test exists). Otherwise d = distance to the stated range box,
                A range that is only planned or a facility capability is NOT evidence: it scores 0
                and is flagged "not tested" (same rule for SoFIE and FLEX-2).
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

import numpy as np
import pandas as pd

from src.catalog_data import LITERATURE_POINTS, O2_SCALE, P_SCALE

FACTORS = ("habitat", "o2_pressure", "flow", "data", "suppression")
DEFAULT_WEIGHTS: dict[str, float] = {"habitat": 0.25, "o2_pressure": 0.30, "flow": 0.15,
                                     "data": 0.15, "suppression": 0.15}
HABITAT = {"solid": 1.0, "droplet": 0.6, "gas": 0.3}
DATA = {"table": 1.0, "telemetry": 0.8, "video": 0.5, "none": 0.0}
UNTESTED_FACTOR = 0.0   # evidence = tested only: planned / facility-capability ranges score 0
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


def o2_pressure_score(row: Mapping, measured: np.ndarray | None = None) -> tuple[float, bool]:
    """`measured`: (n, 2) array of real (O2, p_atm) tests for this investigation, if any."""
    if measured is not None and len(measured):
        best = 0.0
        for o2_t, p_t in TARGETS:
            d = np.hypot((measured[:, 0] - o2_t) / O2_SCALE, (measured[:, 1] - p_t) / P_SCALE).min()
            best = max(best, math.exp(-0.5 * float(d) ** 2))
        return best, False
    if is_untested(row, "range_basis"):
        return 0.0, False   # flagged separately as "not tested"
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


def is_untested(row: Mapping, basis_col: str) -> bool:
    basis = row.get(basis_col)
    return isinstance(basis, str) and basis != "tested"


def measured_by_psi(points: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """psi_id -> array of measured (O2, p) tests, from coverage_points (assumed pressure excluded)."""
    if points is None:
        return {}
    ok = points[points["x_o2"].notna() & points["p_atm"].notna()]
    return {psi: g[["x_o2", "p_atm"]].to_numpy(dtype=float) for psi, g in ok.groupby("source")}


def flow_score(row: Mapping) -> tuple[float, bool]:
    if is_untested(row, "flow_basis"):
        return 0.0, False
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


def score_row(row: Mapping, weights: Mapping[str, float] | None = None,
              measured: np.ndarray | None = None) -> dict[str, Any]:
    """Score one investigation; returns per-factor score, contribution and missing flags."""
    w = normalise_weights(weights)
    out: dict[str, Any] = {"missing": []}
    total = 0.0
    for f in FACTORS:
        s, missing = o2_pressure_score(row, measured) if f == "o2_pressure" else SCORERS[f](row)
        out[f"{f}_score"] = round(s, 4)
        out[f"{f}_points"] = round(100 * w[f] * s, 2)
        total += 100 * w[f] * s
        if missing:
            out["missing"].append(f)
    untested = [b for b in ("range_basis", "flow_basis") if is_untested(row, b)]
    out["not_tested"] = ", ".join(sorted({str(row.get(b)) for b in untested}))
    out["o2p_method"] = ("nearest measured test" if measured is not None and len(measured)
                         else "not tested (" + str(row.get("range_basis")) + ")" if is_untested(row, "range_basis")
                         else "stated range" if "o2_pressure" not in out["missing"] else "missing")
    out["score"] = round(total, 2)
    return out


def rank_investigations(inv: pd.DataFrame, weights: Mapping[str, float] | None = None,
                        points: pd.DataFrame | None = None) -> pd.DataFrame:
    """Deterministic ranking: score desc, then name asc. Never mutates `inv`.
    Pass `points` (src.coverage.coverage_points()) so PSI-backed rows are scored on real tests."""
    measured = measured_by_psi(points)
    rows = []
    for rec in inv.to_dict("records"):
        s = score_row(rec, weights, measured.get(str(rec.get("psi_id"))))
        rows.append({"key": rec["key"], "name": rec["name"], "fuel_type": rec["fuel_type"], **s,
                     "data_missing": ", ".join(s.pop("missing")) or ""})
    out = pd.DataFrame(rows).sort_values(["score", "name"], ascending=[False, True], kind="mergesort")
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def main() -> pd.DataFrame:
    from src.catalog import INVESTIGATIONS_PATH, build_investigations

    inv = build_investigations() if not INVESTIGATIONS_PATH.exists() else pd.read_csv(INVESTIGATIONS_PATH)
    from src.coverage import coverage_points

    ranked = rank_investigations(inv, points=coverage_points())
    cols = ["rank", "name", "score"] + [f"{f}_points" for f in FACTORS] + ["o2p_method", "data_missing"]
    print(ranked[cols].to_string(index=False))
    return ranked


if __name__ == "__main__":
    main()
