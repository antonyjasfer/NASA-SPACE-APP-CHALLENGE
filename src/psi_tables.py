"""Loaders for the extra PSI experimental tables (CLAUDE.md "Extra PSI tables").

Run: python -m src.psi_tables   (prints a summary of every table)
Rules: headers are whitespace-stripped; non-test rows (notes, "Pre-test flow visulation data...",
all-blank rows) are dropped and counted; raw text is kept next to every parsed number; nothing is
interpolated or converted beyond explicit unit changes (%, bar -> atm).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ingest import ROOT

RAW = ROOT / "data" / "raw"
BAR_TO_ATM = 1 / 1.01325
FILES = {
    "PSI-25": "PSI-25_Experimental table_BASS-II.csv",
    "PSI-26": "PSI-26_Experimental table_BASS.csv",
    "PSI-10": "PSI-10_Experimental table_Flame Design_Inverse Flame.csv",
    "PSI-159": "PSI-159_Experimental table_CFI-G_Normal Flames.csv",
    "PSI-98": "PSI-98_Experimental table_SAFFIRE-1.csv",
    "PSI-99": "PSI-99_Experimental table_SAFFIRE-2.csv",
    "PSI-100": "PSI-100_Experimental table_SAFFIRE-3.csv",
}
# Unsigned on purpose: every quantity here is non-negative, so "21.5-21.7" must not yield -21.7.
_NUM = re.compile(r"(?<![\d.])\d*\.?\d+")
_NOTE = re.compile(r"\((?:note|see)[^)]*\)", re.IGNORECASE)


@dataclass
class PSITable:
    psi_id: str
    name: str
    data: pd.DataFrame
    n_raw_rows: int
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return sum(self.dropped.values())


def read_raw(psi_id: str) -> pd.DataFrame:
    """Read a table as text with stripped headers (utf-8; BOM tolerated)."""
    df = pd.read_csv(RAW / FILES[psi_id], encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip())


def _drop_non_tests(df: pd.DataFrame, id_col: str, pattern: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep rows whose id matches `pattern`; count what was removed and why."""
    blank = df.eq("").all(axis=1)
    keep = df[id_col].str.fullmatch(pattern) & ~blank
    dropped = {"all_blank": int(blank.sum()), "not_a_test_row": int((~keep & ~blank).sum())}
    return df[keep].reset_index(drop=True), {k: v for k, v in dropped.items() if v}


def parse_first_number(text: str | None) -> tuple[float, bool]:
    """'~ 21.5' -> (21.5, True); '22.1 to 22.0' -> (22.1, False); '20 cm/s' -> (20.0, False)."""
    if text is None or not str(text).strip():
        return np.nan, False
    clean = _NOTE.sub("", str(text))
    m = _NUM.search(clean)
    return (float(m.group()) if m else np.nan), "~" in clean


def parse_range(text: str | None) -> tuple[float, float]:
    """All numbers in a cell -> (min, max). '21.5 - 21.7' and '21.5-21.7' -> (21.5, 21.7);
    parenthesised notes such as '(Note 2)' are ignored."""
    nums = [float(x) for x in _NUM.findall(_NOTE.sub("", str(text or "")))]
    return (min(nums), max(nums)) if nums else (np.nan, np.nan)


def _check_fraction(values: pd.Series, label: str) -> None:
    v = values.dropna()
    if ((v < 0) | (v > 1)).any():
        raise ValueError(f"{label}: O2 fraction outside 0..1: {v[(v < 0) | (v > 1)].tolist()}")


# --------------------------------------------------------------------------- tables
def load_bass2() -> PSITable:
    raw = read_raw("PSI-25")
    df, dropped = _drop_non_tests(raw, "Test #", r"[A-Z]\d+[A-Za-z]?")
    out = pd.DataFrame({
        "test_id": df["Test #"], "date": df["Date"], "pi": df["PI"],
        "material": df["Fuel Sample Material"],
        "x_o2": pd.to_numeric(df["Calibrated  initial O2 % by vol"], errors="raise") / 100.0,
        "x_o2_final": pd.to_numeric(df["Calibrated final O2 % by vol"], errors="coerce") / 100.0,
        "p_atm": np.nan,             # no pressure column in PSI-25
        "p_assumed_atm": 1.0,        # ISS cabin; for plotting ONLY, never for analysis
        "p_assumed": True,
        "flow_cms": np.nan,          # fan/air "display" values are settings, not cm/s
        "fan_display_raw": df["Fan display"], "air_display_raw": df["Air display"],
        "flow_restrictor": df["Flow restrictor"],
    })
    _check_fraction(out["x_o2"], "PSI-25")
    return PSITable("PSI-25", "BASS-II", out, len(raw), dropped)


def load_bass() -> PSITable:
    raw = read_raw("PSI-26")
    df, dropped = _drop_non_tests(raw, "Sample #", r"\d+")
    out = df.rename(columns={"Date": "date", "Test #": "test_id", "As Run Test #": "as_run_test_no",
                             "Sample #": "sample_no", "Fuel Sample Material": "material"})
    return PSITable("PSI-26", "BASS", out, len(raw), dropped)


def load_inverse_flame() -> PSITable:
    raw = read_raw("PSI-10")
    df, dropped = _drop_non_tests(raw, "Test", r"\w+")
    x = pd.to_numeric(df["XO2"], errors="coerce")
    out = pd.DataFrame({
        "test_id": df["Test"], "day": df["Day"], "flame_type": df["Flame Type"],
        "x_o2": x, "x_o2_raw": df["XO2"], "x_o2_not_constant": x.isna(),  # 'RAMP' / 'LOW O2'
        "p_atm": pd.to_numeric(df["pmean (atm)"], errors="raise"),
        "fuel": "ethylene",
    })
    _check_fraction(out["x_o2"], "PSI-10")
    return PSITable("PSI-10", "Flame Design (inverse flames)", out, len(raw), dropped)


def load_cfig() -> PSITable:
    raw = read_raw("PSI-159")
    df, dropped = _drop_non_tests(raw, "Test", r"[\w ]+")
    end = df["Self-ext or flow term"].map({"SE": "Self-extinguished", "FT": "Flow terminated"})
    end = end.where(end.notna(), np.where(df["Hot ignition"].eq("No"), "No hot ignition", None))
    out = pd.DataFrame({
        "test_id": df["Test"], "fuel": df["Fuel"],
        "x_o2": pd.to_numeric(df["XO2 before test"], errors="raise"),
        "p_atm": pd.to_numeric(df["approx p (bar)"], errors="raise") * BAR_TO_ATM,
        "p_approx": True,  # header says "approx p (bar)"
        "hot_ignition": df["Hot ignition"].eq("Yes"),
        "cool_flame_raw": df["Cool flame?"],
        # blank = not recorded (unknown), kept as None rather than False
        "cool_flame": df["Cool flame?"].str.lower().map(
            lambda v: None if not v else v.startswith("yes")),
        "end_raw": df["Self-ext or flow term"], "outcome": end,
    })
    _check_fraction(out["x_o2"], "PSI-159")
    return PSITable("PSI-159", "CFI-G (normal flames)", out, len(raw), dropped)


def _load_saffire(psi_id: str, name: str, o2_col: str | None, flow_col: str,
                  extra: dict[str, str] | None = None) -> PSITable:
    raw = read_raw(psi_id)
    df, dropped = _drop_non_tests(raw, raw.columns[0], r"S\d+|\d+-\d+")
    if df[raw.columns[0]].duplicated().any():
        raise ValueError(f"{psi_id}: duplicate sample ids")
    o2 = [parse_first_number(v) for v in df[o2_col]] if o2_col else [(np.nan, False)] * len(df)
    rng = [parse_range(v) for v in df[o2_col]] if o2_col else [(np.nan, np.nan)] * len(df)
    out = pd.DataFrame({
        "test_id": df[raw.columns[0]], "material": df["Material"].replace("", np.nan),
        "x_o2": [v / 100 for v, _ in o2], "x_o2_min": [a / 100 for a, _ in rng],
        "x_o2_max": [b / 100 for _, b in rng], "x_o2_approx": [a for _, a in o2],
        "x_o2_raw": df[o2_col] if o2_col else "",
        "p_atm": np.nan,
        "flow_cms": [parse_first_number(v)[0] for v in df[flow_col]], "flow_raw": df[flow_col],
    })
    for new_col, raw_col in (extra or {}).items():   # built from the same filtered frame: no realignment
        out[new_col] = df[raw_col].to_numpy()
    for col in ("x_o2", "x_o2_min", "x_o2_max"):
        _check_fraction(out[col], f"{psi_id} {col}")
    if (out["x_o2_min"] > out["x_o2_max"]).any():
        raise ValueError(f"{psi_id}: O2 min > max")
    return PSITable(psi_id, name, out, len(raw), dropped)


def load_saffire1() -> PSITable:
    return _load_saffire("PSI-98", "Saffire-I", "Percent O2", "Air Flow (cm/s)")


def load_saffire2() -> PSITable:
    # header '?-g' = micro-g (lost mu character)
    return _load_saffire("PSI-99", "Saffire-II", "Percent O2 (Note 1)", "Air Flow (cm/s)",
                         extra={"ug_spread_raw": "?-g Spread Length"})


def load_saffire3() -> PSITable:
    return _load_saffire("PSI-100", "Saffire-III", None, "Air Flow")


LOADERS = {"PSI-25": load_bass2, "PSI-26": load_bass, "PSI-10": load_inverse_flame,
           "PSI-159": load_cfig, "PSI-98": load_saffire1, "PSI-99": load_saffire2,
           "PSI-100": load_saffire3}


def load_all() -> dict[str, PSITable]:
    return {k: f() for k, f in LOADERS.items()}


def main() -> None:
    for t in load_all().values():
        d = t.data
        o2 = d["x_o2"] if "x_o2" in d else pd.Series(dtype=float)
        p = d["p_atm"] if "p_atm" in d else pd.Series(dtype=float)
        print(f"{t.psi_id:8s} {t.name:31s} raw={t.n_raw_rows:4d} tests={len(d):4d} dropped={t.dropped or 0} "
              f"O2=[{o2.min():.3f}, {o2.max():.3f}] p_atm=[{p.min():.3f}, {p.max():.3f}]")


if __name__ == "__main__":
    main()
