"""Saffire-III 1 Hz telemetry -> small, honest replay file for the app.

Run: python -m src.telemetry   (reads the xlsx once; the app only reads the small CSV)
Source: data/raw/Saffire-III 1 Hz Telemetry Data.xlsx (PSI-100, DOI 10.60555/040q-4h36). Raw file untouched.

Cleaning rules (documented, no unit invented):
- Both sheets are time-multiplexed: in most rows a channel is not sampled and holds a placeholder.
  CygnusData placeholder = exactly 0 (cabin pressure ~14 psi / air ~20 degC can never read 0);
  SaffireData placeholder = the channel's start-of-file value, repeated exactly (e.g. radiometer
  -4002.47, anemometer -76.19, igniter current -3.05). Placeholders -> missing.
- Cygnus channels keep their published units. Channels with < 50 real samples are dropped
  (FIZ 1 obscuration: none; FIZ 1 scatterometer: 2).
- Radiometers/anemometers have no published unit: shown as change from baseline in raw counts
  (unitless), baseline = median of the first 60 s of REAL samples. (The first 60 s of the file are
  placeholders only, so a file-start baseline would be meaningless.)
- Ignition is marked only where an igniter current shows a clear pulse (> 0.5 A after placeholders
  are removed); the file does not say which PSI-100 sample each igniter lit.
- Downsampling keeps every k-th real sample (no averaging or interpolation), <= 2,000 per channel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest import PROCESSED_DIR, ROOT

RAW_XLSX = ROOT / "data" / "raw" / "Saffire-III 1 Hz Telemetry Data.xlsx"
SMALL_PATH = PROCESSED_DIR / "saffire3_telemetry_small.csv"
EVENTS_PATH = PROCESSED_DIR / "saffire3_ignition_events.csv"
DOI = "10.60555/040q-4h36"
MAX_POINTS = 2000
MIN_REAL_SAMPLES = 50
BASELINE_S = 60
IGNITION_CURRENT_A = 0.5
UNITLESS = "change from baseline (raw counts, unitless)"
CAPTION = "Raw radiometer/anemometer counts are unitless; units are not published in PSI-100."

CYGNUS_CHANNELS: dict[str, tuple[str, str]] = {   # raw column -> (display name, unit as published)
    "PCM Air Temp (degC)": ("Cabin air temperature", "degC"),
    "PCM Press EPS 1 (psi)": ("Cabin pressure EPS 1", "psi"),
    "PCM Press ECM 2 (psi)": ("Cabin pressure ECM 2", "psi"),
    "PCM Press EPS 3 (psi)": ("Cabin pressure EPS 3", "psi"),
    "FIZ1 Smoke Detector Obscuration  (V)": ("FIZ 1 smoke obscuration", "V"),
    "FIZ 2 Smoke Detector Obscuration (V)": ("FIZ 2 smoke obscuration", "V"),
    "FIZ 1 Smoke Detector Scatterometer (V)": ("FIZ 1 smoke scatterometer", "V"),
    "FIZ 2 Smoke Detector Scatterometer (V)": ("FIZ 2 smoke scatterometer", "V"),
}
SAFFIRE_COUNT_CHANNELS = [f"Radiometer{i}Value" for i in range(1, 5)] + ["Anemometer1", "Anemometer2"]
IGNITERS = {f"Igniter {i}": f"Sample{i}IgniterI" for i in range(1, 10)}


def read_sheets(path=RAW_XLSX) -> tuple[pd.DataFrame, pd.DataFrame]:
    cyg = pd.read_excel(path, sheet_name="CygnusData", usecols="A:L", engine="openpyxl")
    saf = pd.read_excel(path, sheet_name="SaffireData", engine="openpyxl")
    cyg = cyg[cyg["CygnusTimestamp"] > 1e12]          # drops a corrupt negative-timestamp row
    return cyg.reset_index(drop=True), saf


def _downsample(t: pd.Series, v: pd.Series) -> tuple[pd.Series, pd.Series]:
    ok = v.notna()
    t, v = t[ok], v[ok]
    step = max(1, int(np.ceil(len(v) / MAX_POINTS)))
    return t.iloc[::step], v.iloc[::step]


def cygnus_long(cyg: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col, (name, unit) in CYGNUS_CHANNELS.items():
        v = cyg[col].where(cyg[col] != 0)              # 0 = not sampled in this row
        if v.notna().sum() < MIN_REAL_SAMPLES:
            continue
        t, vv = _downsample(cyg["Relative Time (s)"], v)
        frames.append(pd.DataFrame({"channel": name, "unit": unit, "t_s": t.to_numpy(), "value": vv.to_numpy(),
                                    "sheet": "CygnusData", "raw_column": col}))
    return pd.concat(frames, ignore_index=True)


def saffire_long(saf: pd.DataFrame) -> pd.DataFrame:
    frames = []
    t_all = saf["Relative Time (s)"]
    for col in SAFFIRE_COUNT_CHANNELS:
        v = saf[col].where(saf[col] != saf[col].iloc[0])  # start-of-file placeholder -> missing
        real_t = t_all[v.notna()]
        if real_t.empty:
            continue
        baseline = v[v.notna() & (t_all <= real_t.min() + BASELINE_S)].median()
        t, vv = _downsample(t_all, v - baseline)
        name = col.replace("Value", "")
        frames.append(pd.DataFrame({"channel": name, "unit": UNITLESS, "t_s": t.to_numpy(), "value": vv.to_numpy(),
                                    "sheet": "SaffireData", "raw_column": col}))
    return pd.concat(frames, ignore_index=True)


def ignition_events(saf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in IGNITERS.items():
        v = saf[col].where(saf[col] != saf[col].iloc[0])
        on = saf.loc[v > IGNITION_CURRENT_A, "Relative Time (s)"]
        if len(on):
            rows.append({"igniter": name, "t_start_s": float(on.min()), "t_end_s": float(on.max()),
                         "n_samples": int(len(on)), "peak_current_A": float(v.max()), "raw_column": col})
    return pd.DataFrame(rows, columns=["igniter", "t_start_s", "t_end_s", "n_samples", "peak_current_A", "raw_column"])


def main() -> pd.DataFrame:
    cyg, saf = read_sheets()
    small = pd.concat([cygnus_long(cyg), saffire_long(saf)], ignore_index=True)
    small.to_csv(SMALL_PATH, index=False)
    events = ignition_events(saf)
    events.to_csv(EVENTS_PATH, index=False)
    size_mb = SMALL_PATH.stat().st_size / 1e6
    print(small.groupby(["channel", "unit"]).size().to_string())
    print(f"\n{len(small)} rows, {size_mb:.2f} MB -> {SMALL_PATH.relative_to(ROOT)}")
    print(events.to_string(index=False))
    return small


if __name__ == "__main__":
    main()
