"""Load and clean the FLEX droplet-combustion table (PSI-69, DOI 10.60555/mbq8-0451).

Run: python -m src.ingest
Applies every FLEX rule in CLAUDE.md "Data rules". Rows are never added or dropped
(274 in -> 274 out); problems are turned into NaN or flag columns instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "PSI-69_Experimental table_FLEX.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "flex_clean.csv"
DICT_PATH = PROCESSED_DIR / "flex_data_dictionary.md"

SOURCE = "NASA PSI-69 FLEX experimental table, DOI 10.60555/mbq8-0451"
ENCODING = "cp1252"
EXPECTED_ROWS = 274
MMHG_PER_ATM = 760.0
MISSING_TOKENS = ("", "–", "-", "—")  # en dash is how the file marks missing values

# Raw header -> clean name. Subscripts were lost in the raw headers: "O" is O2, "N" is N2,
# "CO" is CO2 (not carbon monoxide), "He" is helium.
COLUMN_MAP: dict[str, str] = {
    "FLEX Test #": "test_no",
    "FLEX Identifier": "flex_id",
    "Test Date": "test_date",
    "Test GMT": "test_gmt",
    "Fuel": "fuel",
    "Ambient pressure; mmHg": "p_mmhg",
    "O initial ambient composition; mole fraction": "x_o2",
    "N initial ambient composition; mole fraction": "x_n2",
    "CO initial ambient composition; mole fraction": "x_co2",
    "He initial ambient composition; mole fraction": "x_he",
    "Droplet initial diameter; mm": "d0_mm",
    "Visible flame extinction diameter; mm": "d_ext_mm",
    "Burning rate; mm": "K_mm2_s",  # header says mm, units are really mm^2/s
    "Burn time; s": "burn_time_s",
    "Test end": "test_end",
}
NUMERIC_COLS = ["test_no", "p_mmhg", "x_o2", "x_n2", "x_co2", "x_he",
                "d0_mm", "d_ext_mm", "K_mm2_s", "burn_time_s"]
OUTCOMES = ("Extinction", "Disruption", "Completion")
ZERO_PRESSURE_IDS = ("C11M201",)
DUPLICATE_IDS = ("193F001",)
FLEX_YEARS = (2009, 2011)  # range present in the raw file (verified); guards %y century errors

DATA_DICTIONARY: list[tuple[str, str, str]] = [
    ("test_no", "-", "FLEX test sequence number (raw 'FLEX Test #')"),
    ("flex_id", "-", "FLEX test identifier (raw 'FLEX Identifier')"),
    ("test_date", "date (YYYY-MM-DD)", "Test date, parsed from raw M/D/YYYY (50 rows are written M/D/YY; all 2009-2011). 80 distinct dates; used as the CV group"),
    ("test_gmt", "hh:mm:ss GMT", "Test start time, kept as text"),
    ("fuel", "-", "Droplet fuel: Methanol or Heptane"),
    ("p_mmhg", "mmHg", "Ambient pressure. C11M201 raw value 0 -> NaN (physically impossible)"),
    ("p_atm", "atm", "p_mmhg / 760"),
    ("x_o2", "mole fraction", "Initial ambient O2 (raw header 'O ...', subscript lost)"),
    ("x_n2", "mole fraction", "Initial ambient N2 (raw header 'N ...')"),
    ("x_co2", "mole fraction", "Initial ambient CO2 (raw header 'CO ...'; CO2, not carbon monoxide)"),
    ("x_he", "mole fraction", "Initial ambient helium"),
    ("d0_mm", "mm", "Droplet initial diameter. '–'/blank -> NaN"),
    ("d_ext_mm", "mm", "Visible flame extinction diameter. '–'/blank -> NaN"),
    ("K_mm2_s", "mm^2/s", "Burning-rate constant (raw header 'Burning rate; mm'; true units mm^2/s)"),
    ("burn_time_s", "s", "Burn time. '~6', '~9' -> 6, 9 with burn_time_approx=True"),
    ("burn_time_approx", "bool", "True where the raw burn time was written as approximate ('~')"),
    ("test_end", "-", "Outcome: Extinction / Disruption / Completion"),
    ("p_was_zero", "bool", "True where the raw pressure was 0 and was set to NaN"),
    ("dup_id", "bool", "True where flex_id occurs more than once (193F001). Both rows kept"),
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV as text, exactly as stored (no NA inference)."""
    return pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=False)


def validate_schema(raw: pd.DataFrame) -> None:
    missing = [c for c in COLUMN_MAP if c not in raw.columns]
    extra = [c for c in raw.columns if c not in COLUMN_MAP]
    if missing or extra:
        raise ValueError(f"FLEX schema changed. missing={missing} extra={extra}")


def _parse_burn_time(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = raw.str.strip()
    approx = text.str.startswith("~")
    value = pd.to_numeric(text.str.lstrip("~").replace(list(MISSING_TOKENS), np.nan), errors="raise")
    return value, approx


def parse_test_date(raw: pd.Series) -> pd.Series:
    """Raw dates are M/D/YYYY, except 50 rows written M/D/YY (all 2009-2011). Parse each
    format explicitly; anything matching neither raises."""
    two_digit = raw.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}")
    long_form = pd.to_datetime(raw.where(~two_digit), format="%m/%d/%Y")
    short_form = pd.to_datetime(raw.where(two_digit), format="%m/%d/%y")
    parsed = long_form.fillna(short_form)
    if parsed.isna().any() or not parsed.dt.year.between(*FLEX_YEARS).all():
        raise ValueError(f"test_date outside FLEX years {FLEX_YEARS} or unparseable")
    return parsed.dt.date


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the CLAUDE.md FLEX rules. Returns a new frame with the same row count."""
    validate_schema(raw)
    df = raw.rename(columns=COLUMN_MAP).copy()
    df = df.apply(lambda s: s.str.strip())

    df["burn_time_s"], df["burn_time_approx"] = _parse_burn_time(df["burn_time_s"])
    for col in NUMERIC_COLS:
        if col == "burn_time_s":
            continue
        # errors="raise": any token we have not explicitly handled stops the pipeline.
        df[col] = pd.to_numeric(df[col].replace(list(MISSING_TOKENS), np.nan), errors="raise")
    df["test_no"] = df["test_no"].astype("Int64")

    df["p_was_zero"] = df["p_mmhg"].eq(0)
    df.loc[df["p_was_zero"], "p_mmhg"] = np.nan
    df["p_atm"] = df["p_mmhg"] / MMHG_PER_ATM

    df["test_date"] = parse_test_date(df["test_date"])
    df["dup_id"] = df["flex_id"].duplicated(keep=False)

    unknown = set(df["test_end"]) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"Unknown test_end values: {unknown}")
    if len(df) != len(raw):
        raise AssertionError("Cleaning changed the row count")

    ordered = [COLUMN_MAP[c] for c in COLUMN_MAP]
    ordered.insert(ordered.index("p_mmhg") + 1, "p_atm")
    return df[ordered + ["burn_time_approx", "p_was_zero", "dup_id"]]


def write_data_dictionary(path: Path = DICT_PATH) -> None:
    lines = [
        "# FLEX clean table — data dictionary",
        "",
        f"Source: {SOURCE} (file `data/raw/PSI-69_Experimental table_FLEX.csv`, cp1252).",
        f"Produced by `python -m src.ingest` -> `data/processed/flex_clean.csv`. "
        f"{EXPECTED_ROWS} rows in, {EXPECTED_ROWS} rows out; no rows added, dropped or imputed.",
        "",
        "| column | unit | meaning | source |",
        "|---|---|---|---|",
    ]
    lines += [f"| `{c}` | {u} | {m} | PSI-69, DOI 10.60555/mbq8-0451 |" for c, u, m in DATA_DICTIONARY]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Read the processed CSV back with proper dtypes (used by features/models/coverage)."""
    df = pd.read_csv(path, parse_dates=["test_date"])
    df["test_date"] = df["test_date"].dt.date
    return df


def main() -> pd.DataFrame:
    raw = load_raw()
    assert len(raw) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} raw rows, got {len(raw)}"
    print(f"Raw schema: {raw.shape[0]} rows x {raw.shape[1]} cols ({ENCODING})")
    for col in raw.columns:
        print(f"  {col!r} -> {COLUMN_MAP[col]}")
    df = clean(raw)
    assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} clean rows, got {len(df)}"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    write_data_dictionary()
    print(f"\nClean: {len(df)} rows -> {CLEAN_PATH.relative_to(ROOT)}")
    print(df.dtypes.to_string())
    print("\nMissing per column:", {k: int(v) for k, v in df.isna().sum().items() if v})
    print("Outcomes:", df["test_end"].value_counts().to_dict())
    print("Flags: burn_time_approx", int(df["burn_time_approx"].sum()),
          "| p_was_zero", df.loc[df["p_was_zero"], "flex_id"].tolist(),
          "| dup_id", df.loc[df["dup_id"], "flex_id"].tolist())
    print("Distinct test dates:", df["test_date"].nunique())
    return df


if __name__ == "__main__":
    main()
