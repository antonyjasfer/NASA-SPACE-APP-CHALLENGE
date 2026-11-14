import numpy as np
import pandas as pd
import pytest

from src.ingest import (
    COLUMN_MAP,
    EXPECTED_ROWS,
    RAW_PATH,
    clean,
    load_raw,
    parse_test_date,
)


def test_raw_file_needs_cp1252():
    with pytest.raises(UnicodeDecodeError):
        pd.read_csv(RAW_PATH, encoding="utf-8")
    assert load_raw().shape == (EXPECTED_ROWS, 15)


def test_row_count_preserved_no_rows_invented(raw, flex):
    assert len(raw) == len(flex) == EXPECTED_ROWS
    assert flex["test_no"].tolist() == raw["FLEX Test #"].astype(int).tolist()
    assert flex["flex_id"].tolist() == raw["FLEX Identifier"].str.strip().tolist()


def test_renames(flex):
    assert set(COLUMN_MAP.values()) <= set(flex.columns)
    assert "K_mm2_s" in flex.columns and "Burning rate; mm" not in flex.columns
    assert not any(";" in c for c in flex.columns)


def test_gas_columns_are_o2_n2_co2_he(flex):
    total = flex[["x_o2", "x_n2", "x_co2", "x_he"]].sum(axis=1)
    assert total.between(0.97, 1.02).all()
    assert flex["x_o2"].between(0.12, 0.34).all()


def test_en_dash_and_blank_become_nan(raw, flex):
    for raw_col, col in [("Droplet initial diameter; mm", "d0_mm"),
                         ("Visible flame extinction diameter; mm", "d_ext_mm"),
                         ("Burning rate; mm", "K_mm2_s")]:
        missing = raw[raw_col].str.strip().isin(["", "–"])
        assert missing.any() or col == "K_mm2_s"
        assert flex.loc[missing.values, col].isna().all()
        assert flex.loc[~missing.values, col].notna().all()


def test_approximate_burn_times(raw, flex):
    approx = raw["Burn time; s"].str.startswith("~")
    assert approx.sum() == 2
    assert flex["burn_time_approx"].sum() == 2
    assert sorted(flex.loc[flex["burn_time_approx"], "burn_time_s"]) == [6.0, 9.0]


def test_zero_pressure_becomes_nan(flex):
    row = flex[flex["flex_id"] == "C11M201"]
    assert len(row) == 1
    assert np.isnan(row["p_mmhg"].iloc[0]) and np.isnan(row["p_atm"].iloc[0])
    assert row["p_was_zero"].iloc[0]
    assert flex["p_was_zero"].sum() == 1
    assert (flex["p_mmhg"].dropna() > 0).all()


def test_duplicate_id_flagged_not_dropped(flex):
    dup = flex[flex["dup_id"]]
    assert dup["flex_id"].tolist() == ["193F001", "193F001"]
    assert flex["dup_id"].sum() == 2


def test_p_atm_conversion(flex):
    np.testing.assert_allclose(flex["p_atm"], flex["p_mmhg"] / 760.0)


def test_outcomes_and_fuels(flex):
    assert flex["test_end"].value_counts().to_dict() == {"Extinction": 186, "Disruption": 61, "Completion": 27}
    assert flex["fuel"].value_counts().to_dict() == {"Methanol": 157, "Heptane": 117}


def test_test_date_parsing_two_formats(flex):
    s = pd.Series(["3/5/2009", "10/24/09"])
    assert [str(d) for d in parse_test_date(s)] == ["2009-03-05", "2009-10-24"]
    assert flex["test_date"].nunique() == 80
    assert str(min(flex["test_date"])) == "2009-03-05"


def test_unknown_token_raises(raw):
    bad = raw.copy()
    bad.loc[0, "Burning rate; mm"] = "n/a"
    with pytest.raises(ValueError):
        clean(bad)


def test_schema_change_raises(raw):
    with pytest.raises(ValueError, match="schema"):
        clean(raw.rename(columns={"Test end": "Outcome"}))


def test_out_of_range_year_raises():
    with pytest.raises(ValueError, match="FLEX years"):
        parse_test_date(pd.Series(["3/5/1999"]))
    with pytest.raises(ValueError):
        parse_test_date(pd.Series(["2009-03-05"]))
