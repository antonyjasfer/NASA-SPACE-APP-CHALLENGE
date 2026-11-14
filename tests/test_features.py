import numpy as np
import pandas as pd
import pytest

from src.features import LOI, MODEL_FEATURES, build_features, d2law_burn_time


def test_d2law_burn_time():
    d0 = pd.Series([2.0, 3.0, 2.0, 2.0])
    k = pd.Series([0.5, 0.6, np.nan, 0.0])
    out = d2law_burn_time(d0, k)
    assert out.iloc[0] == pytest.approx(8.0)
    assert out.iloc[1] == pytest.approx(15.0)
    assert np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])


def test_d2law_only_where_k_exists(feats):
    assert feats.loc[feats["K_mm2_s"].isna(), "d2law_burn_time_s"].isna().all()
    has = feats["K_mm2_s"].notna() & feats["d0_mm"].notna()
    np.testing.assert_allclose(feats.loc[has, "d2law_burn_time_s"],
                               feats.loc[has, "d0_mm"] ** 2 / feats.loc[has, "K_mm2_s"])


def test_physics_features(feats):
    np.testing.assert_allclose(feats["d0_mm2"], feats["d0_mm"] ** 2)
    np.testing.assert_allclose(feats["po2_atm"], feats["x_o2"] * feats["p_atm"])
    expected = feats["x_o2"] - feats["fuel"].map({"Methanol": 0.12, "Heptane": 0.15})
    np.testing.assert_allclose(feats["o2_margin"], expected)
    assert LOI == {"Methanol": 0.12, "Heptane": 0.15}


def test_fuel_one_hot(feats):
    assert (feats["fuel_methanol"] + feats["fuel_heptane"] == 1).all()
    assert feats["fuel_methanol"].sum() == 157


def test_features_exclude_outcomes():
    for leak in ("d_ext_mm", "K_mm2_s", "burn_time_s", "d2law_burn_time_s", "test_end"):
        assert leak not in MODEL_FEATURES


def test_build_features_does_not_mutate_or_add_rows(flex):
    before = flex.copy()
    out = build_features(flex)
    pd.testing.assert_frame_equal(flex, before)
    assert len(out) == len(flex)


def test_unknown_fuel_raises(flex):
    with pytest.raises(ValueError):
        build_features(flex.assign(fuel="Decane"))
