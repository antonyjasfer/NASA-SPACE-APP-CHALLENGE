import numpy as np
import pandas as pd

from src.coverage import coverage_table, disruption_contrast
from src.features import FUELS, LOI, MODEL_FEATURES
from src.models import (
    NEAREST_FEATURES,
    NearestTestIndex,
    fit_cqr,
    group_splits,
    make_hgb_regressor,
    predict_interval,
)


def test_group_kfold_never_shares_a_test_date(feats):
    splits = group_splits(feats)
    assert len(splits) == 5
    seen_test = []
    for tr, te in splits:
        assert not set(feats["test_date"].iloc[tr]) & set(feats["test_date"].iloc[te])
        seen_test.extend(te)
    assert sorted(seen_test) == list(range(len(feats)))  # every row tested exactly once


def test_group_kfold_k_subset_never_shares_a_test_date(feats):
    sub = feats[feats["K_mm2_s"].notna()].reset_index(drop=True)
    for tr, te in group_splits(sub):
        assert not set(sub["test_date"].iloc[tr]) & set(sub["test_date"].iloc[te])


def test_k_model_monotonic_in_o2(feats):
    sub = feats[feats["K_mm2_s"].notna()]
    model = make_hgb_regressor().fit(sub[MODEL_FEATURES], sub["K_mm2_s"])
    base = sub.iloc[[0, 50, 120]][MODEL_FEATURES]
    for _, row in base.iterrows():
        fuel = FUELS[0] if row["fuel_methanol"] == 1 else FUELS[1]
        grid = pd.DataFrame([row] * 23)
        grid["x_o2"] = np.linspace(0.12, 0.34, 23)
        grid["po2_atm"] = grid["x_o2"] * grid["p_atm"]
        grid["o2_margin"] = grid["x_o2"] - LOI[fuel]
        pred = model.predict(grid)
        assert (np.diff(pred) >= -1e-12).all()


def test_nearest_test_distance_zero_for_real_test(feats):
    index = NearestTestIndex.fit(feats, NEAREST_FEATURES)
    hit = index.query(feats.iloc[[5]])
    assert hit["nearest_distance"].iloc[0] == 0.0
    assert hit["nearest_test_no"].iloc[0] == feats["test_no"].iloc[5]
    far = feats.iloc[[5]].copy()
    far["x_o2"], far["p_atm"] = 0.34, 0.4  # outside anything FLEX tested
    assert index.query(far)["nearest_distance"].iloc[0] > 1.0


def test_cqr_interval_ordered(feats):
    sub = feats[feats["K_mm2_s"].notna()].reset_index(drop=True)
    cqr = fit_cqr(sub[MODEL_FEATURES], sub["K_mm2_s"], sub["test_date"])
    lo, hi = predict_interval(cqr, sub[MODEL_FEATURES].head(20))
    assert cqr["q_hat"] >= 0 and (hi >= lo).all()


def test_coverage_counts_every_real_test(flex):
    table = coverage_table(flex)
    assert table["n"].sum() == len(flex)
    assert table[["Extinction", "Disruption", "Completion"]].sum().to_dict() == {
        "Extinction": 186, "Disruption": 61, "Completion": 27}
    summary = disruption_contrast(flex)
    assert summary["high_o2"]["n_tests"] == int((flex["x_o2"] >= 0.25).sum())


def test_cqr_interval_nonnegative_and_ordered_everywhere(feats):
    sub = feats[feats["K_mm2_s"].notna()].reset_index(drop=True)
    cqr = fit_cqr(sub[MODEL_FEATURES], sub["K_mm2_s"], sub["test_date"])
    lo, hi = predict_interval(cqr, feats[MODEL_FEATURES])
    assert (lo >= 0).all() and (hi >= lo).all()
