"""P2: PSI table parsing, catalog provenance, ranking, coverage points and recommender."""
import numpy as np
import pandas as pd
import pytest

from src.catalog import (
    MAX_CARD_WORDS,
    build_investigations,
    build_literature,
    investigation_cards,
    special_cards,
    word_count,
)
from src.catalog_data import INVESTIGATIONS, MISCONCEPTIONS
from src.coverage import coverage_points, recommend_experiments
from src.psi_tables import load_all, parse_first_number, parse_range
from src.ranking import DEFAULT_WEIGHTS, FACTORS, rank_investigations, score_row

O2P = ["o2_min", "o2_max", "p_min_atm", "p_max_atm"]
FLOW = ["flow_min_cms", "flow_max_cms"]


@pytest.fixture(scope="module")
def tables():
    return load_all()


@pytest.fixture(scope="module")
def inv():
    return build_investigations()


@pytest.fixture(scope="module")
def points():
    return coverage_points()


# --------------------------------------------------------------------------- PSI tables
def test_parsers():
    assert parse_range("21.5-21.7") == (21.5, 21.7)          # unspaced hyphen is not a minus sign
    assert parse_range("21.5 (Note 2)") == (21.5, 21.5)      # note numbers ignored
    assert parse_first_number(".33 cm") == (0.33, False)
    assert parse_first_number("~ 21.5") == (21.5, True)
    assert parse_first_number("22.1 to 22.0") == (22.1, False)
    assert parse_first_number("20 cm/s") == (20.0, False)
    assert np.isnan(parse_first_number("")[0])
    assert parse_range("21.5 - 21.7") == (21.5, 21.7)


def test_non_test_rows_dropped_and_logged(tables):
    assert tables["PSI-98"].dropped == {"not_a_test_row": 2}   # pre-test flow visualisation rows
    assert tables["PSI-99"].dropped == {"not_a_test_row": 6}   # flow-vis rows + 4 notes
    assert tables["PSI-26"].dropped == {"not_a_test_row": 1}   # "SOOT Cleaning"
    for t in tables.values():
        assert len(t.data) + t.n_dropped == t.n_raw_rows


def test_verified_ranges_match_claude_md(tables):
    b2 = tables["PSI-25"].data
    assert (len(b2), b2["x_o2"].min(), b2["x_o2"].max()) == (129, 0.136, 0.222)
    assert b2["p_atm"].isna().all() and b2["flow_cms"].isna().all()
    inv_f = tables["PSI-10"].data
    assert round(inv_f["x_o2"].min(), 2) == 0.21 and round(inv_f["x_o2"].max(), 2) == 0.85
    assert round(inv_f["p_atm"].min(), 2) == 0.43 and round(inv_f["p_atm"].max(), 2) == 3.02
    assert inv_f["x_o2_not_constant"].sum() == 11 and len(inv_f) == 120
    cf = tables["PSI-159"].data
    assert len(cf) == 179 and cf["x_o2"].min() == pytest.approx(0.1797, abs=1e-4)
    assert cf["x_o2"].max() == pytest.approx(0.3997, abs=1e-4)
    assert (cf["cool_flame"] == True).sum() == 7 and cf["cool_flame"].isna().sum() == 171  # noqa: E712
    assert cf["hot_ignition"].sum() == 132
    assert cf["end_raw"].value_counts().to_dict() == {"SE": 102, "-": 46, "FT": 31}


def test_saffire_material_blank_is_not_filled(tables):
    s2 = tables["PSI-99"].data.set_index("test_id")
    assert pd.isna(s2.loc["2-6", "material"])
    assert s2.loc["2-5", "ug_spread_raw"] == "2.1 mm/s" and s2.loc["2-6", "ug_spread_raw"] == "2.6 mm/s"


# --------------------------------------------------------------------------- catalog provenance
def test_catalog_size_and_columns(inv):
    assert 20 <= len(inv) <= 24 and inv["key"].is_unique
    required = ["psi_id", "name", "fuel_type", "fuels", "objective", "what_varied", *O2P, *FLOW,
                "sensors", "data_available", "key_result", "source", "source_page"]
    assert set(required) <= set(inv.columns)
    assert set(inv["fuel_type"]) <= {"droplet", "gas", "solid"}
    assert set(inv["data_available"]) <= {"table", "telemetry", "video", "none"}


def test_every_filled_range_has_a_source(inv):
    for _, r in inv.iterrows():
        if r[O2P].notna().any():
            assert isinstance(r["range_source"], str) and r["range_source"], r["key"]
            assert r["range_basis"] in ("tested", "planned", "facility capability (MDCA), not tested"), r["key"]
        if r[FLOW].notna().any():
            assert isinstance(r["flow_source"], str) and r["flow_source"], r["key"]
        assert isinstance(r["source"], str) and r["source"], r["key"]


def test_psi_ranges_are_computed_not_typed(inv):
    by_key = inv.set_index("key")
    assert by_key.loc["bass2", "o2_min"] == 0.136 and pd.isna(by_key.loc["bass2", "p_min_atm"])
    assert by_key.loc["cfig", "range_source"].startswith("PSI-159")
    assert by_key.loc["flex", "p_min_atm"] == pytest.approx(0.692, abs=1e-3)
    typed = {d["key"] for d in INVESTIGATIONS if any(k in d for k in O2P)}
    assert typed.isdisjoint({"flex", "bass2", "cfig", "flame_design", "saffire1", "saffire2", "saffire3"})


def test_guide_ranges_cite_pages(inv):
    guide = inv[inv["range_source"].fillna("").str.startswith("GUIDE")]
    assert set(guide["key"]) == {"flex2", "bass", "sofie"}
    assert guide["range_source"].str.contains(r"p\.\d+").all()


def test_literature_is_labelled_and_separate(points):
    lit = build_literature()
    assert set(lit["kind"]) == {"literature", "exploration_target"}
    assert lit["citation"].str.len().gt(20).all()
    assert lit["value_verified"].isin([True, False]).all()
    targets = lit[lit["kind"] == "exploration_target"].set_index("point_id")
    assert targets.loc["target_moon_34_8_2", "p_atm"] == 0.558
    assert targets.loc["target_challenge_30_10_2", "p_atm"] == 0.694
    assert points["source"].str.startswith("PSI-").all()   # literature never mixed into coverage


# --------------------------------------------------------------------------- cards
def test_cards_word_limits_and_provenance(inv):
    cards = investigation_cards(inv) + special_cards()
    assert len(cards) == len(inv) + 4
    for c in cards:
        assert c["generated_by"] and c["citation"]
        for k in ("objective", "what_varied", "key_result", "fire_safety_implication"):
            assert word_count(c[k]) <= MAX_CARD_WORDS, (c["card_id"], k)
        assert all(str(v).lower() not in ("nan", "none", "") for v in c.values()), c["card_id"]
    by_id = {c["card_id"]: c for c in cards}
    assert by_id["inv_saffire2"]["key_result"].startswith("Micro-g spread 2.1 mm/s")
    assert by_id["inv_lsp"]["key_result"] == "Not stated in the available sources."
    misc = next(c for c in cards if c["card_id"] == "three_misconceptions")
    assert all(m in misc["key_result"] for m in MISCONCEPTIONS) and "p. 35" in misc["citation"]


# --------------------------------------------------------------------------- ranking
def test_ranking_deterministic_and_order_independent(inv):
    a = rank_investigations(inv)
    b = rank_investigations(inv.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(a, b)
    assert a["score"].is_monotonic_decreasing and a["score"].between(0, 100).all()


def test_o2p_uses_nearest_measured_test_not_box(inv, points):
    r = rank_investigations(inv, points=points).set_index("key")
    assert r.loc["flame_design", "o2p_method"] == "nearest measured test"
    assert r.loc["sofie", "o2p_method"] == "stated range"
    box = rank_investigations(inv).set_index("key")   # without points: box (bound)
    assert r.loc["flame_design", "o2_pressure_score"] < box.loc["flame_design", "o2_pressure_score"]


def test_recommender_rejects_bad_top_n(points):
    with pytest.raises(ValueError):
        recommend_experiments(points, top_n=0)


def test_breakdown_sums_to_score(inv):
    r = rank_investigations(inv)
    pts = r[[f"{f}_points" for f in FACTORS]].sum(axis=1)
    np.testing.assert_allclose(pts, r["score"], atol=0.05)


def test_slider_effects(inv):
    only_habitat = rank_investigations(inv, {f: 0.0 for f in FACTORS} | {"habitat": 1.0})
    n_solid = int((inv["fuel_type"] == "solid").sum())
    assert (only_habitat.head(n_solid)["fuel_type"] == "solid").all()
    assert (only_habitat.iloc[n_solid:]["fuel_type"] != "solid").all()
    assert only_habitat["score"].max() == 100
    only_o2p = rank_investigations(inv, {f: 0.0 for f in FACTORS} | {"o2_pressure": 1.0})
    assert {"flex", "cfig", "flame_design"} <= set(only_o2p.head(3)["key"])
    up = rank_investigations(inv, {"data": 1.0}).set_index("key")["score"]
    base = rank_investigations(inv).set_index("key")["score"]
    assert up["bass2"] > base["bass2"] and up["sofie"] < base["sofie"]   # table vs no data


def test_missing_gives_zero_and_flag():
    row = {"fuel_type": "gas", "data_available": "none", "suppression_detection": None}
    s = score_row(row)
    assert s["o2_pressure_score"] == 0 and s["flow_score"] == 0 and s["suppression_score"] == 0
    assert set(s["missing"]) == {"o2_pressure", "flow", "suppression"}


def test_bad_weights_rejected():
    with pytest.raises(ValueError):
        score_row({}, {"habitat": -1})
    with pytest.raises(ValueError):
        score_row({}, {"vibes": 1})
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- coverage + recommender
def test_coverage_points_are_real_tests_only(points, tables):
    expected = 274 + sum(len(tables[k].data) for k in ("PSI-25", "PSI-10", "PSI-159", "PSI-98", "PSI-99", "PSI-100"))
    assert len(points) == expected
    assert not points.loc[points["p_assumed"], "p_atm"].notna().any()   # assumed p never stored as measured


def test_recommender_prioritises_moon_base_gap(points):
    picks = recommend_experiments(points)
    assert len(picks) == 5 and picks["rank"].tolist() == [1, 2, 3, 4, 5]
    top = picks.iloc[0]
    d = np.hypot((top["x_o2"] - 0.34) / 0.05, (top["p_atm"] - 0.558) / 0.2)
    assert d <= 0.5 and top["score"] >= 0.9
    assert picks["explanation"].str.len().gt(40).all()
    assert len(set(picks["fuel_type"])) >= 2   # diverse across fuel types


def test_recommender_droplet_gap_is_below_flex_pressures(points):
    pick = recommend_experiments(points, top_n=1, fuel_types=("droplet",)).iloc[0]
    assert pick["x_o2"] >= 0.30 and pick["p_atm"] < points.loc[points["source"] == "PSI-69", "p_atm"].min()


def test_recommender_ignores_assumed_pressure(points):
    pick = recommend_experiments(points, top_n=1, fuel_types=("solid",)).iloc[0]
    assert pd.isna(pick["gap_units"]) and "no PSI test" in pick["nearest_real_test"]
