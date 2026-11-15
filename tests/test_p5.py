"""P5 stretch: cool-flame flag, partial-gravity heuristic, next-experiment analogue, app start-up."""
from pathlib import Path

import pandas as pd
import pytest

from src import coolflame, partial_gravity
from src.coverage import NEXT_PATH, coverage_points, recommend_experiments


def test_cool_flame_counts_keep_unknown_separate():
    c = coolflame.counts()
    assert c == {"yes": 7, "no": 1, "unknown": 171}
    obs = coolflame.observed()
    assert len(obs) == 8 and set(obs["cool_flame"]) == {"yes", "no"}
    assert (obs["source"] == "PSI-159").all() and obs[["fuel", "x_o2", "p_atm", "test_id"]].notna().all().all()


def test_cool_flame_card_cites_guide_and_psi():
    card = coolflame.card()
    assert "PSI-159" in card["citation"] and "p. 33-34, 36" in card["citation"]
    assert "unknown, not 'no'" in card["key_result"]
    assert all(len(str(card[k]).split()) <= 80 for k in ("key_result", "fire_safety_implication"))


def test_grashof_scaling_factor():
    assert partial_gravity.pressure_factor(0.16) == pytest.approx(0.4)
    assert partial_gravity.pressure_factor(0.38) == pytest.approx(0.616, abs=1e-3)
    assert partial_gravity.equivalent_pressure(0.558, 0.16) == pytest.approx(0.223, abs=1e-3)
    assert partial_gravity.equivalent_pressure(0.694, 0.16) == pytest.approx(0.278, abs=1e-3)
    with pytest.raises(ValueError):
        partial_gravity.pressure_factor(0)


def test_equivalent_points_are_labelled_heuristic():
    eq = partial_gravity.equivalent_points()
    assert len(eq) == 4 and eq["heuristic"].all() and set(eq["body"]) == {"Moon", "Mars"}
    assert "Heuristic" in partial_gravity.CAPTION and "no partial-gravity" in partial_gravity.CAPTION


def test_next_experiments_top5_unchanged_plus_separate_analogue():
    saved = pd.read_csv(NEXT_PATH)
    fresh = recommend_experiments(coverage_points())
    top5 = saved[saved["kind"] == "recommended (evidence gap)"]
    assert top5[["rank", "fuel_type", "x_o2", "p_atm"]].reset_index(drop=True).equals(
        fresh[["rank", "fuel_type", "x_o2", "p_atm"]])
    six = saved[saved["kind"] == "ground-test analogue (heuristic)"]
    assert len(six) == 1 and six.iloc[0]["rank"] == 6
    assert six.iloc[0]["x_o2"] == 0.34 and six.iloc[0]["p_atm"] == pytest.approx(0.223, abs=1e-3)
    assert "heuristic" in six.iloc[0]["explanation"]


def test_app_starts_without_secrets_and_shows_heptane_note(monkeypatch):
    from streamlit.testing.v1 import AppTest

    for v in ("MODEL_API_KEY", "META_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Cool-flame note" in m.value for m in at.markdown)   # default fuel is Heptane
    at.toggle[0].set_value(True).run()
    assert not at.exception and any("Heuristic" in c.value for c in at.caption)


def test_saffire3_small_file_exists_is_small_and_invents_no_units():
    from src import telemetry

    assert telemetry.SMALL_PATH.exists() and telemetry.SMALL_PATH.stat().st_size < 5_000_000
    small = pd.read_csv(telemetry.SMALL_PATH)
    assert small.groupby("channel").size().max() <= telemetry.MAX_POINTS
    published = {"psi", "degC", "V"}
    cyg = small[small["sheet"] == "CygnusData"]
    assert set(cyg["unit"]) <= published
    for raw, unit in zip(cyg["raw_column"], cyg["unit"]):
        assert f"({unit.lower()})" in raw.lower()                 # unit copied from the column header
    counts = small[small["sheet"] == "SaffireData"]
    assert set(counts["unit"]) == {telemetry.UNITLESS}
    assert counts["channel"].str.match(r"^(Radiometer[1-4]|Anemometer[12])$").all()
    assert small["value"].notna().all() and not (cyg["value"] == 0).any()   # placeholders removed


def test_saffire3_ignitions_are_clear_pulses():
    from src import telemetry

    ev = pd.read_csv(telemetry.EVENTS_PATH)
    assert set(ev["igniter"]) == {"Igniter 1", "Igniter 2"}
    assert (ev["peak_value_unit_unlabelled"] > telemetry.IGNITION_THRESHOLD).all() and (ev["n_samples"] < 30).all()
    assert not any(c.endswith("_A") for c in ev.columns)   # no unit invented for the igniter channel
