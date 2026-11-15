"""Fire-Margin — Streamlit entry point.  Run: streamlit run app/app.py

Loads only pre-built files (data/processed, models/, data/knowledge); nothing is retrained at
startup. The app starts with no secrets: the chatbot then shows retrieved NASA passages only.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Streamlit Cloud runs app/app.py; make `src` importable
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

from src import coolflame, partial_gravity, telemetry  # noqa: E402
from src.catalog_data import LITERATURE_POINTS  # noqa: E402
from src.features import MODEL_FEATURES, build_features  # noqa: E402
from src.knowledge import Retriever, load_chunks  # noqa: E402
from src.lessons import LESSONS  # noqa: E402
from src.llm_config import connection_report, load_providers  # noqa: E402
from src.models import NEAREST_FEATURES, predict_interval  # noqa: E402
from src.rag import STARTER_QUESTIONS, answer  # noqa: E402
from src.ranking import DEFAULT_WEIGHTS, FACTORS, rank_investigations  # noqa: E402

PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"
FLEX_DOI = "https://doi.org/10.60555/mbq8-0451"

# ------------------------------------------------------------------ design tokens (app/theme.py)
from theme import (  # noqa: E402
    BG,
    BORDER,
    CSS,
    CYAN,
    FLAME,
    LILAC,
    MUTED,
    PANEL,
    ROSE,
    SAND,
    STEEL,
    TEXT,
)

NIGHT, OXYGEN, PLASMA, DANGER, CALM, PALE = BG, CYAN, LILAC, ROSE, STEEL, SAND
MAX_HISTORY = 8
TARGET_SHORT = {"target_moon_34_8_2": "Moon base 34% / 8.2 psia", "target_challenge_30_10_2": "30% / 10.2 psi"}
FUEL_COLOR = {"solid": FLAME, "droplet": OXYGEN, "gas": PLASMA}
OUTCOME_COLOR = {"Disruption": DANGER, "Extinction": CALM, "Completion": PALE}
OUTCOME_SYMBOL = {"Disruption": "x", "Extinction": "circle", "Completion": "square"}   # never colour alone
FACTOR_LABEL = {"habitat": "Habitat relevance", "o2_pressure": "O2 / pressure near targets",
                "flow": "Flow overlap 0-30 cm/s", "data": "Data available", "suppression": "Suppression / detection"}
GRID = "#22324D"


def plot_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=height,
                      font={"family": "Inter, sans-serif", "color": TEXT, "size": 12},
                      hoverlabel={"font": {"family": "IBM Plex Mono, monospace"}})
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def lede(text: str) -> None:
    st.markdown(f'<p class="fm-lede">{html.escape(text)}</p>', unsafe_allow_html=True)


# ------------------------------------------------------------------ cached data
@st.cache_data(show_spinner=False)
def load_tables() -> dict:
    cards = json.loads((PROC / "insight_cards.json").read_text())
    return {
        "inv": pd.read_csv(PROC / "investigations.csv"),
        "points": pd.read_csv(PROC / "coverage_points.csv"),
        "next": pd.read_csv(PROC / "next_experiments.csv"),
        "lit": pd.DataFrame(LITERATURE_POINTS),
        "cards": {c["card_id"]: c for c in cards},
        "flex": pd.read_csv(PROC / "flex_clean.csv"),
        "summary": json.loads((PROC / "flex_coverage_summary.json").read_text()),
        "metrics": json.loads((MODELS / "metrics.json").read_text()),
        "cool": coolflame.observed(),
        "pg": partial_gravity.equivalent_points(),
        "telemetry": pd.read_csv(telemetry.SMALL_PATH) if telemetry.SMALL_PATH.exists() else None,
        "ignitions": pd.read_csv(telemetry.EVENTS_PATH) if telemetry.EVENTS_PATH.exists() else None,
    }


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    names = ("outcome_logistic_multinomial", "k_ridge", "k_interval_cqr", "nearest_test")
    return {n: joblib.load(MODELS / f"{n}.joblib") for n in names}


@st.cache_resource(show_spinner=False)
def load_retriever() -> Retriever:
    return Retriever(load_chunks())


class _NotCacheable(Exception):
    def __init__(self, payload: dict):
        super().__init__("not cacheable")
        self.payload = payload


@st.cache_data(show_spinner=False, ttl=24 * 3600, max_entries=500)
def _cached_answer(question: str) -> dict:
    result = answer(question, load_retriever()).to_dict()
    if result["status"] not in ("answered", "refused"):   # never pin a temporary outage for 24 h
        raise _NotCacheable(result)
    return result


def get_answer(question: str) -> dict:
    try:
        return _cached_answer(question)
    except _NotCacheable as exc:
        return exc.payload


def a_link(text: str, url: str) -> str:
    """HTML anchor with escaped text and URL (markdown links do not render inside raw HTML)."""
    return (f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">'
            f"{html.escape(text)}</a>")


def fmt_range(lo, hi, pct: bool = False) -> str:
    f = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.2f}")
    if pd.isna(lo) and pd.isna(hi):
        return "not stated"
    a, b = ("?" if pd.isna(lo) else f(lo)), ("?" if pd.isna(hi) else f(hi))
    return a if a == b else f"{a} to {b}"


# ------------------------------------------------------------------ header / footer
def kpis(t: dict) -> list[tuple[str, str, str]]:
    """(value, label, source) computed from the data files, never typed in."""
    hi = t["summary"]["high_o2"]
    pts, lit = t["points"], t["lit"]
    solid_max = pts.loc[pts["fuel_type"] == "solid", "x_o2"].max()
    vi = lit[lit["point_id"].astype(str).str.startswith("saffire_vi_")]
    flex_min_p = t["flex"]["p_atm"].min()
    moon_p = float(lit.loc[lit["point_id"] == "target_moon_34_8_2", "p_atm"].iloc[0])
    return [
        (f"{hi['disruption_share']:.0%}", f"Disruption at >= 25% O2 ({hi['n_disruption']}/{hi['n_tests']} FLEX tests)",
         "PSI-69 FLEX table, DOI 10.60555/mbq8-0451 (data/processed/flex_coverage_summary.json)"),
        (f"{solid_max:.1%}", "Max O2 for solid materials in PSI",
         "PSI-25 BASS-II and PSI-98/99 Saffire I-II tables (data/processed/coverage_points.csv)"),
        (f"{len(vi)}", f"Solid samples near Moon-base conditions (Saffire-VI, {vi['x_o2'].min():.0%}-{vi['x_o2'].max():.0%} O2)",
         "Urban et al., ICES-2024-365; NTRS 20240007488 test matrix (verified literature, not PSI)"),
        (f"{flex_min_p:.2f} atm", f"Lowest FLEX pressure vs {moon_p:.2f} atm Moon base",
         "PSI-69 FLEX table; target from NASA/TM-2013-217377 (34% O2 / 8.2 psia)"),
    ]


def header(t: dict) -> None:
    st.title("Fire-Margin")
    st.markdown(
        '<p class="fm-mission">Evidence-graded fire safety for Moon-base atmospheres: every NASA microgravity '
        "combustion test we could trace, mapped against the 34% oxygen / 8.2 psia habitat NASA plans, with "
        "the gaps and the next experiment to close them.</p>", unsafe_allow_html=True)
    for col, (value, label, source) in zip(st.columns(4), kpis(t)):
        col.metric(label, value, help=f"Source: {source}")
    with st.expander("Data & AI disclosure"):
        st.markdown(
            "- **Data**: NASA Physical Sciences Informatics (PSI) tables, the ISS Researcher's Guide to "
            "Combustion Science (NP-2015-10-034-JSC) and team-verified literature. No row is synthetic or "
            "interpolated; every number cites a PSI ID, DOI or paper.\n"
            "- **Models**: logistic regression (outcome) and ridge regression (burning rate K), validated with "
            "GroupKFold by test date; metrics in `models/metrics.json`.\n"
            "- **AI**: Claude Code (Anthropic) wrote the code and the insight cards from NASA text; the "
            "Evidence Console uses Meta (NVIDIA NIM as fallback) and may only quote the retrieved passages. "
            "Full log: `docs/AI_DISCLOSURE.md`.\n"
            "- No NASA logos or insignia are used; NASA is credited as the data source only.")


def footer(t: dict) -> None:
    st.divider()
    psi = t["inv"].dropna(subset=["psi_id"])[["psi_id", "name"]]
    items = ["PSI-69 FLEX — DOI " + a_link("10.60555/mbq8-0451", FLEX_DOI)]
    items += [html.escape(f"{r.psi_id} {r.name}") for r in psi.itertuples() if r.psi_id != "PSI-69"]
    lit = t["lit"][t["lit"]["kind"] == "literature"].drop_duplicates("citation")
    items += [a_link(str(r.citation), r.url) for r in lit.itertuples() if isinstance(r.url, str)]
    st.markdown('<p class="fm-src">Sources: ' + " | ".join(items) + ". NASA is credited as the data source "
                "only; no NASA logos or insignia are used.</p>", unsafe_allow_html=True)


# ------------------------------------------------------------------ tab 1: catalog
def tab_catalog(t: dict) -> None:
    st.subheader("Which NASA investigations tell us most about a Moon-base fire?")
    lede(f"{len(t['inv'])} investigations ranked by a transparent score. Move the weights to re-rank; open any row for its "
         "insight card, score breakdown and the source of every range. Planned or facility-capability ranges "
         "are not evidence and score zero.")
    cols = st.columns(len(FACTORS))
    weights = {f: c.slider(FACTOR_LABEL[f], 0.0, 1.0, float(DEFAULT_WEIGHTS[f]), 0.05, key=f"w_{f}")
               for f, c in zip(FACTORS, cols)}
    if sum(weights.values()) == 0:
        st.warning("All weights are zero, so every investigation scores 0. Raise at least one weight.")
    ranked = rank_investigations(t["inv"], weights, points=t["points"])
    inv = t["inv"].set_index("key")
    table = ranked.assign(
        O2=[fmt_range(inv.at[k, "o2_min"], inv.at[k, "o2_max"], pct=True) for k in ranked["key"]],
        pressure_atm=[fmt_range(inv.at[k, "p_min_atm"], inv.at[k, "p_max_atm"]) for k in ranked["key"]],
        basis=[inv.at[k, "range_basis"] if isinstance(inv.at[k, "range_basis"], str) else "" for k in ranked["key"]],
    )[["rank", "name", "fuel_type", "score", "O2", "pressure_atm", "basis", "data_missing"]]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100,
                                                                         format="%.1f")})
    for r in ranked.itertuples():
        row, card = inv.loc[r.key], t["cards"].get(f"inv_{r.key}", {})
        with st.expander(f"{r.rank}. {r.name}  —  {r.score:.1f}"):
            left, right = st.columns([3, 2])
            with left:
                for label, k in (("Objective", "objective"), ("What varied", "what_varied"),
                                 ("Key result", "key_result"), ("Fire-safety implication", "fire_safety_implication")):
                    st.markdown(f"**{label}.** {card.get(k, 'Not stated in the available sources.')}")
                st.markdown(f'<p class="fm-src">{html.escape(card.get("citation", ""))} · written by '
                            f'{html.escape(card.get("generated_by", ""))}</p>', unsafe_allow_html=True)
            with right:
                st.markdown("**Score breakdown**")
                for f in FACTORS:
                    st.progress(min(1.0, getattr(r, f"{f}_score")),
                                text=f"{FACTOR_LABEL[f]}: {getattr(r, f'{f}_points'):.1f} pts")
                basis = row["range_basis"] if isinstance(row["range_basis"], str) else "no stated range"
                src = row["range_source"] if isinstance(row["range_source"], str) else "no source states a range"
                st.markdown(f"**O2 / pressure range basis**: {basis}  \n{src}  \n"
                            f"**Method**: {r.o2p_method}")
                if isinstance(row["flow_source"], str):
                    st.markdown(f"**Flow source**: {row['flow_source']}")
                if r.data_missing:
                    st.caption(f"Data missing (scored 0): {r.data_missing}")


# ------------------------------------------------------------------ tab 2: predictor
def predict(fuel: str, x_o2: float, p_atm: float, x_co2: float, x_he: float, d0: float) -> dict:
    m = load_models()
    row = pd.DataFrame([{"fuel": fuel, "x_o2": x_o2, "p_atm": p_atm, "x_co2": x_co2, "x_he": x_he,
                         "x_n2": max(0.0, 1 - x_o2 - x_co2 - x_he), "d0_mm": d0, "K_mm2_s": np.nan}])
    feats = build_features(row)
    clf = m["outcome_logistic_multinomial"]
    probs = dict(zip(clf.classes_, clf.predict_proba(feats[MODEL_FEATURES])[0]))
    k = float(m["k_ridge"].predict(feats[MODEL_FEATURES])[0])
    lo, hi = predict_interval(m["k_interval_cqr"], feats[MODEL_FEATURES])
    near = m["nearest_test"].query(feats[NEAREST_FEATURES]).iloc[0]
    return {"probs": probs, "k": k, "k_lo": float(lo[0]), "k_hi": float(hi[0]),
            "distance": float(near["nearest_distance"]), "nearest_id": near["nearest_flex_id"]}


def completion_note(t: dict) -> str:
    n = t["metrics"]["data"]["outcome_counts"]["Completion"]
    recalls = [t["metrics"]["outcome_3class"][m]["pooled_oof"]["per_class_recall"]["Completion"]
               for m in ("hist_gradient_boosting", "logistic_multinomial")]
    return f"{n} tests, out-of-fold recall {min(recalls):.2f}-{max(recalls):.2f}"


def risk_band(p: float) -> tuple[str, str]:
    """Display band for disruption probability (presentation thresholds, not validated limits)."""
    if p >= 0.5:
        return "High", "high"
    if p >= 0.25:
        return "Elevated", "elevated"
    return "Low", "low"


def tab_predictor(t: dict) -> None:
    st.subheader("What would a burning droplet do in this atmosphere?")
    lede("Set the atmosphere in the input panel. The models were trained on 274 FLEX droplet tests (PSI-69) and "
         "validated on unseen test days; outside the tested envelope the numbers are extrapolations.")
    inputs, results = st.columns([1, 2], gap="large")
    with inputs, st.container(border=True):
        fuel = st.selectbox("Fuel", ["Heptane", "Methanol"])
        x_o2 = st.slider("Oxygen (mole fraction)", 0.10, 0.40, 0.21, 0.01)
        p_atm = st.slider("Pressure (atm)", 0.30, 3.10, 1.00, 0.01)
        d0 = st.slider("Initial droplet diameter d0 (mm)", 1.0, 5.0, 3.0, 0.1)
        x_co2 = st.slider("CO2 diluent (mole fraction)", 0.0, 0.70, 0.0, 0.01)
        x_he = st.slider("Helium diluent (mole fraction)", 0.0, 0.45, 0.0, 0.01)
    with results:
        if x_o2 + x_co2 + x_he > 1.0:
            st.error("O2 + CO2 + He add up to more than 1. Lower one of them; the rest is nitrogen.")
            return
        res = predict(fuel, x_o2, p_atm, x_co2, x_he, d0)
        flex = t["flex"][t["flex"]["fuel"] == fuel]
        outside = [name for name, v, s in (("O2", x_o2, flex["x_o2"]), ("pressure", p_atm, flex["p_atm"]),
                                           ("d0", d0, flex["d0_mm"])) if not (s.min() <= v <= s.max())]
        if outside or res["distance"] > 1.5:
            why = f"{', '.join(outside)} outside the tested {fuel} range" if outside else "far from every test"
            banner(f"Outside the tested envelope ({why}). Nearest real test {res['nearest_id']} is "
                   f"{res['distance']:.2f} standard units away: treat these numbers as extrapolation.")
        else:
            banner(f"Inside the tested envelope. Nearest real FLEX test: {res['nearest_id']}, "
                   f"{res['distance']:.2f} standard units away.", calm=True)
        p_dis = float(res["probs"].get("Disruption", 0.0))
        label, cls = risk_band(p_dis)
        dis = t["metrics"]["disruption_binary"]["logistic"]["mean"]
        st.markdown(f'<span class="fm-risk {cls}">Disruption risk: {label} ({p_dis:.0%})</span>',
                    unsafe_allow_html=True)
        st.caption(f"Bands: Low < 25%, Elevated 25-50%, High >= 50% (display bands, not validated limits). "
                   f"Disruption model on unseen days: ROC-AUC {dis['roc_auc']['mean']:.2f}, "
                   f"balanced accuracy {dis['balanced_accuracy']['mean']:.2f}.")
        for c in ("Extinction", "Disruption", "Completion"):
            p = res["probs"].get(c, 0.0)
            note = f" (low confidence: {completion_note(t)})" if c == "Completion" else ""
            st.progress(float(p), text=f"{c}: {p:.0%}{note}")
        if fuel == "Heptane":
            banner(f"Cool-flame note: {coolflame.GUIDE_COOL_FLAME} ({coolflame.GUIDE_COOL_FLAME_CITE}). "
                   "The outcome model does not predict cool flames.", calm=True)
        comp_r = t["metrics"]["d2law_consistency"]["completion_only"]["pearson_r"]
        if res["k"] > 0:
            m1, m2 = st.columns(2)
            m1.metric("Burning-rate constant K (mm²/s)", f"{res['k']:.3f}",
                      help=f"10-90% conformal interval {res['k_lo']:.3f} to {res['k_hi']:.3f} mm²/s (ridge model)")
            m2.metric("Burn-time upper limit d0²/K (s)", f"{d0 ** 2 / res['k']:.1f}",
                      help=f"Full burn-out time; matches measured burn time only for Completion (r = {comp_r:.2f}).")
            st.caption(f"K 10-90% interval: {res['k_lo']:.3f} to {res['k_hi']:.3f} mm²/s. d0²/K is an upper limit: "
                       f"extinction or disruption ends a burn earlier (r = {comp_r:.2f} for Completion only).")
            st.plotly_chart(d2_curve(d0, res), width="stretch")
        else:
            banner("The K model gives no positive burning rate here: these conditions are far outside the FLEX "
                   "data, so no burn time is shown.")


def banner(text: str, calm: bool = False) -> None:
    st.markdown(f'<div class="fm-banner{" calm" if calm else ""}">{html.escape(text)}</div>', unsafe_allow_html=True)


def d2_curve(d0: float, res: dict) -> go.Figure:
    fig = go.Figure()
    for k, name, dash in ((res["k_hi"], "K high (90%)", "dot"), (res["k"], "K predicted", "solid"),
                          (res["k_lo"], "K low (10%)", "dot")):
        if k <= 0:
            continue
        t_end = d0 ** 2 / k
        tt = np.linspace(0, t_end, 60)
        fig.add_scatter(x=tt, y=d0 ** 2 - k * tt, name=name, mode="lines",
                        line={"color": FLAME if dash == "solid" else MUTED, "dash": dash, "width": 2})
    plot_layout(fig, 320)
    fig.update_layout(title="d² law: d(t)² = d0² − K·t (upper limit on burn time)",
                      margin={"l": 50, "r": 16, "t": 48, "b": 40}, xaxis_title="time (s)", yaxis_title="d² (mm²)",
                      legend={"orientation": "h", "y": -0.3})
    return fig


# ------------------------------------------------------------------ tab 3: map
def oxygen_margin_map(t: dict, show: set[str], show_pg: bool = False) -> go.Figure:
    pts, lit = t["points"], t["lit"]
    nxt = t["next"][t["next"]["kind"] == "recommended (evidence gap)"]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.2, x1=3.2, y0=22.2, y1=42, fillcolor=FLAME, opacity=0.06,
                  line_width=0, layer="below")
    fig.add_annotation(x=np.log10(2.2), y=23.2, text="Above 22.2% O2: no PSI solid-material test",
                       showarrow=False, font={"color": FLAME, "size": 12})
    flex = pts[(pts["source"] == "PSI-69") & pts["p_atm"].notna()]
    if "droplet" in show:
        for outcome, g in flex.groupby("outcome"):
            fig.add_scatter(x=g["p_atm"], y=g["x_o2"] * 100, mode="markers", name=f"FLEX droplet: {outcome}",
                            marker={"color": OUTCOME_COLOR.get(str(outcome), MUTED), "size": 7, "opacity": 0.85,
                                    "symbol": OUTCOME_SYMBOL.get(str(outcome), "circle")},
                            hovertext=g["test_id"].astype(str) + " (PSI-69)", hoverinfo="text+x+y")
    for ft in ("gas", "solid"):
        if ft not in show:
            continue
        g = pts[(pts["fuel_type"] == ft) & pts["x_o2"].notna() & pts["p_atm"].notna()]
        if len(g):
            fig.add_scatter(x=g["p_atm"], y=g["x_o2"] * 100, mode="markers", name=f"PSI {ft} tests",
                            marker={"color": FUEL_COLOR[ft], "size": 7, "opacity": 0.75, "symbol": "circle"},
                            hovertext=g["source"] + " " + g["test_id"].astype(str), hoverinfo="text+x+y")
        a = pts[(pts["fuel_type"] == ft) & pts["x_o2"].notna() & pts["p_atm"].isna() & pts["p_assumed"]]
        if len(a):
            fig.add_scatter(x=a["p_assumed_atm"], y=a["x_o2"] * 100, mode="markers",
                            name=f"PSI {ft} tests, pressure assumed 1 atm",
                            marker={"color": FUEL_COLOR[ft], "size": 8, "symbol": "circle-open", "line": {"width": 1.5}},
                            hovertext=a["source"] + " " + a["test_id"].astype(str) + " (pressure not measured)",
                            hoverinfo="text+y")
    if "literature" in show:
        L = lit[lit["kind"] == "literature"]
        for measured, g in ((True, L[L["p_atm"].notna()]), (False, L[L["p_atm"].isna() & L["p_assumed_atm"].notna()])):
            if not len(g):
                continue
            x = g["p_atm"] if measured else g["p_assumed_atm"]
            suffix = "" if measured else ", pressure assumed 1 atm"
            fig.add_scatter(x=x, y=g["x_o2"] * 100, mode="markers", name=f"Verified literature (not PSI){suffix}",
                            marker={"color": FLAME, "size": 12, "symbol": "diamond" if measured else "diamond-open",
                                    "line": {"color": TEXT if measured else FLAME, "width": 1.5}},
                            hovertext=g["label"] + " — " + g["citation"] + ("" if measured else " (pressure not stated)"),
                            hoverinfo="text+y")
    cool = t["cool"]
    if "gas" in show and len(cool):
        for status, sym in (("yes", "triangle-up"), ("no", "triangle-up-open")):
            g = cool[cool["cool_flame"] == status]
            fig.add_scatter(x=g["p_atm"], y=g["x_o2"] * 100, mode="markers", name=f"CFI-G cool flame: {status}",
                            marker={"color": OXYGEN, "size": 13, "symbol": sym, "line": {"color": TEXT, "width": 1}},
                            hovertext=("PSI-159 " + g["test_id"].astype(str) + ": " + g["fuel"].astype(str)
                                       + f", cool flame {status}"), hoverinfo="text+x+y")
    if show_pg:
        pg = t["pg"]
        fig.add_scatter(x=pg["p_1g_equiv_atm"], y=pg["x_o2"] * 100, mode="markers+text",
                        name="1-g partial-gravity equivalents (heuristic)",
                        marker={"color": TEXT, "size": 14, "symbol": "circle-open", "line": {"width": 2}},
                        text=pg["body"], textposition="bottom center", textfont={"color": MUTED},
                        hovertext=pg["label"] + " — 1-g equivalent " + pg["p_1g_equiv_atm"].astype(str)
                        + " atm (heuristic)", hoverinfo="text")
    tg = lit[lit["kind"] == "exploration_target"]
    for r in tg.itertuples():   # target zones: +/- 2 O2 points, +/- 10% pressure (display band only)
        fig.add_shape(type="rect", x0=r.p_atm * 0.9, x1=r.p_atm * 1.1, y0=r.x_o2 * 100 - 2,
                      y1=r.x_o2 * 100 + 2, xref="x", yref="y", line={"color": TEXT, "width": 1, "dash": "dot"},
                      fillcolor=TEXT, opacity=0.08, layer="below")
    fig.add_scatter(x=tg["p_atm"], y=tg["x_o2"] * 100, mode="markers+text", name="Exploration targets",
                    marker={"color": TEXT, "size": 16, "symbol": "x-thin", "line": {"width": 3, "color": TEXT}},
                    text=tg["point_id"].map(TARGET_SHORT).fillna(tg["label"]).tolist(),
                    textposition=["bottom left" if "challenge" in str(i) else "middle right" for i in tg["point_id"]],
                    textfont={"color": TEXT}, hovertext=tg["label"], hoverinfo="text")
    fig.add_scatter(x=nxt["p_atm"], y=nxt["x_o2"] * 100, mode="markers+text", name="Recommended next tests",
                    marker={"color": "#FFFFFF", "size": 15, "symbol": "star", "line": {"color": FLAME, "width": 1.5}},
                    text=[f"#{r}" for r in nxt["rank"]], textposition="top center", textfont={"color": TEXT},
                    hovertext=nxt["explanation"], hoverinfo="text")
    share = t["summary"]["high_o2"]
    fig.add_annotation(x=np.log10(0.72), y=34.6, ax=np.log10(1.1), ay=33.2, xref="x", yref="y", axref="x", ayref="y",
                       xanchor="left",
                       text=f"At O2 ≥ 25%: {share['n_disruption']} of {share['n_tests']} FLEX droplets "
                            f"burst ({share['disruption_share']:.0%})", showarrow=True, arrowcolor=DANGER,
                       font={"color": DANGER, "size": 13}, arrowwidth=1.5)
    plot_layout(fig, 680)
    fig.update_layout(margin={"l": 60, "r": 16, "t": 24, "b": 48}, clickmode="event+select",
                      xaxis={"title": "Ambient pressure (atm, log scale)", "type": "log",
                             "range": [np.log10(0.2 if show_pg else 0.3), np.log10(3.2)], "gridcolor": "#22324D",
                             "tickvals": [0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3]},
                      yaxis={"title": "Oxygen (% by volume)", "range": [10, 42], "gridcolor": "#22324D"},
                      legend={"orientation": "h", "y": -0.14, "font": {"size": 11}, "bgcolor": PANEL,
                              "bordercolor": BORDER, "borderwidth": 1})
    return fig


def tab_map(t: dict) -> None:
    st.subheader("Where does NASA have fire evidence near a Moon-base atmosphere?")
    lede("Every real tested condition in oxygen x pressure, the two exploration-atmosphere targets (dotted "
         "boxes) and the recommended next experiments (stars). Click or box-select points to see their details "
         "below; marker shape, not only colour, tells categories apart.")
    c1, c2 = st.columns([2, 1])
    opts = {"droplet": "Droplets (FLEX)", "gas": "Gas flames", "solid": "Solid materials", "literature": "Literature"}
    chosen = c1.multiselect("Show", list(opts), default=list(opts), format_func=opts.get)
    show_pg = c2.toggle("Show partial-gravity equivalents")
    fig = oxygen_margin_map(t, set(chosen), show_pg)
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points", "box"),
                            key=f"omap_{'-'.join(sorted(chosen))}_{int(show_pg)}")
    if show_pg:
        st.caption(partial_gravity.CAPTION)
    point_details(fig, event)
    st.markdown("**Recommended next experiments** (gap to the nearest evidence x closeness to a target; "
                "PSI tests count fully, verified literature counts half)")
    st.dataframe(t["next"][["rank", "kind", "fuel_type", "x_o2", "p_atm", "score", "explanation"]], hide_index=True,
                 width="stretch")
    saffire3_replay(t)
    st.caption("Open markers: BASS-II and Saffire I-II have no pressure column; they are drawn at an assumed "
               "ISS-cabin 1 atm. BASS (PSI-26) and Saffire-III have no O2 value and are not plotted.")


def point_details(fig: go.Figure, event) -> None:
    pts = []
    try:
        pts = list(event.selection.points) if event and event.selection else []
    except AttributeError:
        pts = []
    with st.container(border=True):
        if not pts:
            st.markdown('<p class="fm-status">Point details: click a marker or drag a box on the map.</p>',
                        unsafe_allow_html=True)
            return
        rows = []
        for p in pts[:50]:
            cn = p.get("curve_number")
            if cn is None or not 0 <= cn < len(fig.data):   # stale selection after a layer change
                continue
            trace = fig.data[cn]
            hover = trace.hovertext if trace is not None else None
            idx = p.get("point_index", p.get("point_number"))
            label = hover[idx] if hover is not None and not isinstance(hover, str) and idx is not None else hover
            rows.append({"layer": trace.name, "detail": label,
                         "pressure_atm": round(float(p["x"]), 3), "O2_percent": round(float(p["y"]), 2)})
        if not rows:
            st.markdown('<p class="fm-status">Selection cleared after a layer change: select again.</p>',
                        unsafe_allow_html=True)
            return
        st.markdown(f'<p class="fm-status">{len(rows)} selected point(s)</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def saffire3_replay(t: dict) -> None:
    tel, ign = t.get("telemetry"), t.get("ignitions")
    if tel is None or tel.empty:
        return
    with st.expander("Saffire-III: a real fire in a spacecraft (PSI-100)"):
        channels = sorted(tel["channel"].unique())
        default = [c for c in ("Radiometer1", "Cabin air temperature", "FIZ 2 smoke obscuration") if c in channels]
        chosen = st.multiselect("Channels", channels, default=default, key="s3_channels")
        t_max = float(tel["t_s"].max())
        t_now = st.slider("Replay time (s since telemetry start)", 0.0, t_max, t_max, 10.0, key="s3_time")
        if not chosen:
            st.info("Pick at least one channel to replay.")
            return
        sub = tel[tel["channel"].isin(chosen) & (tel["t_s"] <= t_now)]
        fig = go.Figure()
        for (name, unit), g in sub.groupby(["channel", "unit"]):
            fig.add_scatter(x=g["t_s"], y=g["value"], mode="lines", name=f"{name} [{unit}]")
        if ign is not None:
            for i, r in enumerate(ign.sort_values("t_start_s").itertuples()):
                if r.t_start_s <= t_now:
                    fig.add_vline(x=r.t_start_s, line={"color": FLAME, "dash": "dash"},
                                  annotation_text=f"{r.igniter} channel pulse", annotation_font_color=FLAME,
                                  annotation_position="top left" if i % 2 == 0 else "top right")
        plot_layout(fig, 420)
        fig.update_layout(margin={"l": 50, "r": 16, "t": 32, "b": 40}, xaxis_title="time since telemetry start (s)",
                          xaxis_range=[0, t_max], legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, width="stretch")
        if len({u for u in sub["unit"]}) > 1:
            st.caption("Channels with different units share one axis; compare shapes and timing, not magnitudes.")
        st.caption(telemetry.CAPTION + " Units shown are exactly as labelled in the file (psi, degC, V). "
                   "Multiplexed placeholder values (0 or the start-of-file value) were treated as missing; "
                   "dashed lines mark igniter-channel pulses (Sample*IgniterI above 0.5; unit not labelled in the file). Source: PSI-100 Saffire-III telemetry, DOI ")
        st.markdown(a_link(f"{telemetry.DOI}", f"https://doi.org/{telemetry.DOI}"), unsafe_allow_html=True)


# ------------------------------------------------------------------ tab 4: chatbot
STATUS_LINE = {
    "answered": "Answered from {n} NASA passages",
    "refused": "No NASA evidence found",
    "unavailable": "AI unavailable: showing {n} NASA passages",
    "unverified": "AI answer not verifiable against the evidence: showing {n} NASA passages",
}


def _queue_starter() -> None:
    q = st.session_state.get("starter_pill")
    if q:
        st.session_state["pending_q"] = q
    st.session_state["starter_pill"] = None


def citation_chips(sources: list[dict]) -> None:
    for s in sources:
        label = s["cite"] if len(s["cite"]) <= 70 else s["cite"][:67] + "...]"
        with st.popover(label, width="stretch"):
            st.markdown(f"**{s['cite']}**")
            st.caption(f"Retrieval similarity {s['score']}")
            st.write(s["text"])
            if s["url"]:
                st.markdown(a_link("Open source", s["url"]), unsafe_allow_html=True)


def render_answer(a: dict) -> None:
    n = len(a["sources"])
    st.markdown(f'<p class="fm-status">{STATUS_LINE.get(a["status"], "{n} passages").format(n=n)}</p>',
                unsafe_allow_html=True)
    if a["status"] == "answered":
        st.markdown(a["text"])
    elif a["status"] == "refused":
        st.markdown("I don't have NASA evidence for that. Try asking about oxygen limits, droplets, solid "
                    "materials, suppression or past spacecraft fires.")
    else:
        note = ("The AI service is not reachable right now, so here are the NASA passages that match your "
                "question." if a["status"] == "unavailable" else
                "The AI's answer cited sources outside the retrieved NASA evidence, so only the passages are shown.")
        banner(note, calm=True)
        for s in a["sources"]:
            with st.container(border=True):
                st.markdown(f"**{s['cite']}**")
                st.write(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))
    if a["sources"]:
        citation_chips(a["sources"])


def tab_ask() -> None:
    st.subheader("Evidence Console")
    lede("Ask a fire-safety question. Answers come only from the ISS Researcher's Guide, the insight cards and "
         "verified literature, and every claim carries a citation chip you can open to read the exact passage. "
         "Questions outside that evidence are declined.")
    if not load_providers():
        banner("No AI key is configured, so the console shows the matching NASA passages with citations instead "
               "of a written summary. Add MODEL_API_KEY (Meta) or NVIDIA_API_KEY in Streamlit secrets to enable it.",
               calm=True)
    history = st.session_state.setdefault("chat", [])
    st.pills("Suggested questions", STARTER_QUESTIONS, key="starter_pill", on_change=_queue_starter)
    if st.button("Clear conversation", disabled=not history):
        history.clear()
        st.rerun()
    question = st.chat_input("Ask about fire in space habitats", max_chars=300) or st.session_state.pop("pending_q", None)
    if question:
        with st.spinner("Searching NASA evidence..."):
            history.append((question, get_answer(question)))
        del history[:-MAX_HISTORY]
    for q, a in reversed(history):
        with st.chat_message("user", avatar=":material/person:"):
            st.write(q)
        with st.chat_message("assistant", avatar=":material/menu_book:"):
            render_answer(a)


# ------------------------------------------------------------------ tab 5: lessons
def tab_lessons() -> None:
    st.subheader("What past fires taught, and what it gave back to Earth")
    lede("Three fires that shaped spacecraft safety, and three technologies that came back to Earth. Each "
         "entry links its NASA source.")
    for kind, title in (("incident", "Fires that changed spaceflight"), ("spinoff", "On Earth today")):
        st.markdown(f"#### {title}")
        for x in [x for x in LESSONS if x["kind"] == kind]:
            year = f" ({x['year']})" if x["year"] else ""
            st.markdown(f"**{x['title']}{year}.** {x['what_happened']}  \n*{x['lesson']}*")
            link = " — " + a_link("NASA source", x["url"]) if x["url"] else ""
            st.markdown(f'<p class="fm-src">{html.escape(x["source"])}{link}</p>', unsafe_allow_html=True)


def sidebar_ai_check() -> None:
    with st.sidebar.expander("AI connection check"):
        st.caption("Runs one tiny live request per provider. Keys are never shown, only whether one was found.")
        if st.button("Test connection", key="ai_check"):
            with st.spinner("Contacting providers..."):
                rows = connection_report()
            for r in rows:
                status = "OK" if r.get("ok") else "FAILED"
                st.markdown(f"**{r['provider']}** — {status}")
                st.text(f"base_url: {r['base_url']}\nmodel: {r['model']}\napi_style: {r['api_style']}\n"
                        f"key found: {r['key_found']}")
                if r.get("error_class"):
                    st.text(f"error: {r['error_class']} (HTTP {r.get('status_code')})")
                st.code(r["result"], language=None)
                if r.get("hint"):
                    st.warning(r["hint"])
                if r.get("available_models"):
                    st.text("Available models:\n" + "\n".join(r["available_models"]))
                if r.get("available_models_error"):
                    st.text("Model listing failed: " + r["available_models_error"])


def main() -> None:
    st.set_page_config(page_title="Fire-Margin", page_icon=":material/local_fire_department:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    t = load_tables()
    sidebar_ai_check()
    header(t)
    tabs = st.tabs(["Oxygen-Margin Map", "Ask Fire-Margin", "Flame Predictor", "Catalog & Ranking",
                    "Lessons & Earth Impact"])
    with tabs[0]:
        tab_map(t)
    with tabs[1]:
        tab_ask()
    with tabs[2]:
        tab_predictor(t)
    with tabs[3]:
        tab_catalog(t)
    with tabs[4]:
        tab_lessons()
    footer(t)


main()
