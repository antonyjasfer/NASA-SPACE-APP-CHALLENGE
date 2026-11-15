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

from src import coolflame, partial_gravity, telemetry  # noqa: E402
from src.catalog_data import LITERATURE_POINTS  # noqa: E402
from src.features import MODEL_FEATURES, build_features  # noqa: E402
from src.knowledge import Retriever, load_chunks  # noqa: E402
from src.lessons import LESSONS  # noqa: E402
from src.llm_config import load_providers  # noqa: E402
from src.models import NEAREST_FEATURES, predict_interval  # noqa: E402
from src.rag import STARTER_QUESTIONS, answer  # noqa: E402
from src.ranking import DEFAULT_WEIGHTS, FACTORS, rank_investigations  # noqa: E402

PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"
FLEX_DOI = "https://doi.org/10.60555/mbq8-0451"

# ------------------------------------------------------------------ design tokens
NIGHT, PANEL, TEXT, MUTED = "#0F1A2B", "#17243A", "#E6E1D6", "#9AA6B8"
FLAME, OXYGEN, PLASMA, DANGER, CALM, PALE = "#FFB03B", "#5EC8E5", "#B79CFF", "#FF5A5F", "#7FB3D5", "#E8D27A"
MAX_HISTORY = 8
TARGET_SHORT = {"target_moon_34_8_2": "Moon base 34% / 8.2 psia", "target_challenge_30_10_2": "30% / 10.2 psi"}
FUEL_COLOR = {"solid": FLAME, "droplet": OXYGEN, "gas": PLASMA}
OUTCOME_COLOR = {"Disruption": DANGER, "Extinction": CALM, "Completion": PALE}
FACTOR_LABEL = {"habitat": "Habitat relevance", "o2_pressure": "O2 / pressure near targets",
                "flow": "Flow overlap 0-30 cm/s", "data": "Data available", "suppression": "Suppression / detection"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label {{ font-family: 'IBM Plex Sans', sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: .01em; }}
h1 {{ font-size: 3.1rem !important; font-weight: 700 !important; line-height: 1.0 !important; margin-bottom: .2rem; }}
.fm-finding {{ font-size: 1.12rem; line-height: 1.55; max-width: 62rem; border-left: 3px solid {FLAME};
              padding: .2rem 0 .2rem 1rem; margin: .6rem 0 1rem 0; }}
.fm-muted {{ color: {MUTED}; font-size: .92rem; max-width: 62rem; }}
.fm-src {{ color: {MUTED}; font-size: .82rem; }}
[data-testid="stExpander"] {{ border-color: #2A3B57 !important; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


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
def header(t: dict) -> None:
    st.title("Fire-Margin")
    st.markdown(
        '<p class="fm-muted">Evidence-graded fire safety for Moon-base atmospheres. NASA plans lunar and '
        "Mars habitats at about 34% oxygen and 8.2 psia, far richer in oxygen than the ISS. Fire-Margin "
        "maps every NASA microgravity combustion test we could trace, shows where the evidence stops, and "
        "recommends the experiment that would close the gap.</p>", unsafe_allow_html=True)
    finding = html.escape(t["cards"]["pressure_gap"]["key_result"])
    st.markdown(f'<div class="fm-finding">{finding}</div>', unsafe_allow_html=True)
    with st.expander("Data & AI disclosure"):
        st.markdown(
            "- **Data**: NASA Physical Sciences Informatics (PSI) tables, the ISS Researcher's Guide to "
            "Combustion Science (NP-2015-10-034-JSC) and team-verified literature. No row is synthetic or "
            "interpolated; every number cites a PSI ID, DOI or paper.\n"
            "- **Models**: logistic regression (outcome) and ridge regression (burning rate K), validated with "
            "GroupKFold by test date; metrics in `models/metrics.json`.\n"
            "- **AI**: Claude Code (Anthropic) wrote the code and the insight cards from NASA text; the "
            "chatbot uses Meta Llama (NVIDIA NIM as fallback) and may only quote the retrieved passages. "
            "Full log: `docs/AI_DISCLOSURE.md`.")


def footer(t: dict) -> None:
    st.divider()
    psi = t["inv"].dropna(subset=["psi_id"])[["psi_id", "name"]]
    items = ["PSI-69 FLEX — DOI " + a_link("10.60555/mbq8-0451", FLEX_DOI)]
    items += [html.escape(f"{r.psi_id} {r.name}") for r in psi.itertuples() if r.psi_id != "PSI-69"]
    lit = t["lit"][t["lit"]["kind"] == "literature"].drop_duplicates("citation")
    items += [a_link(str(r.citation), r.url) for r in lit.itertuples() if isinstance(r.url, str)]
    st.markdown('<p class="fm-src">Sources: ' + " · ".join(items) + "</p>", unsafe_allow_html=True)


# ------------------------------------------------------------------ tab 1: catalog
def tab_catalog(t: dict) -> None:
    st.subheader("Which NASA investigations tell us most about a Moon-base fire?")
    st.caption("Move the weights to see how the ranking changes. Ranges that are only planned or a facility "
               "capability are not evidence and score zero on the O2 and flow factors.")
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


def tab_predictor(t: dict) -> None:
    st.subheader("What would a burning droplet do in this atmosphere?")
    st.caption("Trained on 274 FLEX droplet tests (PSI-69). Outside the tested envelope these are extrapolations.")
    a, b, c = st.columns(3)
    fuel = a.selectbox("Fuel", ["Heptane", "Methanol"])
    x_o2 = a.slider("Oxygen (mole fraction)", 0.10, 0.40, 0.21, 0.01)
    p_atm = b.slider("Pressure (atm)", 0.30, 3.10, 1.00, 0.01)
    d0 = b.slider("Initial droplet diameter d0 (mm)", 1.0, 5.0, 3.0, 0.1)
    x_co2 = c.slider("CO2 diluent (mole fraction)", 0.0, 0.70, 0.0, 0.01)
    x_he = c.slider("Helium diluent (mole fraction)", 0.0, 0.45, 0.0, 0.01)
    if x_o2 + x_co2 + x_he > 1.0:
        st.error("O2 + CO2 + He add up to more than 1. Lower one of them; the rest is nitrogen.")
        return
    res = predict(fuel, x_o2, p_atm, x_co2, x_he, d0)
    if fuel == "Heptane":
        st.info(f"Cool-flame note: {coolflame.GUIDE_COOL_FLAME} ({coolflame.GUIDE_COOL_FLAME_CITE}). "
                "The outcome model does not predict cool flames.")
    flex = t["flex"][t["flex"]["fuel"] == fuel]
    outside = [name for name, v, s in (("O2", x_o2, flex["x_o2"]), ("pressure", p_atm, flex["p_atm"]),
                                       ("d0", d0, flex["d0_mm"])) if not (s.min() <= v <= s.max())]
    if outside or res["distance"] > 1.5:
        why = f"{', '.join(outside)} outside the tested {fuel} range" if outside else "far from every test"
        st.warning(f"Outside the tested envelope ({why}). Nearest real test {res['nearest_id']} is "
                   f"{res['distance']:.2f} standard units away: treat these numbers as extrapolation.")
    else:
        st.info(f"Nearest real FLEX test: {res['nearest_id']}, {res['distance']:.2f} standard units away.")
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Outcome probabilities** (logistic model)")
        for cls in ("Extinction", "Disruption", "Completion"):
            p = res["probs"].get(cls, 0.0)
            note = f" — low confidence ({completion_note(t)})" if cls == "Completion" else ""
            st.progress(float(p), text=f"{cls}: {p:.0%}{note}")
        if res["k"] > 0:
            st.markdown(f"**Burning-rate constant K**: {res['k']:.3f} mm²/s  \n"
                        f"10-90% conformal interval: {res['k_lo']:.3f} to {res['k_hi']:.3f} mm²/s")
        else:
            st.warning("The K model gives no positive burning rate here: these conditions are far outside "
                       "the FLEX data, so no burn time is shown.")
        comp_r = t["metrics"]["d2law_consistency"]["completion_only"]["pearson_r"]
        if res["k"] > 0:
            st.markdown(f"**Burn-time upper limit d0²/K**: {d0 ** 2 / res['k']:.1f} s")
            st.caption(f"Full burn-out time. It matches measured burn time only for droplets that burn to "
                       f"completion (r = {comp_r:.2f}); extinction or disruption ends a burn earlier.")
    with right:
        st.plotly_chart(d2_curve(d0, res), width="stretch")


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
    fig.update_layout(title="d² law: d(t)² = d0² − K·t (upper limit on burn time)", template="plotly_dark",
                      paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=340, margin={"l": 50, "r": 10, "t": 50, "b": 40},
                      xaxis_title="time (s)", yaxis_title="d² (mm²)", legend={"orientation": "h", "y": -0.25})
    return fig


# ------------------------------------------------------------------ tab 3: map
def oxygen_margin_map(t: dict, show: set[str], show_pg: bool = False) -> go.Figure:
    pts, lit = t["points"], t["lit"]
    nxt = t["next"][t["next"]["kind"] == "recommended (evidence gap)"]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.2, x1=3.2, y0=22.2, y1=42, fillcolor=FLAME, opacity=0.07, line_width=0,
                  layer="below")
    fig.add_annotation(x=np.log10(2.2), y=23.2, text="Above 22.2% O2: no PSI solid-material test",
                       showarrow=False, font={"color": FLAME, "size": 12})
    flex = pts[(pts["source"] == "PSI-69") & pts["p_atm"].notna()]
    if "droplet" in show:
        for outcome, g in flex.groupby("outcome"):
            fig.add_scatter(x=g["p_atm"], y=g["x_o2"] * 100, mode="markers", name=f"FLEX droplet: {outcome}",
                            marker={"color": OUTCOME_COLOR.get(str(outcome), MUTED), "size": 7, "opacity": 0.8},
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
    fig.add_scatter(x=tg["p_atm"], y=tg["x_o2"] * 100, mode="markers+text", name="Exploration targets",
                    marker={"color": TEXT, "size": 16, "symbol": "x-thin", "line": {"width": 3, "color": TEXT}},
                    text=tg["point_id"].map(TARGET_SHORT).fillna(tg["label"]).tolist(), textposition="middle right",
                    textfont={"color": TEXT}, hovertext=tg["label"], hoverinfo="text")
    fig.add_scatter(x=nxt["p_atm"], y=nxt["x_o2"] * 100, mode="markers+text", name="Recommended next tests",
                    marker={"color": "#FFFFFF", "size": 15, "symbol": "star", "line": {"color": FLAME, "width": 1.5}},
                    text=[f"#{r}" for r in nxt["rank"]], textposition="top center", textfont={"color": TEXT},
                    hovertext=nxt["explanation"], hoverinfo="text")
    share = t["summary"]["high_o2"]
    fig.add_annotation(x=np.log10(0.71), y=33.8, ax=np.log10(1.9), ay=31, xref="x", yref="y", axref="x", ayref="y",
                       text=f"At O2 ≥ 25%: {share['n_disruption']} of {share['n_tests']} FLEX droplets "
                            f"burst ({share['disruption_share']:.0%})", showarrow=True, arrowcolor=DANGER,
                       font={"color": DANGER, "size": 13}, arrowwidth=1.5)
    fig.update_layout(template="plotly_dark", paper_bgcolor=NIGHT, plot_bgcolor=PANEL, height=620,
                      margin={"l": 60, "r": 20, "t": 30, "b": 60},
                      xaxis={"title": "Ambient pressure (atm, log scale)", "type": "log",
                             "range": [np.log10(0.2 if show_pg else 0.3), np.log10(3.2)], "gridcolor": "#22324D",
                             "tickvals": [0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3]},
                      yaxis={"title": "Oxygen (% by volume)", "range": [10, 42], "gridcolor": "#22324D"},
                      legend={"orientation": "h", "y": -0.16, "font": {"size": 11}})
    return fig


def tab_map(t: dict) -> None:
    st.subheader("Where does NASA have fire evidence near a Moon-base atmosphere?")
    opts = {"droplet": "Droplets (FLEX)", "gas": "Gas flames", "solid": "Solid materials", "literature": "Literature"}
    chosen = st.multiselect("Show", list(opts), default=list(opts), format_func=opts.get)
    show_pg = st.toggle("Show partial-gravity equivalents")
    st.plotly_chart(oxygen_margin_map(t, set(chosen), show_pg), width="stretch")
    if show_pg:
        st.caption(partial_gravity.CAPTION)
    st.markdown("**Recommended next experiments** (gap to the nearest evidence × closeness to a target; "
                "PSI tests count fully, verified literature counts half)")
    st.dataframe(t["next"][["rank", "kind", "fuel_type", "x_o2", "p_atm", "score", "explanation"]], hide_index=True,
                 width="stretch")
    saffire3_replay(t)
    st.caption("Open circles: BASS-II and Saffire I-II have no pressure column; they are drawn at an assumed "
               "ISS-cabin 1 atm. BASS (PSI-26) and Saffire-III have no O2 value and are not plotted.")


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
                                  annotation_text=f"{r.igniter} current pulse", annotation_font_color=FLAME,
                                  annotation_position="top left" if i % 2 == 0 else "top right")
        fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=420,
                          margin={"l": 50, "r": 10, "t": 30, "b": 40}, xaxis_title="time since telemetry start (s)",
                          xaxis_range=[0, t_max], legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, width="stretch")
        if len({u for u in sub["unit"]}) > 1:
            st.caption("Channels with different units share one axis; compare shapes and timing, not magnitudes.")
        st.caption(telemetry.CAPTION + " Units shown are exactly as labelled in the file (psi, degC, V). "
                   "Multiplexed placeholder values (0 or the start-of-file value) were treated as missing; "
                   "dashed lines mark igniter current pulses. Source: PSI-100 Saffire-III telemetry, DOI ")
        st.markdown(a_link(f"{telemetry.DOI}", f"https://doi.org/{telemetry.DOI}"), unsafe_allow_html=True)


# ------------------------------------------------------------------ tab 4: chatbot
def tab_ask() -> None:
    st.subheader("Ask Fire-Margin")
    if not load_providers():
        st.info("No AI key is configured, so answers show the matching NASA passages with citations instead "
                "of a written summary. Add META_API_KEY (or NVIDIA_API_KEY) in Streamlit secrets to enable it.")
    st.caption("Answers come only from the Researcher's Guide, the insight cards and verified literature. "
               "Questions outside that evidence are declined.")
    history = st.session_state.setdefault("chat", [])
    picked = None
    for row in (STARTER_QUESTIONS[:3], STARTER_QUESTIONS[3:]):
        for q, c in zip(row, st.columns(3)):
            if c.button(q, width="stretch"):
                picked = q
    question = st.chat_input("Ask about fire in space habitats", max_chars=300) or picked
    if question:
        history.append((question, get_answer(question)))
        del history[:-MAX_HISTORY]
    for q, a in reversed(history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.markdown(a["text"])
            if a["sources"]:
                with st.expander(f"{len(a['sources'])} source passages"):
                    for s in a["sources"]:
                        link = f" ([link]({s['url']}))" if s["url"] else ""
                        st.markdown(f"**{s['cite']}**{link} — similarity {s['score']}  \n{s['text']}")


# ------------------------------------------------------------------ tab 5: lessons
def tab_lessons() -> None:
    st.subheader("What past fires taught, and what it gave back to Earth")
    for kind, title in (("incident", "Fires that changed spaceflight"), ("spinoff", "On Earth today")):
        st.markdown(f"#### {title}")
        for x in [x for x in LESSONS if x["kind"] == kind]:
            year = f" ({x['year']})" if x["year"] else ""
            st.markdown(f"**{x['title']}{year}.** {x['what_happened']}  \n*{x['lesson']}*")
            link = " — " + a_link("NASA source", x["url"]) if x["url"] else ""
            st.markdown(f'<p class="fm-src">{html.escape(x["source"])}{link}</p>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Fire-Margin", page_icon="🔥", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    t = load_tables()
    header(t)
    tabs = st.tabs(["Catalog & Ranking", "Flame Predictor", "Oxygen-Margin Map", "Ask Fire-Margin",
                    "Lessons & Earth Impact"])
    with tabs[0]:
        tab_catalog(t)
    with tabs[1]:
        tab_predictor(t)
    with tabs[2]:
        tab_map(t)
    with tabs[3]:
        tab_ask()
    with tabs[4]:
        tab_lessons()
    footer(t)


main()
