"""Fire-Margin — Streamlit entry point. Run: streamlit run app/app.py

P1 scaffold: five tabs from CLAUDE.md as placeholders. Content lands in phases P2-P4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Streamlit Cloud runs app/app.py; make `src` importable
    sys.path.insert(0, str(ROOT))

TABS = [
    ("Catalog & Ranking", "PSI combustion investigations, AI insight cards and a transparent weighted score."),
    ("Flame Predictor", "Fuel, O2, pressure, diluent and droplet size -> outcome risk, K, burn time, "
                        "uncertainty and distance to the nearest real FLEX test."),
    ("Oxygen-Margin Map", "Every tested condition in O2 x pressure space against the 34% / 8.2 psia "
                          "and 30% / 10.2 psi exploration-atmosphere targets."),
    ("Ask Fire-Margin", "RAG chatbot grounded only on NASA sources; every answer cites a PSI ID / DOI."),
    ("Lessons & Earth Impact", "Apollo 1, Apollo 13, Mir 1997 and NASA fire-safety spinoffs."),
]


def main() -> None:
    st.set_page_config(page_title="Fire-Margin", page_icon="🔥", layout="wide")
    st.title("Fire-Margin")
    st.caption("Evidence-graded fire safety for Moon-base atmospheres — NASA Space Apps 2026, "
               "\"Flame in Freefall\".")
    for tab, (name, blurb) in zip(st.tabs([n for n, _ in TABS]), TABS):
        with tab:
            st.subheader(name)
            st.info(f"Coming soon. {blurb}")


main()
