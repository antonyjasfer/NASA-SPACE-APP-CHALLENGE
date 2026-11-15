"""Fire-Margin visual system: tokens + one CSS block. Presentation only.

Contrast (WCAG): on BG TEXT 14.8, MUTED 8.5, CYAN 9.1, FLAME 6.1; on PANEL all >= 5.6 -> all >= 4.5:1.
"""
from __future__ import annotations

BG, PANEL, BORDER = "#0E1A2E", "#14233A", "#26395A"
TEXT, MUTED = "#E8EDF4", "#A9B6C9"
FLAME, CYAN = "#FF6B35", "#4CC9F0"           # one warm accent (warnings/highlights), one cool (data)
LILAC, SAND, ROSE, STEEL = "#C3B1FF", "#E9D48B", "#FF8A8A", "#8FB8DE"   # categorical data colours

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {{ --fm-bg:{BG}; --fm-panel:{PANEL}; --fm-border:{BORDER}; --fm-text:{TEXT}; --fm-muted:{MUTED};
        --fm-flame:{FLAME}; --fm-cyan:{CYAN}; }}
html, body, [class*="css"], .stMarkdown, p, li, label, button, input, textarea {{ font-family: 'Inter', sans-serif; }}
code, .fm-num, [data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ padding: 24px 32px 48px 32px !important; max-width: 1440px; }}
h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif !important; letter-spacing: -0.01em; color: {TEXT}; }}
h1 {{ font-size: 2.5rem !important; font-weight: 700 !important; margin: 0 0 8px 0 !important; padding: 0 !important; }}
h3 {{ font-size: 1.25rem !important; font-weight: 600 !important; margin: 8px 0 !important; }}
.fm-mission {{ color: {MUTED}; font-size: 1rem; line-height: 1.5; max-width: 72ch; margin: 0 0 16px 0; }}
.fm-lede {{ color: {MUTED}; font-size: .95rem; line-height: 1.5; max-width: 80ch; margin: 0 0 16px 0;
           border-left: 2px solid {CYAN}; padding: 0 0 0 12px; }}
.fm-src {{ color: {MUTED}; font-size: .8rem; line-height: 1.5; }}
.fm-banner {{ border: 1px solid {BORDER}; border-left: 3px solid {FLAME}; background: {PANEL};
             padding: 8px 16px; margin: 8px 0 16px 0; border-radius: 4px; font-size: .9rem; color: {TEXT}; }}
.fm-banner.calm {{ border-left-color: {CYAN}; }}
.fm-status {{ font-family: 'IBM Plex Mono', monospace; font-size: .8rem; color: {MUTED}; margin: 0 0 8px 0; }}
.fm-risk {{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-weight: 500; padding: 4px 12px;
           border: 1px solid {BORDER}; border-radius: 4px; margin: 0 0 8px 0; }}
.fm-risk.high {{ border-color: {FLAME}; color: {FLAME}; }}
.fm-risk.elevated {{ border-color: {SAND}; color: {SAND}; }}
.fm-risk.low {{ border-color: {CYAN}; color: {CYAN}; }}
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 16px; height: 100%; }}
[data-testid="stMetricValue"] {{ font-size: 1.9rem !important; color: {TEXT}; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .85rem !important; white-space: normal !important; }}
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {BORDER} !important; }}
[data-testid="stChatMessage"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 12px 16px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 16px; font-weight: 500; }}
button:focus-visible, [role="tab"]:focus-visible, a:focus-visible {{ outline: 2px solid {CYAN} !important; outline-offset: 2px; }}
a {{ color: {CYAN} !important; }}
@media (max-width: 640px) {{
  .block-container {{ padding: 72px 16px 40px 16px !important; }}
  h1 {{ font-size: 1.9rem !important; }}
  [data-testid="stMetricValue"] {{ font-size: 1.5rem !important; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""
