"""Visual identity: colors, Plotly chart template, and CSS polish.

Color roles
-----------
* ACCENT (metallic gold) is the brand color: UI highlights and single-series charts.
* CATEGORICAL is a validated 8-slot palette (colorblind-separation checked against
  the dark chart surface). Slots are assigned to entities in a FIXED order so a
  filter never repaints a series.
* SEQUENTIAL / ORDINAL ramps are single-hue blue (dark = low, bright = high).
* POSITIVE / NEGATIVE are reserved for gains and losses.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Surfaces & ink
BG = "#0b0c0e"
SURFACE = "#14161a"
SURFACE_2 = "#1b1e23"
BORDER = "#2a2e35"
GRID = "#22252b"
TEXT = "#ebe8e2"
TEXT_2 = "#a9acb3"
MUTED = "#6d727b"

ACCENT = "#d6a84e"        # metallic gold
ACCENT_SOFT = "rgba(214,168,78,0.14)"
PRIOR = "#6d727b"         # prior-period comparison series (neutral steel)

CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
POSITIVE = "#3987e5"
NEGATIVE = "#e66767"

SEQUENTIAL = ["#16202e", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]
AGING_COLORS = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]  # fresh -> oldest (brightest)

CONDITION_COLORS = {"New": CATEGORICAL[0], "Used": CATEGORICAL[1]}
VEHICLE_TYPE_COLORS = {"Truck": CATEGORICAL[0], "Trailer": CATEGORICAL[2]}
SEGMENT_COLORS = {
    "Owner-Operator": CATEGORICAL[0], "Small Fleet": CATEGORICAL[1], "Regional Carrier": CATEGORICAL[2],
    "National Fleet": CATEGORICAL[3], "Construction & Vocational": CATEGORICAL[4], "Agriculture": CATEGORICAL[5],
    "Government & Municipal": CATEGORICAL[6],
}

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
HEADING_FONT = "'Barlow Semi Condensed', Inter, sans-serif"


def register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT, size=12, color=TEXT_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CATEGORICAL,
        margin=dict(l=8, r=12, t=36, b=8),
        title=dict(font=dict(family=HEADING_FONT, size=15, color=TEXT), x=0, xanchor="left",
                   yref="container", y=0.985, yanchor="top"),
        xaxis=dict(automargin=True, gridcolor=GRID, linecolor=BORDER, zeroline=False, ticks="", showgrid=False,
                   tickfont=dict(color=TEXT_2), title=dict(font=dict(color=MUTED, size=11))),
        yaxis=dict(automargin=True, gridcolor=GRID, linecolor=BORDER, zeroline=False, ticks="", gridwidth=1,
                   tickfont=dict(color=TEXT_2), title=dict(font=dict(color=MUTED, size=11))),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text="",
                    font=dict(color=TEXT_2, size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(family=FONT, color=TEXT, size=12)),
        hovermode="closest",
        barcornerradius=3,
        bargap=0.28,
        separators=".,",
        geo=dict(bgcolor="rgba(0,0,0,0)", landcolor=SURFACE_2, subunitcolor=BORDER, countrycolor=BORDER,
                 lakecolor=BG, showlakes=False, coastlinecolor=BORDER),
    )
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    template.data.scatter = [go.Scatter(line=dict(width=2), marker=dict(size=8))]
    pio.templates["ironledger"] = template
    pio.templates.default = "ironledger"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&display=swap');

/* ---------- layout ---------- */
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0f1013 0%, #0b0c0e 100%);
                             border-right: 1px solid {BORDER}; }}
[data-testid="stSidebarNavSeparator"] {{ border-color: {BORDER}; }}
hr {{ border-color: {BORDER} !important; margin: 0.6rem 0 1rem 0 !important; }}

/* ---------- typography ---------- */
h1, h2, h3, h4 {{ font-family: {HEADING_FONT} !important; letter-spacing: 0.01em; }}
.il-eyebrow {{ color: {ACCENT}; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.18em;
               text-transform: uppercase; margin-bottom: 0.1rem; }}
.il-title {{ font-family: {HEADING_FONT}; font-size: 2.1rem; font-weight: 700; color: {TEXT};
             line-height: 1.1; margin: 0; }}
.il-subtitle {{ color: {TEXT_2}; font-size: 0.92rem; margin-top: 0.25rem; }}
.il-section {{ display:flex; align-items:baseline; gap:0.75rem; margin: 1.4rem 0 0.4rem 0;
               border-bottom: 1px solid {BORDER}; padding-bottom: 0.35rem; }}
.il-section h3 {{ font-size: 1.05rem !important; font-weight: 600; text-transform: uppercase;
                  letter-spacing: 0.12em; color: {TEXT}; margin: 0 !important; padding: 0 !important; }}
.il-section span {{ color: {MUTED}; font-size: 0.8rem; }}
.il-context {{ color: {TEXT_2}; font-size: 0.8rem; background: {SURFACE}; border: 1px solid {BORDER};
               border-left: 3px solid {ACCENT}; border-radius: 6px; padding: 0.45rem 0.8rem; margin: 0.6rem 0 0.9rem 0; }}
.il-context b {{ color: {TEXT}; font-weight: 600; }}
.il-note {{ color: {MUTED}; font-size: 0.78rem; }}

/* ---------- KPI cards (native st.metric) ---------- */
[data-testid="stMetric"] {{ background: linear-gradient(180deg, {SURFACE_2} 0%, {SURFACE} 100%);
                           border: 1px solid {BORDER} !important; border-radius: 10px !important;
                           padding: 0.85rem 1rem 0.7rem 1rem !important; }}
[data-testid="stMetric"]:hover {{ border-color: #3a3f48 !important; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{ white-space: normal !important;
    overflow: visible !important; text-overflow: clip !important; }}
[data-testid="stMetricLabel"] p {{ color: {TEXT_2} !important; font-size: 0.72rem !important; font-weight: 600 !important;
                                   text-transform: uppercase; letter-spacing: 0.1em; }}
[data-testid="stMetricValue"] {{ font-family: {HEADING_FONT}; font-weight: 600; font-size: 1.85rem !important;
                                 color: {TEXT}; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}

/* ---------- containers, tabs, tables ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {BORDER} !important; border-radius: 10px !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ font-size: 0.82rem; font-weight: 600; letter-spacing: 0.06em;
                                text-transform: uppercase; color: {TEXT_2}; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
[data-testid="stExpander"] details {{ border-color: {BORDER} !important; background: {SURFACE}; }}

/* ---------- sidebar brand footer ---------- */
.il-side-label {{ color: {MUTED}; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.16em;
                  text-transform: uppercase; margin: 0.4rem 0 0.2rem 0; }}
.il-side-foot {{ color: {MUTED}; font-size: 0.72rem; line-height: 1.5; border-top: 1px solid {BORDER};
                 padding-top: 0.6rem; margin-top: 0.8rem; }}
.il-side-foot b {{ color: {TEXT_2}; }}

/* ---------- empty state ---------- */
.il-empty {{ border: 1px dashed {BORDER}; border-radius: 10px; padding: 2rem; text-align: center;
             color: {TEXT_2}; background: {SURFACE}; }}
.il-empty h4 {{ color: {TEXT}; margin-bottom: 0.3rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
