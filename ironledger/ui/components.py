"""Page furniture: headers, section titles, KPI cards, context banner, empty states."""

from __future__ import annotations

import html
import math

import pandas as pd
import streamlit as st

from ironledger.analytics import FilterState
from ironledger.config import AS_OF_DATE


# ---------------------------------------------------------------------------
# Number formatting (NaN-safe: missing values render as an em dash)
# ---------------------------------------------------------------------------
def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def money(x, compact: bool = True, decimals: int = 1) -> str:
    if _bad(x):
        return "—"
    sign = "−" if x < 0 else ""
    a = abs(float(x))
    if compact:
        for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= div:
                return f"{sign}${a / div:,.{decimals}f}{suf}"
    return f"{sign}${a:,.0f}"


def number(x, decimals: int = 0) -> str:
    return "—" if _bad(x) else f"{x:,.{decimals}f}"


def pct(x, decimals: int = 1) -> str:
    return "—" if _bad(x) else f"{x * 100:.{decimals}f}%"


# Deltas use an ASCII hyphen: st.metric reads a leading "-" to decide arrow direction/color.
def signed_pct(x, decimals: int = 1) -> str | None:
    return None if _bad(x) else f"{'+' if x >= 0 else '-'}{abs(x) * 100:.{decimals}f}%"


def signed_pts(x, decimals: int = 1) -> str | None:
    return None if _bad(x) else f"{'+' if x >= 0 else '-'}{abs(x) * 100:.{decimals}f} pts"


# ---------------------------------------------------------------------------
# Layout pieces
# ---------------------------------------------------------------------------
def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    left, right = st.columns([5, 1.3], vertical_alignment="bottom")
    left.markdown(
        f"<div class='il-eyebrow'>{html.escape(eyebrow)}</div>"
        f"<div class='il-title'>{html.escape(title)}</div>"
        f"<div class='il-subtitle'>{html.escape(subtitle)}</div>",
        unsafe_allow_html=True,
    )
    with right:
        st.badge(f"Data as of {AS_OF_DATE:%b %d, %Y}", icon=":material/schedule:", color="gray")
        st.badge("Synthetic dataset", icon=":material/science:", color="orange")


def section(title: str, note: str = "") -> None:
    st.markdown(f"<div class='il-section'><h3>{html.escape(title)}</h3><span>{html.escape(note)}</span></div>",
                unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='il-note'>{html.escape(text)}</div>", unsafe_allow_html=True)


def _names(selected: tuple, lookup: dict | None, all_label: str) -> str:
    if not selected:
        return all_label
    names = [lookup.get(s, s) if lookup else s for s in selected]
    return ", ".join(names) if len(names) <= 3 else f"{len(names)} selected"


def filter_context(f: FilterState, dealer_names: dict, rows_in_scope: int | None = None,
                   rows_total: int | None = None, *, dates_apply: bool = True, extra: str = "") -> None:
    """One-line banner stating exactly which slice of data the page is showing."""
    period = (f"<b>{f.start:%b %d, %Y} – {f.end:%b %d, %Y}</b>" if dates_apply
              else f"<b>Snapshot {AS_OF_DATE:%b %d, %Y}</b> (date filter n/a)")
    parts = [
        period,
        f"<b>{html.escape(_names(f.dealerships, dealer_names, 'All dealerships'))}</b>",
        html.escape(_names(f.manufacturers, None, "All manufacturers")),
        html.escape(_names(f.conditions, None, "New & used")),
        html.escape(_names(f.categories, None, "All categories")),
    ]
    scope = ""
    if rows_in_scope is not None and rows_total is not None:
        scope = f" &nbsp;|&nbsp; <b>{rows_in_scope:,}</b> of {rows_total:,} records in scope"
    st.markdown(f"<div class='il-context'>Showing {' · '.join(parts)}{scope}"
                f"{(' &nbsp;|&nbsp; ' + extra) if extra else ''}</div>", unsafe_allow_html=True)


def empty_state(title: str = "No matching records",
                message: str = "No transactions match the current filters. Widen the date range or clear a filter "
                               "in the sidebar.") -> None:
    st.markdown(f"<div class='il-empty'><h4>{html.escape(title)}</h4>{html.escape(message)}</div>",
                unsafe_allow_html=True)


def kpi(container, label: str, value: str, delta: str | None = None, *, help: str | None = None,
        inverse: bool = False, neutral: bool = False, spark: list | pd.Series | None = None,
        spark_type: str = "area") -> None:
    """A KPI card built on native ``st.metric`` (styled by theme CSS)."""
    color = "off" if neutral else ("inverse" if inverse else "normal")
    data = None
    if spark is not None:
        data = [float(v) for v in spark if not pd.isna(v)]
        data = data if len(data) >= 3 else None
    container.metric(label, value, delta, delta_color=color, help=help, border=True,
                     chart_data=data, chart_type=spark_type,
                     delta_arrow="off" if neutral else "auto")


METRIC_DEFINITIONS_MD = """
| Metric | Definition |
|---|---|
| **Vehicle sales revenue** | Σ selling price of vehicles sold. F&I income is *not* included in revenue. |
| **Cost of vehicles sold** | Σ (acquisition cost + reconditioning/make-ready + inbound transportation). |
| **Vehicle gross profit** | Revenue − cost of vehicles sold ("front-end gross"). |
| **Vehicle gross margin** | Vehicle gross profit ÷ revenue. |
| **F&I income** | Finance reserve income + protection-product (warranty/service contract) income ("back-end gross"). |
| **Total gross contribution** | Vehicle gross profit + F&I income. |
| **Operating expenses** | Sales-department payroll (incl. commissions), facility, marketing, utilities, floorplan interest, other. |
| **Est. operating income** | Total gross contribution − operating expenses. Excludes service/parts departments, corporate overhead, depreciation and income tax — it is **not** net income. |
| **Average selling price (ASP)** | Revenue ÷ units sold. |
| **Gross per unit (GPU)** | Vehicle gross profit ÷ units sold. |
| **Days to sell** | Average days between a unit's acquisition and its sale date. |
| **Growth** | (Current − prior) ÷ prior for amounts; margins and rates change in percentage points. Prior period = same dates one year earlier, or the equal-length window immediately before. |
| **Months' supply** | Units in stock ÷ average monthly units sold over the last 90 days (same filters). |
"""


def definitions_expander() -> None:
    with st.expander("Metric definitions", icon=":material/menu_book:"):
        st.markdown(METRIC_DEFINITIONS_MD)
