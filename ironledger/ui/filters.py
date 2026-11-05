"""Global sidebar filters.

The widgets are rendered by ``app.py`` on every run (before the selected page
executes), so their values persist in ``st.session_state`` while users move
between pages. Pages read the resulting ``FilterState`` via ``current_filters()``.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from ironledger.analytics import FilterState
from ironledger.config import AS_OF_DATE, HISTORY_START
from ironledger.data_loader import Dataset
from ironledger.reference_data import VEHICLE_CATEGORIES

PERIODS = ["Year to date", "All history", "Last 12 months", "Last 90 days", "2026", "2025", "2024", "2023",
           "2022", "2021", "Custom range"]
_KEYS = ["f_period", "f_custom", "f_dealers", "f_mfrs", "f_conditions", "f_categories"]
_STATE_KEY = "_filter_state"


def period_bounds(period: str, custom=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = AS_OF_DATE
    if period == "Year to date":
        return pd.Timestamp(AS_OF_DATE.year, 1, 1), end
    if period == "Last 12 months":
        return AS_OF_DATE - pd.DateOffset(years=1) + pd.Timedelta(days=1), end
    if period == "Last 90 days":
        return AS_OF_DATE - pd.Timedelta(days=89), end
    if period.isdigit():
        y = int(period)
        return pd.Timestamp(y, 1, 1), min(pd.Timestamp(y, 12, 31), AS_OF_DATE)
    if period == "Custom range" and custom:
        if isinstance(custom, (list, tuple)) and len(custom) == 2:
            a, b = sorted(pd.Timestamp(x) for x in custom)
            return max(a, HISTORY_START), min(b, AS_OF_DATE)
        if isinstance(custom, (list, tuple)) and len(custom) == 1:  # user mid-selection
            a = pd.Timestamp(custom[0])
            return max(a, HISTORY_START), AS_OF_DATE
    return HISTORY_START, end


def _reset() -> None:
    for k in _KEYS:
        st.session_state.pop(k, None)


def render_sidebar_filters(ds: Dataset) -> FilterState:
    dealers = ds.dealerships.sort_values("dealership_id")
    dealer_label = dict(zip(dealers["dealership_id"], dealers["city"] + " — " + dealers["dealership_name"]))
    manufacturers = sorted(ds.sales["manufacturer"].astype(str).unique())

    st.markdown("<div class='il-side-label'>Global filters</div>", unsafe_allow_html=True)
    period = st.selectbox("Reporting period", PERIODS, key="f_period")
    custom = None
    if period == "Custom range":
        custom = st.date_input("Date range", value=(dt.date(2025, 1, 1), AS_OF_DATE.date()),
                               min_value=HISTORY_START.date(), max_value=AS_OF_DATE.date(), key="f_custom",
                               format="MM/DD/YYYY")
    start, end = period_bounds(period, custom)

    sel_dealers = st.multiselect("Dealership", list(dealer_label), format_func=lambda d: dealer_label[d],
                                 key="f_dealers", placeholder="All dealerships")
    sel_mfrs = st.multiselect("Manufacturer", manufacturers, key="f_mfrs", placeholder="All manufacturers")
    sel_cond = st.pills("Condition", ["New", "Used"], selection_mode="multi", key="f_conditions")
    sel_cats = st.multiselect("Vehicle category", VEHICLE_CATEGORIES, key="f_categories",
                              placeholder="All categories")
    st.button("Reset filters", icon=":material/restart_alt:", on_click=_reset, width="stretch", type="tertiary")

    f = FilterState(
        start=start, end=end, dealerships=tuple(sel_dealers), manufacturers=tuple(sel_mfrs),
        conditions=tuple(sel_cond or ()) if len(sel_cond or ()) < 2 else (), categories=tuple(sel_cats),
    )
    st.session_state[_STATE_KEY] = f
    return f


def current_filters() -> FilterState:
    return st.session_state.get(_STATE_KEY, FilterState())


def dealer_names(ds: Dataset) -> dict[str, str]:
    return dict(zip(ds.dealerships["dealership_id"], ds.dealerships["city"]))
