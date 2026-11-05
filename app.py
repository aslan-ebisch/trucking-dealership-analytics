"""IronLedger Analytics — application entry point.

Run with:   python -m streamlit run app.py

This file only wires the app together: page config, theme, sidebar brand and
global filters, and Streamlit's native multipage navigation. Each page lives in
``views/``; calculations live in ``ironledger/analytics.py``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ironledger.config import APP_SUBTITLE, APP_TAGLINE, AS_OF_DATE, DATA_DIR, GROUP_NAME
from ironledger.ui import theme
from ironledger.ui.components import empty_state
from ironledger.ui.data import get_dataset
from ironledger.ui.filters import render_sidebar_filters

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="IronLedger Analytics",
    page_icon=str(ROOT / "assets" / "icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.register_plotly_template()
theme.inject_css()
st.logo(str(ROOT / "assets" / "logo.svg"), icon_image=str(ROOT / "assets" / "icon.svg"), size="large")

dataset = get_dataset()
if dataset is None:
    st.markdown("<div class='il-eyebrow'>Setup required</div><div class='il-title'>IRONLEDGER</div>"
                f"<div class='il-subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
    st.write("")
    empty_state("No dataset found",
                f"The dashboard reads generated data from '{DATA_DIR}'. Generate it once from the project folder, "
                "then refresh this page.")
    st.code("python generate_data.py", language="bash")
    st.stop()

pages = {
    "Overview": [
        st.Page("views/executive_overview.py", title="Executive Overview", icon=":material/space_dashboard:",
                default=True),
    ],
    "Operations": [
        st.Page("views/sales_performance.py", title="Sales Performance", icon=":material/leaderboard:"),
        st.Page("views/inventory_management.py", title="Inventory Management", icon=":material/local_shipping:"),
    ],
    "Finance & Customers": [
        st.Page("views/financial_analytics.py", title="Financial Analytics", icon=":material/account_balance:"),
        st.Page("views/customer_intelligence.py", title="Customer Intelligence", icon=":material/groups:"),
    ],
    "Data": [
        st.Page("views/data_explorer.py", title="Data Explorer", icon=":material/database:"),
    ],
}
nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    render_sidebar_filters(dataset)
    st.markdown(
        f"<div class='il-side-foot'><b>{GROUP_NAME}</b><br>{APP_TAGLINE}<br>"
        f"Data through {AS_OF_DATE:%b %d, %Y} · {len(dataset.sales):,} transactions<br>"
        "All records are synthetic. Not affiliated with any manufacturer.</div>",
        unsafe_allow_html=True,
    )

nav.run()
