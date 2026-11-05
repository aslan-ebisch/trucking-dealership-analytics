"""Common page bootstrap: every view starts with ``ds, f = setup()``."""

from __future__ import annotations

import streamlit as st

from ironledger.analytics import FilterState
from ironledger.data_loader import Dataset
from ironledger.ui.data import get_dataset
from ironledger.ui.filters import current_filters


def setup() -> tuple[Dataset, FilterState]:
    ds = get_dataset()
    if ds is None:  # app.py normally stops before this; guard for direct execution
        st.error("Dataset not found. Run `python generate_data.py` and refresh.")
        st.stop()
    return ds, current_filters()
