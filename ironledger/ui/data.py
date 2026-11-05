"""Cached data access for the Streamlit app.

``st.cache_resource`` keeps ONE shared, read-only copy of the dataset in memory
for all reruns and sessions (no per-rerun copying, which matters at 100k+ rows).
The cache key is the data source's signature (file names + modification times),
so regenerating the data automatically invalidates the cache.
"""

from __future__ import annotations

import streamlit as st

from ironledger.data_loader import Dataset, DatasetNotFoundError, get_data_source, load_dataset
from ironledger.validation import run_all_checks


@st.cache_resource(show_spinner="Loading IronLedger datasets…", max_entries=2)
def _load(signature: str) -> Dataset:  # signature is only the cache key
    return load_dataset(get_data_source())


def get_dataset() -> Dataset | None:
    """Return the cached dataset, or None if it hasn't been generated yet."""
    source = get_data_source()
    sig = source.signature()
    if ":missing" in sig:
        return None
    try:
        return _load(sig)
    except DatasetNotFoundError:
        return None


@st.cache_data(show_spinner="Running data-quality checks…", max_entries=2)
def quality_report(signature: str) -> list[dict]:
    ds = _load(signature)
    return [r.__dict__ for r in run_all_checks(ds.tables())]


def data_signature() -> str:
    return get_data_source().signature()
