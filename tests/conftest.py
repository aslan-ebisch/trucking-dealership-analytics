"""Shared fixtures.

Tests never depend on the data in ``data/``: a small dataset is generated once
per test session (same engine, smaller size, fixed seed) into a temp folder and
loaded back through the real data-access layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ironledger.data_loader import LocalFileSource, load_dataset  # noqa: E402
from ironledger.generator import GeneratorConfig, generate_all, write_dataset  # noqa: E402

SMALL = GeneratorConfig(n_sales=4_000, n_customers=800, n_salespeople=60, seed=7)


@pytest.fixture(scope="session")
def raw_tables():
    """Tables exactly as produced by the generator (before writing to disk)."""
    return generate_all(SMALL)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, raw_tables):
    d = tmp_path_factory.mktemp("ironledger_data")
    write_dataset(raw_tables, d, SMALL, fmt="parquet")
    return d


@pytest.fixture(scope="session")
def dataset(data_dir):
    """Tables after a Parquet round-trip through the data-access layer."""
    return load_dataset(LocalFileSource(data_dir))
