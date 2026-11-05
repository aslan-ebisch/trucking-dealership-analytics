"""Headless UI tests: every page renders without errors, and sidebar filters change the numbers.

Uses Streamlit's built-in AppTest (no browser needed) against the small test dataset.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ironledger import config

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["views/executive_overview.py", "views/sales_performance.py", "views/inventory_management.py",
         "views/financial_analytics.py", "views/customer_intelligence.py", "views/data_explorer.py"]


@pytest.fixture(autouse=True)
def use_test_data(monkeypatch, data_dir):
    monkeypatch.setattr(config, "DATA_DIR", Path(data_dir))


def _open(page: str, **filters) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    for key, value in filters.items():
        widget = at.selectbox(key=key) if key == "f_period" else at.multiselect(key=key)
        widget.set_value(value)
    at.run()
    at.switch_page(page)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _open(page, f_period="All history")
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) >= 4


@pytest.mark.parametrize("page", PAGES)
def test_page_handles_empty_selection(page):
    # Nashville (D10) opened in 2023, so it has no 2021 sales: pages must show an empty state, not crash.
    at = _open(page, f_period="2021", f_dealers=["D10"])
    assert not at.exception, [e.value for e in at.exception]


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_filters_change_kpis():
    all_units = _metric(_open(PAGES[0], f_period="All history"), "Units sold")
    one_store = _metric(_open(PAGES[0], f_period="All history", f_dealers=["D01"]), "Units sold")
    one_year = _metric(_open(PAGES[0], f_period="2024"), "Units sold")
    as_int = lambda v: int(v.replace(",", ""))
    assert as_int(one_store) < as_int(all_units)
    assert as_int(one_year) < as_int(all_units)


def test_missing_data_shows_setup_screen(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # empty folder
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("generate_data.py" in c.value for c in at.code)
