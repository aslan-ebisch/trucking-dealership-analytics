"""Data-quality tests on a freshly generated dataset (and its Parquet/CSV round-trips)."""

import numpy as np
import pandas as pd
import pytest

from ironledger.data_loader import LocalFileSource, load_dataset
from ironledger.generator import GeneratorConfig, generate_all, write_dataset
from ironledger.validation import ALL_CHECKS, check_gross_profit_identity, check_inventory_matches_sales, \
    check_sale_after_acquisition, run_all_checks

from conftest import SMALL


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__)
def test_generated_data_passes_check(raw_tables, check):
    result = check(raw_tables)
    assert result.passed, result.detail


def test_loaded_parquet_dataset_passes_all_checks(dataset):
    failed = [r for r in run_all_checks(dataset.tables()) if not r.passed]
    assert not failed, failed


def test_csv_round_trip_passes_all_checks(tmp_path, raw_tables):
    write_dataset(raw_tables, tmp_path, SMALL, fmt="csv")
    ds = load_dataset(LocalFileSource(tmp_path))
    failed = [r for r in run_all_checks(ds.tables()) if not r.passed]
    assert not failed, failed


def test_requested_sizes(raw_tables):
    assert len(raw_tables["sales"]) == SMALL.n_sales
    assert len(raw_tables["customers"]) == SMALL.n_customers
    assert len(raw_tables["salespeople"]) == SMALL.n_salespeople
    assert len(raw_tables["dealerships"]) == 10


def test_deterministic_with_same_seed(raw_tables):
    again = generate_all(SMALL)
    for name, df in raw_tables.items():
        pd.testing.assert_frame_equal(df, again[name], check_like=False)


def test_different_seed_changes_data(raw_tables):
    other = generate_all(GeneratorConfig(n_sales=SMALL.n_sales, n_customers=SMALL.n_customers,
                                         n_salespeople=SMALL.n_salespeople, seed=SMALL.seed + 1))
    assert not raw_tables["sales"]["sale_price"].equals(other["sales"]["sale_price"])


def test_inventory_and_sales_reconcile(raw_tables):
    inv, s = raw_tables["inventory"], raw_tables["sales"]
    assert (inv["status"] == "Sold").sum() == len(s)
    assert set(inv.loc[inv["status"] == "Sold", "unit_id"]) == set(s["unit_id"])
    stock = inv[inv["status"] == "In Stock"]
    assert len(stock) > 0 and stock["sale_id"].isna().all()


def test_reasonable_financials(raw_tables):
    s = raw_tables["sales"]
    margin = s["vehicle_gross_profit"].sum() / s["sale_price"].sum()
    assert 0.03 < margin < 0.15                            # front-end gross margin of a truck dealer
    assert 0.0 < (s["vehicle_gross_profit"] < 0).mean() < 0.25   # some, but not most, deals lose money
    new = s[(s["condition"] == "New") & (s["vehicle_type"] == "Truck")]["sale_price"]
    used = s[(s["condition"] == "Used") & (s["vehicle_type"] == "Truck")]["sale_price"]
    assert new.median() > used.median()                     # new trucks cost more than used
    assert s.groupby("dealership_id").size().idxmax() == "D01"   # Dallas is the flagship


def test_used_trucks_have_meaningful_mileage(raw_tables):
    s = raw_tables["sales"]
    used = s[(s["condition"] == "Used") & (s["vehicle_category"] == "Sleeper Tractor")]
    assert used["mileage"].median() > 150_000


# ----------------------------------------------------------------- the checks catch real corruption
def test_checks_detect_broken_gross_identity(raw_tables):
    bad = dict(raw_tables)
    bad["sales"] = raw_tables["sales"].copy()
    bad["sales"].loc[bad["sales"].index[:3], "vehicle_gross_profit"] += 1
    r = check_gross_profit_identity(bad)
    assert not r.passed and r.failures == 3


def test_checks_detect_sale_before_acquisition(raw_tables):
    bad = dict(raw_tables)
    bad["sales"] = raw_tables["sales"].copy()
    i = bad["sales"].index[0]
    bad["sales"].loc[i, "inventory_acquisition_date"] = bad["sales"].loc[i, "sale_date"] + pd.Timedelta(days=5)
    assert not check_sale_after_acquisition(bad).passed


def test_checks_detect_inventory_contradiction(raw_tables):
    bad = dict(raw_tables)
    bad["inventory"] = raw_tables["inventory"].copy()
    sold_idx = bad["inventory"].index[bad["inventory"]["status"] == "Sold"][0]
    bad["inventory"].loc[sold_idx, "acquisition_cost"] += 500
    assert not check_inventory_matches_sales(bad).passed


def test_write_refuses_to_overwrite(tmp_path, raw_tables):
    write_dataset(raw_tables, tmp_path, SMALL, fmt="parquet")
    with pytest.raises(FileExistsError):
        write_dataset(raw_tables, tmp_path, SMALL, fmt="parquet")
    write_dataset(raw_tables, tmp_path, SMALL, fmt="parquet", force=True)  # explicit overwrite is allowed
