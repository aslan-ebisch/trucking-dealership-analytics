"""Data-quality checks.

The same checks run in three places:
  1. ``generate_data.py`` — a dataset that fails any check is never written.
  2. ``tests/test_data_quality.py`` — pytest asserts every check passes.
  3. The Data Explorer page — shows a live health panel for the loaded data.

Each check returns a ``CheckResult`` with the number of offending rows, so the
result is auditable rather than a bare True/False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from ironledger.config import AS_OF_DATE, HISTORY_START

Tables = Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str
    passed: bool
    failures: int
    detail: str


def _result(name: str, category: str, bad: int | np.integer, ok_detail: str, bad_detail: str) -> CheckResult:
    bad = int(bad)
    return CheckResult(name, category, bad == 0, bad, ok_detail if bad == 0 else f"{bad:,} {bad_detail}")


# --- Uniqueness ------------------------------------------------------------------
def check_unique_sale_ids(t: Tables) -> CheckResult:
    s = t["sales"]["sale_id"]
    return _result("Unique sale IDs", "Uniqueness", s.duplicated().sum() + s.isna().sum(),
                   f"{len(s):,} unique IDs", "duplicate or missing sale IDs")


def check_unique_unit_ids(t: Tables) -> CheckResult:
    u = t["inventory"]["unit_id"]
    return _result("Unique unit IDs", "Uniqueness", u.duplicated().sum() + u.isna().sum(),
                   f"{len(u):,} unique units", "duplicate or missing unit IDs")


def check_unit_sold_once(t: Tables) -> CheckResult:
    d = t["sales"]["unit_id"].duplicated().sum()
    return _result("Each unit sold at most once", "Uniqueness", d, "no unit appears in two sales",
                   "units sold more than once")


# --- Referential integrity ---------------------------------------------------------
def _fk(t: Tables, child: str, col: str, parent: str, pcol: str) -> int:
    return int((~t[child][col].isin(t[parent][pcol])).sum())


def check_foreign_keys(t: Tables) -> CheckResult:
    bad = (_fk(t, "sales", "dealership_id", "dealerships", "dealership_id")
           + _fk(t, "sales", "salesperson_id", "salespeople", "salesperson_id")
           + _fk(t, "sales", "customer_id", "customers", "customer_id")
           + _fk(t, "sales", "unit_id", "inventory", "unit_id")
           + _fk(t, "inventory", "dealership_id", "dealerships", "dealership_id")
           + _fk(t, "salespeople", "dealership_id", "dealerships", "dealership_id")
           + _fk(t, "customers", "primary_dealership_id", "dealerships", "dealership_id")
           + _fk(t, "operating_expenses", "dealership_id", "dealerships", "dealership_id"))
    return _result("Foreign keys resolve", "Referential integrity", bad,
                   "all dealership/salesperson/customer/unit references valid", "orphaned references")


def check_salesperson_store_and_tenure(t: Tables) -> CheckResult:
    s = t["sales"][["sale_date", "dealership_id", "salesperson_id"]].merge(
        t["salespeople"][["salesperson_id", "dealership_id", "hire_date", "termination_date"]],
        on="salesperson_id", how="left", suffixes=("", "_sp"))
    bad = ((s["dealership_id"] != s["dealership_id_sp"])
           | (s["sale_date"] < s["hire_date"])
           | (s["termination_date"].notna() & (s["sale_date"] > s["termination_date"]))).sum()
    return _result("Salesperson employed at store on sale date", "Referential integrity", bad,
                   "every deal credited to an active rep of that store", "deals credited to an ineligible rep")


def check_denormalized_attributes(t: Tables) -> CheckResult:
    s = t["sales"][["dealership_id", "dealership_name", "city", "state", "customer_id", "customer_segment"]]
    d = t["dealerships"].set_index("dealership_id")
    c = t["customers"].set_index("customer_id")
    bad = ((s["dealership_name"].astype(str) != s["dealership_id"].map(d["dealership_name"]).astype(str))
           | (s["city"].astype(str) != s["dealership_id"].map(d["city"]).astype(str))
           | (s["state"].astype(str) != s["dealership_id"].map(d["state"]).astype(str))
           | (s["customer_segment"].astype(str) != s["customer_id"].map(c["customer_segment"]).astype(str))).sum()
    return _result("Denormalized attributes match dimensions", "Referential integrity", bad,
                   "store names/locations and customer segments consistent", "rows disagree with dimension tables")


# --- Dates -------------------------------------------------------------------------------
def check_sale_after_acquisition(t: Tables) -> CheckResult:
    s = t["sales"]
    days = (s["sale_date"] - s["inventory_acquisition_date"]).dt.days
    bad = ((days < 0) | (days != s["days_in_inventory"])).sum()
    return _result("Sale date on/after acquisition date", "Dates", bad,
                   "days_in_inventory = sale_date − acquisition_date for every deal", "impossible date sequences")


def check_sale_dates_in_window(t: Tables) -> CheckResult:
    s = t["sales"]
    opened = s["dealership_id"].map(t["dealerships"].set_index("dealership_id")["opened_date"])
    bad = ((s["sale_date"] < HISTORY_START) | (s["sale_date"] > AS_OF_DATE) | (s["sale_date"] < opened)).sum()
    return _result("Sale dates inside reporting window", "Dates", bad,
                   f"{HISTORY_START.date()} → {AS_OF_DATE.date()}, after store opening", "out-of-window sales")


def check_model_years(t: Tables) -> CheckResult:
    s = t["sales"]
    bad = ((s["model_year"] > s["sale_date"].dt.year + 1) | (s["model_year"] < 1995)).sum()
    return _result("Plausible model years", "Dates", bad, "model year ≤ sale year + 1", "implausible model years")


# --- Values ------------------------------------------------------------------------------
def check_non_negative_values(t: Tables) -> CheckResult:
    s, inv = t["sales"], t["inventory"]
    cost_cols = ["acquisition_cost", "reconditioning_cost", "transportation_cost", "finance_income",
                 "warranty_income", "amount_financed", "sales_commission"]
    bad = int((s[cost_cols] < 0).any(axis=1).sum())
    bad += int((s["sale_price"] <= 0).sum())
    bad += int((s["mileage"] < 0).sum()) + int((inv["mileage"] < 0).sum())
    bad += int((inv[["acquisition_cost", "reconditioning_cost", "transportation_cost"]] < 0).any(axis=1).sum())
    return _result("No negative costs, prices or mileage", "Values", bad,
                   "all costs ≥ 0, prices > 0, mileage ≥ 0", "rows with invalid negative values")


def check_mileage_semantics(t: Tables) -> CheckResult:
    s = t["sales"]
    trucks = s["vehicle_type"] == "Truck"
    bad = ((trucks & (s["condition"] == "Used") & ~(s["mileage"] > 1_000))
           | (trucks & (s["condition"] == "New") & ~(s["mileage"] < 1_000))
           | (~trucks & s["mileage"].notna())).sum()
    return _result("Mileage consistent with condition", "Values", bad,
                   "used trucks > 1,000 mi; new < 1,000 mi; trailers not metered", "inconsistent odometer readings")


def check_price_ranges(t: Tables) -> CheckResult:
    s = t["sales"]
    bad = ((s["sale_price"] < 3_000) | (s["sale_price"] > 450_000)).sum()
    return _result("Selling prices in plausible range", "Values", bad, "$3k – $450k", "prices out of range")


# --- Financial identities -----------------------------------------------------------------
def check_gross_profit_identity(t: Tables) -> CheckResult:
    s = t["sales"]
    calc = s["sale_price"] - s["acquisition_cost"] - s["reconditioning_cost"] - s["transportation_cost"]
    bad = (calc != s["vehicle_gross_profit"]).sum()
    return _result("Vehicle gross = price − acquisition − recon − transport", "Financial identities", bad,
                   "identity holds to the dollar", "rows violate the gross-profit identity")


def check_margin_identity(t: Tables) -> CheckResult:
    s = t["sales"]
    bad = (~np.isclose(s["vehicle_gross_margin"], s["vehicle_gross_profit"] / s["sale_price"], atol=1e-5)).sum()
    return _result("Gross margin = gross / price", "Financial identities", bad, "identity holds",
                   "rows with inconsistent margin")


def check_contribution_identity(t: Tables) -> CheckResult:
    s = t["sales"]
    bad = ((s["vehicle_gross_profit"] + s["finance_income"] + s["warranty_income"])
           != s["total_gross_contribution"]).sum()
    bad += int(((~s["financing_arranged"].astype(bool)) & ((s["finance_income"] > 0) | (s["amount_financed"] > 0))).sum())
    bad += int((s["protection_product"].isna() & (s["warranty_income"] > 0)).sum())
    return _result("F&I income tracked separately & consistently", "Financial identities", bad,
                   "contribution = vehicle gross + finance + warranty; no F&I income without a product",
                   "rows with inconsistent F&I income")


def check_opex_identity(t: Tables) -> CheckResult:
    o = t["operating_expenses"]
    parts = ["payroll_expense", "facility_expense", "marketing_expense", "utilities_expense",
             "floorplan_interest_expense", "other_operating_expense"]
    bad = int((o[parts].sum(axis=1) != o["total_operating_expense"]).sum())
    bad += int(o.duplicated(["dealership_id", "month"]).sum())
    comm = (t["sales"].assign(month=t["sales"]["sale_date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["dealership_id", "month"], observed=True)["sales_commission"].sum())
    pay = o.set_index(["dealership_id", "month"])["payroll_expense"]
    bad += int((pay.reindex(comm.index).fillna(0) < comm).sum())
    return _result("Operating expenses sum & cover commissions", "Financial identities", bad,
                   "line items sum to total; payroll ≥ commissions earned", "inconsistent expense rows")


# --- Inventory consistency ------------------------------------------------------------------
def check_inventory_matches_sales(t: Tables) -> CheckResult:
    inv, s = t["inventory"], t["sales"]
    sold = inv[inv["status"] == "Sold"].set_index("unit_id")
    bad = int((inv["status"] == "Sold").sum() != len(s))
    bad += int(((inv["status"] == "Sold") != inv["sale_id"].notna()).sum())
    m = s.set_index("unit_id")
    joined = sold.join(m, rsuffix="_s", how="left")
    cols = [("sale_id", "sale_id_s"), ("sale_date", "sale_date_s"), ("dealership_id", "dealership_id_s"),
            ("acquisition_date", "inventory_acquisition_date"), ("acquisition_cost", "acquisition_cost_s"),
            ("reconditioning_cost", "reconditioning_cost_s"), ("transportation_cost", "transportation_cost_s"),
            ("manufacturer", "manufacturer_s"), ("model", "model_s"), ("model_year", "model_year_s"),
            ("condition", "condition_s"), ("days_in_inventory", "days_in_inventory_s")]
    for a, b in cols:
        bad += int((joined[a].astype(str) != joined[b].astype(str)).sum())
    return _result("Inventory records agree with sales", "Inventory consistency", bad,
                   "every sold unit matches its sale (store, dates, costs, vehicle)", "mismatched unit/sale fields")


def check_stock_units(t: Tables) -> CheckResult:
    inv = t["inventory"]
    stock = inv[inv["status"] == "In Stock"]
    age = (AS_OF_DATE - stock["acquisition_date"]).dt.days
    bad = ((stock["acquisition_date"] > AS_OF_DATE) | (age != stock["days_in_inventory"])
           | stock["sale_date"].notna() | ~(stock["list_price"] > 0)).sum()
    return _result("Current stock is valid", "Inventory consistency", bad,
                   f"{len(stock):,} units on hand, ages computed to {AS_OF_DATE.date()}", "invalid in-stock units")


ALL_CHECKS: list[Callable[[Tables], CheckResult]] = [
    check_unique_sale_ids, check_unique_unit_ids, check_unit_sold_once,
    check_foreign_keys, check_salesperson_store_and_tenure, check_denormalized_attributes,
    check_sale_after_acquisition, check_sale_dates_in_window, check_model_years,
    check_non_negative_values, check_mileage_semantics, check_price_ranges,
    check_gross_profit_identity, check_margin_identity, check_contribution_identity, check_opex_identity,
    check_inventory_matches_sales, check_stock_units,
]


def run_all_checks(tables: Tables) -> list[CheckResult]:
    out = []
    for check in ALL_CHECKS:
        try:
            out.append(check(tables))
        except Exception as exc:  # a crashing check is a failing check, never a silent pass
            out.append(CheckResult(check.__name__, "Error", False, -1, f"check raised {type(exc).__name__}: {exc}"))
    return out
