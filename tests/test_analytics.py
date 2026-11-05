"""Unit tests for the analytics engine using small hand-built frames with known answers."""

import math

import numpy as np
import pandas as pd
import pytest

from ironledger import analytics as A


def make_sales(rows):
    """Build a minimal sales frame. Each row: (date, store, mfr, cond, category, price, acq, recon, transport)."""
    recs = []
    for i, (date, store, mfr, cond, cat, price, acq, recon, trans) in enumerate(rows):
        gross = price - acq - recon - trans
        recs.append({
            "sale_id": f"S{i:03d}", "sale_date": pd.Timestamp(date), "dealership_id": store, "manufacturer": mfr,
            "condition": cond, "vehicle_category": cat, "sale_price": price, "acquisition_cost": acq,
            "reconditioning_cost": recon, "transportation_cost": trans, "vehicle_gross_profit": gross,
            "finance_income": 1_000 if i % 2 == 0 else 0, "warranty_income": 500,
            "total_gross_contribution": gross + (1_000 if i % 2 == 0 else 0) + 500,
            "days_in_inventory": 10 * (i + 1), "financing_arranged": i % 2 == 0, "protection_product": "GAP",
            "customer_id": f"C{i % 2}", "customer_segment": "Small Fleet", "salesperson_id": f"SP{i % 2}",
            "sales_commission": 300,
        })
    return pd.DataFrame(recs)


@pytest.fixture
def sales():
    return make_sales([
        ("2024-01-15", "D01", "Peterbilt", "New", "Sleeper Tractor", 200_000, 180_000, 1_000, 3_000),
        ("2024-02-10", "D01", "Kenworth", "Used", "Day Cab Tractor", 100_000, 85_000, 4_000, 1_000),
        ("2025-01-20", "D02", "Peterbilt", "Used", "Dump Truck", 80_000, 82_000, 2_000, 1_000),   # a loss
        ("2025-02-05", "D02", "Mack", "New", "Sleeper Tractor", 150_000, 140_000, 500, 2_500),
    ])


# ---------------------------------------------------------------------------- core KPIs
def test_revenue_units_and_gross(sales):
    assert A.total_revenue(sales) == 530_000
    assert A.units_sold(sales) == 4
    # gross = Σ(price − acquisition − recon − transport)
    expected_gross = (16_000) + (10_000) + (-5_000) + (7_000)
    assert A.vehicle_gross_profit(sales) == expected_gross
    assert A.cost_of_vehicles_sold(sales) + A.vehicle_gross_profit(sales) == A.total_revenue(sales)


def test_margin_asp_gpu_days(sales):
    assert A.gross_margin(sales) == pytest.approx(28_000 / 530_000)
    assert A.average_selling_price(sales) == pytest.approx(132_500)
    assert A.gross_per_unit(sales) == pytest.approx(7_000)
    assert A.average_days_to_sell(sales) == pytest.approx(25)


def test_fi_income_is_separate_from_revenue(sales):
    k = A.kpi_summary(sales)
    assert k["fi_income"] == 2 * 1_000 + 4 * 500
    assert k["total_contribution"] == k["gross"] + k["fi_income"]
    assert k["revenue"] == 530_000  # F&I never added into revenue
    assert k["loss_rate"] == pytest.approx(0.25)


def test_empty_frame_is_safe(sales):
    empty = sales.iloc[0:0]
    k = A.kpi_summary(empty)
    assert k["revenue"] == 0 and k["units"] == 0
    assert math.isnan(k["margin"]) and math.isnan(k["asp"]) and math.isnan(k["days_to_sell"])
    assert A.performance_by(empty, "manufacturer").empty


def test_safe_div_and_pct_change():
    assert math.isnan(A.safe_div(5, 0))
    assert math.isnan(A.safe_div(5, float("nan")))
    assert A.pct_change(110, 100) == pytest.approx(0.10)
    assert A.pct_change(-50, -100) == pytest.approx(0.50)  # improvement from a larger loss
    assert math.isnan(A.pct_change(10, 0))


# ---------------------------------------------------------------------------- period comparison
def test_prior_period_modes():
    assert A.prior_period("2025-01-01", "2025-03-31", "yoy") == (pd.Timestamp("2024-01-01"),
                                                                  pd.Timestamp("2024-03-31"))
    ps, pe = A.prior_period("2025-01-11", "2025-01-20", "sequential")
    assert (ps, pe) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-10"))
    assert (pe - ps) == (pd.Timestamp("2025-01-20") - pd.Timestamp("2025-01-11"))


def test_compare_periods_yoy_growth(sales):
    cmp = A.compare_periods(sales, "2025-01-01", "2025-02-28", mode="yoy")
    assert cmp.comparable
    assert cmp.current["revenue"] == 230_000 and cmp.prior["revenue"] == 300_000
    assert cmp.change("revenue") == pytest.approx(230_000 / 300_000 - 1)
    assert cmp.change("units") == pytest.approx(0.0)
    # margins compare in percentage points, not relative %
    assert cmp.change("margin") == pytest.approx(2_000 / 230_000 - 26_000 / 300_000)


def test_compare_periods_before_history_is_not_comparable(sales):
    cmp = A.compare_periods(sales, "2021-01-01", "2021-06-30", mode="yoy")
    assert not cmp.comparable
    assert math.isnan(cmp.change("revenue"))


def test_compare_periods_clips_to_as_of(sales):
    cmp = A.compare_periods(sales, "2026-01-01", "2027-12-31", mode="yoy")
    assert cmp.current_end == A.AS_OF_DATE
    assert cmp.prior_end == A.AS_OF_DATE - pd.DateOffset(years=1)  # equal-length windows


def test_same_store_excludes_stores_opened_later(sales):
    dealers = pd.DataFrame({"dealership_id": ["D01", "D02"],
                            "opened_date": pd.to_datetime(["2010-01-01", "2024-06-01"])})
    cmp = A.compare_periods(sales, "2025-01-01", "2025-02-28", dealerships=dealers, same_store=True)
    assert cmp.current["units"] == 0  # D02 opened inside the prior window -> excluded
    assert cmp.prior["units"] == 2


# ---------------------------------------------------------------------------- filters
def test_filter_dimensions_and_dates(sales):
    f = A.FilterState(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2025-12-31"),
                      manufacturers=("Peterbilt",))
    out = A.apply_filters(sales, f)
    assert set(out["manufacturer"]) == {"Peterbilt"} and len(out) == 2
    f2 = A.FilterState(start=pd.Timestamp("2024-01-15"), end=pd.Timestamp("2024-02-10"))
    assert len(A.apply_filters(sales, f2)) == 2  # both endpoints inclusive
    f3 = A.FilterState(dealerships=("D02",), conditions=("Used",))
    assert list(A.apply_filters(sales, f3)["sale_id"]) == ["S002"]
    f4 = A.FilterState(categories=("Box Truck",))
    assert A.apply_filters(sales, f4).empty


def test_product_filter_flag():
    assert not A.FilterState(dealerships=("D01",)).has_product_filter
    assert A.FilterState(conditions=("Used",)).has_product_filter
    assert A.FilterState(categories=("Dump Truck",)).has_product_filter


def test_operating_expense_proration():
    opex = pd.DataFrame({"dealership_id": ["D01", "D01"], "month": pd.to_datetime(["2025-01-01", "2025-02-01"])})
    for col in A.OPEX_LINES:
        opex[col] = 3_100.0
    opex["total_operating_expense"] = 3_100.0 * len(A.OPEX_LINES)
    f = A.FilterState(start=pd.Timestamp("2025-01-17"), end=pd.Timestamp("2025-02-28"))
    out = A.filter_operating_expenses(opex, f)
    # Jan 17–31 = 15 of 31 days; February fully covered
    assert out["payroll_expense"].tolist() == pytest.approx([3_100 * 15 / 31, 3_100])
    assert out["total_operating_expense"].sum() == pytest.approx(out[list(A.OPEX_LINES)].sum().sum())


# ---------------------------------------------------------------------------- time series
def test_monthly_summary_fills_gaps_and_flags_partials(sales):
    m = A.monthly_summary(sales, "2024-01-10", "2024-04-30")
    assert list(m["month"].dt.month) == [1, 2, 3, 4]
    assert m.loc[m["month"].dt.month == 3, "units"].item() == 0
    assert m["is_partial"].tolist() == [True, False, False, False]
    assert m["revenue"].sum() == 300_000


def test_complete_quarters():
    df = pd.DataFrame({"quarter": pd.to_datetime(["2026-01-01", "2026-04-01", "2026-07-01", "2026-10-01"])})
    out = A.complete_quarters(df, "quarter", "2026-01-01", "2026-12-31")
    assert len(out) == 3  # Q4 2026 runs past the as-of date
    out = A.complete_quarters(df, "quarter", "2026-02-01", "2026-09-30")
    assert list(out["quarter"].dt.month) == [4, 7]


# ---------------------------------------------------------------------------- inventory
def test_aging_bucket_boundaries():
    b = A.aging_bucket(pd.Series([0, 30, 31, 60, 61, 90, 91, 120, 121, 400]))
    assert list(b.astype(str)) == ["0–30 days", "0–30 days", "31–60 days", "31–60 days", "61–90 days",
                                   "61–90 days", "91–120 days", "91–120 days", "Over 120 days", "Over 120 days"]


def test_inventory_trend_from_unit_records():
    inv = pd.DataFrame({
        "acquisition_date": pd.to_datetime(["2025-01-05", "2025-01-20", "2025-02-10"]),
        "sale_date": pd.to_datetime(["2025-02-15", None, "2025-02-20"]),
        "total_cost_basis": [100.0, 200.0, 300.0],
    })
    t = A.inventory_trend(inv, "2025-01-01", "2025-03-31")
    assert t["units"].tolist() == [2, 1, 1]  # Jan: both; Feb: only unsold unit; Mar: same
    assert t["cost_basis"].tolist() == [300.0, 200.0, 200.0]


def test_inventory_summary_months_supply():
    stock = pd.DataFrame({"acquisition_cost": [100.0] * 6, "transportation_cost": [0.0] * 6,
                          "total_cost_basis": [110.0] * 6, "list_price": [130.0] * 6,
                          "days_in_inventory": [10, 40, 70, 95, 130, 200]})
    sales = pd.DataFrame({"sale_date": pd.date_range(end=A.AS_OF_DATE, periods=9, freq="10D")})
    s = A.inventory_summary(stock, sales, floorplan_rate=0.12)
    assert s["units"] == 6 and s["units_over_120"] == 2
    assert s["pct_over_90"] == pytest.approx(0.5)
    assert s["months_supply"] == pytest.approx(6 / (9 / (90 / 30.4375)))
    assert s["monthly_carry_cost"] == pytest.approx(600 * 0.12 / 12)
    assert s["potential_gross"] == pytest.approx(6 * 20)


# ---------------------------------------------------------------------------- performance & customers
def test_performance_by_and_leaderboard(sales):
    p = A.performance_by(sales, "manufacturer").set_index("manufacturer")
    assert p.loc["Peterbilt", "units"] == 2 and p.loc["Peterbilt", "revenue"] == 280_000
    assert p["revenue_share"].sum() == pytest.approx(1.0)
    lb = A.leaderboard(p.reset_index(), "Avg gross per unit", min_units=1)
    assert lb.iloc[0]["manufacturer"] == "Kenworth" and lb["rank"].tolist() == [1, 2, 3]
    assert len(A.leaderboard(p.reset_index(), "Units sold", min_units=2)) == 1


def test_revenue_concentration_equal_customers():
    cs = pd.DataFrame({"revenue": [100.0] * 20})
    c = A.revenue_concentration(cs)
    assert c["top10_share"] == pytest.approx(0.5)
    assert c["hhi"] == pytest.approx(500)  # 20 equal shares of 5% -> 20 * 25
    assert c["curve"]["revenue_share"].iloc[-1] == pytest.approx(1.0)


def test_repeat_customer_stats():
    cs = pd.DataFrame({"units": [1, 2, 5, 1], "revenue": [10.0, 20.0, 50.0, 20.0]})
    r = A.repeat_customer_stats(cs)
    assert r["buyers"] == 4 and r["repeat_buyers"] == 2
    assert r["repeat_rate"] == pytest.approx(0.5) and r["repeat_revenue_share"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------- statements & utilities
def test_pnl_statement_reconciles(sales):
    opex = pd.DataFrame({c: [1_000.0] for c in A.OPEX_LINES})
    opex["total_operating_expense"] = 1_000.0 * len(A.OPEX_LINES)
    pnl = A.pnl_statement(sales, opex).set_index("line_item")["amount"]
    cost_lines = pnl[["Vehicle acquisition cost", "Reconditioning & make-ready", "Inbound transportation"]].sum()
    assert pnl["Vehicle sales revenue"] + cost_lines == pnl["Vehicle gross profit"]
    assert pnl["Total gross contribution"] == pnl["Vehicle gross profit"] + pnl["Finance reserve income"] + \
        pnl["Protection product income"]
    assert pnl["Est. operating income"] == pnl["Total gross contribution"] + pnl["Total operating expenses"]
    no_opex = A.pnl_statement(sales, None)
    assert "Est. operating income" not in set(no_opex["line_item"])  # never invented without expense data


def test_search_frame(sales):
    assert len(A.search_frame(sales, "peterbilt")) == 2
    assert len(A.search_frame(sales, "")) == len(sales)
    assert A.search_frame(sales, "zzz-not-there").empty


def test_profile_table_counts_missing():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
    p = A.profile_table(df).set_index("column")
    assert p.loc["a", "missing"] == 1 and p.loc["b", "non_null"] == 2
    assert p.loc["a", "missing_pct"] == pytest.approx(1 / 3)
