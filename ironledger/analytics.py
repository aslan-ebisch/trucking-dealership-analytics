"""Business and financial calculations.

Pure functions over pandas DataFrames — no Streamlit, no I/O — so every number
on the dashboard can be unit-tested.

Metric definitions (also shown in the app and README)
-----------------------------------------------------
Vehicle sales revenue     Σ sale_price
Cost of vehicles sold     Σ (acquisition_cost + reconditioning_cost + transportation_cost)
Vehicle gross profit      revenue − cost of vehicles sold   (front-end gross)
Vehicle gross margin      vehicle gross profit ÷ revenue
F&I income                Σ (finance_income + warranty_income)   (back-end gross;
                          NOT part of revenue, so it is never double counted)
Total gross contribution  vehicle gross profit + F&I income
Operating expenses        payroll (incl. commissions) + facility + marketing + utilities
                          + floorplan interest + other   (sales department only)
Est. operating income     total gross contribution − operating expenses
Average selling price     revenue ÷ units
Gross per unit (GPU)      vehicle gross profit ÷ units
Days to sell              mean(sale_date − inventory acquisition date)
Growth                    (current − prior) ÷ |prior|; margins compare in points

Gross profit is not net profit: operating income here excludes service/parts
departments, corporate overhead, depreciation and income taxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ironledger.config import AS_OF_DATE, HISTORY_START

AGING_BUCKETS = ["0–30 days", "31–60 days", "61–90 days", "91–120 days", "Over 120 days"]
_AGING_EDGES = [-1, 30, 60, 90, 120, np.inf]

OPEX_LINES = {
    "payroll_expense": "Payroll & commissions",
    "facility_expense": "Facility & occupancy",
    "marketing_expense": "Marketing",
    "utilities_expense": "Utilities",
    "floorplan_interest_expense": "Floorplan interest",
    "other_operating_expense": "Other operating",
}


# ---------------------------------------------------------------------------
# Safe math
# ---------------------------------------------------------------------------
def safe_div(num: float, den: float) -> float:
    """Division that returns NaN (not an error or inf) when the denominator is 0/NaN."""
    try:
        if den is None or pd.isna(den) or den == 0:
            return float("nan")
        return float(num) / float(den)
    except (TypeError, ZeroDivisionError):
        return float("nan")


def pct_change(current: float, prior: float) -> float:
    """Relative change; NaN when the prior value is zero/missing (growth is undefined)."""
    return safe_div(current - prior, abs(prior)) if prior is not None and not pd.isna(prior) else float("nan")


def _group(df: pd.DataFrame, by):
    return df.groupby(by, observed=True, sort=False)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterState:
    """Global filter selection. Empty tuples mean 'all'."""

    start: pd.Timestamp = HISTORY_START
    end: pd.Timestamp = AS_OF_DATE
    dealerships: tuple[str, ...] = ()
    manufacturers: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    categories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_product_filter(self) -> bool:
        """True when filters slice *within* a store (store-level expenses then can't be allocated)."""
        return bool(self.manufacturers or self.conditions or self.categories)

    @property
    def is_full_range(self) -> bool:
        return self.start <= HISTORY_START and self.end >= AS_OF_DATE


def filter_dimensions(df: pd.DataFrame, f: FilterState, *, include_dealer: bool = True) -> pd.DataFrame:
    """Apply dealership / manufacturer / condition / category filters (not dates)."""
    mask = pd.Series(True, index=df.index)
    if include_dealer and f.dealerships and "dealership_id" in df:
        mask &= df["dealership_id"].isin(f.dealerships)
    if f.manufacturers and "manufacturer" in df:
        mask &= df["manufacturer"].isin(f.manufacturers)
    if f.conditions and "condition" in df:
        mask &= df["condition"].isin(f.conditions)
    if f.categories and "vehicle_category" in df:
        mask &= df["vehicle_category"].isin(f.categories)
    return df[mask]


def filter_dates(df: pd.DataFrame, start, end, col: str = "sale_date") -> pd.DataFrame:
    """Inclusive date-range filter (whole days)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return df[(df[col] >= start) & (df[col] <= end)]


def apply_filters(sales: pd.DataFrame, f: FilterState) -> pd.DataFrame:
    return filter_dates(filter_dimensions(sales, f), f.start, f.end)


def filter_operating_expenses(opex: pd.DataFrame, f: FilterState) -> pd.DataFrame:
    """Store filter + date range. Months partly inside the range are prorated by days,
    so expenses always cover exactly the same days as the sales they are matched with."""
    o = opex[opex["dealership_id"].isin(f.dealerships)] if f.dealerships else opex
    start, end = pd.Timestamp(f.start).normalize(), pd.Timestamp(f.end).normalize()
    month_end = o["month"] + pd.offsets.MonthEnd(0)
    overlap_start = o["month"].where(o["month"] > start, start)
    overlap_end = month_end.where(month_end < end, end)
    days_in = (overlap_end - overlap_start).dt.days + 1
    frac = (days_in.clip(lower=0) / o["month"].dt.days_in_month).clip(upper=1.0)
    o = o[frac > 0].copy()
    frac = frac[frac > 0]
    cols = list(OPEX_LINES) + ["total_operating_expense"]
    o[cols] = o[cols].mul(frac, axis=0)
    o["proration"] = frac
    return o


# ---------------------------------------------------------------------------
# Core KPIs
# ---------------------------------------------------------------------------
def total_revenue(s: pd.DataFrame) -> float:
    return float(s["sale_price"].sum())


def units_sold(s: pd.DataFrame) -> int:
    return int(len(s))


def cost_of_vehicles_sold(s: pd.DataFrame) -> float:
    return float((s["acquisition_cost"] + s["reconditioning_cost"] + s["transportation_cost"]).sum())


def vehicle_gross_profit(s: pd.DataFrame) -> float:
    return float(s["vehicle_gross_profit"].sum())


def gross_margin(s: pd.DataFrame) -> float:
    return safe_div(vehicle_gross_profit(s), total_revenue(s))


def average_selling_price(s: pd.DataFrame) -> float:
    return safe_div(total_revenue(s), units_sold(s))


def gross_per_unit(s: pd.DataFrame) -> float:
    return safe_div(vehicle_gross_profit(s), units_sold(s))


def average_days_to_sell(s: pd.DataFrame) -> float:
    return float(s["days_in_inventory"].mean()) if len(s) else float("nan")


def fi_income(s: pd.DataFrame) -> float:
    return float((s["finance_income"] + s["warranty_income"]).sum())


def total_gross_contribution(s: pd.DataFrame) -> float:
    return vehicle_gross_profit(s) + fi_income(s)


def kpi_summary(s: pd.DataFrame) -> dict[str, float]:
    return {
        "revenue": total_revenue(s),
        "units": units_sold(s),
        "cogs": cost_of_vehicles_sold(s),
        "gross": vehicle_gross_profit(s),
        "margin": gross_margin(s),
        "asp": average_selling_price(s),
        "gpu": gross_per_unit(s),
        "days_to_sell": average_days_to_sell(s),
        "fi_income": fi_income(s),
        "fi_per_unit": safe_div(fi_income(s), units_sold(s)),
        "total_contribution": total_gross_contribution(s),
        "finance_penetration": float(s["financing_arranged"].mean()) if len(s) else float("nan"),
        "protection_penetration": float(s["protection_product"].notna().mean()) if len(s) else float("nan"),
        "loss_rate": float((s["vehicle_gross_profit"] < 0).mean()) if len(s) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Period-over-period comparison
# ---------------------------------------------------------------------------
@dataclass
class PeriodComparison:
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    prior_start: pd.Timestamp
    prior_end: pd.Timestamp
    current: dict
    prior: dict
    comparable: bool
    note: str

    def change(self, metric: str) -> float:
        """Relative change for amounts; percentage-point change for ratio metrics."""
        if not self.comparable:
            return float("nan")
        cur, pri = self.current.get(metric), self.prior.get(metric)
        if metric in RATIO_METRICS:
            return float("nan") if pd.isna(cur) or pd.isna(pri) else float(cur - pri)
        return pct_change(cur, pri)


RATIO_METRICS = {"margin", "finance_penetration", "protection_penetration", "loss_rate"}


def prior_period(start, end, mode: str = "yoy") -> tuple[pd.Timestamp, pd.Timestamp]:
    """'yoy' = same calendar dates one year earlier (removes seasonality);
    'sequential' = the equally long window immediately before."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if mode == "yoy":
        return start - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
    length = end - start
    p_end = start - pd.Timedelta(days=1)
    return p_end - length, p_end


def compare_periods(dim_filtered_sales: pd.DataFrame, start, end, *, mode: str = "yoy",
                    dealerships: pd.DataFrame | None = None, same_store: bool = False,
                    data_start: pd.Timestamp = HISTORY_START, as_of: pd.Timestamp = AS_OF_DATE) -> PeriodComparison:
    """Compare a window with its prior comparable window.

    * The current window is clipped to the data's as-of date, so a window running
      into the future is never compared against a complete prior window.
    * If the prior window starts before recorded history, growth is reported as
      not available rather than as a misleading jump from zero.
    * ``same_store`` keeps only stores open for the whole prior window, so
      growth from newly opened locations doesn't inflate the comparison.
    """
    start = max(pd.Timestamp(start).normalize(), data_start)
    end = min(pd.Timestamp(end).normalize(), as_of)
    p_start, p_end = prior_period(start, end, mode)
    s = dim_filtered_sales
    note = f"vs {p_start:%b %d, %Y} – {p_end:%b %d, %Y}"
    if same_store and dealerships is not None:
        eligible = dealerships.loc[dealerships["opened_date"] <= p_start, "dealership_id"]
        s = s[s["dealership_id"].isin(eligible)]
        note += " · same-store"
    comparable = p_start >= data_start
    if not comparable:
        note = "Prior period precedes recorded history — growth not available"
    cur = kpi_summary(filter_dates(s, start, end))
    pri = kpi_summary(filter_dates(s, p_start, p_end)) if comparable else {}
    return PeriodComparison(start, end, p_start, p_end, cur, pri, comparable, note)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def monthly_summary(s: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Monthly revenue / gross / units, including empty months, with a flag for
    months only partly covered by the selected window (so charts can mark them)."""
    start = pd.Timestamp(start if start is not None else (s["sale_date"].min() if len(s) else HISTORY_START))
    end = pd.Timestamp(end if end is not None else (s["sale_date"].max() if len(s) else AS_OF_DATE))
    end = min(end, AS_OF_DATE)
    months = pd.date_range(start.to_period("M").to_timestamp(), end.to_period("M").to_timestamp(), freq="MS")
    month_col = s["sale_date"].dt.to_period("M").dt.to_timestamp()
    g = s.groupby(month_col).agg(
        revenue=("sale_price", "sum"), gross=("vehicle_gross_profit", "sum"), units=("sale_id", "count"),
        fi_income=("finance_income", "sum"), warranty=("warranty_income", "sum"),
    )
    g = g.reindex(months, fill_value=0)
    g.index.name = "month"
    g["fi_income"] = g["fi_income"] + g.pop("warranty")
    g["margin"] = np.where(g["revenue"] > 0, g["gross"] / g["revenue"].where(g["revenue"] > 0), np.nan)
    g["gpu"] = np.where(g["units"] > 0, g["gross"] / g["units"].where(g["units"] > 0), np.nan)
    month_end = g.index + pd.offsets.MonthEnd(0)
    g["is_partial"] = (g.index < start.normalize()) | (month_end > end.normalize())
    return g.reset_index()


def add_prior_year(monthly: pd.DataFrame, all_monthly: pd.DataFrame, col: str) -> pd.DataFrame:
    """Attach the same month one year earlier (from an unclipped monthly series)."""
    lookup = all_monthly.set_index("month")[col]
    out = monthly.copy()
    out[f"{col}_prior_year"] = (out["month"] - pd.DateOffset(years=1)).map(lookup)
    return out


def complete_quarters(df: pd.DataFrame, col: str, start, end) -> pd.DataFrame:
    """Keep only quarters fully inside [start, end] (and on/before the as-of date).

    Summed metrics (revenue, units) for a partial quarter look like a collapse —
    e.g. Q4 2026 holds only October — so total-based trend charts drop them.
    """
    q = pd.to_datetime(df[col])
    q_end = q + pd.offsets.QuarterEnd(0)
    last = min(pd.Timestamp(end).normalize(), AS_OF_DATE)
    return df[(q >= pd.Timestamp(start).normalize()) & (q_end <= last)]


def mix_over_time(s: pd.DataFrame, dim: str, freq: str = "Q", value: str = "units") -> pd.DataFrame:
    """Share of units (or revenue) by a dimension per period — long format."""
    period = s["sale_date"].dt.to_period(freq).dt.to_timestamp()
    v = s["sale_price"] if value == "revenue" else pd.Series(1, index=s.index)
    g = v.groupby([period, s[dim].astype(str)]).sum().rename("value").reset_index()
    g.columns = ["period", dim, "value"]
    g["share"] = g["value"] / g.groupby("period")["value"].transform("sum")
    return g


# ---------------------------------------------------------------------------
# Dimension performance
# ---------------------------------------------------------------------------
def performance_by(s: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Standard KPI block for any dimension (dealership, manufacturer, rep, ...)."""
    cols = [by] if isinstance(by, str) else list(by)
    if s.empty:
        return pd.DataFrame(columns=cols + ["units", "revenue", "gross", "margin", "gpu", "asp", "avg_days",
                                           "fi_income", "total_contribution", "revenue_share", "loss_rate"])
    tmp = s.assign(_fi=s["finance_income"] + s["warranty_income"], _loss=(s["vehicle_gross_profit"] < 0))
    g = _group(tmp, cols).agg(
        units=("sale_price", "size"), revenue=("sale_price", "sum"), gross=("vehicle_gross_profit", "sum"),
        avg_days=("days_in_inventory", "mean"), fi_income=("_fi", "sum"), loss_rate=("_loss", "mean"),
    ).reset_index()
    for c in cols:
        g[c] = g[c].astype(str)
    g["margin"] = g["gross"] / g["revenue"]
    g["gpu"] = g["gross"] / g["units"]
    g["asp"] = g["revenue"] / g["units"]
    g["total_contribution"] = g["gross"] + g["fi_income"]
    g["revenue_share"] = g["revenue"] / g["revenue"].sum()
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def dealership_performance(s: pd.DataFrame, dealerships: pd.DataFrame) -> pd.DataFrame:
    p = performance_by(s, "dealership_id")
    d = dealerships[["dealership_id", "dealership_name", "city", "state", "region", "latitude", "longitude",
                     "opened_date"]]
    return p.merge(d, on="dealership_id", how="left")


def salesperson_performance(s: pd.DataFrame, salespeople: pd.DataFrame) -> pd.DataFrame:
    p = performance_by(s, "salesperson_id")
    if not s.empty:
        comm = _group(s, "salesperson_id")["sales_commission"].sum()
        comm.index = comm.index.astype(str)
        p["commission"] = p["salesperson_id"].map(comm)
    else:
        p["commission"] = pd.Series(dtype=float)
    sp = salespeople[["salesperson_id", "salesperson_name", "dealership_id", "dealership_name", "role",
                      "employment_status", "hire_date"]]
    return p.merge(sp, on="salesperson_id", how="left")


LEADERBOARD_SORTS = {
    "Revenue": "revenue",
    "Units sold": "units",
    "Vehicle gross profit": "gross",
    "Avg gross per unit": "gpu",
}


def leaderboard(perf: pd.DataFrame, sort_label: str, min_units: int = 0, top_n: int | None = None) -> pd.DataFrame:
    col = LEADERBOARD_SORTS[sort_label]
    out = perf[perf["units"] >= min_units].sort_values([col, "revenue"], ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.head(top_n) if top_n else out


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def current_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    """Units physically in stock at the as-of date (derived from the unit master)."""
    return inv[inv["status"] == "In Stock"]


def aging_bucket(days: pd.Series) -> pd.Series:
    return pd.cut(days, bins=_AGING_EDGES, labels=AGING_BUCKETS)


def inventory_summary(stock: pd.DataFrame, sales: pd.DataFrame, as_of: pd.Timestamp = AS_OF_DATE,
                      floorplan_rate: float | None = None, lookback_days: int = 90) -> dict[str, float]:
    """KPIs for current stock. ``sales`` should be filtered by the same dimensions
    (store/brand/condition/category) so months-of-supply compares like with like."""
    recent = sales[sales["sale_date"] > as_of - pd.Timedelta(days=lookback_days)]
    monthly_pace = len(recent) / (lookback_days / 30.4375)
    units = len(stock)
    cost_basis = float(stock["total_cost_basis"].sum())
    out = {
        "units": units,
        "acquisition_cost": float(stock["acquisition_cost"].sum()),
        "cost_basis": cost_basis,
        "list_value": float(stock["list_price"].sum()),
        "avg_age": float(stock["days_in_inventory"].mean()) if units else float("nan"),
        "pct_over_90": float((stock["days_in_inventory"] > 90).mean()) if units else float("nan"),
        "units_over_120": int((stock["days_in_inventory"] > 120).sum()),
        "months_supply": safe_div(units, monthly_pace),
        "monthly_sales_pace": monthly_pace,
        "potential_gross": float(stock["list_price"].sum() - cost_basis),
    }
    out["monthly_carry_cost"] = (float((stock["acquisition_cost"] + stock["transportation_cost"]).sum())
                                 * floorplan_rate / 12) if floorplan_rate else float("nan")
    return out


def aging_distribution(stock: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    b = aging_bucket(stock["days_in_inventory"]).rename("aging_bucket")
    keys = [b] if by is None else [stock[by].astype(str), b]
    g = stock.groupby(keys, observed=False).agg(units=("unit_id", "count"), cost_basis=("total_cost_basis", "sum"))
    return g.reset_index()


def inventory_trend(inv: pd.DataFrame, start=HISTORY_START, end=AS_OF_DATE) -> pd.DataFrame:
    """Month-end units and cost basis on hand, reconstructed from the unit master.

    A unit is on hand at date t if acquired on/before t and not yet sold at t.
    This proves inventory history is *derived* from records, not invented.
    """
    month_ends = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")
    if len(month_ends) == 0 or inv.empty:
        return pd.DataFrame(columns=["month_end", "units", "cost_basis"])
    acq = inv["acquisition_date"].to_numpy()
    sold = inv["sale_date"].fillna(pd.Timestamp("2262-01-01")).to_numpy()
    me = month_ends.to_numpy()
    on_hand = (acq[:, None] <= me[None, :]) & (sold[:, None] > me[None, :])
    cost = inv["total_cost_basis"].to_numpy(dtype=float)
    return pd.DataFrame({"month_end": month_ends, "units": on_hand.sum(axis=0), "cost_basis": cost @ on_hand})


# ---------------------------------------------------------------------------
# Financial statements
# ---------------------------------------------------------------------------
def pnl_statement(s: pd.DataFrame, opex: pd.DataFrame | None) -> pd.DataFrame:
    """Sales-department P&L. If ``opex`` is None (e.g. product filters active, so
    store-level expenses cannot be allocated), the statement stops at total gross."""
    revenue = total_revenue(s)
    units = units_sold(s)
    rows = [
        ("Vehicle sales revenue", revenue, "revenue"),
        ("Vehicle acquisition cost", -float(s["acquisition_cost"].sum()), "cost"),
        ("Reconditioning & make-ready", -float(s["reconditioning_cost"].sum()), "cost"),
        ("Inbound transportation", -float(s["transportation_cost"].sum()), "cost"),
        ("Vehicle gross profit", vehicle_gross_profit(s), "subtotal"),
        ("Finance reserve income", float(s["finance_income"].sum()), "income"),
        ("Protection product income", float(s["warranty_income"].sum()), "income"),
        ("Total gross contribution", total_gross_contribution(s), "subtotal"),
    ]
    if opex is not None:
        for col, label in OPEX_LINES.items():
            rows.append((label, -float(opex[col].sum()), "expense"))
        total_opex = float(opex["total_operating_expense"].sum())
        rows.append(("Total operating expenses", -total_opex, "subtotal_expense"))
        rows.append(("Est. operating income", total_gross_contribution(s) - total_opex, "total"))
    df = pd.DataFrame(rows, columns=["line_item", "amount", "kind"])
    df["pct_of_revenue"] = df["amount"] / revenue if revenue else np.nan
    df["per_unit"] = df["amount"] / units if units else np.nan
    return df


def monthly_operating_income(s: pd.DataFrame, opex_f: pd.DataFrame) -> pd.DataFrame:
    month = s["sale_date"].dt.to_period("M").dt.to_timestamp()
    g = s.assign(_c=s["total_gross_contribution"]).groupby(month)["_c"].sum().rename("gross_contribution")
    e = opex_f.groupby("month")["total_operating_expense"].sum().rename("operating_expense")
    out = pd.concat([g, e], axis=1).fillna(0).sort_index()
    out.index.name = "month"
    out["operating_income"] = out["gross_contribution"] - out["operating_expense"]
    return out.reset_index()


def store_operating_income(s: pd.DataFrame, opex_f: pd.DataFrame, dealerships: pd.DataFrame) -> pd.DataFrame:
    g = performance_by(s, "dealership_id")[["dealership_id", "units", "revenue", "gross", "total_contribution"]]
    e = opex_f.groupby("dealership_id")[list(OPEX_LINES) + ["total_operating_expense"]].sum().reset_index()
    out = g.merge(e, on="dealership_id", how="outer").fillna(0)
    out["operating_income"] = out["total_contribution"] - out["total_operating_expense"]
    out["operating_margin"] = out["operating_income"] / out["revenue"].where(out["revenue"] > 0)
    out = out.merge(dealerships[["dealership_id", "dealership_name", "city"]], on="dealership_id", how="left")
    return out.sort_values("operating_income", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
def customer_summary(s: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """One row per purchasing customer within the filtered sales."""
    cols = ["customer_id", "units", "revenue", "gross", "fi_income", "first_purchase", "last_purchase",
            "stores_used"]
    if s.empty:
        return pd.DataFrame(columns=cols + ["customer_name", "customer_segment", "home_state", "fleet_size",
                                           "avg_transaction"])
    tmp = s.assign(_fi=s["finance_income"] + s["warranty_income"], customer_id=s["customer_id"].astype(str))
    g = tmp.groupby("customer_id").agg(
        units=("sale_price", "size"), revenue=("sale_price", "sum"), gross=("vehicle_gross_profit", "sum"),
        fi_income=("_fi", "sum"), first_purchase=("sale_date", "min"), last_purchase=("sale_date", "max"),
        stores_used=("dealership_id", "nunique"),
    ).reset_index()
    c = customers[["customer_id", "customer_name", "customer_segment", "home_state", "fleet_size", "credit_tier",
                   "primary_dealership_name"]]
    g = g.merge(c, on="customer_id", how="left")
    g["avg_transaction"] = g["revenue"] / g["units"]
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def repeat_customer_stats(cs: pd.DataFrame) -> dict[str, float]:
    buyers = len(cs)
    repeat = cs[cs["units"] >= 2]
    return {
        "buyers": buyers,
        "repeat_buyers": len(repeat),
        "repeat_rate": safe_div(len(repeat), buyers),
        "repeat_revenue_share": safe_div(repeat["revenue"].sum(), cs["revenue"].sum()),
        "avg_units_per_buyer": safe_div(cs["units"].sum(), buyers),
        "revenue_per_buyer": safe_div(cs["revenue"].sum(), buyers),
    }


def revenue_concentration(cs: pd.DataFrame) -> dict:
    """Lorenz curve + concentration ratios (top-10 share, top-20% share, HHI)."""
    if cs.empty or cs["revenue"].sum() <= 0:
        return {"curve": pd.DataFrame({"customer_share": [0.0, 1.0], "revenue_share": [0.0, 1.0]}),
                "top10_share": float("nan"), "top20pct_share": float("nan"), "hhi": float("nan"),
                "customers_for_half": 0}
    r = np.sort(cs["revenue"].to_numpy(dtype=float))[::-1]
    shares = r / r.sum()
    cum = np.cumsum(shares)
    n = len(r)
    curve = pd.DataFrame({"customer_share": np.concatenate([[0], np.arange(1, n + 1) / n]),
                          "revenue_share": np.concatenate([[0], cum])})
    return {
        "curve": curve,
        "top10_share": float(shares[:10].sum()),
        "top20pct_share": float(shares[: max(1, int(np.ceil(n * 0.2)))].sum()),
        "hhi": float((shares ** 2).sum() * 10_000),
        "customers_for_half": int(np.searchsorted(cum, 0.5) + 1),
    }


def segment_performance(s: pd.DataFrame, customers_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    p = performance_by(s, "customer_segment")
    if customers_summary is not None and not customers_summary.empty:
        b = customers_summary.groupby("customer_segment").agg(
            buyers=("customer_id", "count"), repeat_buyers=("units", lambda u: int((u >= 2).sum())))
        p = p.merge(b, left_on="customer_segment", right_index=True, how="left")
        p["repeat_rate"] = p["repeat_buyers"] / p["buyers"]
        p["units_per_buyer"] = p["units"] / p["buyers"]
    p["finance_penetration"] = p["customer_segment"].map(
        s.groupby(s["customer_segment"].astype(str))["financing_arranged"].mean()) if not s.empty else np.nan
    return p


def customer_history(s: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    h = s[s["customer_id"].astype(str) == customer_id]
    return h.sort_values("sale_date", ascending=False)


# ---------------------------------------------------------------------------
# Table utilities (Data Explorer, inventory search)
# ---------------------------------------------------------------------------
def search_frame(df: pd.DataFrame, query: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Case-insensitive substring search across text-like columns."""
    q = (query or "").strip().lower()
    if not q or df.empty:
        return df
    cols = list(columns) if columns else [c for c in df.columns
                                          if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype)
                                          or pd.api.types.is_string_dtype(df[c])]
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask |= df[c].astype(str).str.lower().str.contains(q, regex=False, na=False)
    return df[mask]


def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level profile: type, completeness, cardinality, example value."""
    rows = []
    n = len(df)
    for c in df.columns:
        col = df[c]
        missing = int(col.isna().sum())
        non_null = col.dropna()
        rows.append({
            "column": c,
            "dtype": str(col.dtype),
            "non_null": n - missing,
            "missing": missing,
            "missing_pct": missing / n if n else 0.0,
            "unique": int(col.nunique(dropna=True)),
            "example": str(non_null.iloc[0]) if len(non_null) else "—",
        })
    return pd.DataFrame(rows)
