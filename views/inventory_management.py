"""Page 3 — Inventory Management: current stock, aging, carrying cost, unit search.

All figures derive from the unit master (inventory.status == 'In Stock'), the same
records that sales are drawn from — never from independent totals.
"""

import streamlit as st

from ironledger import analytics as A
from ironledger.reference_data import market_index
from ironledger.config import AS_OF_DATE
from ironledger.ui import charts as C
from ironledger.ui import components as U
from ironledger.ui import theme as T
from ironledger.ui.filters import dealer_names
from ironledger.ui.page import setup

ds, f = setup()
U.page_header("Operations", "Inventory Management",
              "Units on the lot, how long they've been there, and what they cost to carry.")

inv_dim = A.filter_dimensions(ds.inventory, f)
stock = A.current_inventory(inv_dim).copy()
all_stock = A.current_inventory(ds.inventory)
U.filter_context(f, dealer_names(ds), len(stock), len(all_stock), dates_apply=False,
                 extra="inventory is a point-in-time snapshot; store/brand/condition/category filters apply")
if stock.empty:
    U.empty_state("No units in stock", "No units on hand match the current filters.")
    st.stop()

stock["aging_bucket"] = A.aging_bucket(stock["days_in_inventory"])
rate = float(market_index([AS_OF_DATE], "floorplan_rate")[0])
summary = A.inventory_summary(stock, A.filter_dimensions(ds.sales, f), floorplan_rate=rate)

r1 = st.columns(4)
U.kpi(r1[0], "Units in stock", U.number(summary["units"]))
U.kpi(r1[1], "Inventory cost basis", U.money(summary["cost_basis"]),
      help="Acquisition + reconditioning + transportation capitalized into the units on hand.")
U.kpi(r1[2], "Average age", f"{summary['avg_age']:.0f} days", help="Days since acquisition, as of the snapshot.")
U.kpi(r1[3], "Months' supply", U.number(summary["months_supply"], 1),
      help="Units on hand ÷ average monthly units sold over the last 90 days (same filters).")
r2 = st.columns(4)
U.kpi(r2[0], "Acquisition cost", U.money(summary["acquisition_cost"]), help="Purchase/invoice cost only.")
U.kpi(r2[1], "Aged over 90 days", U.pct(summary["pct_over_90"]),
      f"{summary['units_over_120']:,} units over 120 days", neutral=True)
U.kpi(r2[2], "Floorplan carry / mo", U.money(summary["monthly_carry_cost"]),
      help=f"(Acquisition + transport cost) × {rate:.2%} annual floorplan rate ÷ 12.")
U.kpi(r2[3], "Potential front gross", U.money(summary["potential_gross"]),
      help="Σ (list price − cost basis) if every unit sold at sticker. An upper bound, not a forecast.")

U.section("Aging profile", "older buckets = brighter = more carrying cost and markdown risk")
left, right = st.columns([1, 1.3])
with left:
    C.show(C.aging_bars(A.aging_distribution(stock), A.AGING_BUCKETS, title="Units by days in stock"), height=380)
with right:
    by_store = A.aging_distribution(stock.assign(store=stock["city"].astype(str)), by="store")
    C.show(C.stacked_bars_h(by_store, "store", "aging_bucket", "units",
                            colors=dict(zip(A.AGING_BUCKETS, T.AGING_COLORS)), order=A.AGING_BUCKETS,
                            title="Aging by dealership"), height=380)

U.section("Stock composition")
left, right = st.columns(2)
with left:
    g = stock.groupby([stock["manufacturer"].astype(str), stock["condition"].astype(str)]).size()
    df = g.rename("units").reset_index()
    C.show(C.stacked_bars_h(df, "manufacturer", "condition", "units", colors=T.CONDITION_COLORS,
                            order=["New", "Used"], title="Units by manufacturer · new vs used"), height=400)
with right:
    g = stock.groupby([stock["vehicle_category"].astype(str), stock["condition"].astype(str)]).size()
    df = g.rename("units").reset_index()
    C.show(C.stacked_bars_h(df, "vehicle_category", "condition", "units", colors=T.CONDITION_COLORS,
                            order=["New", "Used"], title="Units by vehicle type · new vs used"), height=400)

left, right = st.columns(2)
with left:
    val = stock.groupby(stock["city"].astype(str))["total_cost_basis"].sum().rename("cost").reset_index()
    C.show(C.ranked_bars(val, "city", "cost", title="Inventory investment by dealership"), height=380)
with right:
    trend = A.inventory_trend(inv_dim)
    trend["is_partial"] = False
    trend = trend.rename(columns={"month_end": "month"})
    C.show(C.monthly_bars(trend, "units", money=False, color=T.CATEGORICAL[0],
                          title="Units on hand at month-end",
                          subtitle="reconstructed from acquisition & sale dates in the unit master"), height=380)

# ---- unit search ---------------------------------------------------------------------------
U.section("Unit search", "every unit currently on the lot")
c1, c2, c3 = st.columns([2.2, 2, 1.2])
query = c1.text_input("Search", placeholder="Unit ID, model, store, source…", key="inv_q",
                      label_visibility="collapsed")
buckets = c2.multiselect("Aging", A.AGING_BUCKETS, key="inv_buckets", placeholder="All aging buckets",
                         label_visibility="collapsed")
sort = c3.selectbox("Sort", ["Oldest first", "Newest first", "Highest cost", "Lowest cost"], key="inv_sort",
                    label_visibility="collapsed")
view = stock
if buckets:
    view = view[view["aging_bucket"].isin(buckets)]
view = A.search_frame(view, query, ["unit_id", "manufacturer", "model", "dealership_name", "city",
                                    "vehicle_category", "acquisition_source", "condition"])
sort_map = {"Oldest first": ("days_in_inventory", False), "Newest first": ("days_in_inventory", True),
            "Highest cost": ("total_cost_basis", False), "Lowest cost": ("total_cost_basis", True)}
col, asc = sort_map[sort]
view = view.sort_values(col, ascending=asc)
cols = ["unit_id", "manufacturer", "model", "model_year", "vehicle_category", "condition", "mileage",
        "dealership_name", "city", "acquisition_source", "acquisition_date", "days_in_inventory", "aging_bucket",
        "acquisition_cost", "total_cost_basis", "list_price"]
st.dataframe(
    view[cols], hide_index=True, height=440,
    column_config={
        "unit_id": "Unit ID", "manufacturer": "Make", "model": "Model",
        "model_year": st.column_config.NumberColumn("Year", format="%d"),
        "vehicle_category": "Category", "condition": "Cond.",
        "mileage": st.column_config.NumberColumn("Mileage", format="%,d", help="Blank for trailers (not metered)."),
        "dealership_name": "Dealership", "city": "City", "acquisition_source": "Source",
        "acquisition_date": st.column_config.DateColumn("Acquired", format="MMM D, YYYY"),
        "days_in_inventory": st.column_config.NumberColumn("Days", format="%d"),
        "aging_bucket": "Aging",
        "acquisition_cost": st.column_config.NumberColumn("Acq. cost", format="dollar"),
        "total_cost_basis": st.column_config.NumberColumn("Cost basis", format="dollar"),
        "list_price": st.column_config.NumberColumn("List price", format="dollar"),
    },
)
c1, c2 = st.columns([4, 1], vertical_alignment="center")
c1.caption(f"{len(view):,} of {len(stock):,} units shown")
c2.download_button("Export CSV", view[cols].to_csv(index=False).encode(), "ironledger_inventory.csv",
                   "text/csv", icon=":material/download:", width="stretch")

aged = stock[stock["days_in_inventory"] > 120].sort_values("days_in_inventory", ascending=False)
with st.expander(f"Aged-unit watchlist — {len(aged):,} units over 120 days", icon=":material/warning:"):
    carry = (aged["acquisition_cost"] + aged["transportation_cost"]) * rate / 365 * aged["days_in_inventory"]
    st.dataframe(
        aged.assign(carry_to_date=carry)[["unit_id", "manufacturer", "model", "model_year", "condition", "city",
                                          "days_in_inventory", "total_cost_basis", "list_price", "carry_to_date"]],
        hide_index=True,
        column_config={
            "days_in_inventory": st.column_config.NumberColumn("Days", format="%d"),
            "model_year": st.column_config.NumberColumn("Year", format="%d"),
            "total_cost_basis": st.column_config.NumberColumn("Cost basis", format="dollar"),
            "list_price": st.column_config.NumberColumn("List price", format="dollar"),
            "carry_to_date": st.column_config.NumberColumn("Est. floorplan interest to date", format="dollar"),
        },
    )
