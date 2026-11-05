"""Page 1 — Executive Overview: group-level KPIs, growth and mix."""

import pandas as pd
import streamlit as st

from ironledger import analytics as A
from ironledger.ui import charts as C
from ironledger.ui import components as U
from ironledger.ui import theme as T
from ironledger.ui.filters import dealer_names
from ironledger.ui.page import setup

ds, f = setup()
U.page_header("Overview", "Executive Overview",
              "Vehicle sales performance across the dealership network, with period-over-period comparisons.")

s_dim = A.filter_dimensions(ds.sales, f)
s = A.filter_dates(s_dim, f.start, f.end)
U.filter_context(f, dealer_names(ds), len(s), len(ds.sales))

if s.empty:
    U.empty_state()
    st.stop()

# ---- comparison controls -------------------------------------------------------
c1, c2, _ = st.columns([2.2, 1.6, 4], vertical_alignment="center")
mode_label = c1.segmented_control("Compare against", ["Prior year", "Prior period"], default="Prior year",
                                  key="exec_cmp_mode", label_visibility="collapsed") or "Prior year"
same_store = c2.toggle("Same-store basis", key="exec_same_store",
                       help="Only include dealerships that were open for the entire comparison period, so growth "
                            "from newly opened locations (Denver 2022, Nashville 2023) doesn't inflate results.")
cmp = A.compare_periods(s_dim, f.start, f.end, mode="yoy" if mode_label == "Prior year" else "sequential",
                        dealerships=ds.dealerships, same_store=same_store)
cur = cmp.current
U.note((cmp.note if cmp.comparable else f"⚠ {cmp.note}")
       + (" — KPI cards use the same-store basis; charts include every store in scope." if same_store else ""))

monthly = A.monthly_summary(s, f.start, f.end)
spark_rev = monthly["revenue"].tail(24)
spark_gross = monthly["gross"].tail(24)

# ---- KPI cards ------------------------------------------------------------------
r1 = st.columns(4)
U.kpi(r1[0], "Vehicle sales revenue", U.money(cur["revenue"]), U.signed_pct(cmp.change("revenue")),
      spark=spark_rev, help="Σ selling price. Excludes F&I income.")
U.kpi(r1[1], "Vehicle gross profit", U.money(cur["gross"]), U.signed_pct(cmp.change("gross")),
      spark=spark_gross, help="Revenue − acquisition − reconditioning − transportation. Not net profit.")
U.kpi(r1[2], "Vehicle gross margin", U.pct(cur["margin"]), U.signed_pts(cmp.change("margin")),
      spark=monthly["margin"].tail(24), spark_type="line", help="Vehicle gross profit ÷ revenue.")
U.kpi(r1[3], "Units sold", U.number(cur["units"]), U.signed_pct(cmp.change("units")),
      spark=monthly["units"].tail(24), spark_type="bar")

r2 = st.columns(4)
U.kpi(r2[0], "Average selling price", U.money(cur["asp"], decimals=1), U.signed_pct(cmp.change("asp")))
U.kpi(r2[1], "Avg days to sell", U.number(cur["days_to_sell"], 1), U.signed_pct(cmp.change("days_to_sell")),
      inverse=True, help="Average days from inventory acquisition to sale. Lower is better.")
U.kpi(r2[2], "Revenue growth", U.signed_pct(cmp.change("revenue")) or "n/a",
      f"Prior: {U.money(cmp.prior.get('revenue'))}" if cmp.comparable else "No comparable prior period",
      neutral=True, help=f"Current period revenue vs comparable prior period ({cmp.note}).")
U.kpi(r2[3], "Gross profit growth", U.signed_pct(cmp.change("gross")) or "n/a",
      f"Prior: {U.money(cmp.prior.get('gross'))}" if cmp.comparable else "No comparable prior period",
      neutral=True, help=f"Vehicle gross profit vs comparable prior period ({cmp.note}).")

# ---- trends ---------------------------------------------------------------------------
U.section("Monthly trend", "bars = selected period · line = same month one year earlier")
all_monthly = A.monthly_summary(s_dim)
m = A.add_prior_year(A.add_prior_year(monthly, all_monthly, "revenue"), all_monthly, "gross")
left, right = st.columns(2)
with left:
    C.show(C.monthly_bars(m, "revenue", prior_col="revenue_prior_year", title="Monthly vehicle sales revenue"))
with right:
    C.show(C.monthly_bars(m, "gross", prior_col="gross_prior_year", title="Monthly vehicle gross profit",
                          color=T.CATEGORICAL[0]))

# ---- breakdowns -----------------------------------------------------------------------------
U.section("Where revenue comes from")
dp = A.dealership_performance(s, ds.dealerships)
dp["label"] = dp["city"]
mp = A.performance_by(s, "manufacturer")
left, right = st.columns(2)
with left:
    C.show(C.ranked_bars(dp, "label", "revenue", title="Revenue by dealership",
                         hover_cols={"dealership_name": "Store", "units": "Units"}), height=380)
with right:
    C.show(C.ranked_bars(mp, "manufacturer", "revenue", title="Revenue by manufacturer",
                         hover_cols={"units": "Units"}), height=380)

left, right = st.columns(2)
with left:
    cp = A.performance_by(s, ["vehicle_category"])
    C.show(C.ranked_bars(cp, "vehicle_category", "units", title="Units sold by vehicle category", money=False,
                         color=T.CATEGORICAL[0], hover_cols={"revenue": "Revenue ($)"}), height=360)
with right:
    mix = A.mix_over_time(s, "condition", freq="Q")
    new_share = (s["condition"] == "New").mean()
    C.show(C.stacked_share(mix, "period", "condition", colors=T.CONDITION_COLORS, order=["New", "Used"],
                           title="New vs used mix (share of units, quarterly)",
                           subtitle=f"New {new_share:.0%} · Used {1 - new_share:.0%} of units in period"),
           height=360)

# ---- highlights -----------------------------------------------------------------------------------
U.section("Highlights", "computed from the filtered data")
cond = A.performance_by(s, "condition").set_index("condition")


def md(txt: str) -> str:
    return txt.replace("$", r"\$")  # a bare "$" would start LaTeX math in st.markdown


h1, h2, h3 = st.columns(3)
with h1.container(border=True):
    top = dp.iloc[0]
    st.markdown(md(f"**Top store:** {top['dealership_name']} ({top['city']}) generated "
                f"**{U.money(top['revenue'])}**, {top['revenue_share']:.0%} of revenue in scope."))
with h2.container(border=True):
    best = mp[mp["units"] >= 30].sort_values("gpu", ascending=False)
    if not best.empty:
        b = best.iloc[0]
        st.markdown(md(f"**Richest gross per unit:** {b['manufacturer']} at **{U.money(b['gpu'])}** per unit "
                    f"({U.pct(b['margin'])} margin) across {b['units']:,} units."))
with h3.container(border=True):
    if "Used" in cond.index:
        u = cond.loc["Used"]
        st.markdown(md(f"**Used-vehicle risk:** {u['loss_rate']:.0%} of used deals sold below total cost; "
                    f"used gross per unit **{U.money(u['gpu'])}** vs "
                    f"{U.money(cond.loc['New', 'gpu']) if 'New' in cond.index else '—'} for new."))

U.definitions_expander()
