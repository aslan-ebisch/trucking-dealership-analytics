"""Page 2 — Sales Performance: salesperson leaderboard, location comparison, mix."""

import pandas as pd
import streamlit as st

from ironledger import analytics as A
from ironledger.ui import charts as C
from ironledger.ui import components as U
from ironledger.ui import theme as T
from ironledger.ui.filters import dealer_names
from ironledger.ui.page import setup

ds, f = setup()
U.page_header("Operations", "Sales Performance",
              "Who is selling, where, and what — ranked by revenue, volume and profitability.")

s = A.apply_filters(ds.sales, f)
U.filter_context(f, dealer_names(ds), len(s), len(ds.sales))
if s.empty:
    U.empty_state()
    st.stop()

k = A.kpi_summary(s)
sp_perf = A.salesperson_performance(s, ds.salespeople)
cols = st.columns(5)
U.kpi(cols[0], "Units sold", U.number(k["units"]))
U.kpi(cols[1], "Revenue", U.money(k["revenue"]))
U.kpi(cols[2], "Gross / unit", U.money(k["gpu"], decimals=1), help="Vehicle gross profit ÷ units.")
U.kpi(cols[3], "Selling reps", U.number(len(sp_perf)), help="Salespeople with at least one deal in scope.")
months = max(1.0, (min(f.end, s["sale_date"].max()) - max(f.start, s["sale_date"].min())).days / 30.4375)
U.kpi(cols[4], "Units / rep / mo", U.number(k["units"] / max(len(sp_perf), 1) / months, 1),
      help="Average monthly productivity of reps active in scope (industry norm ≈ 3–6).")

tab_lb, tab_loc, tab_mix = st.tabs(["Salesperson leaderboard", "Location comparison", "Mix & trends"])

# =========================================================================================
with tab_lb:
    c1, c2, c3 = st.columns([3, 2, 1.4], vertical_alignment="bottom")
    sort_label = c1.segmented_control("Rank by", list(A.LEADERBOARD_SORTS), default="Revenue",
                                      key="lb_sort") or "Revenue"
    min_units = c2.slider("Minimum units", 0, 100, 10, step=5, key="lb_min",
                          help="Hides reps with very few deals so per-unit averages aren't dominated by noise.")
    active_only = c3.toggle("Active reps only", value=False, key="lb_active")
    lb = A.leaderboard(sp_perf, sort_label, min_units=min_units)
    if active_only:
        lb = lb[lb["employment_status"] == "Active"].copy()
        lb["rank"] = range(1, len(lb) + 1)

    metric_col = A.LEADERBOARD_SORTS[sort_label]
    left, right = st.columns([1.05, 1])
    with left:
        st.dataframe(
            lb[["rank", "salesperson_name", "dealership_name", "role", "units", "revenue", "gross", "gpu", "margin",
                "commission", "employment_status"]],
            hide_index=True, height=520,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "salesperson_name": st.column_config.TextColumn("Salesperson"),
                "dealership_name": st.column_config.TextColumn("Store"),
                "role": st.column_config.TextColumn("Role"),
                "units": st.column_config.NumberColumn("Units", format="%d"),
                "revenue": st.column_config.ProgressColumn("Revenue", format="dollar", min_value=0,
                                                           max_value=float(sp_perf["revenue"].max())),
                "gross": st.column_config.NumberColumn("Vehicle gross", format="dollar"),
                "gpu": st.column_config.NumberColumn("Gross / unit", format="dollar"),
                "margin": st.column_config.NumberColumn("Margin", format="percent"),
                "commission": st.column_config.NumberColumn("Commission", format="dollar"),
                "employment_status": st.column_config.TextColumn("Status"),
            },
        )
    with right:
        C.show(C.ranked_bars(lb.head(15), "salesperson_name", metric_col, money=metric_col != "units",
                             title=f"Top 15 by {sort_label.lower()}",
                             hover_cols={"dealership_name": "Store", "units": "Units"}), height=520)

    U.section("Compare salespeople", "pick up to 5 reps")
    default = list(lb["salesperson_id"].head(3))
    names = dict(zip(sp_perf["salesperson_id"], sp_perf["salesperson_name"] + " · " + sp_perf["dealership_name"]))
    chosen = st.multiselect("Salespeople", list(names), default=default, format_func=names.get, max_selections=5,
                            key="cmp_reps", label_visibility="collapsed")
    if chosen:
        q = s[s["salesperson_id"].isin(chosen)].assign(
            quarter=lambda d: d["sale_date"].dt.to_period("Q").dt.to_timestamp())
        trend = A.performance_by(q, ["quarter", "salesperson_id"])
        trend["quarter"] = pd.to_datetime(trend["quarter"])
        trend = A.complete_quarters(trend, "quarter", f.start, f.end)
        trend["rep"] = trend["salesperson_id"].map(names)
        order = [names[c] for c in chosen]
        rep_colors = {names[c]: T.CATEGORICAL[i] for i, c in enumerate(chosen)}
        l2, r2 = st.columns(2)
        with l2:
            C.show(C.lines(trend, "quarter", "revenue", "rep", money=True, order=order, colors=rep_colors,
                           title="Quarterly revenue", subtitle="complete quarters only"))
        with r2:
            C.show(C.lines(trend, "quarter", "gpu", "rep", money=True, order=order, colors=rep_colors,
                           title="Quarterly gross per unit"))

# =========================================================================================
with tab_loc:
    dp = A.dealership_performance(s, ds.dealerships)
    left, right = st.columns([1.15, 1])
    with left:
        C.show(C.store_map(dp, title="Network footprint", subtitle="bubble area = revenue in scope"), height=430)
    with right:
        C.show(C.bubble(dp, "units", "gpu", "revenue", "city", x_title="Units sold", y_title="Gross per unit",
                        y_money=True, title="Volume vs profitability",
                        subtitle="upper-right = high volume and rich gross"), height=430)
    st.dataframe(
        dp[["dealership_name", "city", "state", "units", "revenue", "revenue_share", "gross", "margin", "gpu", "asp",
            "avg_days", "fi_income"]],
        hide_index=True,
        column_config={
            "dealership_name": "Dealership", "city": "City", "state": "State",
            "units": st.column_config.NumberColumn("Units", format="%d"),
            "revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
            "revenue_share": st.column_config.ProgressColumn("Share", format="percent", min_value=0, max_value=1),
            "gross": st.column_config.NumberColumn("Vehicle gross", format="dollar"),
            "margin": st.column_config.NumberColumn("Margin", format="percent"),
            "gpu": st.column_config.NumberColumn("Gross / unit", format="dollar"),
            "asp": st.column_config.NumberColumn("ASP", format="dollar"),
            "avg_days": st.column_config.NumberColumn("Days to sell", format="%.0f"),
            "fi_income": st.column_config.NumberColumn("F&I income", format="dollar"),
        },
    )
    U.section("Location trend comparison")
    pick = st.multiselect("Dealerships", list(dp["dealership_id"]), default=list(dp["dealership_id"].head(4)),
                          format_func=dealer_names(ds).get, max_selections=6, key="cmp_stores",
                          label_visibility="collapsed")
    if pick:
        q = s[s["dealership_id"].isin(pick)].assign(quarter=lambda d: d["sale_date"].dt.to_period("Q").dt.to_timestamp())
        t = A.performance_by(q, ["quarter", "dealership_id"])
        t["quarter"] = pd.to_datetime(t["quarter"])
        t = A.complete_quarters(t, "quarter", f.start, f.end)
        t["store"] = t["dealership_id"].map(dealer_names(ds))
        # colors bound to the store, not its rank, so selections never repaint survivors
        all_ids = list(ds.dealerships["dealership_id"])
        store_colors = {dealer_names(ds)[d]: T.CATEGORICAL[all_ids.index(d) % 8] for d in pick}
        order = [dealer_names(ds)[d] for d in pick]
        l2, r2 = st.columns(2)
        with l2:
            C.show(C.lines(t, "quarter", "units", "store", order=order, colors=store_colors,
                           title="Quarterly units sold", subtitle="complete quarters only"))
        with r2:
            C.show(C.lines(t, "quarter", "margin", "store", pct=True, order=order, colors=store_colors,
                           title="Quarterly vehicle gross margin"))

# =========================================================================================
with tab_mix:
    mfr = A.performance_by(s, "manufacturer")
    top = list(mfr["manufacturer"].head(7))
    q = s.assign(mfr=s["manufacturer"].astype(str).where(s["manufacturer"].astype(str).isin(top), "Other"))
    mix = A.mix_over_time(q, "mfr", freq="Q")
    mfr_colors = {m: T.CATEGORICAL[i] for i, m in enumerate(top)}
    mfr_colors["Other"] = T.MUTED
    C.show(C.stacked_share(mix, "period", "mfr", colors=mfr_colors, order=top + ["Other"],
                           title="Manufacturer share of units, quarterly",
                           subtitle="top 7 manufacturers in scope; the rest folded into Other"), height=380)

    left, right = st.columns(2)
    with left:
        C.show(C.ranked_bars(mfr, "manufacturer", "gpu", title="Gross per unit by manufacturer",
                             hover_cols={"units": "Units"}), height=380)
    with right:
        cat = A.performance_by(s, "vehicle_category")
        C.show(C.ranked_bars(cat, "vehicle_category", "revenue", title="Revenue by vehicle category",
                             color=T.CATEGORICAL[0], hover_cols={"units": "Units"}), height=380)

    pivot = (s.assign(store=s["dealership_id"].astype(str).map(dealer_names(ds)))
             .pivot_table(index="store", columns="vehicle_category", values="sale_id", aggfunc="count",
                          observed=True).fillna(0))
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    C.show(C.heatmap(pivot, title="Units sold — dealership × vehicle category",
                     subtitle="reveals each store's specialization", colorbar_title="units"), height=440)

    m = A.monthly_summary(s, f.start, f.end)
    by_cond = (s.assign(month=s["sale_date"].dt.to_period("M").dt.to_timestamp())
               .groupby(["month", s["condition"].astype(str)]).size().rename("units").reset_index())
    C.show(C.lines(by_cond, "month", "units", "condition", colors=T.CONDITION_COLORS, order=["New", "Used"],
                   title="Monthly units sold — new vs used"), height=320)
