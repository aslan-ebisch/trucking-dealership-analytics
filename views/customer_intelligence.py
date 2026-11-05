"""Page 5 — Customer Intelligence: segments, loyalty, concentration, account lookup."""

import pandas as pd
import streamlit as st

from ironledger import analytics as A
from ironledger.reference_data import CUSTOMER_SEGMENTS
from ironledger.ui import charts as C
from ironledger.ui import components as U
from ironledger.ui import theme as T
from ironledger.ui.filters import dealer_names
from ironledger.ui.page import setup

ds, f = setup()
U.page_header("Customers", "Customer Intelligence",
              "Who buys, how often, and how concentrated revenue is. All accounts are fictional businesses.")

s = A.apply_filters(ds.sales, f)
U.filter_context(f, dealer_names(ds), len(s), len(ds.sales))
if s.empty:
    U.empty_state()
    st.stop()

cs = A.customer_summary(s, ds.customers)
rep = A.repeat_customer_stats(cs)
conc = A.revenue_concentration(cs)

r = st.columns(6)
U.kpi(r[0], "Buyers", U.number(rep["buyers"]))
U.kpi(r[1], "Repeat buyers", U.pct(rep["repeat_rate"]), f"{rep['repeat_buyers']:,} accounts", neutral=True,
      help="Share of purchasing customers with 2+ units in scope.")
U.kpi(r[2], "Avg deal size", U.money(A.average_selling_price(s), decimals=1))
U.kpi(r[3], "Revenue / buyer", U.money(rep["revenue_per_buyer"], decimals=1))
U.kpi(r[4], "Top-10 revenue share", U.pct(conc["top10_share"]),
      help="Share of revenue from the ten largest accounts in scope.")
U.kpi(r[5], "HHI", U.number(conc["hhi"]),
      help="Herfindahl–Hirschman index of customer revenue shares (0–10,000). Under 1,500 = unconcentrated.")

# ---- segments -----------------------------------------------------------------------------
U.section("Customer segments")
seg = A.segment_performance(s, cs)
seg_order = [x for x in CUSTOMER_SEGMENTS if x in set(seg["customer_segment"])]
left, right = st.columns(2)
with left:
    C.show(C.ranked_bars(seg, "customer_segment", "revenue", title="Revenue by customer segment",
                         hover_cols={"units": "Units", "buyers": "Buyers"}), height=360)
with right:
    C.show(C.ranked_bars(seg, "customer_segment", "units_per_buyer", money=False, color=T.CATEGORICAL[0],
                         decimals=1, title="Units per purchasing customer",
                         subtitle="fleets buy repeatedly; owner-operators buy once or twice"), height=360)
    # decimals for non-money bars: show one decimal
st.dataframe(
    seg[["customer_segment", "buyers", "units", "revenue", "revenue_share", "asp", "gpu", "repeat_rate",
         "finance_penetration"]],
    hide_index=True,
    column_config={
        "customer_segment": "Segment", "buyers": st.column_config.NumberColumn("Buyers", format="%d"),
        "units": st.column_config.NumberColumn("Units", format="%d"),
        "revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
        "revenue_share": st.column_config.ProgressColumn("Share", format="percent", min_value=0, max_value=1),
        "asp": st.column_config.NumberColumn("Avg transaction", format="dollar"),
        "gpu": st.column_config.NumberColumn("Gross / unit", format="dollar"),
        "repeat_rate": st.column_config.NumberColumn("Repeat rate", format="percent"),
        "finance_penetration": st.column_config.NumberColumn("Dealer-financed", format="percent"),
    },
)

left, right = st.columns(2)
with left:
    pv = pd.crosstab(s["customer_segment"].astype(str), s["vehicle_category"].astype(str), normalize="index")
    pv = pv.reindex([x for x in seg_order if x in pv.index])
    C.show(C.heatmap(pv, pct=True, title="What each segment buys", subtitle="share of segment's units"),
           height=420)
with right:
    mix = A.mix_over_time(s, "customer_segment", freq="Q", value="revenue")
    C.show(C.stacked_share(mix, "period", "customer_segment", colors=T.SEGMENT_COLORS, order=seg_order,
                           value_label="dollars", title="Segment share of revenue, quarterly"), height=420)

# ---- concentration & geography -------------------------------------------------------------------
U.section("Concentration & geography")
left, right = st.columns(2)
with left:
    C.show(C.lorenz(conc["curve"], title="Revenue concentration (Lorenz curve)",
                    subtitle=f"top 20% of customers = {U.pct(conc['top20pct_share'])} of revenue · "
                             f"{conc['customers_for_half']:,} accounts make up half"), height=400)
with right:
    geo = cs.groupby("home_state").agg(revenue=("revenue", "sum"), customers=("customer_id", "count")).reset_index()
    C.show(C.state_choropleth(geo, "home_state", "revenue", title="Revenue by customer home state",
                              subtitle="national fleets headquartered outside the footprint buy here too"),
           height=400)

# ---- top accounts --------------------------------------------------------------------------------------
U.section("Top accounts")
top = cs.head(25)
st.dataframe(
    top[["customer_name", "customer_segment", "home_state", "fleet_size", "units", "revenue", "gross",
         "avg_transaction", "first_purchase", "last_purchase", "stores_used"]],
    hide_index=True, height=420,
    column_config={
        "customer_name": "Customer", "customer_segment": "Segment", "home_state": "HQ",
        "fleet_size": st.column_config.NumberColumn("Fleet size", format="%d"),
        "units": st.column_config.NumberColumn("Units", format="%d"),
        "revenue": st.column_config.ProgressColumn("Revenue", format="dollar", min_value=0,
                                                   max_value=float(top["revenue"].max())),
        "gross": st.column_config.NumberColumn("Vehicle gross", format="dollar"),
        "avg_transaction": st.column_config.NumberColumn("Avg deal", format="dollar"),
        "first_purchase": st.column_config.DateColumn("First", format="MMM YYYY"),
        "last_purchase": st.column_config.DateColumn("Latest", format="MMM YYYY"),
        "stores_used": st.column_config.NumberColumn("Stores", format="%d"),
    },
)

# ---- account lookup ----------------------------------------------------------------------------------------
U.section("Account lookup", "purchase history for any customer in scope")
labels = dict(zip(cs["customer_id"], cs["customer_name"] + " · " + cs["customer_segment"].astype(str)))
cid = st.selectbox("Customer", list(labels), format_func=labels.get, key="cust_pick",
                   label_visibility="collapsed", placeholder="Type to search accounts…")
if cid:
    row = cs.set_index("customer_id").loc[cid]
    meta = ds.customers.set_index("customer_id").loc[cid]
    hist = A.customer_history(s, cid)
    c = st.columns(5)
    U.kpi(c[0], "Units purchased", U.number(row["units"]))
    U.kpi(c[1], "Revenue", U.money(row["revenue"]))
    U.kpi(c[2], "Vehicle gross", U.money(row["gross"]))
    U.kpi(c[3], "F&I income", U.money(row["fi_income"]))
    U.kpi(c[4], "Customer since", f"{meta['customer_since']:%b %Y}", f"Credit tier {meta['credit_tier']}",
          neutral=True)
    st.caption(f"{meta['customer_name']} · {meta['customer_segment']} · HQ {meta['home_state']} · "
               f"fleet ≈ {meta['fleet_size']:,} units · primary store {meta['primary_dealership_name']} · "
               f"account status {meta['account_status']}")
    left, right = st.columns([1, 1.4])
    with left:
        C.show(C.purchase_timeline(hist, title="Purchases over time"), height=340)
    with right:
        st.dataframe(
            hist[["sale_date", "sale_id", "manufacturer", "model", "model_year", "condition", "dealership_name",
                  "sale_price", "vehicle_gross_profit", "financing_arranged", "salesperson_name"]],
            hide_index=True, height=340,
            column_config={"sale_date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                           "sale_id": "Sale", "manufacturer": "Make", "model": "Model",
                           "model_year": st.column_config.NumberColumn("Year", format="%d"),
                           "condition": "Cond.", "dealership_name": "Store",
                           "sale_price": st.column_config.NumberColumn("Price", format="dollar"),
                           "vehicle_gross_profit": st.column_config.NumberColumn("Gross", format="dollar"),
                           "financing_arranged": st.column_config.CheckboxColumn("Financed"),
                           "salesperson_name": "Salesperson"})
