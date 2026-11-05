"""Page 4 — Financial Analytics: P&L, gross-to-operating-income waterfall, profitability."""

import pandas as pd
import streamlit as st

from ironledger import analytics as A
from ironledger.ui import charts as C
from ironledger.ui import components as U
from ironledger.ui import theme as T
from ironledger.ui.filters import dealer_names
from ironledger.ui.page import setup

ds, f = setup()
U.page_header("Finance", "Financial Analytics",
              "From vehicle revenue to estimated sales-department operating income.")

s = A.apply_filters(ds.sales, f)
U.filter_context(f, dealer_names(ds), len(s), len(ds.sales))
if s.empty:
    U.empty_state()
    st.stop()

# Store-level expenses can only be matched to store-level sales. With a product filter
# active (e.g. "Used only"), allocating rent or payroll to that slice would be invented.
opex = None if f.has_product_filter else A.filter_operating_expenses(ds.operating_expenses, f)
k = A.kpi_summary(s)
pnl = A.pnl_statement(s, opex)
op_income = pnl.loc[pnl["line_item"] == "Est. operating income", "amount"]
op_income = float(op_income.iloc[0]) if len(op_income) else float("nan")

r1 = st.columns(4)
U.kpi(r1[0], "Vehicle sales revenue", U.money(k["revenue"]))
U.kpi(r1[1], "Cost of vehicles sold", U.money(k["cogs"]), help="Acquisition + reconditioning + transportation.")
U.kpi(r1[2], "Vehicle gross profit", U.money(k["gross"]), f"{U.pct(k['margin'])} margin", neutral=True)
U.kpi(r1[3], "Gross per unit", U.money(k["gpu"], decimals=1))
r2 = st.columns(4)
U.kpi(r2[0], "F&I income", U.money(k["fi_income"]), f"{U.money(k['fi_per_unit'], compact=False)} per unit",
      neutral=True, help="Finance reserve + protection products. Tracked separately — not part of revenue.")
U.kpi(r2[1], "Total gross", U.money(k["total_contribution"]), help="Vehicle gross + F&I income.")
U.kpi(r2[2], "Est. operating income", U.money(op_income),
      None if opex is None else f"{U.pct(A.safe_div(op_income, k['revenue']))} of revenue", neutral=True,
      help="Total gross contribution − sales-department operating expenses. Not net income.")
U.kpi(r2[3], "Deals sold at a loss", U.pct(k["loss_rate"]), help="Share of deals with negative vehicle gross.")

if opex is None:
    st.info("Operating expenses are recorded per **dealership**, not per manufacturer, condition or category. "
            "With a product filter active the statement stops at total gross contribution rather than "
            "inventing an expense allocation. Clear those filters to see operating income.",
            icon=":material/info:")

# ---- P&L & waterfall ---------------------------------------------------------------------
U.section("Sales-department P&L", "expenses prorated by day when the period covers part of a month")
left, right = st.columns([1, 1.25])
with left:
    def _style(row):
        if row["kind"] in ("subtotal", "subtotal_expense"):
            return [f"background-color: {T.SURFACE_2}; color: {T.TEXT}"] * len(row)
        if row["kind"] == "total":
            return [f"background-color: {T.ACCENT_SOFT}; color: {T.ACCENT}"] * len(row)
        return [f"color: {T.TEXT_2}"] * len(row)

    table = pnl.copy()
    styled = (table.style.apply(_style, axis=1)
              .format({"amount": lambda v: U.money(v, compact=False),
                       "pct_of_revenue": lambda v: U.pct(v), "per_unit": lambda v: U.money(v, compact=False)}))
    st.dataframe(styled, hide_index=True, height=min(38 * (len(table) + 1), 640),
                 column_order=["line_item", "amount", "pct_of_revenue", "per_unit"],
                 column_config={"line_item": st.column_config.TextColumn("Line item", width="medium"),
                                "amount": "Amount", "pct_of_revenue": "% of revenue", "per_unit": "Per unit"})
with right:
    steps = [("Vehicle gross", k["gross"], "absolute"),
             ("Finance reserve", float(s["finance_income"].sum()), "relative"),
             ("Protection products", float(s["warranty_income"].sum()), "relative"),
             ("Total gross", k["total_contribution"], "total")]
    if opex is not None:
        for col, label in A.OPEX_LINES.items():
            steps.append((label.replace(" & ", " & "), -float(opex[col].sum()), "relative"))
        steps.append(("Operating income", op_income, "total"))
    wf = pd.DataFrame(steps, columns=["label", "amount", "measure"])
    C.show(C.waterfall(wf, title="Gross-to-operating-income bridge",
                       subtitle="blue adds · red subtracts · gold = subtotal"), height=460)

# ---- trends ---------------------------------------------------------------------------------
U.section("Profitability over time")
left, right = st.columns(2)
with left:
    if opex is not None:
        moi = A.monthly_operating_income(s, opex)
        C.show(C.signed_bars(moi, "month", "operating_income", title="Monthly est. operating income",
                             subtitle="contribution − operating expenses; red = loss month"))
    else:
        m = A.monthly_summary(s, f.start, f.end)
        C.show(C.monthly_bars(m, "gross", title="Monthly vehicle gross profit", color=T.CATEGORICAL[0]))
with right:
    q = s.assign(quarter=s["sale_date"].dt.to_period("Q").dt.to_timestamp())
    t = A.performance_by(q, ["quarter", "condition"])
    t["quarter"] = pd.to_datetime(t["quarter"])
    t = A.complete_quarters(t, "quarter", f.start, f.end) if f.end - f.start > pd.Timedelta(days=200) else t
    C.show(C.lines(t, "quarter", "gpu", "condition", money=True, colors=T.CONDITION_COLORS, order=["New", "Used"],
                   title="Gross per unit — new vs used, quarterly",
                   subtitle="used gross is exposed to market value swings while units sit in stock"))

# ---- profitability by dimension -----------------------------------------------------------------
U.section("Profitability by dimension")
tab_m, tab_d, tab_c = st.tabs(["Manufacturer", "Dealership", "Vehicle category"])
money_cfg = {
    "units": st.column_config.NumberColumn("Units", format="%d"),
    "revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
    "gross": st.column_config.NumberColumn("Vehicle gross", format="dollar"),
    "margin": st.column_config.NumberColumn("Margin", format="percent"),
    "gpu": st.column_config.NumberColumn("Gross / unit", format="dollar"),
    "fi_income": st.column_config.NumberColumn("F&I income", format="dollar"),
    "total_contribution": st.column_config.NumberColumn("Total contribution", format="dollar"),
    "loss_rate": st.column_config.NumberColumn("Loss deals", format="percent"),
}
cols = ["units", "revenue", "gross", "margin", "gpu", "fi_income", "total_contribution", "loss_rate"]
with tab_m:
    mp = A.performance_by(s, "manufacturer")
    l, r = st.columns([1.3, 1])
    l.dataframe(mp[["manufacturer"] + cols], hide_index=True, column_config={"manufacturer": "Manufacturer",
                                                                              **money_cfg})
    with r:
        C.show(C.ranked_bars(mp, "manufacturer", "margin", pct=True, money=False,
                             title="Vehicle gross margin by manufacturer"), height=380)
with tab_d:
    if opex is not None:
        so = A.store_operating_income(s, opex, ds.dealerships)
        l, r = st.columns([1.3, 1])
        l.dataframe(
            so[["dealership_name", "units", "revenue", "gross", "total_contribution", "payroll_expense",
                "floorplan_interest_expense", "total_operating_expense", "operating_income", "operating_margin"]],
            hide_index=True,
            column_config={"dealership_name": "Dealership", **money_cfg,
                           "payroll_expense": st.column_config.NumberColumn("Payroll", format="dollar"),
                           "floorplan_interest_expense": st.column_config.NumberColumn("Floorplan int.",
                                                                                      format="dollar"),
                           "total_operating_expense": st.column_config.NumberColumn("Total opex", format="dollar"),
                           "operating_income": st.column_config.NumberColumn("Est. op. income", format="dollar"),
                           "operating_margin": st.column_config.NumberColumn("Op. margin", format="percent")})
        with r:
            C.show(C.ranked_bars(so.assign(city=so["city"]), "city", "operating_income",
                                 title="Est. operating income by dealership"), height=380)
    else:
        dp = A.dealership_performance(s, ds.dealerships)
        st.dataframe(dp[["dealership_name"] + cols], hide_index=True,
                     column_config={"dealership_name": "Dealership", **money_cfg})
with tab_c:
    cp = A.performance_by(s, "vehicle_category")
    l, r = st.columns([1.3, 1])
    l.dataframe(cp[["vehicle_category"] + cols], hide_index=True,
                column_config={"vehicle_category": "Category", **money_cfg})
    with r:
        C.show(C.ranked_bars(cp, "vehicle_category", "gpu", title="Gross per unit by vehicle category"),
               height=380)

if opex is not None:
    U.section("Operating expense structure")
    l, r = st.columns(2)
    mix = pd.DataFrame({"line": list(A.OPEX_LINES.values()),
                        "amount": [float(opex[c].sum()) for c in A.OPEX_LINES]})
    with l:
        C.show(C.ranked_bars(mix, "line", "amount", title="Operating expenses by type",
                             color=T.CATEGORICAL[1]), height=320)
    with r:
        total_c = k["total_contribution"]
        mix["absorption"] = mix["amount"] / total_c if total_c else float("nan")
        C.show(C.ranked_bars(mix, "line", "absorption", pct=True, money=False, color=T.CATEGORICAL[1],
                             title="Expense as % of total gross contribution"), height=320)

U.definitions_expander()
