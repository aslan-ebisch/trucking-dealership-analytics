"""Page 6 — Data Explorer: inspect, filter, profile and export the raw datasets."""

import pandas as pd
import streamlit as st

from ironledger import analytics as A
from ironledger.data_loader import LocalFileSource, get_data_source
from ironledger.ui import components as U
from ironledger.ui.data import data_signature, quality_report
from ironledger.ui.filters import dealer_names
from ironledger.ui.page import setup

ds, f = setup()
U.page_header("Data", "Data Explorer",
              "The structured datasets behind every number in this application — browse, profile and export.")

DESCRIPTIONS = {
    "sales": "Fact table — one row per vehicle sale. Revenue, costs, gross, F&I income, customer, rep and unit.",
    "inventory": "Unit master — one row per physical truck/trailer ever acquired. status = 'Sold' (linked to a "
                 "sale) or 'In Stock' (current lot). Inventory KPIs are computed from this table.",
    "customers": "Dimension — fictional commercial accounts with segment, HQ state, fleet size and credit tier.",
    "salespeople": "Dimension — sales roster with hire/termination dates. Every sale is credited to a rep "
                   "employed at that store on the sale date.",
    "dealerships": "Dimension — the 10 stores: location, franchises carried, opening date, lot capacity.",
    "operating_expenses": "Monthly sales-department expenses per store. Payroll includes the commissions "
                          "earned on that month's deals; floorplan interest is charged on actual stock on hand.",
}
EXPECTED_MISSING = {
    "mileage": "trailers are not odometer-metered",
    "protection_product": "no protection product purchased on that deal",
    "list_price": "only in-stock units carry an asking price",
    "sale_id": "unsold (in-stock) units have no sale",
    "sale_date": "unsold (in-stock) units have no sale",
    "termination_date": "rep is still employed",
}

tables = ds.tables()
name = st.segmented_control("Dataset", list(tables), default="sales", key="dx_table",
                            format_func=lambda t: t.replace("_", " ").title()) or "sales"
df = tables[name]
st.caption(DESCRIPTIONS[name])

source = get_data_source()
info = source.file_info().set_index("table") if isinstance(source, LocalFileSource) else None
c = st.columns(5)
U.kpi(c[0], "Rows", U.number(len(df)))
U.kpi(c[1], "Columns", U.number(df.shape[1]))
U.kpi(c[2], "In-memory size", f"{df.memory_usage(deep=True).sum() / 1e6:,.1f} MB",
      help="pandas memory footprint, using categorical types for repeated text.")
if info is not None:
    U.kpi(c[3], "Stored as", info.at[name, "format"], f"{info.at[name, 'size_mb']:,.2f} MB on disk", neutral=True)
U.kpi(c[4], "Missing cells", U.number(int(df.isna().sum().sum())),
      help="Structural blanks (e.g. trailer mileage) are expected — see the profile tab.")

tab_browse, tab_profile, tab_stats, tab_quality, tab_model = st.tabs(
    ["Browse & export", "Schema & profile", "Statistics", "Data quality", "Data model"])

# =====================================================================================================
with tab_browse:
    view = df
    if name in ("sales", "inventory"):
        apply_global = st.toggle("Apply sidebar filters", value=True, key="dx_global",
                                 help="Sales: all filters. Inventory: dealership/manufacturer/condition/category.")
        if apply_global:
            view = A.apply_filters(view, f) if name == "sales" else A.filter_dimensions(view, f)
            U.filter_context(f, dealer_names(ds), len(view), len(df), dates_apply=(name == "sales"))

    c1, c2, c3 = st.columns([2.4, 1.6, 1])
    query = c1.text_input("Search all text columns", key=f"dx_q_{name}", placeholder="e.g. Peterbilt, Tulsa, C00042…")
    sort_col = c2.selectbox("Sort by", ["(none)"] + list(df.columns), key=f"dx_sort_{name}")
    descending = c3.toggle("Descending", value=True, key=f"dx_desc_{name}")

    filter_cols = st.multiselect("Column filters", list(df.columns), key=f"dx_fcols_{name}",
                                 placeholder="Add a column to filter on…")
    for col in filter_cols:
        s = view[col]
        if pd.api.types.is_bool_dtype(s):
            pick = st.multiselect(col, [True, False], default=[True, False], key=f"dx_f_{name}_{col}")
            view = view[s.isin(pick)]
        elif pd.api.types.is_datetime64_any_dtype(s):
            lo, hi = s.min(), s.max()
            if pd.notna(lo):
                rng = st.date_input(col, value=(lo.date(), hi.date()), min_value=lo.date(), max_value=hi.date(),
                                    key=f"dx_f_{name}_{col}")
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    view = view[(s >= pd.Timestamp(rng[0])) & (s <= pd.Timestamp(rng[1]))]
        elif pd.api.types.is_numeric_dtype(s):
            lo, hi = float(s.min()), float(s.max())
            if lo < hi:
                a, b = st.slider(col, lo, hi, (lo, hi), key=f"dx_f_{name}_{col}")
                view = view[s.between(a, b) | (s.isna() & (a == lo) & (b == hi))]
        else:
            opts = sorted(s.dropna().astype(str).unique())
            if len(opts) <= 300:
                pick = st.multiselect(col, opts, key=f"dx_f_{name}_{col}", placeholder="Any value")
                if pick:
                    view = view[s.astype(str).isin(pick)]
            else:
                txt = st.text_input(f"{col} contains", key=f"dx_f_{name}_{col}")
                if txt:
                    view = view[s.astype(str).str.contains(txt, case=False, regex=False, na=False)]

    view = A.search_frame(view, query)
    if sort_col != "(none)":
        view = view.sort_values(sort_col, ascending=not descending)

    LIMIT = 10_000
    date_cfg = {c: st.column_config.DateColumn(c, format="YYYY-MM-DD")
                for c in view.columns if pd.api.types.is_datetime64_any_dtype(view[c])}
    st.dataframe(view.head(LIMIT), hide_index=True, height=480, column_config=date_cfg)
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.caption(f"{len(view):,} matching rows" + (f" · showing first {LIMIT:,} (export includes all)"
                                                  if len(view) > LIMIT else ""))
    c2.download_button("Export CSV", view.to_csv(index=False).encode(), f"ironledger_{name}.csv", "text/csv",
                       icon=":material/download:", width="stretch", key=f"dx_dl_{name}")

# =====================================================================================================
with tab_profile:
    prof = A.profile_table(df)
    prof["note"] = prof.apply(lambda r: EXPECTED_MISSING.get(r["column"], "") if r["missing"] else "", axis=1)
    st.dataframe(prof, hide_index=True, height=min(36 * (len(prof) + 1), 720),
                 column_config={"missing_pct": st.column_config.ProgressColumn("Missing %", format="percent",
                                                                               min_value=0, max_value=1),
                                "note": st.column_config.TextColumn("Why blanks are expected", width="large")})

# =====================================================================================================
with tab_stats:
    num = df.select_dtypes("number")
    if num.shape[1]:
        st.markdown("**Numeric columns**")
        st.dataframe(num.describe().T.round(2), height=min(36 * (num.shape[1] + 1), 720))
    cat = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object]
    if cat:
        st.markdown("**Top values**")
        col = st.selectbox("Column", cat, key=f"dx_topcol_{name}")
        vc = df[col].astype(str).value_counts().head(20).rename_axis(col).reset_index(name="rows")
        vc["share"] = vc["rows"] / len(df)
        st.dataframe(vc, hide_index=True,
                     column_config={"share": st.column_config.ProgressColumn("Share", format="percent",
                                                                             min_value=0, max_value=1)})

# =====================================================================================================
with tab_quality:
    results = pd.DataFrame(quality_report(data_signature()))
    passed = int(results["passed"].sum())
    c = st.columns(3)
    U.kpi(c[0], "Checks passed", f"{passed} / {len(results)}")
    U.kpi(c[1], "Failing rows", U.number(int(results.loc[~results["passed"], "failures"].clip(lower=0).sum())))
    U.kpi(c[2], "Overall health", "Healthy" if passed == len(results) else "Attention needed")
    results["status"] = results["passed"].map({True: "✅ Pass", False: "❌ Fail"})
    st.dataframe(results[["status", "category", "name", "detail"]], hide_index=True,
                 column_config={"status": "Status", "category": "Category", "name": "Check",
                                "detail": st.column_config.TextColumn("Result", width="large")})
    st.caption("The same checks run automatically in `generate_data.py` (bad data is never written) and in the "
               "pytest suite (`tests/test_data_quality.py`).")

# =====================================================================================================
with tab_model:
    l, r = st.columns([1.4, 1])
    with l:
        st.graphviz_chart("""
digraph G {
  graph [bgcolor="transparent", rankdir=LR, pad=0.3, nodesep=0.4];
  node  [shape=box, style="rounded,filled", fillcolor="#1b1e23", color="#2a2e35", fontcolor="#ebe8e2",
         fontname="Helvetica", fontsize=11];
  edge  [color="#6d727b", fontcolor="#a9acb3", fontname="Helvetica", fontsize=9, arrowsize=0.6];
  dealerships [label="dealerships\\n10 stores", color="#d6a84e"];
  salespeople; customers; inventory [label="inventory\\nunit master"]; sales [label="sales\\nfact table", color="#d6a84e"];
  operating_expenses [label="operating_expenses\\nstore × month"];
  dealerships -> salespeople [label="dealership_id"];
  dealerships -> customers [label="primary_dealership_id"];
  dealerships -> inventory [label="dealership_id"];
  dealerships -> operating_expenses [label="dealership_id"];
  dealerships -> sales [label="dealership_id"];
  salespeople -> sales [label="salesperson_id"];
  customers -> sales [label="customer_id"];
  inventory -> sales [label="unit_id (1:0..1)"];
}""", width="stretch")
    with r:
        meta = ds.metadata or {}
        st.markdown("**Generation metadata**")
        params = meta.get("parameters", {})
        st.markdown(
            f"- Generated: `{meta.get('generated_at_utc', 'n/a')}`\n"
            f"- Generator version: `{meta.get('generator_version', 'n/a')}`\n"
            f"- Random seed: `{params.get('seed', 'n/a')}` (same seed ⇒ identical data)\n"
            f"- Window: `{params.get('start', '?')}` → `{params.get('as_of', '?')}`")
        if info is not None:
            st.markdown("**Storage**")
            st.dataframe(info.reset_index(), hide_index=True,
                         column_config={"size_mb": st.column_config.NumberColumn("Size (MB)", format="%.2f")})
            st.caption("Parquet is columnar and compressed — typically ~4× smaller than CSV and much faster to "
                       "load. The same files can be uploaded unchanged to Azure Blob Storage / Data Lake.")
