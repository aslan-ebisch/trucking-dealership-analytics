"""Reusable Plotly figures in the IronLedger style.

Rules applied throughout: one y-axis per chart (never dual axes), a legend
whenever there are two or more series, single-series charts in the brand
accent, categorical colors bound to entities (not rank), and full-precision
values in tooltips even when axes are abbreviated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ironledger.ui import theme as T

PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False}


def show(fig: go.Figure, height: int = 340, key: str | None = None) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def money_scale(values) -> tuple[float, str]:
    """Pick a divisor/suffix so money axes read $12.4M rather than 12400000."""
    vmax = float(np.nanmax(np.abs(np.asarray(values, dtype=float)))) if len(values) else 0.0
    if vmax >= 1e9:
        return 1e9, "B"
    if vmax >= 1e6:
        return 1e6, "M"
    if vmax >= 1e3:
        return 1e3, "K"
    return 1.0, ""


def _title(fig: go.Figure, title: str | None, subtitle: str | None = None) -> None:
    if title:
        fig.update_layout(title=dict(text=title))
        if subtitle:  # Plotly's native subtitle reserves its own space (no clipping)
            fig.update_layout(title=dict(subtitle=dict(text=subtitle, font=dict(size=11, color=T.MUTED))),
                              margin=dict(t=76))


def _period_axis(fig: go.Figure, x) -> None:
    """Label quarterly series as 'Q1 2026' instead of month names."""
    xs = pd.to_datetime(pd.Series(x)).dropna().unique()
    if len(xs) > 1 and all(t.day == 1 and t.month in (1, 4, 7, 10) for t in pd.to_datetime(xs)):
        fig.update_xaxes(tickformat="Q%q<br>%Y", dtick="M3", tick0=pd.Timestamp(min(xs)))


def empty_figure(message: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=T.MUTED, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------------------------
def ranked_bars(df: pd.DataFrame, label: str, value: str, *, title: str | None = None, subtitle: str | None = None,
                money: bool = True, pct: bool = False, color: str = T.ACCENT, max_bars: int = 15,
                hover_cols: dict[str, str] | None = None, decimals: int | None = None) -> go.Figure:
    """Horizontal ranked bars (largest on top) with end-of-bar value labels."""
    if df.empty:
        return empty_figure()
    d = df.nlargest(max_bars, value).iloc[::-1]
    vals = d[value].astype(float)
    if pct:
        div, suf, prefix = 0.01, "%", ""
    elif money:
        div, suf = money_scale(vals)
        prefix = "$"
    else:
        div, suf, prefix = 1.0, "", ""
    scaled = vals / div
    if decimals is None:
        decimals = 1 if (money or pct) else 0
    text = [f"{prefix}{v:,.{decimals}f}{suf}" for v in scaled]
    custom = [d[value]]
    hover = f"<b>%{{y}}</b><br>{value.replace('_', ' ').title()}: " + (
        "%{customdata[0]:.1%}" if pct else ("$%{customdata[0]:,.0f}" if money else "%{customdata[0]:,.2~f}"))
    for i, (col, lbl) in enumerate((hover_cols or {}).items(), start=1):
        custom.append(d[col])
        hover += f"<br>{lbl}: %{{customdata[{i}]}}"
    fig = go.Figure(go.Bar(
        x=scaled, y=d[label].astype(str), orientation="h", marker_color=color, text=text,
        textposition="outside", textfont=dict(color=T.TEXT_2, size=11), cliponaxis=False,
        customdata=np.column_stack(custom), hovertemplate=hover + "<extra></extra>",
    ))
    fig.update_xaxes(showgrid=True, gridcolor=T.GRID, tickprefix=prefix, ticksuffix=suf,
                     range=[min(0, scaled.min() * 1.15), max(scaled.max() * 1.18, 0.0001)])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=T.TEXT, size=12))
    _title(fig, title, subtitle)
    return fig


def monthly_bars(m: pd.DataFrame, value: str, *, title: str | None = None, subtitle: str | None = None,
                 money: bool = True, prior_col: str | None = None, color: str = T.ACCENT,
                 current_label: str = "Selected period", prior_label: str = "Same month, prior year") -> go.Figure:
    """Monthly bars with partial months faded and an optional prior-year line."""
    if m.empty:
        return empty_figure()
    vals = m[value].astype(float)
    div, suf = money_scale(pd.concat([vals, m[prior_col]]).dropna() if prior_col else vals) if money else (1.0, "")
    prefix = "$" if money else ""
    opacity = np.where(m["is_partial"], 0.4, 1.0)
    fmt = "$%{customdata:,.0f}" if money else "%{customdata:,.0f}"
    fig = go.Figure()
    fig.add_bar(x=m["month"], y=vals / div, name=current_label, marker=dict(color=color, opacity=opacity),
                customdata=vals, hovertemplate=f"%{{x|%b %Y}}<br>{current_label}: {fmt}<extra></extra>")
    if prior_col and prior_col in m and m[prior_col].notna().any():
        fig.add_scatter(x=m["month"], y=m[prior_col] / div, name=prior_label, mode="lines",
                        line=dict(color=T.TEXT_2, width=1.6, shape="spline", smoothing=0.6),
                        customdata=m[prior_col],
                        hovertemplate=f"%{{x|%b %Y}}<br>{prior_label}: {fmt}<extra></extra>")
    fig.update_yaxes(showgrid=True, tickprefix=prefix, ticksuffix=suf, rangemode="tozero")
    fig.update_xaxes(tickformat="%b<br>%Y", dtick="M3" if len(m) > 18 else "M1")
    fig.update_layout(hovermode="x unified", bargap=0.18, showlegend=bool(prior_col))
    if m["is_partial"].any():
        fig.add_annotation(text="faded bars = partial month in selection", xref="paper", yref="paper", x=0, y=-0.22,
                           showarrow=False, font=dict(size=10, color=T.MUTED), xanchor="left")
    _title(fig, title, subtitle)
    return fig


def stacked_share(df: pd.DataFrame, x: str, series: str, *, colors: dict[str, str] | None = None,
                  title: str | None = None, subtitle: str | None = None, order: list[str] | None = None,
                  value_label: str = "Units") -> go.Figure:
    """100% stacked columns — composition over time."""
    if df.empty:
        return empty_figure()
    keys = order or sorted(df[series].unique())
    fig = go.Figure()
    for i, k in enumerate(keys):
        d = df[df[series] == k]
        if d.empty:
            continue
        fig.add_bar(x=d[x], y=d["share"], name=k,
                    marker=dict(color=(colors or {}).get(k, T.CATEGORICAL[i % 8]), line=dict(width=1, color=T.BG)),
                    customdata=d["value"],
                    hovertemplate=f"{k}<br>%{{x|%b %Y}}: %{{y:.1%}} (%{{customdata:,.0f}} {value_label.lower()})<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.2, legend_traceorder="normal")
    fig.update_yaxes(tickformat=".0%", showgrid=True, range=[0, 1])
    _period_axis(fig, df[x])
    _title(fig, title, subtitle)
    return fig


def stacked_bars_h(df: pd.DataFrame, label: str, series: str, value: str, *, colors: dict[str, str] | None = None,
                   order: list[str] | None = None, title: str | None = None, subtitle: str | None = None,
                   money: bool = False, label_order: list[str] | None = None) -> go.Figure:
    """Horizontal stacked bars (e.g. stock by store split by aging bucket)."""
    if df.empty:
        return empty_figure()
    keys = order or sorted(df[series].astype(str).unique())
    totals = df.groupby(label)[value].sum().sort_values()
    y_order = label_order or list(totals.index)
    div, suf = money_scale(totals.values) if money else (1.0, "")
    fig = go.Figure()
    for i, k in enumerate(keys):
        d = df[df[series].astype(str) == k].set_index(label).reindex(y_order)
        fig.add_bar(y=y_order, x=d[value].fillna(0) / div, name=k, orientation="h",
                    marker=dict(color=(colors or {}).get(k, T.CATEGORICAL[i % 8]), line=dict(width=1, color=T.BG)),
                    customdata=d[value].fillna(0),
                    hovertemplate=f"<b>%{{y}}</b><br>{k}: " + ("$%{customdata:,.0f}" if money else "%{customdata:,.0f}")
                    + "<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.3, legend_traceorder="normal")
    fig.update_xaxes(showgrid=True, tickprefix="$" if money else "", ticksuffix=suf)
    fig.update_yaxes(tickfont=dict(color=T.TEXT, size=12))
    _title(fig, title, subtitle)
    return fig


def lines(df: pd.DataFrame, x: str, y: str, series: str, *, colors: dict[str, str] | None = None,
          title: str | None = None, subtitle: str | None = None, money: bool = False, pct: bool = False,
          order: list[str] | None = None) -> go.Figure:
    if df.empty:
        return empty_figure()
    keys = order or list(dict.fromkeys(df[series].astype(str)))
    div, suf = money_scale(df[y]) if money else (1.0, "")
    fmt = "$%{customdata:,.0f}" if money else ("%{customdata:.1%}" if pct else "%{customdata:,.1f}")
    fig = go.Figure()
    for i, k in enumerate(keys):
        d = df[df[series].astype(str) == k].sort_values(x)
        fig.add_scatter(x=d[x], y=d[y] / div, name=k, mode="lines+markers",
                        line=dict(color=(colors or {}).get(k, T.CATEGORICAL[i % 8]), width=2),
                        marker=dict(size=6), customdata=d[y],
                        hovertemplate=f"{k}: {fmt}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(showgrid=True, tickprefix="$" if money else "", ticksuffix=suf,
                     tickformat=".0%" if pct else None)
    _period_axis(fig, df[x])
    _title(fig, title, subtitle)
    return fig


def heatmap(pivot: pd.DataFrame, *, title: str | None = None, subtitle: str | None = None, fmt: str = ",.0f",
            pct: bool = False, colorbar_title: str = "") -> go.Figure:
    if pivot.empty:
        return empty_figure()
    z = pivot.to_numpy(dtype=float)
    text = [[("" if np.isnan(v) else (f"{v:.0%}" if pct else f"{v:{fmt}}")) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c).replace(' ', '<br>', 1) for c in pivot.columns], y=[str(i) for i in pivot.index], colorscale=[
            [i / (len(T.SEQUENTIAL) - 1), c] for i, c in enumerate(T.SEQUENTIAL)],
        text=text, texttemplate="%{text}", textfont=dict(size=11, color=T.TEXT), xgap=2, ygap=2,
        customdata=[[str(c) for c in pivot.columns]] * len(pivot.index),
        hovertemplate="%{y} · %{customdata}: %{text}<extra></extra>",
        colorbar=dict(title=dict(text=colorbar_title, font=dict(size=11, color=T.MUTED)), thickness=10,
                      outlinewidth=0, tickfont=dict(color=T.TEXT_2, size=10), tickformat=".0%" if pct else "~s"),
    ))
    fig.update_xaxes(side="top", tickangle=0, tickfont=dict(size=11))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=T.TEXT, size=11))
    _title(fig, title, subtitle)
    if title:
        fig.update_layout(margin=dict(t=96 if subtitle else 80))
    return fig


def waterfall(steps: pd.DataFrame, *, title: str | None = None, subtitle: str | None = None) -> go.Figure:
    """steps: columns label, amount, measure ('relative' | 'total')."""
    div, suf = money_scale(steps["amount"])
    text = [f"{'−' if a < 0 else ''}${abs(a) / div:,.1f}{suf}" for a in steps["amount"]]
    fig = go.Figure(go.Waterfall(
        x=steps["label"], y=steps["amount"] / div, measure=steps["measure"], text=text, textposition="outside",
        textfont=dict(color=T.TEXT_2, size=11), cliponaxis=False,
        increasing=dict(marker=dict(color=T.POSITIVE)), decreasing=dict(marker=dict(color=T.NEGATIVE)),
        totals=dict(marker=dict(color=T.ACCENT)), connector=dict(line=dict(color=T.BORDER, width=1)),
        customdata=steps["amount"], hovertemplate="%{x}: $%{customdata:,.0f}<extra></extra>",
    ))
    fig.update_yaxes(showgrid=True, tickprefix="$", ticksuffix=suf)
    fig.update_xaxes(tickfont=dict(size=11, color=T.TEXT))
    _title(fig, title, subtitle)
    return fig


def signed_bars(df: pd.DataFrame, x: str, y: str, *, title: str | None = None, subtitle: str | None = None) -> go.Figure:
    """Monthly bars colored by sign (gain vs loss)."""
    if df.empty:
        return empty_figure()
    div, suf = money_scale(df[y])
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y] / div, marker_color=np.where(df[y] >= 0, T.POSITIVE, T.NEGATIVE), customdata=df[y],
        hovertemplate="%{x|%b %Y}: $%{customdata:,.0f}<extra></extra>"))
    fig.update_yaxes(showgrid=True, tickprefix="$", ticksuffix=suf, zeroline=True, zerolinecolor=T.BORDER)
    fig.update_xaxes(tickformat="%b<br>%Y")
    fig.update_layout(bargap=0.15)
    _title(fig, title, subtitle)
    return fig


def lorenz(curve: pd.DataFrame, *, title: str | None = None, subtitle: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
                    line=dict(color=T.MUTED, width=1), hoverinfo="skip")
    fig.add_scatter(x=curve["customer_share"], y=curve["revenue_share"], mode="lines", name="Actual",
                    line=dict(color=T.ACCENT, width=2.4), fill="tonexty", fillcolor=T.ACCENT_SOFT,
                    hovertemplate="Top %{x:.0%} of customers → %{y:.1%} of revenue<extra></extra>")
    fig.update_xaxes(tickformat=".0%", title="Share of customers (ranked by revenue)", showgrid=True, range=[0, 1])
    fig.update_yaxes(tickformat=".0%", title="Cumulative share of revenue", showgrid=True, range=[0, 1.02])
    _title(fig, title, subtitle)
    return fig


def bubble(df: pd.DataFrame, x: str, y: str, size: str, label: str, *, title: str | None = None,
           subtitle: str | None = None, x_title: str = "", y_title: str = "", x_money: bool = False,
           y_money: bool = False) -> go.Figure:
    if df.empty:
        return empty_figure()
    sz = df[size].astype(float)
    sizeref = 2.0 * sz.max() / (46 ** 2) if sz.max() > 0 else 1
    fig = go.Figure(go.Scatter(
        x=df[x], y=df[y], mode="markers+text", text=df[label], textposition="top center",
        textfont=dict(size=10, color=T.TEXT_2),
        marker=dict(size=sz, sizemode="area", sizeref=sizeref, sizemin=6, color=T.ACCENT, opacity=0.8,
                    line=dict(width=2, color=T.BG)),
        customdata=np.column_stack([df[size]]),
        hovertemplate=("<b>%{text}</b><br>" + f"{x_title}: " + ("$%{x:,.0f}" if x_money else "%{x:,.0f}")
                       + f"<br>{y_title}: " + ("$%{y:,.0f}" if y_money else "%{y:,.1f}")
                       + "<br>Revenue: $%{customdata[0]:,.0f}<extra></extra>"),
    ))
    fig.update_xaxes(title=x_title, showgrid=True, tickprefix="$" if x_money else "")
    fig.update_yaxes(title=y_title, showgrid=True, tickprefix="$" if y_money else "")
    _title(fig, title, subtitle)
    return fig


def store_map(df: pd.DataFrame, *, title: str | None = None, subtitle: str | None = None) -> go.Figure:
    """Dealership locations sized by revenue."""
    if df.empty:
        return empty_figure()
    sz = df["revenue"].astype(float)
    fig = go.Figure(go.Scattergeo(
        lat=df["latitude"], lon=df["longitude"], text=df["city"], mode="markers+text", textposition="top center",
        textfont=dict(color=T.TEXT_2, size=10),
        marker=dict(size=sz, sizemode="area", sizeref=2.0 * sz.max() / (40 ** 2), sizemin=5, color=T.ACCENT,
                    opacity=0.85, line=dict(width=1.5, color=T.BG)),
        customdata=np.column_stack([df["dealership_name"], df["revenue"], df["units"], df["gpu"]]),
        hovertemplate="<b>%{customdata[0]}</b> · %{text}<br>Revenue: $%{customdata[1]:,.0f}"
                      "<br>Units: %{customdata[2]:,}<br>Gross/unit: $%{customdata[3]:,.0f}<extra></extra>",
    ))
    fig.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor=T.SURFACE_2,
                    showsubunits=True, subunitcolor="#30343c", bgcolor="rgba(0,0,0,0)",
                    lataxis_range=[27, 44], lonaxis_range=[-108, -84])
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    _title(fig, title, subtitle)
    return fig


def state_choropleth(df: pd.DataFrame, state: str, value: str, *, title: str | None = None,
                     subtitle: str | None = None) -> go.Figure:
    if df.empty:
        return empty_figure()
    fig = go.Figure(go.Choropleth(
        locations=df[state], z=df[value], locationmode="USA-states",
        colorscale=[[i / (len(T.SEQUENTIAL) - 1), c] for i, c in enumerate(T.SEQUENTIAL)],
        marker_line_color=T.BG, marker_line_width=0.6,
        colorbar=dict(thickness=10, outlinewidth=0, tickprefix="$", tickformat="~s",
                      tickfont=dict(color=T.TEXT_2, size=10)),
        hovertemplate="%{location}: $%{z:,.0f}<extra></extra>",
    ))
    fig.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor=T.SURFACE,
                    showsubunits=True, subunitcolor=T.BORDER, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    _title(fig, title, subtitle)
    return fig


def aging_bars(dist: pd.DataFrame, buckets: list[str], *, title: str | None = None,
               subtitle: str | None = None) -> go.Figure:
    """Units per aging bucket, ordinal blue ramp (older = brighter)."""
    d = dist.set_index("aging_bucket").reindex(buckets).fillna(0).reset_index()
    total = d["units"].sum()
    fig = go.Figure(go.Bar(
        x=d["aging_bucket"], y=d["units"], marker_color=T.AGING_COLORS,
        text=[f"{u:,.0f} · {u / total:.0%}" if total else "0" for u in d["units"]],
        textposition="outside", textfont=dict(color=T.TEXT_2, size=11), cliponaxis=False,
        customdata=d["cost_basis"],
        hovertemplate="%{x}<br>Units: %{y:,}<br>Cost basis: $%{customdata:,.0f}<extra></extra>"))
    fig.update_yaxes(showgrid=True, rangemode="tozero")
    fig.update_xaxes(tickfont=dict(color=T.TEXT, size=12))
    _title(fig, title, subtitle)
    return fig


def purchase_timeline(h: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    if h.empty:
        return empty_figure()
    fig = go.Figure()
    for cond, color in T.CONDITION_COLORS.items():
        d = h[h["condition"].astype(str) == cond]
        if d.empty:
            continue
        fig.add_scatter(x=d["sale_date"], y=d["sale_price"], mode="markers", name=cond,
                        marker=dict(color=color, size=10, line=dict(width=2, color=T.BG)),
                        customdata=np.column_stack([d["manufacturer"].astype(str), d["model"].astype(str),
                                                    d["model_year"], d["dealership_name"].astype(str)]),
                        hovertemplate="%{x|%b %d, %Y}<br>%{customdata[2]} %{customdata[0]} %{customdata[1]}"
                                      "<br>$%{y:,.0f} · %{customdata[3]}<extra></extra>")
    fig.update_yaxes(showgrid=True, tickprefix="$", tickformat="~s", rangemode="tozero")
    _title(fig, title)
    return fig
