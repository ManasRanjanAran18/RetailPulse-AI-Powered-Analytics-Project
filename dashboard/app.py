"""RetailPulse - interactive analytics dashboard (Streamlit).

Run locally:   streamlit run dashboard/app.py
The app reads pre-computed artifacts from data/processed/. If they are missing
(fresh clone without artifacts) it builds them automatically on first launch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (ACTIVE_WINDOW_DAYS, CHURN_HORIZON_DAYS, PROCESSED_DIR,  # noqa: E402
                        RAW_DIR, REPORTS_DIR, SERVICE_LEVEL)
from src.inventory.optimizer import optimize_inventory  # noqa: E402

st.set_page_config(page_title="RetailPulse", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")


# ---------- HORIZONTAL THEME SELECTOR ----------
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = "Light"

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"]:has(
        div[data-testid="stRadio"]
    ) {
        justify-content: flex-end;
    }

    div[data-testid="stRadio"] > div {
        flex-direction: row;
        align-items: center;
        gap: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

_, theme_col = st.columns([5, 2])

with theme_col:
    theme = st.radio(
        "🎨 Theme",
        ["☀️ Light", "🌙 Dark", "🔵 Blue"],
        horizontal=True,
        key="theme_selector",
        label_visibility="collapsed",
        index=["Light", "Dark", "Blue"].index(
            st.session_state.selected_theme
        )
    )

st.session_state.selected_theme = {
    "☀️ Light": "Light",
    "🌙 Dark": "Dark",
    "🔵 Blue": "Blue"
}[theme]

# ---------- THEME COLORS ----------
themes = {
    "Light": {
        "bg": "#FFFFFF",
        "sidebar": "#F1F5F9",
        "text": "#1F2937",
        "card": "#F8FAFC",
        "border": "#E2E8F0",
        "grid": "rgba(100,116,139,0.18)",
    },
    "Dark": {
        "bg": "#0E1117",
        "sidebar": "#161B22",
        "text": "#F1F5F9",
        "card": "#1E293B",
        "border": "#334155",
        "grid": "rgba(148,163,184,0.18)",
    },
    "Blue": {
        "bg": "#EAF3FF",
        "sidebar": "#D6E8FF",
        "text": "#12345A",
        "card": "#FFFFFF",
        "border": "#BBD6F5",
        "grid": "rgba(18,52,90,0.15)",
    }
}

colors = themes[st.session_state.selected_theme]

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {colors["bg"]};
        color: {colors["text"]};
    }}

    [data-testid="stHeader"] {{
        background-color: {colors["bg"]};
        color: {colors["text"]};
    }}

    [data-testid="stHeader"] button,
    [data-testid="stHeader"] svg {{
        color: {colors["text"]};
        fill: {colors["text"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {colors["sidebar"]};
    }}

    [data-testid="stMetricValue"] > div {{
        font-size: 1.7rem;
        overflow: visible;
        text-overflow: clip;
        white-space: normal;
    }}

    [data-testid="stMetricLabel"] p {{
        white-space: normal;
    }}

    [data-testid="stMetric"] {{
        background-color: {colors["card"]};
        border: 1px solid {colors["border"]};
        border-left: 4px solid #4F46E5;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}

    .stApp h1, .stApp h2, .stApp h3,
    .stApp p, .stApp label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {{
        color: {colors["text"]};
    }}

    .stApp h1 {{
        font-weight: 800;
        letter-spacing: -0.5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ---------- CHART PALETTE (consistent across all sections) ----------
PALETTE = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EC4899",
           "#8B5CF6", "#14B8A6", "#F97316"]
PRIMARY, SECONDARY, ACCENT = PALETTE[0], PALETTE[1], PALETTE[3]
NEUTRAL = "#64748B"
SEQ_SCALE = ["#E0E7FF", "#A5B4FC", "#6366F1", "#4338CA", "#312E81"]

STATUS_COLORS = {"Critical": "#EF4444", "Reorder": "#F59E0B",
                 "Healthy": "#10B981", "Overstock": "#3B82F6"}
RISK_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
DRIFT_COLORS = {"OK": "#10B981", "WARN": "#F59E0B", "ALERT": "#EF4444", "n/a": "#94A3B8"}
SEGMENT_ACTIONS = {
    "Champions": "Reward & retain: early access, loyalty perks, referral programmes.",
    "Loyal Customers": "Grow basket size: cross-sell, bundles, tier upgrades.",
    "At Risk": "Win back now: personalised offers, reminders, feedback outreach.",
    "Hibernating": "Low-cost reactivation: seasonal campaigns; stop heavy spend if no response.",
}


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply the shared, theme-aware look to every chart."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text"]),
        title_font=dict(size=16, color=colors["text"]),
        legend=dict(font=dict(color=colors["text"])),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor=colors["grid"], zeroline=False,
                     tickfont=dict(color=colors["text"]),
                     title_font=dict(color=colors["text"]))
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False,
                     tickfont=dict(color=colors["text"]),
                     title_font=dict(color=colors["text"]))
    return fig


# ------------------------------------------------------------------ data layer
@st.cache_resource(show_spinner="First launch: building analytics artifacts (about a minute)…")
def bootstrap() -> bool:
    """Make sure processed artifacts exist; build them (lean models) if not."""
    if not (PROCESSED_DIR / "forecast_future.csv").exists():
        from src.pipeline import run_pipeline
        run_pipeline(regenerate=not (RAW_DIR / "transactions.csv").exists(),
                     skip_prophet=True, skip_lstm=True)
    return True


@st.cache_data(show_spinner=False)
def read_csv(name: str, dates: tuple = ()) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name, parse_dates=list(dates))


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def money(x: float) -> str:
    return f"${x:,.0f}"


def download(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode(), filename, "text/csv")


# ------------------------------------------------------------------- sections
def page_overview(tx: pd.DataFrame) -> None:
    st.subheader("Business overview")
    c1, c2, c3 = st.columns([2, 2, 1])
    d_min, d_max = tx["order_date"].min().date(), tx["order_date"].max().date()
    rng = c1.date_input("Date range", (d_min, d_max), min_value=d_min, max_value=d_max)
    cats = sorted(tx["category"].unique())
    sel_cats = c2.multiselect("Categories", cats, default=cats)
    if not isinstance(rng, tuple) or len(rng) != 2:
        st.info("Select a start and end date.")
        return
    mask = (tx["order_date"].dt.date >= rng[0]) & (tx["order_date"].dt.date <= rng[1]) \
        & tx["category"].isin(sel_cats)
    df = tx[mask]
    if df.empty:
        st.warning("No data for this selection.")
        return

    end = df["order_date"].max()
    recent_cust = df[df["order_date"] > end - pd.Timedelta(days=90)]["customer_id"].nunique()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Revenue", money(df["revenue"].sum()))
    k2.metric("Orders", f"{df['order_id'].nunique():,}")
    k3.metric("Units sold", f"{int(df['quantity'].sum()):,}")
    k4.metric("Avg order value", money(df["revenue"].mean()))
    k5.metric("Active customers (90d)", f"{recent_cust:,}")

    df = df.assign(month=df["order_date"].dt.to_period("M").dt.to_timestamp())
    monthly = (df.groupby("month", as_index=False)
                 .agg(revenue=("revenue", "sum"), units=("quantity", "sum")))
    fig = px.area(monthly, x="month", y="revenue", title="Monthly revenue",
                  color_discrete_sequence=[PRIMARY])
    fig.update_traces(line=dict(width=2.5), fillcolor=hex_to_rgba(PRIMARY, 0.25))
    style(fig, 340)
    fig.update_layout(yaxis_title="Revenue ($)", xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    by_cat = df.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue")
    fig = px.bar(by_cat, x="revenue", y="category", orientation="h", title="Revenue by category",
                 color="revenue", color_continuous_scale=SEQ_SCALE)
    style(fig, 320)
    fig.update_layout(yaxis_title=None, coloraxis_showscale=False)
    a.plotly_chart(fig, width="stretch")
    by_store = df.groupby("store", as_index=False)["revenue"].sum()
    fig = px.pie(by_store, names="store", values="revenue", hole=0.5, title="Revenue by channel",
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      marker=dict(line=dict(color=colors["bg"], width=2)))
    style(fig, 320)
    b.plotly_chart(fig, width="stretch")

    # ---- NEW: category trend + weekday pattern
    c, d = st.columns([3, 2])
    cat_month = df.groupby(["month", "category"], as_index=False)["revenue"].sum()
    fig = px.area(cat_month, x="month", y="revenue", color="category",
                  title="Monthly revenue by category (stacked)",
                  color_discrete_sequence=PALETTE)
    style(fig, 340)
    fig.update_layout(yaxis_title="Revenue ($)", xaxis_title=None,
                      legend=dict(orientation="h", y=-0.2))
    c.plotly_chart(fig, width="stretch")

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wk = (df.assign(weekday=df["order_date"].dt.day_name())
            .groupby("weekday", as_index=False)["revenue"].sum())
    wk["weekday"] = pd.Categorical(wk["weekday"], categories=order, ordered=True)
    wk = wk.sort_values("weekday")
    fig = px.bar(wk, x="weekday", y="revenue", title="Revenue by day of week",
                 color="revenue", color_continuous_scale=SEQ_SCALE)
    style(fig, 340)
    fig.update_layout(xaxis_title=None, yaxis_title="Revenue ($)", coloraxis_showscale=False)
    d.plotly_chart(fig, width="stretch")

    top = (df.groupby("product_id", as_index=False)
             .agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
             .sort_values("revenue", ascending=False).head(10))
    st.markdown("**Top 10 products by revenue**")
    top["revenue"] = top["revenue"].round(0)
    st.dataframe(top, width="stretch", hide_index=True)


def page_forecast() -> None:
    st.subheader("Demand forecasting - hybrid ensemble (GBM + Prophet + LSTM)")
    daily = read_csv("daily_sales.csv", ("date",))
    back = read_csv("forecast_backtest.csv", ("date",))
    fut = read_csv("forecast_future.csv", ("date",))
    metrics = read_csv("forecast_metrics.csv")
    model_cols = [c for c in back.columns if c not in ("date", "actual", "ensemble")]

    best = metrics.set_index("model")["mae"].idxmin()
    ens = metrics.set_index("model").loc["ensemble (hybrid)"]
    base = metrics.set_index("model").drop(index="ensemble (hybrid)")["mae"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ensemble MAE (units/day)", f"{ens['mae']:.1f}",
              delta=f"{ens['mae'] - base.min():+.1f} vs best single model", delta_color="inverse")
    k2.metric("Ensemble WAPE", f"{ens['wape'] * 100:.1f}%")
    k3.metric("Next-30-day demand", f"{fut['ensemble'].sum():,.0f} units")
    k4.metric("Best model (MAE)", best.title())

    horizon_view = st.radio("View", ["Forecast (next 30 days)", "Backtest (last 60 days)"],
                            horizontal=True)
    if horizon_view.startswith("Forecast"):
        hist = daily.tail(120)
        fig = go.Figure()
        fig.add_scatter(x=hist["date"], y=hist["units"], name="Actual",
                        line=dict(color=NEUTRAL, width=2))
        fig.add_scatter(x=fut["date"], y=fut["ensemble_upper"], line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fut["date"], y=fut["ensemble_lower"], fill="tonexty",
                        fillcolor=hex_to_rgba(PRIMARY, 0.18), line=dict(width=0),
                        name="90% interval", hoverinfo="skip")
        fig.add_scatter(x=fut["date"], y=fut["ensemble"], name="Ensemble forecast",
                        line=dict(color=PRIMARY, width=3))
        style(fig, 420)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="Units / day", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

        # ---- NEW: cumulative forecast demand
        a, b = st.columns(2)
        cum = fut[["date", "ensemble", "ensemble_lower", "ensemble_upper"]].copy()
        for c in ("ensemble", "ensemble_lower", "ensemble_upper"):
            cum[c] = cum[c].cumsum()
        fig = go.Figure()
        fig.add_scatter(x=cum["date"], y=cum["ensemble_upper"], line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=cum["date"], y=cum["ensemble_lower"], fill="tonexty",
                        fillcolor=hex_to_rgba(SECONDARY, 0.2), line=dict(width=0),
                        name="90% interval", hoverinfo="skip")
        fig.add_scatter(x=cum["date"], y=cum["ensemble"], name="Cumulative demand",
                        line=dict(color=SECONDARY, width=3))
        style(fig, 330)
        fig.update_layout(title="Cumulative expected demand (next 30 days)",
                          yaxis_title="Units", legend=dict(orientation="h", y=-0.2))
        a.plotly_chart(fig, width="stretch")

        # ---- NEW: weekday demand pattern from history
        order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pat = (daily.assign(day=daily["date"].dt.day_name().str[:3])
                    .groupby("day", as_index=False)["units"].mean())
        pat["day"] = pd.Categorical(pat["day"], categories=order, ordered=True)
        pat = pat.sort_values("day")
        fig = px.bar(pat, x="day", y="units", title="Average daily units by weekday (history)",
                     color="units", color_continuous_scale=SEQ_SCALE)
        style(fig, 330)
        fig.update_layout(xaxis_title=None, yaxis_title="Avg units / day",
                          coloraxis_showscale=False)
        b.plotly_chart(fig, width="stretch")

        num_cols = ["ensemble", "ensemble_lower", "ensemble_upper"] + model_cols
        show = fut[["date"] + num_cols].copy()
        show[num_cols] = show[num_cols].round(1)
        show["date"] = show["date"].dt.date
        with st.expander("Forecast table"):
            st.dataframe(show, width="stretch", hide_index=True)
        download(show, "Download forecast CSV", "retailpulse_forecast.csv")
    else:
        picks = st.multiselect("Models to display", model_cols, default=model_cols)
        pre = daily[daily["date"] < back["date"].min()].tail(60)
        fig = go.Figure()
        fig.add_scatter(x=pre["date"], y=pre["units"], name="History",
                        line=dict(color="#94A3B8"))
        fig.add_scatter(x=back["date"], y=back["actual"], name="Actual",
                        line=dict(color=colors["text"], width=3))
        for i, m in enumerate(picks):
            fig.add_scatter(x=back["date"], y=back[m], name=m.upper(),
                            line=dict(color=PALETTE[(i + 1) % len(PALETTE)], dash="dot"))
        fig.add_scatter(x=back["date"], y=back["ensemble"], name="Ensemble",
                        line=dict(color=PRIMARY, width=3))
        style(fig, 420)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="Units / day", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Backtest accuracy (60-day holdout)** - lower is better")
    m = metrics.copy()
    for c in ("mae", "rmse", "bias"):
        m[c] = m[c].round(2)
    m["wape"] = (m["wape"] * 100).round(1).astype(str) + "%"
    st.dataframe(m, width="stretch", hide_index=True)

    # ---- NEW: model comparison + residual distribution
    a, b = st.columns(2)
    cmp_df = metrics.copy()
    cmp_df["type"] = np.where(cmp_df["model"].str.contains("ensemble"), "Ensemble", "Single model")
    fig = px.bar(cmp_df.sort_values("mae"), x="model", y="mae", color="type", text="mae",
                 title="Model comparison - MAE (lower is better)",
                 color_discrete_map={"Ensemble": PRIMARY, "Single model": "#94A3B8"})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    style(fig, 340)
    fig.update_layout(xaxis_title=None, yaxis_title="MAE (units/day)",
                      legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")

    resid = back.assign(residual=back["actual"] - back["ensemble"])
    fig = px.histogram(resid, x="residual", nbins=20, title="Ensemble error distribution (actual - forecast)",
                       color_discrete_sequence=[SECONDARY])
    fig.add_vline(x=0, line_dash="dash", line_color=colors["text"])
    style(fig, 340)
    fig.update_layout(xaxis_title="Residual (units/day)", yaxis_title="Days", bargap=0.08)
    b.plotly_chart(fig, width="stretch")

    summ = read_json(str(REPORTS_DIR / "pipeline_summary.json")).get("forecast", {})
    if summ.get("weights"):
        w = ", ".join(f"{k.upper()} {v * 100:.0f}%" for k, v in summ["weights"].items())
        st.caption(f"Ensemble weights (inverse-MAE): {w}")


def page_segments() -> None:
    st.subheader("Customer segmentation (RFM + K-Means)")
    seg = read_csv("customer_segments.csv")
    summ = read_csv("segment_summary.csv")
    ksearch = read_csv("segmentation_k_search.csv")
    seg_colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sorted(seg["segment"].unique()))}

    k1, k2, k3 = st.columns(3)
    k1.metric("Customers segmented", f"{len(seg):,}")
    k2.metric("Segments", seg["segment"].nunique())
    k3.metric("Top segment revenue share", f"{summ['revenue_share_pct'].max():.0f}%")

    a, b = st.columns([3, 2])
    sample = seg.sample(min(len(seg), 1500), random_state=1)
    fig = px.scatter(sample, x="frequency", y="monetary", color="segment", log_x=True, log_y=True,
                     hover_data=["customer_id", "recency"], opacity=0.7,
                     color_discrete_map=seg_colors, title="Customers by frequency vs. spend")
    style(fig, 400)
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")
    fig = px.bar(summ, x="segment", y="customers", color="segment", text="customers",
                 color_discrete_map=seg_colors, title="Customers per segment")
    style(fig, 400)
    fig.update_layout(showlegend=False, xaxis_title=None)
    b.plotly_chart(fig, width="stretch")

    # ---- NEW: revenue share donut + RFM radar
    c, d = st.columns(2)
    fig = px.pie(summ, names="segment", values="revenue_share_pct", hole=0.5,
                 color="segment", color_discrete_map=seg_colors, title="Revenue share by segment")
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      marker=dict(line=dict(color=colors["bg"], width=2)))
    style(fig, 380)
    fig.update_layout(showlegend=False)
    c.plotly_chart(fig, width="stretch")

    prof = seg.groupby("segment")[["recency", "frequency", "monetary"]].mean()
    prof_n = prof / prof.max()
    axes = ["Recency (days)", "Frequency", "Monetary"]
    fig = go.Figure()
    for s in prof_n.index:
        vals = prof_n.loc[s].tolist()
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], name=s,
                                      fill="toself", line=dict(color=seg_colors[s], width=2),
                                      fillcolor=hex_to_rgba(seg_colors[s], 0.18)))
    style(fig, 380)
    fig.update_layout(title="RFM profile by segment (normalised)",
                      polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(visible=True, range=[0, 1],
                                                 gridcolor=colors["grid"]),
                                 angularaxis=dict(gridcolor=colors["grid"])),
                      legend=dict(orientation="h", y=-0.1))
    d.plotly_chart(fig, width="stretch")

    st.markdown("**Segment profile**")
    st.dataframe(summ, width="stretch", hide_index=True)
    st.markdown("**Recommended actions**")
    for name in summ["segment"]:
        st.markdown(f"- **{name}** - {SEGMENT_ACTIONS.get(name, 'Review segment behaviour.')}")

    with st.expander("How was k=4 chosen? (silhouette / elbow)"):
        fig = go.Figure()
        fig.add_scatter(x=ksearch["k"], y=ksearch["inertia"], name="Inertia", yaxis="y",
                        mode="lines+markers", line=dict(color=PRIMARY, width=3))
        fig.add_scatter(x=ksearch["k"], y=ksearch["silhouette"], name="Silhouette", yaxis="y2",
                        mode="lines+markers", line=dict(color=ACCENT, width=3))
        style(fig, 320)
        fig.update_layout(xaxis_title="k", yaxis=dict(title="Inertia"),
                          yaxis2=dict(title="Silhouette", overlaying="y", side="right",
                                      showgrid=False),
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("k=4 balances statistical fit with four actionable business personas.")
    download(seg, "Download segments CSV", "retailpulse_segments.csv")


def page_churn() -> None:
    st.subheader("Churn prediction")
    scores = read_csv("churn_scores.csv")
    imp = read_csv("churn_feature_importance.csv")
    seg = read_csv("customer_segments.csv")[["customer_id", "segment"]]
    summary = read_json(str(REPORTS_DIR / "pipeline_summary.json")).get("churn", {})
    scores = scores.merge(seg, on="customer_id", how="left")
    scores["segment"] = scores["segment"].fillna("Unknown")
    active = scores[scores["active"]]

    st.caption(f"Definition: an active customer (bought in the last {ACTIVE_WINDOW_DAYS} days) is "
               f"predicted to churn if they make no purchase in the next {CHURN_HORIZON_DAYS} days.")
    if summary:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ROC-AUC", f"{summary['roc_auc']:.3f}")
        k2.metric("Precision (churn)", f"{summary['precision']:.2f}")
        k3.metric("Recall (churn)", f"{summary['recall']:.2f}")
        k4.metric("Baseline churn rate", f"{summary['base_rate'] * 100:.0f}%")

    high = active[active["risk_band"] == "High"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Active customers scored", f"{len(active):,}")
    k2.metric("High-risk customers", f"{len(high):,}")
    k3.metric("Revenue at risk (lifetime spend)", money(high["monetary"].sum()))

    a, b = st.columns(2)
    counts = active["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    counts.columns = ["risk_band", "customers"]
    fig = px.bar(counts, x="risk_band", y="customers", color="risk_band", text="customers",
                 color_discrete_map=RISK_COLORS, title="Active customers by risk band")
    style(fig, 340)
    fig.update_layout(showlegend=False, xaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    top_imp = imp.head(10).iloc[::-1]
    fig = px.bar(top_imp, x="importance", y="feature", orientation="h",
                 title="What drives churn (top features)",
                 color="importance", color_continuous_scale=SEQ_SCALE)
    style(fig, 340)
    fig.update_layout(yaxis_title=None, coloraxis_showscale=False)
    b.plotly_chart(fig, width="stretch")

    # ---- NEW: probability distribution + revenue at risk by segment
    c, d = st.columns(2)
    fig = px.histogram(active, x="churn_probability", color="risk_band", nbins=30,
                       color_discrete_map=RISK_COLORS, barmode="overlay", opacity=0.8,
                       category_orders={"risk_band": ["Low", "Medium", "High"]},
                       title="Churn probability distribution")
    style(fig, 340)
    fig.update_layout(xaxis_title="Predicted churn probability", yaxis_title="Customers",
                      legend=dict(orientation="h", y=-0.2), bargap=0.05)
    c.plotly_chart(fig, width="stretch")

    risk_seg = (high.groupby("segment", as_index=False)["monetary"].sum()
                    .sort_values("monetary", ascending=False))
    fig = px.bar(risk_seg, x="segment", y="monetary", title="Revenue at risk by segment (high-risk)",
                 color="monetary", color_continuous_scale=["#FDE68A", "#F97316", "#DC2626"])
    style(fig, 340)
    fig.update_layout(xaxis_title=None, yaxis_title="Lifetime spend ($)",
                      coloraxis_showscale=False)
    d.plotly_chart(fig, width="stretch")

    st.markdown("**Highest-risk active customers**")
    n = st.slider("Rows to show", 10, 200, 25, step=5)
    table = high.head(n)[["customer_id", "segment", "churn_probability", "recency",
                          "frequency", "monetary", "risk_band"]]
    st.dataframe(table, width="stretch", hide_index=True)
    download(high, "Download all high-risk customers", "retailpulse_high_risk_customers.csv")


def page_inventory() -> None:
    st.subheader("Inventory optimisation")
    inv = read_csv("inventory.csv")
    demand = read_csv("product_demand_forecast.csv")
    products = read_csv("products.csv")[["product_id", "product_name", "category"]]

    sl = st.slider("Target service level", 0.80, 0.995, float(SERVICE_LEVEL), step=0.005,
                   format="%.3f", help="Probability of not stocking out during lead time.")
    plan = optimize_inventory(inv, demand, service_level=sl).merge(products, on="product_id")

    counts = plan["status"].value_counts()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Critical (below safety stock)", int(counts.get("Critical", 0)))
    k2.metric("Reorder now", int(counts.get("Reorder", 0)))
    k3.metric("Healthy", int(counts.get("Healthy", 0)))
    k4.metric("Overstock", int(counts.get("Overstock", 0)))
    k5.metric("Capital tied in excess", money(plan["excess_value"].sum()))

    a, b = st.columns([2, 3])
    sc = counts.reindex(["Critical", "Reorder", "Healthy", "Overstock"]).fillna(0).reset_index()
    sc.columns = ["status", "products"]
    fig = px.bar(sc, x="status", y="products", color="status", text="products",
                 color_discrete_map=STATUS_COLORS, title="Products by stock status")
    style(fig, 360)
    fig.update_layout(showlegend=False, xaxis_title=None)
    a.plotly_chart(fig, width="stretch")

    ranked = plan.sort_values("days_of_cover").head(15)
    fig = go.Figure()
    fig.add_bar(x=ranked["product_id"], y=ranked["current_stock"], name="Current stock",
                marker_color=[STATUS_COLORS[s] for s in ranked["status"]])
    fig.add_scatter(x=ranked["product_id"], y=ranked["reorder_point"], name="Reorder point",
                    mode="markers", marker=dict(symbol="line-ew-open", size=22,
                                                color=colors["text"], line=dict(width=3)))
    style(fig, 360)
    fig.update_layout(title="15 lowest-cover products: stock vs. reorder point",
                      legend=dict(orientation="h", y=-0.2))
    b.plotly_chart(fig, width="stretch")

    # ---- NEW: status mix by category + days-of-cover distribution
    c, d = st.columns(2)
    cat_status = plan.groupby(["category", "status"]).size().reset_index(name="products")
    fig = px.bar(cat_status, x="category", y="products", color="status",
                 color_discrete_map=STATUS_COLORS, barmode="stack",
                 category_orders={"status": ["Critical", "Reorder", "Healthy", "Overstock"]},
                 title="Stock status mix by category")
    style(fig, 340)
    fig.update_layout(xaxis_title=None, yaxis_title="Products",
                      legend=dict(orientation="h", y=-0.25))
    c.plotly_chart(fig, width="stretch")

    cover = plan.replace([np.inf, -np.inf], np.nan).dropna(subset=["days_of_cover"]).copy()
    if not cover.empty:
        cover["days_of_cover"] = cover["days_of_cover"].clip(upper=cover["days_of_cover"].quantile(0.98))
        fig = px.histogram(cover, x="days_of_cover", color="status", nbins=30,
                           color_discrete_map=STATUS_COLORS, barmode="stack",
                           category_orders={"status": ["Critical", "Reorder", "Healthy", "Overstock"]},
                           title="Days of cover distribution")
        style(fig, 340)
        fig.update_layout(xaxis_title="Days of cover", yaxis_title="Products",
                          legend=dict(orientation="h", y=-0.25), bargap=0.05)
        d.plotly_chart(fig, width="stretch")

    status_filter = st.multiselect("Filter by status", list(STATUS_COLORS),
                                   default=["Critical", "Reorder"])
    cols = ["product_id", "product_name", "category", "status", "current_stock", "days_of_cover",
            "lead_time_days", "safety_stock", "reorder_point", "eoq",
            "recommended_order_qty", "order_cost_estimate", "supplier"]
    view = plan[plan["status"].isin(status_filter)][cols].sort_values("days_of_cover")
    st.dataframe(view, width="stretch", hide_index=True)
    orders = plan[plan["recommended_order_qty"] > 0][cols]
    k1, k2 = st.columns(2)
    k1.metric("Suggested purchase orders", len(orders))
    k2.metric("Estimated PO value", money(orders["order_cost_estimate"].sum()))
    download(orders, "Download purchase-order suggestions", "retailpulse_purchase_orders.csv")
    st.caption("Safety stock = z·σ·√lead-time. Reorder point = forecast demand × lead-time + safety stock. "
               "EOQ = √(2·D·S / H). Order-up-to level = reorder point + EOQ.")


def page_monitoring() -> None:
    st.subheader("Model monitoring & drift detection")
    summ = read_json(str(REPORTS_DIR / "pipeline_summary.json"))
    quality = read_json(str(PROCESSED_DIR / "quality_report.json"))
    if summ:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Last pipeline run (UTC)", summ.get("generated_at", "n/a")[:16].replace("T", " "))
        k2.metric("Pipeline runtime", f"{summ.get('runtime_sec', 0):.0f}s")
        k3.metric("Forecast models", ", ".join(m.upper() for m in summ.get("forecast", {}).get("models", [])))
        d = summ.get("drift", {})
        k4.metric("Drift alerts", d.get("transactions_alerts", 0) + d.get("feature_alerts", 0)
                  + d.get("forecast_error_breaches", 0))

    # ---- NEW: overall drift status summary across both drift tables
    drift_tx = read_csv("drift_transactions.csv").assign(source="Transactions")
    drift_cf = read_csv("drift_customer_features.csv").assign(source="Customer features")
    drift_all = pd.concat([drift_tx, drift_cf], ignore_index=True)
    status_counts = drift_all.groupby(["source", "status"]).size().reset_index(name="features")
    fig = px.bar(status_counts, x="source", y="features", color="status", barmode="stack",
                 color_discrete_map=DRIFT_COLORS, text="features",
                 category_orders={"status": ["OK", "WARN", "ALERT", "n/a"]},
                 title="Drift status summary (features per status)")
    style(fig, 300)
    fig.update_layout(xaxis_title=None, yaxis_title="Features",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    tabs = st.tabs(["Transaction drift", "Customer-feature drift", "Forecast error", "Data quality"])
    for tab, name, title in ((tabs[0], "drift_transactions.csv", "Transactions: first 12 months vs. last 90 days"),
                             (tabs[1], "drift_customer_features.csv", "Customer features: 6 months ago vs. today")):
        with tab:
            df = read_csv(name)
            fig = px.bar(df, x="psi", y="feature", orientation="h", color="status",
                         color_discrete_map=DRIFT_COLORS, title=title)
            fig.add_vline(x=0.10, line_dash="dash", line_color=DRIFT_COLORS["WARN"])
            fig.add_vline(x=0.25, line_dash="dash", line_color=DRIFT_COLORS["ALERT"])
            style(fig, 320)
            fig.update_layout(yaxis_title=None, xaxis_title="Population Stability Index")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df, width="stretch", hide_index=True)
            st.caption("PSI < 0.10 stable · 0.10-0.25 moderate shift (WARN) · > 0.25 significant shift (ALERT).")
    with tabs[2]:
        err = read_csv("drift_forecast_error.csv", ("date",))
        fig = go.Figure()
        fig.add_scatter(x=err["date"], y=err["abs_error"], name="Abs. error",
                        line=dict(color=SECONDARY), opacity=0.5)
        fig.add_scatter(x=err["date"], y=err["rolling_error"], name="14-day rolling error",
                        line=dict(width=3, color=PRIMARY))
        fig.add_scatter(x=err["date"], y=err["upper_limit"], name="Control limit",
                        line=dict(dash="dash", color=DRIFT_COLORS["ALERT"]))
        style(fig, 360)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Units",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

        # ---- NEW: error distribution
        fig = px.histogram(err, x="abs_error", nbins=30, title="Distribution of absolute forecast error",
                           color_discrete_sequence=[ACCENT])
        fig.add_vline(x=float(err["abs_error"].mean()), line_dash="dash", line_color=colors["text"],
                      annotation_text="mean", annotation_position="top")
        style(fig, 300)
        fig.update_layout(xaxis_title="Absolute error (units)", yaxis_title="Days", bargap=0.05)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Days where the rolling error exceeds the control limit: {int(err['breach'].sum())}. "
                   "Sustained breaches should trigger retraining.")
    with tabs[3]:
        if quality:
            st.json(quality)


# ------------------------------------------------------------------------ main
def main() -> None:
    bootstrap()
    tx = read_csv("transactions_clean.csv", ("order_date",))

    with st.sidebar:
        st.markdown("## 📈 RetailPulse")
        st.caption("AI-Powered Customer Analytics & Demand Forecasting")
        page = st.radio("Navigate", ["Overview", "Demand forecast", "Customer segments",
                                     "Churn risk", "Inventory", "Monitoring"],
                        label_visibility="collapsed")
        st.divider()
        st.caption(f"Data: {tx['order_date'].min():%b %Y} - {tx['order_date'].max():%b %Y}  \n"
                   f"{tx['customer_id'].nunique():,} customers · {tx['product_id'].nunique()} products")
        st.caption("Synthetic demo dataset. Plug in your own CSVs via `data/raw/`.")

    st.title("RetailPulse")
    {"Overview": lambda: page_overview(tx),
     "Demand forecast": page_forecast,
     "Customer segments": page_segments,
     "Churn risk": page_churn,
     "Inventory": page_inventory,
     "Monitoring": page_monitoring}[page]()


main()