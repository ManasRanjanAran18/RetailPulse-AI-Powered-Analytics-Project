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

PALETTE = px.colors.qualitative.Set2
STATUS_COLORS = {"Critical": "#d62728", "Reorder": "#ff7f0e",
                 "Healthy": "#2ca02c", "Overstock": "#1f77b4"}
RISK_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
SEGMENT_ACTIONS = {
    "Champions": "Reward & retain: early access, loyalty perks, referral programmes.",
    "Loyal Customers": "Grow basket size: cross-sell, bundles, tier upgrades.",
    "At Risk": "Win back now: personalised offers, reminders, feedback outreach.",
    "Hibernating": "Low-cost reactivation: seasonal campaigns; stop heavy spend if no response.",
}


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

    monthly = (df.assign(month=df["order_date"].dt.to_period("M").dt.to_timestamp())
                 .groupby("month", as_index=False).agg(revenue=("revenue", "sum"),
                                                       units=("quantity", "sum")))
    fig = px.area(monthly, x="month", y="revenue", title="Monthly revenue",
                  color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Revenue ($)",
                      xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    by_cat = df.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue")
    fig = px.bar(by_cat, x="revenue", y="category", orientation="h", title="Revenue by category",
                 color_discrete_sequence=[PALETTE[1]])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    by_store = df.groupby("store", as_index=False)["revenue"].sum()
    fig = px.pie(by_store, names="store", values="revenue", hole=0.45, title="Revenue by channel",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    b.plotly_chart(fig, width="stretch")

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
        fig.add_scatter(x=hist["date"], y=hist["units"], name="Actual", line=dict(color="#444"))
        fig.add_scatter(x=fut["date"], y=fut["ensemble_upper"], line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fut["date"], y=fut["ensemble_lower"], fill="tonexty",
                        fillcolor="rgba(31,119,180,0.18)", line=dict(width=0),
                        name="90% interval", hoverinfo="skip")
        fig.add_scatter(x=fut["date"], y=fut["ensemble"], name="Ensemble forecast",
                        line=dict(color="#1f77b4", width=3))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="Units / day", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
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
        fig.add_scatter(x=pre["date"], y=pre["units"], name="History", line=dict(color="#aaa"))
        fig.add_scatter(x=back["date"], y=back["actual"], name="Actual", line=dict(color="#222", width=3))
        for i, m in enumerate(picks):
            fig.add_scatter(x=back["date"], y=back[m], name=m.upper(),
                            line=dict(color=PALETTE[i + 1], dash="dot"))
        fig.add_scatter(x=back["date"], y=back["ensemble"], name="Ensemble",
                        line=dict(color="#1f77b4", width=3))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="Units / day", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Backtest accuracy (60-day holdout)** - lower is better")
    m = metrics.copy()
    for c in ("mae", "rmse", "bias"):
        m[c] = m[c].round(2)
    m["wape"] = (m["wape"] * 100).round(1).astype(str) + "%"
    st.dataframe(m, width="stretch", hide_index=True)
    summ = read_json(str(REPORTS_DIR / "pipeline_summary.json")).get("forecast", {})
    if summ.get("weights"):
        w = ", ".join(f"{k.upper()} {v * 100:.0f}%" for k, v in summ["weights"].items())
        st.caption(f"Ensemble weights (inverse-MAE): {w}")


def page_segments() -> None:
    st.subheader("Customer segmentation (RFM + K-Means)")
    seg = read_csv("customer_segments.csv")
    summ = read_csv("segment_summary.csv")
    ksearch = read_csv("segmentation_k_search.csv")

    k1, k2, k3 = st.columns(3)
    k1.metric("Customers segmented", f"{len(seg):,}")
    k2.metric("Segments", seg["segment"].nunique())
    k3.metric("Top segment revenue share", f"{summ['revenue_share_pct'].max():.0f}%")

    a, b = st.columns([3, 2])
    sample = seg.sample(min(len(seg), 1500), random_state=1)
    fig = px.scatter(sample, x="frequency", y="monetary", color="segment", log_x=True, log_y=True,
                     hover_data=["customer_id", "recency"], opacity=0.7,
                     color_discrete_sequence=PALETTE, title="Customers by frequency vs. spend")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, width="stretch")
    fig = px.bar(summ, x="segment", y="customers", color="segment", text="customers",
                 color_discrete_sequence=PALETTE, title="Customers per segment")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
                      xaxis_title=None)
    b.plotly_chart(fig, width="stretch")

    st.markdown("**Segment profile**")
    st.dataframe(summ, width="stretch", hide_index=True)
    st.markdown("**Recommended actions**")
    for name in summ["segment"]:
        st.markdown(f"- **{name}** - {SEGMENT_ACTIONS.get(name, 'Review segment behaviour.')}")

    with st.expander("How was k=4 chosen? (silhouette / elbow)"):
        fig = go.Figure()
        fig.add_scatter(x=ksearch["k"], y=ksearch["inertia"], name="Inertia", yaxis="y")
        fig.add_scatter(x=ksearch["k"], y=ksearch["silhouette"], name="Silhouette", yaxis="y2")
        fig.update_layout(height=320, xaxis_title="k", yaxis=dict(title="Inertia"),
                          yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
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
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
                      xaxis_title=None)
    a.plotly_chart(fig, width="stretch")
    top_imp = imp.head(10).iloc[::-1]
    fig = px.bar(top_imp, x="importance", y="feature", orientation="h",
                 title="What drives churn (top features)", color_discrete_sequence=[PALETTE[2]])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None)
    b.plotly_chart(fig, width="stretch")

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
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
                      xaxis_title=None)
    a.plotly_chart(fig, width="stretch")

    ranked = plan.sort_values("days_of_cover").head(15)
    fig = go.Figure()
    fig.add_bar(x=ranked["product_id"], y=ranked["current_stock"], name="Current stock",
                marker_color=[STATUS_COLORS[s] for s in ranked["status"]])
    fig.add_scatter(x=ranked["product_id"], y=ranked["reorder_point"], name="Reorder point",
                    mode="markers", marker=dict(symbol="line-ew-open", size=22, color="#111",
                                                line=dict(width=3)))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10),
                      title="15 lowest-cover products: stock vs. reorder point",
                      legend=dict(orientation="h", y=-0.2))
    b.plotly_chart(fig, width="stretch")

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

    tabs = st.tabs(["Transaction drift", "Customer-feature drift", "Forecast error", "Data quality"])
    for tab, name, title in ((tabs[0], "drift_transactions.csv", "Transactions: first 12 months vs. last 90 days"),
                             (tabs[1], "drift_customer_features.csv", "Customer features: 6 months ago vs. today")):
        with tab:
            df = read_csv(name)
            colors = {"OK": "#2ca02c", "WARN": "#ff7f0e", "ALERT": "#d62728", "n/a": "#999"}
            fig = px.bar(df, x="psi", y="feature", orientation="h", color="status",
                         color_discrete_map=colors, title=title)
            fig.add_vline(x=0.10, line_dash="dash", line_color="#ff7f0e")
            fig.add_vline(x=0.25, line_dash="dash", line_color="#d62728")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None,
                              xaxis_title="Population Stability Index")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df, width="stretch", hide_index=True)
            st.caption("PSI < 0.10 stable · 0.10-0.25 moderate shift (WARN) · > 0.25 significant shift (ALERT).")
    with tabs[2]:
        err = read_csv("drift_forecast_error.csv", ("date",))
        fig = go.Figure()
        fig.add_scatter(x=err["date"], y=err["abs_error"], name="Abs. error", opacity=0.45)
        fig.add_scatter(x=err["date"], y=err["rolling_error"], name="14-day rolling error",
                        line=dict(width=3))
        fig.add_scatter(x=err["date"], y=err["upper_limit"], name="Control limit",
                        line=dict(dash="dash", color="#d62728"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Units",
                          legend=dict(orientation="h", y=1.1))
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
