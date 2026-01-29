# app.py  (Streamlit entrypoint)
# Put this at repo root (or in /app and point Streamlit to it)

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---- your project imports (adjust if your modules are inside a package folder) ----
from engine import BacktestEngine, EngineSpec, DataConfig, IndicatorsConfig, StrategyConfig
from portfolio import PortfolioConfig, CostModel


st.set_page_config(page_title="Backtester", layout="wide")
st.title("Backtester (TA) — Data → Indicators → Strategy → Portfolio → Results")


# --------------------------
# Small plotting helpers
# --------------------------
def plot_line(series: pd.Series, title: str, ylabel: str):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    return fig
import matplotlib.colors as mcolors

def plot_price_indicators_trades(price: pd.Series, indicators: Optional[pd.DataFrame], trades: pd.DataFrame):
    fig = plt.figure(figsize=(12, 5))
    plt.plot(price.index, price.values, label="Close")

    if indicators is not None and not indicators.empty:
        for c in indicators.columns:
            plt.plot(indicators.index, indicators[c].values, label=c)

    if trades is not None and not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"])
        buys = t[t["side"] == "BUY"]
        sells = t[t["side"] == "SELL"]
        if not buys.empty:
            plt.scatter(buys["timestamp"], buys["price"], marker="^", label="BUY")
        if not sells.empty:
            plt.scatter(sells["timestamp"], sells["price"], marker="v", label="SELL")

    plt.title("Price + Indicators + Trades")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    return fig


def plot_cum_vs_bench(cum: pd.Series, bench: Optional[pd.Series]):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(cum.index, cum.values, label="Strategy")
    if bench is not None:
        plt.plot(bench.index, bench.values, label="Benchmark")
    plt.title("Cumulative Returns vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.grid(True)
    return fig


def plot_drawdown_red(dd: pd.Series):
    fig = plt.figure(figsize=(12, 3))
    plt.plot(dd.index, dd.values)
    # fill where drawdown < 0
    plt.fill_between(dd.index, dd.values, 0.0, where=(dd.values < 0.0), alpha=0.35, color="red")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    return fig


def plot_monthly_heatmap(monthly: pd.DataFrame):
    # monthly: index=year, columns=1..12, values=returns
    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()

    if monthly is None or monthly.empty:
        ax.set_title("Monthly returns heatmap (no data)")
        return fig

    data = monthly.values.astype(float)

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # Robust normalization around 0
    if np.isnan(vmin) or np.isnan(vmax):
        ax.set_title("Monthly returns heatmap (no data)")
        return fig

    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        # all positive or all negative: simple norm
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, aspect="auto", norm=norm, cmap="RdYlGn")  # red->yellow->green
    ax.set_title("Monthly Returns Heatmap")
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index.astype(str).tolist())
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    return fig


def plot_yearly_bar(yearly: pd.Series):
    fig = plt.figure(figsize=(12, 3))
    ax = plt.gca()
    if yearly is None or yearly.empty:
        ax.set_title("Yearly returns (no data)")
        return fig

    years = yearly.index.astype(str).tolist()
    vals = yearly.values.astype(float)

    # color by sign
    colors = ["green" if v >= 0 else "red" for v in vals]
    ax.bar(years, vals, color=colors)
    ax.set_title("Yearly Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.grid(True, axis="y")
    return fig


def style_good_bad(df: pd.DataFrame, good_high: bool = True):
    # Simple generic styler: green for good, red for bad, per-cell numeric
    def color(v):
        if pd.isna(v) or not isinstance(v, (int, float, np.floating)):
            return ""
        if good_high:
            return "color: green;" if v > 0 else ("color: red;" if v < 0 else "")
        else:
            return "color: green;" if v < 0 else ("color: red;" if v > 0 else "")

    return df.style.applymap(color)


# --------------------------
# Sidebar inputs
# --------------------------
st.sidebar.header("Data source")
source = st.sidebar.selectbox("Source", ["bmce (upload)", "yfinance"], index=0)

symbol = st.sidebar.text_input("Symbol", value="IAM" if source.startswith("bmce") else "AAPL")
timezone = st.sidebar.text_input("Timezone", value="GMT")
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)

bmce_file = None
if source.startswith("bmce"):
    bmce_file = st.sidebar.file_uploader("Upload BMCE CSV/XLSX", type=["csv", "xlsx"])
else:
    st.sidebar.caption("yfinance needs internet + yfinance in requirements.txt")
    yf_period = st.sidebar.text_input("yfinance period", value="5y")
    yf_interval = st.sidebar.selectbox("yfinance interval", ["1d"], index=0)
    yf_auto_adjust = st.sidebar.checkbox("auto_adjust", value=False)

st.sidebar.header("Strategy: MA Cross")
fast = st.sidebar.number_input("Fast window", min_value=2, max_value=300, value=20, step=1)
slow = st.sidebar.number_input("Slow window", min_value=3, max_value=500, value=50, step=1)
allow_short = st.sidebar.checkbox("Allow short", value=True)

st.sidebar.header("Portfolio")
initial_cash = st.sidebar.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
rebalance_policy = st.sidebar.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)
max_gross = st.sidebar.number_input("Max gross exposure", min_value=0.1, value=1.0, step=0.1)
cash_buffer = st.sidebar.number_input("Cash buffer", min_value=0.0, max_value=0.5, value=0.0, step=0.01)

st.sidebar.header("Costs")
apply_costs = st.sidebar.checkbox("Apply costs", value=False)
if apply_costs:
    brokerage_bps = st.sidebar.number_input("Brokerage (bps)", value=60.0, step=1.0)
    exchange_bps = st.sidebar.number_input("Exchange (bps)", value=10.0, step=1.0)
    settlement_bps = st.sidebar.number_input("Settlement (bps)", value=20.0, step=1.0)
    vat_rate = st.sidebar.number_input("VAT rate", value=0.10, step=0.01)
    slippage_bps = st.sidebar.number_input("Slippage (bps)", value=0.0, step=1.0)
else:
    brokerage_bps = exchange_bps = settlement_bps = slippage_bps = 0.0
    vat_rate = 0.0

st.sidebar.header("Run")
run_btn = st.sidebar.button("Run backtest")


# --------------------------
# Core runner (cached)
# --------------------------
@st.cache_data(show_spinner=False)
def run_engine_cached(
    source_key: str,
    symbol: str,
    timezone: str,
    interval: str,
    # BMCE
    bmce_tmp_path: Optional[str],
    # yfinance
    yf_period: Optional[str],
    yf_interval: Optional[str],
    yf_auto_adjust: Optional[bool],
    # strategy
    fast: int,
    slow: int,
    allow_short: bool,
    # portfolio
    initial_cash: float,
    rebalance_policy: str,
    max_gross: float,
    cash_buffer: float,
    brokerage_bps: float,
    exchange_bps: float,
    settlement_bps: float,
    vat_rate: float,
    slippage_bps: float,
):
    # ---- Data config ----
    if source_key == "bmce":
        data_cfg = DataConfig(
            source="bmce",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            bmce_paths=bmce_tmp_path,  # engine will pass this into BMCEDataSource.load(... paths=...)
        )
    else:
        data_cfg = DataConfig(
            source="yfinance",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            yf_period=yf_period or "max",
            yf_interval=yf_interval or "1d",
            yf_auto_adjust=bool(yf_auto_adjust),
        )

    # ---- Strategy config ----
    strat_cfg = StrategyConfig(
        kind="ma_cross",
        params={
            "fast_window": int(fast),
            "slow_window": int(slow),
            "allow_short": bool(allow_short),
            "nan_policy": "flat",
        },
    )

    # ---- Indicators config ----
    # specs=None => engine infers SMA specs for MA cross
    ind_cfg = IndicatorsConfig(specs=None)

    # ---- Portfolio config ----
    cost_model = CostModel(
        brokerage_bps=float(brokerage_bps),
        exchange_bps=float(exchange_bps),
        settlement_bps=float(settlement_bps),
        slippage_bps=float(slippage_bps),
        vat_rate=float(vat_rate),
    )

    port_cfg = PortfolioConfig(
        allow_short=bool(allow_short),
        initial_cash=float(initial_cash),
        rebalance_policy=str(rebalance_policy),
        max_gross=float(max_gross),
        cash_buffer=float(cash_buffer),
        cost_model=cost_model,
        # fill/marking semantics are inside your portfolio.py (you said mark to close(t+1))
    )

    spec = EngineSpec(
        data=data_cfg,
        indicators=ind_cfg,
        strategy=strat_cfg,
        portfolio=port_cfg,
        periods_per_year=252,
        rf_annual=0.0,
    )

    bundle = BacktestEngine(spec).run()
    return bundle


# --------------------------
# Main page
# --------------------------
if source.startswith("bmce"):
    st.subheader("BMCE upload")
    if bmce_file is None:
        st.info("Upload a BMCE CSV/XLSX then click Run backtest.")
        st.stop()

    # preview file quickly
    try:
        if bmce_file.name.lower().endswith(".csv"):
            preview_df = pd.read_csv(bmce_file)
        else:
            preview_df = pd.read_excel(bmce_file, engine="openpyxl")
        st.caption("File preview (first rows)")
        st.dataframe(preview_df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Could not preview file: {e}")

else:
    st.subheader("yfinance mode")
    st.caption("This requires `yfinance` in requirements.txt. BMCE is recommended for your desk data.")

if not run_btn:
    st.stop()

# --- prepare BMCE temp file path (streamlit cannot access your local windows paths) ---
tmp_path = None
tmp_dir = None
try:
    if source.startswith("bmce"):
        suffix = Path(bmce_file.name).suffix.lower()
        tmp_dir = tempfile.mkdtemp(prefix="bmce_upload_")
        tmp_path = str(Path(tmp_dir) / f"{symbol}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(bmce_file.getbuffer())

    with st.spinner("Running backtest..."):
        bundle = run_engine_cached(
            source_key="bmce" if source.startswith("bmce") else "yfinance",
            symbol=symbol,
            timezone=timezone,
            interval=interval,
            bmce_tmp_path=tmp_path,
            yf_period=(yf_period if not source.startswith("bmce") else None),
            yf_interval=(yf_interval if not source.startswith("bmce") else None),
            yf_auto_adjust=(yf_auto_adjust if not source.startswith("bmce") else None),
            fast=int(fast),
            slow=int(slow),
            allow_short=bool(allow_short),
            initial_cash=float(initial_cash),
            rebalance_policy=str(rebalance_policy),
            max_gross=float(max_gross),
            cash_buffer=float(cash_buffer),
            brokerage_bps=float(brokerage_bps),
            exchange_bps=float(exchange_bps),
            settlement_bps=float(settlement_bps),
            vat_rate=float(vat_rate),
            slippage_bps=float(slippage_bps),
        )

    rep = bundle.report

    # --- Outputs ---
    plots = bundle.report.plots
    tables = bundle.report.tables
    series = bundle.report.series

    st.subheader("Price + Indicators + Trades")
    pp = plots["price_panel"]
    st.pyplot(plot_price_indicators_trades(pp["price"], pp["indicators"], pp["trades"]))

    st.subheader("Cumulative Returns vs Benchmark")
    cvb = plots["cum_vs_bench"]
    st.pyplot(plot_cum_vs_bench(cvb["strategy"], cvb["benchmark"]))

    st.subheader("Drawdown")
    st.pyplot(plot_drawdown_red(plots["drawdown"]))

    st.subheader("Monthly Returns Heatmap")
    st.pyplot(plot_monthly_heatmap(plots["monthly_heatmap"]))

    st.subheader("Yearly Returns")
    st.pyplot(plot_yearly_bar(plots["yearly_bar"]))

    st.subheader("Trades table")
    st.dataframe(tables["trades"], use_container_width=True)

    st.subheader("Time series table")
    st.dataframe(tables["timeseries"].tail(300), use_container_width=True)

    if "strategy_vs_benchmark" in tables:
        st.subheader("Strategy vs Benchmark")
        st.dataframe(tables["strategy_vs_benchmark"], use_container_width=True)

    st.subheader("Headline metrics")
    st.json(rep.metrics)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cumulative returns")
        st.pyplot(plot_line(rep.series["cum_returns"].dropna(), "Cumulative Returns", "Cum return"))
    with c2:
        st.subheader("Drawdown")
        st.pyplot(plot_line(rep.series["drawdown"].dropna(), "Drawdown", "Drawdown"))

    st.subheader("Trades")
    st.dataframe(rep.tables["trades"], use_container_width=True)

    st.subheader("Monthly returns")
    st.dataframe(rep.tables["monthly_returns"], use_container_width=True)

    with st.expander("Debug: feature columns / signals head", expanded=False):
        sym = bundle.md.symbols()[0]
        st.write("Features columns:", list(bundle.feats.features[sym].columns))
        st.write("Signals head:")
        st.dataframe(bundle.signals.signals.head(10), use_container_width=True)

finally:
    # cleanup temp upload file
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if tmp_dir and os.path.isdir(tmp_dir):
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass