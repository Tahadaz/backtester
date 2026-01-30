# app.py  (Streamlit entrypoint)
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from engine import BacktestEngine, EngineSpec, DataConfig, IndicatorsConfig, StrategyConfig
from portfolio import PortfolioConfig, CostModel

from optimize import (
    default_param_catalog_for_your_app,
    run_optimization,
    OptimizeConfig,
    build_date_window_choices_from_uploaded_bmce,
    add_date_window_param,
)

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="Backtester", layout="wide")
st.title("Backtester (TA) — Data → Indicators → Strategy → Portfolio → Results")

# ============================================================
# Plotting helpers
# ============================================================
def plot_pnl_series(pnl: pd.Series) -> plt.Figure:
    fig = plt.figure(figsize=(14, 3))
    ax = plt.gca()
    ax.plot(pnl.index, pnl.values, label="PnL")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("PnL (per bar)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

def plot_cum_pnl_series(cum_pnl: pd.Series) -> plt.Figure:
    fig = plt.figure(figsize=(14, 3))
    ax = plt.gca()
    ax.plot(cum_pnl.index, cum_pnl.values, label="Cumulative PnL")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cum PnL")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

def plot_cum_vs_bench(cum: pd.Series, bench_cum: pd.Series | None) -> plt.Figure:
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.plot(cum.index, cum.values, label="Strategy")
    if bench_cum is not None:
        ax.plot(bench_cum.index, bench_cum.values, label="Benchmark")
    ax.set_title("Cumulative Returns vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.grid(True)
    ax.legend()
    return fig

def plot_drawdown_red(dd: pd.Series) -> plt.Figure:
    fig = plt.figure(figsize=(14, 3))
    ax = plt.gca()
    ax.plot(dd.index, dd.values, label="Drawdown")
    ax.fill_between(dd.index, dd.values, 0.0, where=(dd.values < 0.0), alpha=0.35, color="red")
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    ax.legend()
    return fig

def plot_monthly_heatmap_with_values(monthly: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14, 5))
    ax = plt.gca()

    if monthly is None or monthly.empty:
        ax.set_title("Monthly returns heatmap (no data)")
        return fig

    data = monthly.values.astype(float)
    if np.isnan(data).all():
        ax.set_title("Monthly returns heatmap (no data)")
        return fig

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, aspect="auto", norm=norm, cmap="RdYlGn")
    ax.set_title("Monthly Returns Heatmap")
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index.astype(str).tolist())
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    return fig

def plot_yearly_returns_bar(yearly: pd.Series) -> plt.Figure:
    fig = plt.figure(figsize=(14, 3))
    ax = plt.gca()

    if yearly is None or yearly.empty:
        ax.set_title("Yearly returns (no data)")
        return fig

    years = yearly.index.astype(str).tolist()
    vals = yearly.values.astype(float)
    colors = ["green" if v >= 0 else "red" for v in vals]
    ax.bar(years, vals, color=colors)

    ax.set_title("Yearly Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.grid(True, axis="y")
    return fig

def plot_price_indicators_trades_plotly(
    bars: pd.DataFrame,
    indicators: pd.DataFrame | None,
    trades: pd.DataFrame | None,
    indicator_cols: list[str] | None = None,
) -> go.Figure:
    df = bars.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    if indicators is not None and not indicators.empty:
        ind = indicators.copy()
        if not isinstance(ind.index, pd.DatetimeIndex):
            ind.index = pd.to_datetime(ind.index)
        ind = ind.reindex(df.index)

        cols = [c for c in (indicator_cols or list(ind.columns)) if c in ind.columns]
        for c in cols:
            s = ind[c].astype(float)
            if s.notna().any():
                fig.add_trace(go.Scatter(x=df.index, y=s, mode="lines", name=c, line=dict(width=1)))

    if trades is not None and not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
        t = t.dropna(subset=["timestamp"]).sort_values("timestamp")

        if "side" not in t.columns:
            t["side"] = np.where(t["qty"].astype(float) > 0, "BUY", "SELL")

        if "price" in t.columns:
            y = pd.to_numeric(t["price"], errors="coerce")
        else:
            y = pd.Series(np.nan, index=t.index)

        close_map = df["Close"].reindex(t["timestamp"]).ffill()
        t["y_plot"] = np.where(np.isfinite(y.values), y.values, close_map.values)

        buys = t[t["side"] == "BUY"]
        sells = t[t["side"] == "SELL"]

        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["y_plot"],
                    mode="text",
                    text="BUY",
                    name="BUY",
                    textposition="top center",
                    textfont=dict(size=10, color="#1C60E9"),
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["y_plot"],
                    mode="text",
                    text="SELL",
                    name="SELL",
                    textposition="top center",
                    textfont=dict(size=10, color="#1C60E9"),
                )
            )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

# ============================================================
# Styling helpers (tables)
# ============================================================
def style_curve_vs_bench(df: pd.DataFrame):
    good_high = {"Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "R-Squared"}
    good_low  = {"Annual Volatility", "Max Daily Drawdown", "Max Drawdown Duration"}

    def color_cell(val, metric, col):
        if not isinstance(val, (int, float, np.floating)) or pd.isna(val):
            return ""
        if "Benchmark" in df.columns and col in ("Strategy", "Benchmark"):
            s = df.loc[metric, "Strategy"]
            b = df.loc[metric, "Benchmark"]
            if pd.isna(b) or metric == "R-Squared":
                pass
            else:
                if metric in good_high:
                    return "color: green;" if (col == "Strategy" and s > b) else ("color: red;" if (col == "Strategy" and s < b) else "")
                if metric in good_low:
                    return "color: green;" if (col == "Strategy" and s < b) else ("color: red;" if (col == "Strategy" and s > b) else "")

        if metric in good_high:
            return "color: green;" if val > 0 else ("color: red;" if val < 0 else "")
        if metric in good_low:
            return "color: green;" if val >= 0 else ""
        return ""

    def apply_style(row: pd.Series):
        metric = row.name
        return [color_cell(v, metric, c) for c, v in row.items()]

    return df.style.apply(apply_style, axis=1).format(na_rep="", precision=4)

def style_trade_table(df: pd.DataFrame):
    good_high = {"Trade Winning %", "Average Trade %", "Average Win %", "Best Trade %"}
    good_low  = {"Average Loss %", "Worst Trade %", "Avg Days in Trade"}

    def color(v, metric):
        if metric in ("Worst Trade Date", "Trades"):
            return ""
        if not isinstance(v, (int,float,np.floating)) or pd.isna(v):
            return ""
        if metric in good_high:
            return "color: green;" if v > 0 else ("color: red;" if v < 0 else "")
        if metric in good_low:
            return "color: red;" if v < 0 else ""
        return ""

    def apply(row: pd.Series):
        metric = row.name
        v = row.iloc[0]
        return [color(v, metric)]

    return df.style.apply(apply, axis=1).format(na_rep="", precision=2)

def style_time_table(df: pd.DataFrame):
    good_high = {"Winning Months %", "Average Winning Month %", "Best Month %", "Winning Years %", "Best Year %"}
    good_low  = {"Average Losing Month %", "Worst Month %", "Worst Year %"}

    def color(v, metric):
        if not isinstance(v, (int,float,np.floating)) or pd.isna(v):
            return ""
        if metric in good_high:
            return "color: green;" if v > 0 else ("color: red;" if v < 0 else "")
        if metric in good_low:
            return "color: red;" if v < 0 else ""
        return ""

    def apply(row: pd.Series):
        metric = row.name
        v = row.iloc[0]
        return [color(v, metric)]

    return df.style.apply(apply, axis=1).format(na_rep="", precision=2)

# ============================================================
# Render backtest bundle (used in both Backtest and Optimize)
# ============================================================
def render_bundle(bundle):
    rep = bundle.report
    plots = rep.plots
    tables = rep.tables
    series = rep.series

    st.subheader("Price + Indicators + Trades")
    pp = plots["price_panel"]
    sym0 = bundle.md.symbols()[0]
    bars = bundle.md.bars[sym0]

    fig = plot_price_indicators_trades_plotly(
        bars=bars,
        indicators=pp.get("indicators"),
        trades=pp.get("trades"),
        indicator_cols=pp.get("indicator_cols"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Returns vs Benchmark")
    cvb = plots["cum_vs_bench"]
    st.pyplot(plot_cum_vs_bench(cvb["strategy"], cvb["benchmark"]))

    st.subheader("Drawdown")
    st.pyplot(plot_drawdown_red(plots["drawdown"]))

    st.subheader("PnL")
    if "pnl" in series:
        st.pyplot(plot_pnl_series(series["pnl"]))
    else:
        st.info("PnL series not found in report.series")

    st.subheader("Cumulative PnL")
    if "cum_pnl" in series:
        st.pyplot(plot_cum_pnl_series(series["cum_pnl"]))
    else:
        st.info("cum_pnl series not found in report.series")

    st.subheader("Monthly Returns Heatmap")
    st.pyplot(plot_monthly_heatmap_with_values(plots["monthly_heatmap"]))

    st.subheader("Yearly Returns")
    st.pyplot(plot_yearly_returns_bar(plots["yearly_bar"]))

    st.subheader("Trade Performance (summary)")
    if "trade_performance" in tables:
        st.dataframe(tables["trade_performance"], use_container_width=True)
    else:
        st.info("trade_performance not found in report.tables")

    st.subheader("Trades Ledger (PnL per trade)")
    if "trade_ledger" in tables:
        st.dataframe(tables["trade_ledger"], use_container_width=True)
    else:
        st.info("trade_ledger not found in report.tables")

    if "curve_vs_benchmark" in tables and "trade_summary" in tables and "time_summary" in tables:
        t_curve = tables["curve_vs_benchmark"]
        t_trade = tables["trade_summary"]
        t_time = tables["time_summary"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Curve vs. Benchmark")
            st.dataframe(style_curve_vs_bench(t_curve), use_container_width=True)
        with c2:
            st.markdown("### Trade")
            st.dataframe(style_trade_table(t_trade), use_container_width=True)
        with c3:
            st.markdown("### Time")
            st.dataframe(style_time_table(t_time), use_container_width=True)

    with st.expander("Debug: feature columns / signals head", expanded=False):
        sym = bundle.md.symbols()[0]
        st.write("Features columns:", list(bundle.feats.features[sym].columns))
        st.write("Signals head:")
        st.dataframe(bundle.signals.signals.head(10), use_container_width=True)

# ============================================================
# Spec builder (single source of truth)
# ============================================================
def make_base_spec(
    source_key: str,
    symbol: str,
    timezone: str,
    interval: str,
    bmce_tmp_path: Optional[str],
    start: Optional[str],
    end: Optional[str],
    yf_period: Optional[str],
    yf_interval: Optional[str],
    yf_auto_adjust: Optional[bool],
    strategy_kind: str,
    strategy_params: dict,
    allow_short: bool,
    initial_cash: float,
    rebalance_policy: str,
    sizing_mode: str,
    buy_pct_cash: float,
    sell_pct_shares: float,
    cooldown_bars: int,
    brokerage_bps: float,
    exchange_bps: float,
    settlement_bps: float,
    vat_rate: float,
    slippage_bps: float,
) -> EngineSpec:
    if source_key == "bmce":
        data_cfg = DataConfig(
            source="bmce",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            start=start,
            end=end,
            bmce_paths=bmce_tmp_path,
        )
    else:
        data_cfg = DataConfig(
            source="yfinance",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            start=start,
            end=end,
            yf_period=yf_period or "max",
            yf_interval=yf_interval or "1d",
            yf_auto_adjust=bool(yf_auto_adjust),
        )

    strat_cfg = StrategyConfig(kind=strategy_kind, params=dict(strategy_params or {}))
    ind_cfg = IndicatorsConfig(specs=None)

    cost_model = CostModel(
        brokerage_bps=float(brokerage_bps),
        exchange_bps=float(exchange_bps),
        settlement_bps=float(settlement_bps),
        slippage_bps=float(slippage_bps),
        vat_rate=float(vat_rate),
    )

    # IMPORTANT: this assumes you add cooldown_bars into PortfolioConfig
    port_cfg = PortfolioConfig(
        allow_short=bool(allow_short),
        initial_cash=float(initial_cash),
        rebalance_policy=str(rebalance_policy),
        cost_model=cost_model,
        sizing_mode=str(sizing_mode),
        buy_pct_cash=float(buy_pct_cash),
        sell_pct_shares=float(sell_pct_shares),
        cooldown_bars=int(cooldown_bars),
    )

    return EngineSpec(
        data=data_cfg,
        indicators=ind_cfg,
        strategy=strat_cfg,
        portfolio=port_cfg,
        periods_per_year=252,
        rf_annual=0.0,
    )

# ============================================================
# Strategy param widgets (baseline values)
# ============================================================
def strategy_param_widgets(strategy_kind: str) -> Dict[str, Any]:
    nan_policy = "flat"

    if strategy_kind == "ma_cross":
        fast = st.sidebar.number_input("Fast SMA window", min_value=2, max_value=500, value=20, step=1)
        slow = st.sidebar.number_input("Slow SMA window", min_value=3, max_value=500, value=50, step=1)
        if fast >= slow:
            st.sidebar.warning("Fast must be < Slow; adjusting slow.")
            slow = int(fast) + 1

        return {
            "fast_window": int(fast),
            "slow_window": int(slow),
            "nan_policy": nan_policy,
        }

    # sma_price
    window = st.sidebar.number_input("SMA window", min_value=2, max_value=500, value=50, step=1)
    return {
        "window": int(window),
        "nan_policy": nan_policy,
    }

def strategy_keys_for_kind(kind: str, keys: List[str]) -> List[str]:
    """Filter catalog keys to those relevant to the chosen strategy."""
    out = []
    for k in keys:
        if not k.startswith("strategy."):
            continue
        if kind == "ma_cross" and k in ("strategy.fast_window", "strategy.slow_window"):
            out.append(k)
        if kind == "sma_price" and k in ("strategy.window",):
            out.append(k)
    return out

# ============================================================
# Sidebar: Mode + Data + Strategy + Portfolio (shared)
# ============================================================
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode", ["Backtest", "Optimize"], index=0)

st.sidebar.header("Data source")
source = st.sidebar.selectbox("Source", ["bmce (upload)", "yfinance"], index=0)
source_key = "bmce" if source.startswith("bmce") else "yfinance"

symbol = st.sidebar.text_input("Symbol", value="IAM" if source_key == "bmce" else "AAPL")
timezone = st.sidebar.text_input("Timezone", value="GMT")
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)

bmce_file = None
yf_period = yf_interval = None
yf_auto_adjust = None

if source_key == "bmce":
    bmce_file = st.sidebar.file_uploader("Upload BMCE CSV/XLSX", type=["csv", "xlsx"])
else:
    st.sidebar.caption("yfinance needs internet + yfinance in requirements.txt")
    yf_period = st.sidebar.text_input("yfinance period", value="5y")
    yf_interval = st.sidebar.selectbox("yfinance interval", ["1d"], index=0)
    yf_auto_adjust = st.sidebar.checkbox("auto_adjust", value=False)

# Date range baseline exists for BOTH modes (needed if you optimize it)
st.sidebar.header("Backtest period (baseline)")
use_date_range = st.sidebar.checkbox("Use date range", value=False)
start_str = end_str = None
if use_date_range:
    start_date = st.sidebar.date_input("Start date", value=None)
    end_date = st.sidebar.date_input("End date", value=None)
    if start_date and end_date and start_date > end_date:
        st.sidebar.error("Start date must be <= End date.")
        st.stop()
    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if end_date else None

st.sidebar.header("Strategy")
strategy_kind = st.sidebar.selectbox(
    "Choose strategy",
    options=["ma_cross", "sma_price"],
    index=0,
)

allow_short = st.sidebar.checkbox("Allow short", value=False)

# Baseline strategy params (used in Backtest; used as baseline in Optimize too)
st.sidebar.subheader("Strategy params (baseline)")
baseline_sp = strategy_param_widgets(strategy_kind)
baseline_sp["allow_short"] = bool(allow_short)

st.sidebar.header("Portfolio")
initial_cash = st.sidebar.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
rebalance_policy = st.sidebar.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)

st.sidebar.header("Sizing (baseline)")
sizing_mode = st.sidebar.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=1)
buy_pct_cash = st.sidebar.slider("Buy % of cash per entry", 0.01, 1.00, 0.25, 0.01)
sell_pct_shares = st.sidebar.slider("Sell % of shares per exit", 0.01, 1.00, 1.00, 0.01)

# Cooldown bars (min bars between trades) — requires portfolio.py change
st.sidebar.header("Trade throttling")
cooldown_bars = st.sidebar.number_input("Min bars between trades", min_value=0, value=0, step=1)

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

# ============================================================
# Sidebar: Optimize-specific controls
# ============================================================
opt_cfg = None
catalog = None
active_keys = None
top_k = None

if mode == "Optimize":
    st.sidebar.header("Optimization")

    # 1) choose what to optimize
    catalog = default_param_catalog_for_your_app()
    # optionally remove strategy.kind if present
    catalog.pop("strategy.kind", None)

    st.sidebar.subheader("Select parameters to optimize")

    # show only relevant strategy params + portfolio/data params
    all_keys = list(catalog.keys())

    strategy_keys = strategy_keys_for_kind(strategy_kind, all_keys)
    non_strategy_keys = [k for k in all_keys if not k.startswith("strategy.")]
    selectable = strategy_keys + non_strategy_keys

    # default selection: relevant strategy windows
    default_sel = strategy_keys if strategy_keys else []

    active_keys = st.sidebar.multiselect(
        "Active parameters",
        options=selectable,
        default=default_sel,
    )

    # allow date window optimization (BMCE only, because we can compute windows from uploaded file)
    optimize_dates = st.sidebar.checkbox("Optimize date window (BMCE only)", value=False)

    # 2) choose search mode + show only that mode’s parameters
    st.sidebar.subheader("Search mode")
    search_mode = st.sidebar.selectbox("Mode", ["random", "grid", "wfo"], index=0)

    # mode-specific params
    seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)
    if search_mode == "random":
        n_trials = st.sidebar.number_input("Trials", min_value=10, value=200, step=10)
        grid_max_combos = None
        wfo_train_bars = wfo_test_bars = wfo_step_bars = None
    elif search_mode == "grid":
        # depends on how you implemented grid in optimize.py
        n_trials = None
        grid_max_combos = st.sidebar.number_input("Max grid combinations", min_value=10, value=500, step=10)
        wfo_train_bars = wfo_test_bars = wfo_step_bars = None
    else:  # wfo
        # requires optimize.py implementation
        n_trials = None
        grid_max_combos = None
        wfo_train_bars = st.sidebar.number_input("Train bars", min_value=50, value=252, step=10)
        wfo_test_bars = st.sidebar.number_input("Test bars", min_value=10, value=63, step=1)
        wfo_step_bars = st.sidebar.number_input("Step bars", min_value=1, value=21, step=1)

    # 3) top-k results
    top_k = st.sidebar.number_input("Show top K", min_value=5, value=30, step=5)

    # objective fixed to your request:
    #  - primary: pnl
    #  - secondary: profit/volume (efficiency)
    opt_cfg = OptimizeConfig(
        mode=search_mode,
        n_trials=int(n_trials) if n_trials is not None else None,
        top_k=int(top_k),
        seed=int(seed),
        objective="pnl_then_efficiency",
        grid_max_combos=int(grid_max_combos) if grid_max_combos is not None else None,
        wfo_train_bars=int(wfo_train_bars) if wfo_train_bars is not None else None,
        wfo_test_bars=int(wfo_test_bars) if wfo_test_bars is not None else None,
        wfo_step_bars=int(wfo_step_bars) if wfo_step_bars is not None else None,
        verbose=False,
    )

    run_btn = st.sidebar.button("Run Optimization")

else:
    run_btn = st.sidebar.button("Run backtest")

# ============================================================
# Main page: Data preview + run
# ============================================================
if source_key == "bmce":
    st.subheader("BMCE upload")
    if bmce_file is None:
        st.info("Upload a BMCE CSV/XLSX then click Run.")
        st.stop()

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

# ============================================================
# Run (Backtest or Optimize)
# ============================================================
tmp_path = None
tmp_dir = None

try:
    # Prepare tmp upload file path for BMCE
    if source_key == "bmce":
        suffix = Path(bmce_file.name).suffix.lower()
        tmp_dir = tempfile.mkdtemp(prefix="bmce_upload_")
        tmp_path = str(Path(tmp_dir) / f"{symbol}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(bmce_file.getbuffer())

    # Baseline strategy params
    strategy_params = dict(baseline_sp)

    # Build base spec
    base_spec = make_base_spec(
        source_key=source_key,
        symbol=symbol,
        timezone=timezone,
        interval=interval,
        bmce_tmp_path=tmp_path,
        start=start_str,
        end=end_str,
        yf_period=yf_period,
        yf_interval=yf_interval,
        yf_auto_adjust=yf_auto_adjust,
        strategy_kind=strategy_kind,
        strategy_params=strategy_params,
        allow_short=bool(allow_short),
        initial_cash=float(initial_cash),
        rebalance_policy=str(rebalance_policy),
        sizing_mode=str(sizing_mode),
        buy_pct_cash=float(buy_pct_cash),
        sell_pct_shares=float(sell_pct_shares),
        cooldown_bars=int(cooldown_bars),
        brokerage_bps=float(brokerage_bps),
        exchange_bps=float(exchange_bps),
        settlement_bps=float(settlement_bps),
        vat_rate=float(vat_rate),
        slippage_bps=float(slippage_bps),
    )

    if mode == "Backtest":
        with st.spinner("Running backtest..."):
            bundle = BacktestEngine(base_spec).run()
        render_bundle(bundle)

    else:
        # Optimization mode
        if catalog is None or active_keys is None or opt_cfg is None:
            st.error("Optimization UI state missing. Reload and try again.")
            st.stop()

        # Date window optimization: BMCE only
        if optimize_dates:
            if source_key != "bmce":
                st.warning("Date window optimization is supported for BMCE uploads only.")
            else:
                windows = build_date_window_choices_from_uploaded_bmce(
                    base_data_cfg=base_spec.data,
                    symbol=symbol,
                    min_bars=252,
                    step_bars=21,
                    max_windows=200,
                )
                add_date_window_param(catalog, windows)
                if "data.window" not in active_keys:
                    active_keys = list(active_keys) + ["data.window"]

        with st.spinner("Running optimization..."):
            best, top_df, best_spec = run_optimization(
                base_spec=base_spec,
                catalog=catalog,
                active_keys=active_keys,
                cfg=opt_cfg,
            )

        st.subheader("Optimization results")
        st.json({
            "objective": "pnl_then_efficiency",
            "best_pnl": best.pnl,
            "best_traded_notional": best.traded_notional,
            "best_profit_per_notional": best.efficiency,
            "best_params": best.params,
        })
        st.dataframe(top_df, use_container_width=True)

        st.divider()
        if st.button("Run best configuration (render full report)"):
            with st.spinner("Running best backtest..."):
                bundle = BacktestEngine(best_spec).run()
            render_bundle(bundle)

finally:
    # Cleanup temp file
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
