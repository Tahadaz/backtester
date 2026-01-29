# app.py  (Streamlit entrypoint)
# Put this at repo root (or in /app and point Streamlit to it)

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, List
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# ---- your project imports (adjust if your modules are inside a package folder) ----
from engine import BacktestEngine, EngineSpec, DataConfig, IndicatorsConfig, StrategyConfig
from portfolio import PortfolioConfig, CostModel


st.set_page_config(page_title="Backtester", layout="wide")
st.title("Backtester (TA) — Data → Indicators → Strategy → Portfolio → Results")


# --------------------------
# Small plotting helpers
# --------------------------
def score_report(report, metric: str) -> float:
    """
    metric options:
      - total_net_pnl
      - total_return
      - sharpe
      - max_drawdown (we minimize, so score will be negative of drawdown)
    """
    # 1) Prefer explicit metrics if you added them in results.py
    m = getattr(report, "metrics", {}) or {}
    if metric in m and m[metric] is not None:
        return float(m[metric])

    # 2) Otherwise compute from series/tables defensively
    series = report.series
    tables = report.tables

    if metric == "total_net_pnl":
        if "trade_ledger" in tables and not tables["trade_ledger"].empty and "net_pnl" in tables["trade_ledger"].columns:
            return float(pd.to_numeric(tables["trade_ledger"]["net_pnl"], errors="coerce").fillna(0.0).sum())
        # fallback: equity end - start
        eq = series.get("equity")
        if eq is not None and len(eq) > 1:
            return float(eq.iloc[-1] - eq.iloc[0])
        return np.nan

    if metric == "total_return":
        cum = series.get("cum_returns")
        if cum is not None and len(cum) > 0:
            return float(cum.iloc[-1])
        # fallback
        eq = series.get("equity")
        if eq is not None and len(eq) > 1:
            return float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        return np.nan

    if metric == "sharpe":
        # if you didn't store sharpe in report.metrics yet, approximate from returns
        rets = series.get("returns")
        if rets is None or len(rets) < 5:
            return np.nan
        mu = float(rets.mean())
        sd = float(rets.std(ddof=1))
        if sd == 0:
            return np.nan
        return mu / sd * np.sqrt(252)

    if metric == "max_drawdown":
        dd = series.get("drawdown")
        if dd is None or dd.empty:
            return np.nan
        # dd is negative; bigger drawdown magnitude is worse
        # we return NEGATIVE of min(dd) so "higher is better"
        return float(-dd.min())

    return np.nan

def plot_line(series: pd.Series, title: str, ylabel: str):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    return fig
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


def plot_price_indicators_trades_plotly(
    bars: pd.DataFrame,                    # expects Open/High/Low/Close with DatetimeIndex
    indicators: pd.DataFrame | None,        # SMA cols etc, same index or reindexable
    trades: pd.DataFrame | None,            # timestamp, qty, price (optional)
    indicator_cols: list[str] | None = None,
) -> go.Figure:
    df = bars.copy()
    df = df.sort_index()
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    fig = go.Figure()

    # Candles
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

    # Indicators
    if indicators is not None and not indicators.empty:
        ind = indicators.copy()
        if not isinstance(ind.index, pd.DatetimeIndex):
            ind.index = pd.to_datetime(ind.index)
        ind = ind.reindex(df.index)

        if indicator_cols is not None:
            cols = [c for c in indicator_cols if c in ind.columns]
        else:
            cols = list(ind.columns)
        for c in cols:
            s = ind[c].astype(float)
            if s.notna().any():
                fig.add_trace(go.Scatter(x=df.index, y=s, mode="lines", name=c,line=dict(width=0.5),))

    # Trades
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
        t = t.dropna(subset=["timestamp"]).sort_values("timestamp")

        if "side" not in t.columns:
            t["side"] = np.where(t["qty"].astype(float) > 0, "BUY", "SELL")

        # Use trade price if present else close at that timestamp (ffill on index alignment)
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
                    mode="markers",
                    name="BUY",
                    marker=dict(symbol="triangle-up", size=7, color ="#2AE91C"),
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["y_plot"],
                    mode="markers",
                    name="SELL",
                    marker=dict(symbol="triangle-down", size=7, color="#F73434"),
                )
            )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,   # range slider
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
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
    """
    monthly: DataFrame index=year, columns=1..12, values=monthly return (decimal)
    """
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

    # values in cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    return fig


def plot_yearly_returns_bar(yearly: pd.Series) -> plt.Figure:
    """
    yearly: Series index=year, values=year return (decimal)
    """
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

import numpy as np
import pandas as pd

def _fmt_pct(x):
    if pd.isna(x): return ""
    return f"{100*x:.2f}%" if abs(x) < 2 and abs(x) < 1 else f"{x:.2f}%"  # we store many already in %

def style_curve_vs_bench(df: pd.DataFrame):
    # metrics where higher is better
    good_high = {"Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "R-Squared"}
    # metrics where lower is better
    good_low  = {"Annual Volatility", "Max Daily Drawdown", "Max Drawdown Duration"}

    def color_cell(val, metric, col):
        if not isinstance(val, (int, float, np.floating)) or pd.isna(val):
            return ""
        # if benchmark exists, compare Strategy vs Benchmark
        if "Benchmark" in df.columns and col in ("Strategy", "Benchmark"):
            s = df.loc[metric, "Strategy"]
            b = df.loc[metric, "Benchmark"]
            if pd.isna(b) or metric == "R-Squared":
                # no compare for R2 / missing
                pass
            else:
                if metric in good_high:
                    return "color: green;" if s > b and col == "Strategy" else ("color: red;" if s < b and col == "Strategy" else "")
                if metric in good_low:
                    return "color: green;" if s < b and col == "Strategy" else ("color: red;" if s > b and col == "Strategy" else "")

        # single-column fallback
        if metric in good_high:
            return "color: green;" if val > 0 else ("color: red;" if val < 0 else "")
        if metric in good_low:
            # for drawdown/vol lower is better => if negative drawdown, closer to 0 is better; keep red if very negative
            return "color: green;" if val >= 0 else ""
        return ""

    def apply_style(s: pd.Series):
        metric = s.name
        out = []
        for col, val in s.items():
            out.append(color_cell(val, metric, col))
        return out

    return df.style.apply(apply_style, axis=1).format({
        "Strategy": lambda v: f"{v*100:.2f}%" if df.index.isin(["Total Return","CAGR","Annual Volatility","Max Daily Drawdown"]).any() and isinstance(v,(int,float,np.floating)) else f"{v:.2f}",
        "Benchmark": lambda v: f"{v*100:.2f}%" if isinstance(v,(int,float,np.floating)) and df.index.isin(["Total Return","CAGR","Annual Volatility","Max Daily Drawdown"]).any() else f"{v:.2f}",
    }, na_rep="")

def style_trade_table(df: pd.DataFrame):
    good_high = {"Trade Winning %", "Average Trade %", "Average Win %", "Best Trade %"}
    good_low  = {"Average Loss %", "Worst Trade %", "Avg Days in Trade"}  # avg days not always "bad", but you can keep neutral

    def color(v, metric):
        if metric in ("Worst Trade Date", "Trades"):
            return ""
        if not isinstance(v, (int,float,np.floating)) or pd.isna(v):
            return ""
        if metric in good_high:
            return "color: green;" if v > 0 else ("color: red;" if v < 0 else "")
        if metric in good_low:
            return "color: red;" if v < 0 else ""  # losses are negative => red
        return ""

    def apply(s: pd.Series):
        metric = s.name
        v = s.iloc[0]
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

    def apply(s: pd.Series):
        metric = s.name
        v = s.iloc[0]
        return [color(v, metric)]

    return df.style.apply(apply, axis=1).format(na_rep="", precision=2)


# --------------------------
# Sidebar inputs
# --------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode", ["Backtest", "Optimization"], index=0)

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


st.sidebar.header("Backtest period")

use_date_range = st.sidebar.checkbox("Use date range", value=False)

start_date = None
end_date = None
if use_date_range:
    start_date = st.sidebar.date_input("Start date", value=None)
    end_date = st.sidebar.date_input("End date", value=None)

    # basic validation
    if start_date and end_date and start_date > end_date:
        st.sidebar.error("Start date must be <= End date.")
        st.stop()

# Convert to ISO strings (what your DataConfig expects)
start_str = start_date.isoformat() if start_date else None
end_str = end_date.isoformat() if end_date else None

with st.sidebar:
    st.header("Strategy")

    strategy_kind = st.selectbox(
        "Choose strategy",
        options=["ma_cross", "sma_price"],
        index=0,
        help="ma_cross: SMA fast vs SMA slow. sma_price: Close vs SMA(window).",
    )

    allow_short = st.checkbox("Allow short", value=False)

    if strategy_kind == "ma_cross":
        fast = st.number_input("Fast SMA window", min_value=2, max_value=500, value=20, step=1)
        slow = st.number_input("Slow SMA window", min_value=3, max_value=500, value=50, step=1)
        # enforce fast < slow defensively
        if fast >= slow:
            st.warning("Fast window must be < Slow window. Adjusting automatically.")
            fast = min(int(fast), int(slow) - 1)

        # pack params for engine
        strat_fast = int(fast)
        strat_slow = int(slow)
        strat_window = None

    else:  # "sma_price"
        window = st.number_input("SMA window", min_value=2, max_value=500, value=50, step=1)
        strat_window = int(window)
        strat_fast = None
        strat_slow = None

nan_policy = "flat"

if strategy_kind == "ma_cross":
    strategy_params = {
        "fast_window": int(strat_fast),
        "slow_window": int(strat_slow),
        "allow_short": bool(allow_short),
        "nan_policy": nan_policy,
    }
else:
    strategy_params = {
        "window": int(strat_window),
        "allow_short": bool(allow_short),
        "nan_policy": nan_policy,
    }

st.sidebar.header("Portfolio")
initial_cash = st.sidebar.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
rebalance_policy = st.sidebar.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)
max_gross = st.sidebar.number_input("Max gross exposure", min_value=0.1, value=1.0, step=0.1)
cash_buffer = st.sidebar.number_input("Cash buffer", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
st.sidebar.header("Sizing")
sizing_mode = st.sidebar.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=0)

buy_pct_cash = st.sidebar.slider("Buy % of cash per entry", 0.01, 1.00, 0.25, 0.01)
sell_pct_shares = st.sidebar.slider("Sell % of shares per exit", 0.01, 1.00, 1.00, 0.01)


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
    start: Optional[str],   # <-- add
    end: Optional[str],     # <-- add
    # yfinance
    yf_period: Optional[str],
    yf_interval: Optional[str],
    yf_auto_adjust: Optional[bool],
    # strategy
    allow_short: bool,
    strategy_kind: str,
    strategy_params: dict,
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
            start=start,   # <-- add
            end=end, 
            bmce_paths=bmce_tmp_path,  # engine will pass this into BMCEDataSource.load(... paths=...)
        )
    else:
        data_cfg = DataConfig(
            source="yfinance",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            start=start,   # <-- add
            end=end, 
            yf_period=yf_period or "max",
            yf_interval=yf_interval or "1d",
            yf_auto_adjust=bool(yf_auto_adjust),
        )

    # ---- Strategy config ----
    strat_cfg = StrategyConfig(
        kind=strategy_kind,
        params=dict(strategy_params or {}),
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
        sizing_mode=str(sizing_mode),
        buy_pct_cash=float(buy_pct_cash),
        sell_pct_shares=float(sell_pct_shares),
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

@st.cache_data(show_spinner=False)
def run_one_backtest(spec_dict: dict) -> dict:
    """
    Cache-friendly wrapper: we pass a serializable dict,
    reconstruct EngineSpec inside, run, return lightweight outputs.
    """
    # Reconstruct configs
    data_cfg = DataConfig(**spec_dict["data"])
    ind_cfg = IndicatorsConfig(**spec_dict["indicators"])
    strat_cfg = StrategyConfig(**spec_dict["strategy"])
    port_cfg = PortfolioConfig(**spec_dict["portfolio"])

    spec = EngineSpec(
        data=data_cfg,
        indicators=ind_cfg,
        strategy=strat_cfg,
        portfolio=port_cfg,
        periods_per_year=spec_dict.get("periods_per_year", 252),
        rf_annual=spec_dict.get("rf_annual", 0.0),
        plot_indicators=spec_dict.get("plot_indicators", []),
        benchmark=spec_dict.get("benchmark", None) or None,
    )

    bundle = BacktestEngine(spec).run()

    # Return only what optimizer needs (avoid caching huge objects if possible)
    return {
        "metrics": bundle.report.metrics,
        "series_tail": {
            "equity_last": float(bundle.report.series["equity"].iloc[-1]) if "equity" in bundle.report.series else np.nan
        },
        "tables_meta": {
            "has_trade_ledger": "trade_ledger" in bundle.report.tables,
            "n_trades": int(len(bundle.report.tables["trade_ledger"])) if "trade_ledger" in bundle.report.tables else 0,
        },
        # IMPORTANT: still return report if you want full display for best run (optional)
        "report": bundle.report,
        "bundle_meta": bundle.meta,
    }

# --------------------------
# Main page
# --------------------------
if mode == "Optimization":
    st.title("Optimization")

    st.info("Grid-search runs many backtests. Start small (e.g., 20–100 runs).")

    # --- choose what to optimize ---
    opt_strategy = st.selectbox("Strategy to optimize", ["sma_price", "ma_cross"], index=0)

    metric = st.selectbox(
        "Objective (maximize)",
        ["total_net_pnl", "total_return", "sharpe", "max_drawdown"],
        index=0,
        help="max_drawdown here is scored as (-min drawdown) so higher is better.",
    )

    # --- date range / lookback optimization approach ---
    # simplest: optimize on last N years ending at last available date
    lookbacks = st.multiselect("Lookback windows (years)", [1, 2, 3, 5, 10], default=[1, 2, 3])

    # --- parameter grids ---
    allow_short_opt = st.checkbox("Allow short", value=allow_short)

    if opt_strategy == "sma_price":
        w_min = st.number_input("SMA window min", 2, 500, 10)
        w_max = st.number_input("SMA window max", 2, 500, 200)
        w_step = st.number_input("SMA window step", 1, 50, 5)
        windows = list(range(int(w_min), int(w_max) + 1, int(w_step)))
        grid_size = len(windows) * max(1, len(lookbacks))
        st.caption(f"Grid size: {grid_size} runs")

    else:  # ma_cross
        fast_list = st.multiselect("Fast windows", [5, 10, 15, 20, 30, 40, 50], default=[10, 20])
        slow_list = st.multiselect("Slow windows", [30, 50, 80, 100, 150, 200], default=[50, 100])
        pairs = [(int(f), int(s)) for f in fast_list for s in slow_list if int(f) < int(s)]
        grid_size = len(pairs) * max(1, len(lookbacks))
        st.caption(f"Grid size: {grid_size} runs")

    run_opt = st.button("Run optimization")

    if not run_opt:
        st.stop()

    # --- load your underlying data once so we know the last date ---
    # We’ll reuse your existing single-run logic minimally: call BacktestEngine once with current settings to get last timestamp.
    with st.spinner("Preparing data..."):
        # Use a minimal spec to fetch data and establish last timestamp
        # (We reuse current UI selections; for BMCE this uses uploaded tmp_path already prepared in your backtest flow.
        # If you're in optimization mode, ensure BMCE file is uploaded and tmp_path is created similarly.)
        if source.startswith("bmce"):
            if bmce_file is None:
                st.error("Upload a BMCE file first (sidebar).")
                st.stop()

            # create temp path like in your backtest flow
            suffix = Path(bmce_file.name).suffix.lower()
            tmp_dir = tempfile.mkdtemp(prefix="bmce_upload_")
            tmp_path = str(Path(tmp_dir) / f"{symbol}{suffix}")
            with open(tmp_path, "wb") as f:
                f.write(bmce_file.getbuffer())

            data_cfg_base = {
                "source": "bmce",
                "symbols": [symbol],
                "timezone": timezone,
                "interval": interval,
                "start": None,
                "end": None,
                "bmce_paths": tmp_path,
                "yf_period": "max",
                "yf_interval": "1d",
                "yf_auto_adjust": False,
            }
        else:
            data_cfg_base = {
                "source": "yfinance",
                "symbols": [symbol],
                "timezone": timezone,
                "interval": yf_interval,
                "start": None,
                "end": None,
                "bmce_paths": None,
                "yf_period": yf_period or "max",
                "yf_interval": yf_interval or "1d",
                "yf_auto_adjust": bool(yf_auto_adjust),
            }

        # Quick probe run (cheap) just to get last date from MarketData
        probe_spec = EngineSpec(
            data=DataConfig(**data_cfg_base),
            indicators=IndicatorsConfig(specs=None),
            strategy=StrategyConfig(kind="sma_price", params={"window": 20, "allow_short": allow_short_opt, "nan_policy": "flat"}),
            portfolio=PortfolioConfig(allow_short=allow_short_opt, initial_cash=float(initial_cash), rebalance_policy=str(rebalance_policy),
                                     max_gross=float(max_gross), cash_buffer=float(cash_buffer),
                                     cost_model=CostModel(brokerage_bps=float(brokerage_bps), exchange_bps=float(exchange_bps),
                                                         settlement_bps=float(settlement_bps), slippage_bps=float(slippage_bps), vat_rate=float(vat_rate))),
            periods_per_year=252,
            rf_annual=0.0,
        )
        probe_bundle = BacktestEngine(probe_spec).run()
        sym0 = probe_bundle.md.symbols()[0]
        last_dt = probe_bundle.md.bars[sym0].index.max()

    st.write(f"Last available date in data: **{last_dt.date()}**")

    # --- run grid ---
    results_rows = []
    progress = st.progress(0)
    total = int(grid_size) if grid_size else 1
    k = 0

    def make_start_end_for_lookback(years: int):
        end_dt = last_dt
        start_dt = (end_dt - pd.DateOffset(years=int(years))).to_pydatetime()
        return pd.Timestamp(start_dt).strftime("%Y-%m-%d"), pd.Timestamp(end_dt).strftime("%Y-%m-%d")

    for lb in (lookbacks if lookbacks else [None]):
        if lb is None:
            start_s, end_s = None, None
        else:
            start_s, end_s = make_start_end_for_lookback(int(lb))

        if opt_strategy == "sma_price":
            for w in windows:
                strat_cfg = {"kind": "sma_price", "params": {"window": int(w), "allow_short": bool(allow_short_opt), "nan_policy": "flat"}}
                plot_inds = [f"sma_{int(w)}"]

                spec_dict = {
                    "data": {**data_cfg_base, "start": start_s, "end": end_s},
                    "indicators": {"specs": None, "spec_builder": None, "cache_dir": ".cache/features",
                                   "enable_disk_cache": True, "enable_memory_cache": True, "engine_version": "v1"},
                    "strategy": strat_cfg,
                    "portfolio": {
                        "allow_short": bool(allow_short_opt),
                        "initial_cash": float(initial_cash),
                        "max_gross": float(max_gross),
                        "max_weight_per_asset": None,
                        "cash_buffer": float(cash_buffer),
                        "rebalance_policy": str(rebalance_policy),
                        "fill_price_model": "next_open",
                        "mtm_model": "close_t1",
                        "open_col": "Open",
                        "close_col": "Close",
                        "cost_model": {
                            "brokerage_bps": float(brokerage_bps),
                            "exchange_bps": float(exchange_bps),
                            "settlement_bps": float(settlement_bps),
                            "slippage_bps": float(slippage_bps),
                            "vat_rate": float(vat_rate),
                        },
                        "allow_fractional_shares": False,
                    },
                    "plot_indicators": plot_inds,
                    "periods_per_year": 252,
                    "rf_annual": 0.0,
                }

                out = run_one_backtest(spec_dict)
                rep = out["report"]
                score = score_report(rep, metric)

                results_rows.append({
                    "strategy": "sma_price",
                    "window": int(w),
                    "lookback_years": lb,
                    "start": start_s,
                    "end": end_s,
                    "score": float(score),
                    "total_net_pnl": score_report(rep, "total_net_pnl"),
                    "total_return": score_report(rep, "total_return"),
                    "sharpe": score_report(rep, "sharpe"),
                    "max_drawdown_score": score_report(rep, "max_drawdown"),
                })

                k += 1
                progress.progress(min(1.0, k / total))

        else:
            for fast_w, slow_w in pairs:
                strat_cfg = {"kind": "ma_cross", "params": {"fast_window": int(fast_w), "slow_window": int(slow_w),
                                                           "allow_short": bool(allow_short_opt), "nan_policy": "flat"}}
                plot_inds = [f"sma_{int(fast_w)}", f"sma_{int(slow_w)}"]

                spec_dict = {
                    "data": {**data_cfg_base, "start": start_s, "end": end_s},
                    "indicators": {"specs": None, "spec_builder": None, "cache_dir": ".cache/features",
                                   "enable_disk_cache": True, "enable_memory_cache": True, "engine_version": "v1"},
                    "strategy": strat_cfg,
                    "portfolio": {
                        "allow_short": bool(allow_short_opt),
                        "initial_cash": float(initial_cash),
                        "max_gross": float(max_gross),
                        "max_weight_per_asset": None,
                        "cash_buffer": float(cash_buffer),
                        "rebalance_policy": str(rebalance_policy),
                        "fill_price_model": "next_open",
                        "mtm_model": "close_t1",
                        "open_col": "Open",
                        "close_col": "Close",
                        "cost_model": {
                            "brokerage_bps": float(brokerage_bps),
                            "exchange_bps": float(exchange_bps),
                            "settlement_bps": float(settlement_bps),
                            "slippage_bps": float(slippage_bps),
                            "vat_rate": float(vat_rate),
                        },
                        "allow_fractional_shares": False,
                    },
                    "plot_indicators": plot_inds,
                    "periods_per_year": 252,
                    "rf_annual": 0.0,
                }

                out = run_one_backtest(spec_dict)
                rep = out["report"]
                score = score_report(rep, metric)

                results_rows.append({
                    "strategy": "ma_cross",
                    "fast": int(fast_w),
                    "slow": int(slow_w),
                    "lookback_years": lb,
                    "start": start_s,
                    "end": end_s,
                    "score": float(score),
                    "total_net_pnl": score_report(rep, "total_net_pnl"),
                    "total_return": score_report(rep, "total_return"),
                    "sharpe": score_report(rep, "sharpe"),
                    "max_drawdown_score": score_report(rep, "max_drawdown"),
                })

                k += 1
                progress.progress(min(1.0, k / total))

    res_df = pd.DataFrame(results_rows).sort_values("score", ascending=False).reset_index(drop=True)
    st.subheader("Optimization Results (ranked)")
    st.dataframe(res_df.head(50), use_container_width=True)

    best = res_df.iloc[0].to_dict() if len(res_df) else None
    if best:
        st.success(f"Best score = {best['score']:.4f} | params = {best}")

    # Cleanup BMCE temp dir if created in optimization mode
    if source.startswith("bmce"):
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            if tmp_dir and os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass

    st.stop()

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
            start=start_str,   # <-- add
            end=end_str,       # <-- add
            bmce_tmp_path=tmp_path,
            yf_period=(yf_period if not source.startswith("bmce") else None),
            yf_interval=(yf_interval if not source.startswith("bmce") else None),
            yf_auto_adjust=(yf_auto_adjust if not source.startswith("bmce") else None),
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
            strategy_kind=strategy_kind,
            strategy_params=strategy_params,
        )

    rep = bundle.report

    # --- Outputs ---
    plots = bundle.report.plots
    tables = bundle.report.tables
    series = bundle.report.series

    st.subheader("Price + Indicators + Trades")
    pp = bundle.report.plots["price_panel"]
    sym = bundle.md.symbols()[0]
    bars = bundle.md.bars[sym]  # full OHLCV bars (better than using Close only)

    fig = plot_price_indicators_trades_plotly(
        bars=bundle.md.bars[bundle.md.symbols()[0]],
        indicators=pp["indicators"],
        trades=pp["trades"],
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
        st.info("PnL series not found in report.series (did you add it in ResultsAnalyzer?)")

    st.subheader("Cumulative PnL")
    if "cum_pnl" in series:
        st.pyplot(plot_cum_pnl_series(series["cum_pnl"]))
    else:
        st.info("cum_pnl series not found in report.series (did you add it in ResultsAnalyzer?)")

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
        ledger = tables["trade_ledger"].copy()

        # Optional: nicer formatting
        pct_cols = [c for c in ["return_pct"] if c in ledger.columns]
        money_cols = [c for c in ["gross_pnl", "net_pnl", "entry_cost", "exit_cost"] if c in ledger.columns]
        price_cols = [c for c in ["entry_price", "exit_price"] if c in ledger.columns]

        for c in pct_cols:
            ledger[c] = pd.to_numeric(ledger[c], errors="coerce") * 100.0
        for c in money_cols + price_cols:
            ledger[c] = pd.to_numeric(ledger[c], errors="coerce")

        # Reorder columns if present
        preferred = [
            "entry_time","exit_time","symbol","side","qty",
            "entry_price","exit_price",
            "gross_pnl","entry_cost","exit_cost","net_pnl",
            "return_pct","hold_days"
        ]
        cols = [c for c in preferred if c in ledger.columns] + [c for c in ledger.columns if c not in preferred]
        ledger = ledger[cols]

        st.dataframe(ledger, use_container_width=True)
    else:
        st.info("trade_ledger not found in report.tables")

    
    t_curve = bundle.report.tables["curve_vs_benchmark"]
    t_trade = bundle.report.tables["trade_summary"]
    t_time  = bundle.report.tables["time_summary"]

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
