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
from optimize import (
    default_param_catalog_for_your_app,
    run_optimization,
    OptimizeConfig,
    build_date_window_choices_from_uploaded_bmce,
    add_date_window_param,
)


st.set_page_config(page_title="Backtester", layout="wide")
st.title("Backtester (TA) — Data → Indicators → Strategy → Portfolio → Results")


# --------------------------
# Small plotting helpers
# --------------------------
def render_bundle(bundle):
    rep = bundle.report
    plots = rep.plots
    tables = rep.tables
    series = rep.series
    # paste your existing rendering block here unchanged

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
    max_gross: float,
    cash_buffer: float,
    sizing_mode: str,
    buy_pct_cash: float,
    sell_pct_shares: float,
    brokerage_bps: float,
    exchange_bps: float,
    settlement_bps: float,
    vat_rate: float,
    slippage_bps: float,
) -> EngineSpec:
    # Data config
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

    return EngineSpec(
        data=data_cfg,
        indicators=ind_cfg,
        strategy=strat_cfg,
        portfolio=port_cfg,
        periods_per_year=252,
        rf_annual=0.0,
    )

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
                fig.add_trace(go.Scatter(x=df.index, y=s, mode="lines", name=c,line=dict(width=1),))

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
                    mode="text",
                    text="BUY",
                    name="BUY",
                    textposition="top center",
                    textfont=dict(size=10, color ="#1C60E9"),
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
                    textfont=dict( size=10, color="#1C60E9"),
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
mode = st.sidebar.radio("Choose mode", ["Backtest", "Optimize"], index=0)


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

if mode=="Backtest":

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

if mode == "Optimize":
    st.sidebar.header("Optimization")
    search_mode = st.sidebar.selectbox("Search mode", ["random", "grid"], index=0)
    n_trials = st.sidebar.number_input("Trials (random)", min_value=10, value=200, step=10)
    top_k = st.sidebar.number_input("Show top K", min_value=5, value=30, step=5)
    optimize_dates = st.sidebar.checkbox("Optimize date window", value=False)

    st.sidebar.subheader("Select parameters to optimize")
    # Build a catalog and let the user pick keys
    cat = default_param_catalog_for_your_app()
    # Remove strategy.kind unless you really want it tuned
    cat.pop("strategy.kind", None)

    selectable_keys = list(cat.keys())
    active_keys = st.sidebar.multiselect(
        "Active parameters",
        options=selectable_keys,
        default=["strategy.fast_window", "strategy.slow_window"] if strategy_kind == "ma_cross" else ["strategy.window"],
    )
    st.sidebar.header("Run")
    run_btn = st.sidebar.button("Run Optimization")

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
    sizing_mode: str,
    buy_pct_cash: float,
    sell_pct_shares: float,
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

    if mode == "Backtest":
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
                sizing_mode=str(sizing_mode),
                buy_pct_cash=float(buy_pct_cash),
                sell_pct_shares=float(sell_pct_shares),
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

    elif mode == "Optimize":
        # ---- Optimize mode ----
        base_spec = make_base_spec(
            source_key=source_key,
            symbol=symbol,
            timezone=timezone,
            interval=interval,
            bmce_tmp_path=tmp_path,
            start=start_str,
            end=end_str,
            yf_period=(yf_period if source_key != "bmce" else None),
            yf_interval=(yf_interval if source_key != "bmce" else None),
            yf_auto_adjust=(yf_auto_adjust if source_key != "bmce" else None),
            strategy_kind=strategy_kind,
            strategy_params=strategy_params,
            allow_short=bool(allow_short),
            initial_cash=float(initial_cash),
            rebalance_policy=str(rebalance_policy),
            max_gross=float(max_gross),
            cash_buffer=float(cash_buffer),
            sizing_mode=str(sizing_mode),
            buy_pct_cash=float(buy_pct_cash),
            sell_pct_shares=float(sell_pct_shares),
            brokerage_bps=float(brokerage_bps),
            exchange_bps=float(exchange_bps),
            settlement_bps=float(settlement_bps),
            vat_rate=float(vat_rate),
            slippage_bps=float(slippage_bps),
        )

        cat = default_param_catalog_for_your_app()
        cat.pop("strategy.kind", None)  # optional

        # active_keys, search_mode, n_trials, top_k, optimize_dates
        # must come from the optimize sidebar you already built.
        if optimize_dates and source_key == "bmce":
            windows = build_date_window_choices_from_uploaded_bmce(
                base_data_cfg=base_spec.data,
                symbol=symbol,
                min_bars=252,
                step_bars=21,
                max_windows=200,
            )
            add_date_window_param(cat, windows)
            if "data.window" not in active_keys:
                active_keys = list(active_keys) + ["data.window"]

        opt_cfg = OptimizeConfig(
            mode=search_mode,
            n_trials=int(n_trials),
            top_k=int(top_k),
            seed=42,
            objective="pnl_then_efficiency",
            verbose=False,
        )

        with st.spinner("Running optimization..."):
            best, top_df, best_spec = run_optimization(
                base_spec=base_spec,
                catalog=cat,
                active_keys=active_keys,
                cfg=opt_cfg,
            )

        st.subheader("Optimization results")
        st.write("Best trial:")
        st.json({
            "pnl": best.pnl,
            "traded_notional": best.traded_notional,
            "profit_per_notional": best.efficiency,
            "params": best.params,
        })
        st.dataframe(top_df, use_container_width=True)

        if st.button("Run best configuration"):
            with st.spinner("Running best backtest..."):
                bundle = BacktestEngine(best_spec).run()
            # reuse your exact rendering code below:
            rep = bundle.report
            ...

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
