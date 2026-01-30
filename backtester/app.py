# app.py (Streamlit entrypoint)

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Your project imports
from engine import BacktestEngine, EngineSpec, DataConfig, IndicatorsConfig, StrategyConfig
from portfolio import PortfolioConfig, CostModel

from optimize import (
    OptimizeConfig,
    TrialResult,
    ParamDef,
    default_param_catalog_for_strategy,
    run_optimization,
)

st.set_page_config(page_title="Backtester", layout="wide")
st.title("Backtester (TA) — Backtest / Optimize")


# ============================================================
# Rendering (keep your current rendering block)
# ============================================================

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
# Spec builder
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
    cost_model: CostModel,
) -> EngineSpec:
    # Data
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

    # Strategy
    strat_cfg = StrategyConfig(kind=strategy_kind, params=dict(strategy_params or {}))

    # Indicators
    ind_cfg = IndicatorsConfig(specs=None)

    # Portfolio
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


def apply_best_params_to_spec(base_spec: EngineSpec, best_params: Dict[str, Any]) -> EngineSpec:
    """
    Applies best_params (keys like strategy.fast_window, portfolio.buy_pct_cash, data.window, ...)
    and returns a NEW EngineSpec.
    """
    # --- data ---
    data_cfg = base_spec.data
    if "data.window" in best_params and best_params["data.window"] is not None:
        start, end = best_params["data.window"]
        data_cfg = type(data_cfg)(**{**data_cfg.__dict__, "start": start, "end": end})

    # --- strategy ---
    strat_cfg = base_spec.strategy
    p = dict(strat_cfg.params or {})
    if "strategy.fast_window" in best_params:
        p["fast_window"] = int(best_params["strategy.fast_window"])
    if "strategy.slow_window" in best_params:
        p["slow_window"] = int(best_params["strategy.slow_window"])
    if "strategy.window" in best_params:
        p["window"] = int(best_params["strategy.window"])
    strat_cfg = StrategyConfig(kind=strat_cfg.kind, params=p)

    # --- portfolio ---
    port_cfg = base_spec.portfolio
    pdict = dict(port_cfg.__dict__)
    if "portfolio.buy_pct_cash" in best_params:
        pdict["buy_pct_cash"] = float(best_params["portfolio.buy_pct_cash"])
    if "portfolio.sell_pct_shares" in best_params:
        pdict["sell_pct_shares"] = float(best_params["portfolio.sell_pct_shares"])
    if "portfolio.cooldown_bars" in best_params:
        pdict["cooldown_bars"] = int(best_params["portfolio.cooldown_bars"])
    port_cfg = type(port_cfg)(**pdict)

    return EngineSpec(
        data=data_cfg,
        indicators=base_spec.indicators,
        strategy=strat_cfg,
        portfolio=port_cfg,
        periods_per_year=base_spec.periods_per_year,
        rf_annual=base_spec.rf_annual,
    )


# ============================================================
# BMCE window generator (for optimizing data.window)
# ============================================================
def build_date_windows_from_df(
    df: pd.DataFrame,
    date_col_candidates=("Date", "date", "timestamp", "Datetime", "DATE"),
    min_bars: int = 252,
    step_bars: int = 21,
    max_windows: int = 200,
) -> List[Tuple[str, str]]:
    # find date col
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # fallback: first column
        date_col = df.columns[0]

    dts = pd.to_datetime(df[date_col], errors="coerce")
    dts = dts.dropna().sort_values().reset_index(drop=True)
    if len(dts) < min_bars:
        return []

    out = []
    start = 0
    while True:
        end = start + min_bars - 1
        if end >= len(dts):
            break
        s = dts.iloc[start].date().isoformat()
        e = dts.iloc[end].date().isoformat()
        out.append((s, e))
        if len(out) >= max_windows:
            break
        start += step_bars
    return out


# ============================================================
# Sidebar: common (data + strategy are shared)
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

st.sidebar.header("Strategy")
strategy_kind = st.sidebar.selectbox("Strategy kind", ["ma_cross", "sma_price"], index=0)
allow_short = st.sidebar.checkbox("Allow short", value=False)


# ============================================================
# Backtest mode controls (ONLY shown in Backtest mode)
# ============================================================
start_str = end_str = None
strategy_params: Dict[str, Any] = {}

if mode == "Backtest":
    st.sidebar.header("Backtest period")
    use_date_range = st.sidebar.checkbox("Use date range", value=False)
    if use_date_range:
        start_date = st.sidebar.date_input("Start date", value=None)
        end_date = st.sidebar.date_input("End date", value=None)
        if start_date and end_date and start_date > end_date:
            st.sidebar.error("Start date must be <= End date.")
            st.stop()
        start_str = start_date.isoformat() if start_date else None
        end_str = end_date.isoformat() if end_date else None

    st.sidebar.subheader("Strategy parameters")
    nan_policy = "flat"

    if strategy_kind == "ma_cross":
        fast = st.sidebar.number_input("Fast SMA window", min_value=2, max_value=500, value=20, step=1)
        slow = st.sidebar.number_input("Slow SMA window", min_value=3, max_value=500, value=50, step=1)
        if fast >= slow:
            st.sidebar.warning("Fast must be < Slow. Adjusting fast automatically.")
            fast = min(int(fast), int(slow) - 1)
        strategy_params = {"fast_window": int(fast), "slow_window": int(slow), "allow_short": bool(allow_short), "nan_policy": nan_policy}
    else:
        window = st.sidebar.number_input("SMA window", min_value=2, max_value=500, value=50, step=1)
        strategy_params = {"window": int(window), "allow_short": bool(allow_short), "nan_policy": nan_policy}

    st.sidebar.header("Portfolio")
    initial_cash = st.sidebar.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
    rebalance_policy = st.sidebar.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)
    sizing_mode = st.sidebar.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=0)
    cooldown_bars = st.sidebar.number_input("Min bars between trades (cooldown)", min_value=0, value=0, step=1)

    st.sidebar.header("Sizing")
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

    run_btn = st.sidebar.button("Run backtest")


# ============================================================
# Optimize mode controls (ONLY shown in Optimize mode)
# ============================================================
if mode == "Optimize":
    # We still need baseline values for non-optimized params.
    # Keep them tucked into an expander so the sidebar stays “optimization-only”.
    with st.sidebar.expander("Fixed baseline (used when NOT optimized)", expanded=False):
        initial_cash = st.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
        rebalance_policy = st.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)
        sizing_mode = st.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=0)

        # baseline defaults (will be overridden if you optimize them)
        buy_pct_cash = st.slider("Baseline buy % cash", 0.01, 1.00, 0.25, 0.01)
        sell_pct_shares = st.slider("Baseline sell % shares", 0.01, 1.00, 1.00, 0.01)
        cooldown_bars = st.number_input("Baseline cooldown_bars", min_value=0, value=0, step=1)

        # baseline strategy params
        nan_policy = "flat"
        if strategy_kind == "ma_cross":
            fast0 = st.number_input("Baseline fast_window", min_value=2, max_value=500, value=20, step=1)
            slow0 = st.number_input("Baseline slow_window", min_value=3, max_value=500, value=50, step=1)
            if fast0 >= slow0:
                st.warning("Baseline fast must be < slow. Adjusting fast.")
                fast0 = min(int(fast0), int(slow0) - 1)
            strategy_params = {"fast_window": int(fast0), "slow_window": int(slow0), "allow_short": bool(allow_short), "nan_policy": nan_policy}
        else:
            w0 = st.number_input("Baseline window", min_value=2, max_value=500, value=50, step=1)
            strategy_params = {"window": int(w0), "allow_short": bool(allow_short), "nan_policy": nan_policy}

        apply_costs = st.checkbox("Apply costs", value=False)
        if apply_costs:
            brokerage_bps = st.number_input("Brokerage (bps)", value=60.0, step=1.0)
            exchange_bps = st.number_input("Exchange (bps)", value=10.0, step=1.0)
            settlement_bps = st.number_input("Settlement (bps)", value=20.0, step=1.0)
            vat_rate = st.number_input("VAT rate", value=0.10, step=0.01)
            slippage_bps = st.number_input("Slippage (bps)", value=0.0, step=1.0)
        else:
            brokerage_bps = exchange_bps = settlement_bps = slippage_bps = 0.0
            vat_rate = 0.0

    # Optimization selectors
    st.sidebar.header("Optimization targets")
    catalog = default_param_catalog_for_strategy(strategy_kind)
    selectable_keys = list(catalog.keys())

    active_keys = st.sidebar.multiselect(
        "Select parameters to optimize",
        options=selectable_keys,
        default=["strategy.fast_window", "strategy.slow_window"] if strategy_kind == "ma_cross" else ["strategy.window"],
    )

    # Interval per selected parameter (your request)
    st.sidebar.subheader("Intervals for selected parameters")
    for k in active_keys:
        pdef: ParamDef = catalog[k]
        st.sidebar.markdown(f"**{k}**")

        if pdef.kind == "int":
            lo, hi, step = pdef.domain
            c1, c2, c3 = st.sidebar.columns(3)
            lo2 = c1.number_input(f"{k} min", value=int(lo), step=1, key=f"{k}_min")
            hi2 = c2.number_input(f"{k} max", value=int(hi), step=1, key=f"{k}_max")
            step2 = c3.number_input(f"{k} step", value=int(step), step=1, key=f"{k}_step")
            if lo2 > hi2:
                st.sidebar.error(f"{k}: min must be <= max")
                st.stop()
            pdef.domain = (int(lo2), int(hi2), int(step2))

        elif pdef.kind == "float":
            lo, hi, step = pdef.domain
            c1, c2, c3 = st.sidebar.columns(3)
            lo2 = c1.number_input(f"{k} min", value=float(lo), step=0.01, key=f"{k}_min")
            hi2 = c2.number_input(f"{k} max", value=float(hi), step=0.01, key=f"{k}_max")
            step2 = c3.number_input(f"{k} step", value=float(step), step=0.01, key=f"{k}_step")
            if lo2 > hi2:
                st.sidebar.error(f"{k}: min must be <= max")
                st.stop()
            pdef.domain = (float(lo2), float(hi2), float(step2))

        elif pdef.kind == "choice":
            choices = list(pdef.domain)
            picked = st.sidebar.multiselect(f"{k} choices", choices, default=choices, key=f"{k}_choices")
            pdef.domain = picked

        elif pdef.kind == "date_window":
            # only makes sense for BMCE uploads in your workflow
            if source_key != "bmce":
                st.sidebar.warning("data.window optimization is intended for BMCE uploads.")
                pdef.domain = []
            else:
                min_bars = st.sidebar.number_input("min bars per window", min_value=30, value=252, step=21, key="dw_min_bars")
                step_bars = st.sidebar.number_input("step bars", min_value=1, value=21, step=1, key="dw_step_bars")
                max_windows = st.sidebar.number_input("max windows", min_value=10, value=200, step=10, key="dw_max_windows")
                # domain filled later after file preview is loaded (we’ll compute after reading bmce_file)

        catalog[k] = pdef  # update


    st.sidebar.header("Optimization method")
    method = st.sidebar.selectbox("Method", ["random", "grid"], index=0)
    seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)
    top_k = st.sidebar.number_input("Show top K", min_value=5, value=30, step=5)

    # Method-specific controls
    if method == "random":
        n_trials = st.sidebar.number_input("Trials", min_value=10, value=200, step=10)
    else:
        n_trials = 0

    run_btn = st.sidebar.button("Run optimization")


# ============================================================
# Main page: BMCE preview (shared)
# ============================================================
preview_df = None

if source_key == "bmce":
    st.subheader("BMCE upload")
    if bmce_file is None:
        st.info("Upload a BMCE CSV/XLSX.")
        st.stop()

    try:
        if bmce_file.name.lower().endswith(".csv"):
            preview_df = pd.read_csv(bmce_file)
        else:
            preview_df = pd.read_excel(bmce_file, engine="openpyxl")
        st.caption("File preview (first rows)")
        st.dataframe(preview_df.head(30), use_container_width=True)
    except Exception as e:
        st.error(f"Could not preview file: {e}")
        st.stop()
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
    # Save uploaded BMCE to a temp file so the engine can read it
    if source_key == "bmce":
        suffix = Path(bmce_file.name).suffix.lower()
        tmp_dir = tempfile.mkdtemp(prefix="bmce_upload_")
        tmp_path = str(Path(tmp_dir) / f"{symbol}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(bmce_file.getbuffer())

        # If in optimize mode and data.window is selected, build windows now
        if mode == "Optimize" and "data.window" in active_keys:
            windows = build_date_windows_from_df(
                preview_df,
                min_bars=int(st.session_state.get("dw_min_bars", 252)),
                step_bars=int(st.session_state.get("dw_step_bars", 21)),
                max_windows=int(st.session_state.get("dw_max_windows", 200)),
            )
            if not windows:
                st.warning("Could not build any date windows (maybe too few bars). Removing data.window from optimization.")
                active_keys = [k for k in active_keys if k != "data.window"]
            else:
                catalog["data.window"].domain = windows
                st.sidebar.caption(f"Generated {len(windows)} windows for data.window.")

    # Cost model (shared)
    cost_model = CostModel(
        brokerage_bps=float(brokerage_bps),
        exchange_bps=float(exchange_bps),
        settlement_bps=float(settlement_bps),
        slippage_bps=float(slippage_bps),
        vat_rate=float(vat_rate),
    )

    # Base spec
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
        cost_model=cost_model,
    )

    if mode == "Backtest":
        with st.spinner("Running backtest..."):
            bundle = BacktestEngine(base_spec).run()
        render_bundle(bundle)

    else:
        opt_cfg = OptimizeConfig(
            method=method,
            seed=int(seed),
            n_trials=int(n_trials) if method in ("random") else 0,
            top_k=int(top_k),
            objective="pnl_then_eff",

        )

        with st.spinner("Running optimization..."):
            best, top_df, best_params = run_optimization(
                base_spec=base_spec,
                active_keys=active_keys,
                catalog=catalog,
                cfg=opt_cfg,
            )

        st.subheader("Optimization results")
        st.json({
            "pnl": best.pnl,
            "traded_notional": best.traded_notional,
            "profit_per_notional": best.efficiency,
            "params": best.params,
            "error": best.error,
        })
        st.dataframe(top_df, use_container_width=True)

        st.divider()
        if st.button("Run best configuration backtest"):
            best_spec = apply_best_params_to_spec(base_spec, best.params)
            with st.spinner("Running best backtest..."):
                bundle = BacktestEngine(best_spec).run()
            render_bundle(bundle)

finally:
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
