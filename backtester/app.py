# app.py (Streamlit entrypoint) — FIXED to match your optimize.py API

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from dataclasses import replace
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import plotly.graph_objects as go

# ---- Project imports ----
from engine import BacktestEngine, EngineSpec, DataConfig, IndicatorsConfig, StrategyConfig
from portfolio import PortfolioConfig, CostModel
from optimize import (
    OptimizeConfig,
    TrialResult,
    ParamDef,
    default_param_catalog,
    run_optimization,
)
from data import BMCEDataSource, YahooFinanceDataSource
from results import ResultsAnalyzer

import optimize as _opt_mod
st.sidebar.caption(f"optimize.py loaded from: {_opt_mod.__file__}")
st.sidebar.caption(f"run_optimization: {_opt_mod.run_optimization.__module__}.{_opt_mod.run_optimization.__name__}")

st.set_page_config(page_title="Backtester", layout="wide")
st.title("Backtester (TA) — Backtest / Optimize")

def info_popover(key: str):
    e = EXPLAIN.get(key)
    if not e:
        st.write("")
        return
    with st.popover("ℹ️"):
        st.markdown(f"**{e['title']}**")
        st.write(e["why"])
        if e.get("latex"):
            st.latex(e["latex"])
        if e.get("notes"):
            st.caption(e["notes"])

def metric_with_info(label: str, value: str, explain_key: str):
    c1, c2 = st.columns([6, 1], vertical_alignment="center")
    with c1:
        st.metric(label, value)
    with c2:
        info_popover(explain_key)

# ============================================================
# Metric explanations (only what you asked for)
# ============================================================

EXPLAIN = {
    # --- Plots ---
    "plot.cum_returns": {
        "title": "Cumulative Returns",
        "why": "Shows the compounded performance over time. Easier to compare strategy vs benchmark.",
        "latex": r"CR_t=\prod_{i=1}^{t}(1+r_i)-1",
        "notes": "Uses per-bar equity returns r_i.",
    },
    "plot.drawdown": {
        "title": "Drawdown",
        "why": "Measures peak-to-trough decline. Captures risk / pain / tail behavior.",
        "latex": r"DD_t=\frac{E_t}{\max_{u\le t}E_u}-1,\quad \max DD=\min_t DD_t",
        "notes": "E_t is equity at time t.",
    },

    # --- Trade ledger columns ---
    "ledger.pnl": {
        "title": "Gross PnL / Net PnL / Return %",
        "why": "Gross PnL is price PnL. Net PnL subtracts costs. Return% normalizes net PnL by entry notional.",
        "latex": (
            r"\mathrm{GrossPnL}=\begin{cases}"
            r"q(\,P_{exit}-P_{entry}\,),& \text{LONG}\\"
            r"q(\,P_{entry}-P_{exit}\,),& \text{SHORT}"
            r"\end{cases}"
            "\n"
            r"\mathrm{NetPnL}=\mathrm{GrossPnL}-Cost_{entry}-Cost_{exit}"
            "\n"
            r"\mathrm{Return\%}=\frac{\mathrm{NetPnL}}{q\cdot P_{entry}}"
        ),
        "notes": "q is the closed quantity for that round-trip/lot.",
    },

    # --- Trade performance ---
    "trade.profit_factor": {
        "title": "Profit Factor",
        "why": "Quality of the payoff distribution: how much profit you make per unit of loss (higher is better).",
        "latex": r"\mathrm{PF}=\frac{\sum \mathrm{Wins}}{|\sum \mathrm{Losses}|}",
        "notes": "Computed on net PnL of closed trades.",
    },

    # --- Optimization top candidates ---
    "opt.top": {
        "title": "Optimization metrics (ranking)",
        "why": "You rank candidates by PnL first, then Efficiency. Turnover proxy is traded notional. n_fills is trade count.",
        "latex": (
            r"\mathrm{PnL}=E_T-E_0"
            "\n"
            r"\mathrm{TradedNotional}=\sum_k |\mathrm{qty}_k|\cdot \mathrm{price}_k"
            "\n"
            r"\mathrm{Efficiency}=\frac{\mathrm{PnL}}{\mathrm{TradedNotional}}"
        ),
        "notes": "Ranking rule: sort by (PnL desc, Efficiency desc).",
    },
}

def info_popover(explain_key: str):
    e = EXPLAIN.get(explain_key)
    if not e:
        return
    with st.popover("ℹ️"):
        st.markdown(f"**{e['title']}**")
        st.write(e.get("why", ""))
        if e.get("latex"):
            st.latex(e["latex"])
        if e.get("notes"):
            st.caption(e["notes"])

import plotly.graph_objects as go

INFO = {
    # plots
    "plot.cum_returns": (
        "Cumulative return:  (Π_t (1+r_t)) - 1.\n"
        "Shows compounded growth of equity. Compare to benchmark to judge added value.\n"
        "Benchmark series is aligned by date and forward-filled."
    ),
    "plot.drawdown": (
        "Drawdown: DD_t = Equity_t / max_{u≤t}(Equity_u) - 1.\n"
        "Measures peak-to-trough decline (risk / pain). More negative = worse."
    ),
    # pnl
    "metric.gross_pnl": (
        "Gross PnL (per bar): ΔEquity_t = Equity_t - Equity_{t-1}.\n"
        "This is currency PnL before attributing per-trade costs in a ledger sense."
    ),
    "metric.net_pnl": (
        "Net PnL: Gross PnL minus transaction costs (commissions, VAT, slippage).\n"
        "Used to assess realism when costs are enabled."
    ),
    # trade ledger
    "ledger.return_pct": (
        "Trade return %: net_pnl / entry_notional.\n"
        "Normalizes PnL by capital deployed to compare trades across sizes."
    ),
    "trade.profit_factor": (
        "Profit Factor = (sum of winning trade net PnL) / |sum of losing trade net PnL|.\n"
        ">1 means wins outweigh losses; higher is better."
    ),
    # optimization
    "opt.pnl": "Optimization PnL: final_equity - initial_cash.",
    "opt.traded_notional": "Traded notional: Σ |qty|×price across fills (activity / turnover proxy).",
    "opt.efficiency": (
        "Efficiency = PnL / traded_notional.\n"
        "Secondary objective after PnL: prefers profit per unit of turnover."
    ),
    "opt.n_fills": "Number of fills executed (proxy for trade frequency).",
}

def plot_with_info(title: str, info_key: str, fig, *, key: str, use_container_width: bool = True) -> None:
    """
    Render a plot with a small hover tooltip (ⓘ).
    Works for matplotlib Figure and plotly Figure.
    """
    help_txt = INFO.get(info_key, "No explanation available.")

    h1, h2 = st.columns([0.92, 0.08])
    with h1:
        st.subheader(title)
    with h2:
        # hover tooltip:
        st.button("ⓘ", key=f"help_{key}", help=help_txt)

    if fig is None:
        st.info("No data to plot.")
        return

    # Plotly
    if isinstance(fig, go.Figure):
        st.plotly_chart(fig, use_container_width=use_container_width)
        return

    # Matplotlib fallback
    st.pyplot(fig, use_container_width=use_container_width, clear_figure=False)


def table_with_info(title: str, explain_key: str, df: pd.DataFrame):
    c1, c2 = st.columns([10, 1], vertical_alignment="center")
    with c1:
        st.subheader(title)
    with c2:
        info_popover(explain_key)
    st.dataframe(df, use_container_width=True)


# ============================================================
# Plotting helpers (NORMAL price line + indicators + buy/sell)
# ============================================================
@st.cache_data(show_spinner=False)
def load_benchmark_market_data_cached(
    bench_source_key: str,
    bench_symbol: str,
    timezone: str,
    interval: str,
    bmce_path: str | None,
    start: str | None,
    end: str | None,
    yf_period: str | None,
    yf_interval: str | None,
    yf_auto_adjust: bool | None,
):
    if bench_source_key == "bmce":
        if bmce_path is None:
            raise ValueError("Benchmark BMCE selected but no file path provided.")
        ds = BMCEDataSource(timezone=timezone)
        md = ds.load(
            symbols=[bench_symbol],
            start=start,
            end=end,
            interval=interval,
            paths=bmce_path,
        )
        return md

    # yfinance
    ds = YahooFinanceDataSource(timezone=timezone)
    md = ds.load(
        symbols=[bench_symbol],
        start=start,
        end=end,
        interval=(yf_interval or "1d"),
        auto_adjust=bool(yf_auto_adjust),
        progress=False,
    )
    return md

def plot_price_indicators_trades_line(
    bars: pd.DataFrame,
    indicators: pd.DataFrame | None,
    trades: pd.DataFrame | None,
    indicator_cols: list[str] | None = None,
) -> go.Figure:
    df = bars.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].astype(float), mode="lines", name="Close"))

    if indicators is not None and not indicators.empty:
        ind = indicators.copy()
        if not isinstance(ind.index, pd.DatetimeIndex):
            ind.index = pd.to_datetime(ind.index)
        ind = ind.reindex(df.index)

        cols = [c for c in (indicator_cols or list(ind.columns)) if c in ind.columns]
        for c in cols:
            s = ind[c].astype(float)
            if s.notna().any():
                fig.add_trace(go.Scatter(x=df.index, y=s, mode="lines", name=c))

    if trades is not None and not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
        t = t.dropna(subset=["timestamp"]).sort_values("timestamp")

        if "side" not in t.columns:
            t["side"] = np.where(t["qty"].astype(float) > 0, "BUY", "SELL")

        # y at fill price if exists else close
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
                    mode="markers+text",
                    marker=dict(size=10, symbol="triangle-up"),
                    text="BUY",
                    textposition="top center",
                    name="BUY",
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["y_plot"],
                    mode="markers+text",
                    marker=dict(size=10, symbol="triangle-down"),
                    text="SELL",
                    textposition="bottom center",
                    name="SELL",
                )
            )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
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
    ax.grid(True)
    ax.legend()
    return fig


def plot_drawdown_red(dd: pd.Series) -> plt.Figure:
    fig = plt.figure(figsize=(14, 3))
    ax = plt.gca()
    ax.plot(dd.index, dd.values, label="Drawdown")
    ax.fill_between(dd.index, dd.values, 0.0, where=(dd.values < 0.0), alpha=0.35)
    ax.set_title("Drawdown")
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
    ax.bar(years, vals)
    ax.set_title("Yearly Returns")
    ax.grid(True, axis="y")
    return fig

def render_explain(report, key: str):
    e = report.explain.get(key)
    if not e:
        return
    with st.popover("ℹ️"):
        st.markdown(f"**{e.title}**")
        st.write(e.why)
        if e.latex:
            st.latex(e.latex)
        if e.notes:
            st.caption(e.notes)
        if e.columns:
            st.markdown("**Columns**")
            for c, desc in e.columns.items():
                st.markdown(f"- `{c}`: {desc}")

def metric_card(label: str, value, report, explain_key: str):
    c1, c2 = st.columns([8, 1], vertical_alignment="center")
    with c1:
        st.metric(label, value)
    with c2:
        render_explain(report, explain_key)

def table_with_info(title: str, df: pd.DataFrame, report, explain_key: str):
    c1, c2 = st.columns([8, 1], vertical_alignment="center")
    with c1:
        st.subheader(title)
    with c2:
        render_explain(report, explain_key)
    st.dataframe(df, use_container_width=True)

def plot_with_info(title: str, fig, report, explain_key: str):
    c1, c2 = st.columns([8, 1], vertical_alignment="center")
    with c1:
        st.subheader(title)
    with c2:
        render_explain(report, explain_key)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Render bundle
# ============================================================

def render_bundle(bundle):
    rep = bundle.report
    plots = rep.plots
    tables = rep.tables

    st.subheader("Price + Indicators + Trades")
    pp = plots["price_panel"]
    sym0 = bundle.md.symbols()[0]
    bars = bundle.md.bars[sym0]

    fig = plot_price_indicators_trades_line(
        bars=bars,
        indicators=pp.get("indicators"),
        trades=pp.get("trades"),
        indicator_cols=pp.get("indicator_cols"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Returns vs Benchmark")
    cvb = plots["cum_vs_bench"]
    st.pyplot(plot_cum_vs_bench(cvb["strategy"], cvb.get("benchmark")))


    st.subheader("Drawdown")
    st.pyplot(plot_drawdown_red(plots["drawdown"]))



    st.subheader("Monthly Returns Heatmap")
    st.pyplot(plot_monthly_heatmap_with_values(plots["monthly_heatmap"]))

    st.subheader("Yearly Returns")
    st.pyplot(plot_yearly_returns_bar(plots["yearly_bar"]))

    st.subheader("Trades (fills)")
    if "trades" in tables:
        st.dataframe(tables["trades"], use_container_width=True)

    st.subheader("Trade Ledger (PnL per closed trade)")
        # --- Trade Ledger (explain Gross/Net/Return%) ---
    if "trade_ledger" in tables:
        table_with_info(
            "Trades Ledger (PnL per trade)",
            "ledger.pnl",
            tables["trade_ledger"],
        )
    else:
        st.info("trade_ledger not found in report.tables")


    st.subheader("Trade Performance (summary)")
        # --- Trade Performance (explain Profit Factor) ---
    if "trade_performance" in tables:
        table_with_info(
            "Trade Performance (summary)",
            "trade.profit_factor",
            tables["trade_performance"],
        )
    else:
        st.info("trade_performance not found in report.tables")



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
# BMCE window generator (for optimizing data.window)
# ============================================================

def build_date_windows_from_df(
    df: pd.DataFrame,
    date_col_candidates=("Date", "date", "timestamp", "Datetime", "DATE"),
    min_bars: int = 252,
    step_bars: int = 21,
    max_windows: int = 200,
) -> List[Tuple[str, str]]:
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
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
# Session state helpers
# ============================================================

def ss_get(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ============================================================
# Sidebar: Data source
# ============================================================

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

# Reset optimization UI on strategy change (prevents stale widget keys)
prev_kind = ss_get("prev_strategy_kind", strategy_kind)
if prev_kind != strategy_kind:
    for k in list(st.session_state.keys()):
        if k.endswith("_min") or k.endswith("_max") or k.endswith("_step") or k.endswith("_choices"):
            del st.session_state[k]
    st.session_state["prev_strategy_kind"] = strategy_kind

# ============================================================
# Sidebar: Benchmark (optional)
# ============================================================
st.sidebar.header("Benchmark (optional)")
use_benchmark = st.sidebar.checkbox("Enable benchmark", value=False)

bench_source_key = None
bench_symbol = None
bench_bmce_file = None
bench_yf_period = None
bench_yf_interval = None
bench_yf_auto_adjust = None

if use_benchmark:
    bench_mode = st.sidebar.selectbox(
        "Benchmark source",
        ["same as main source", "yfinance", "bmce (upload)"],
        index=0
    )

    if bench_mode == "same as main source":
        bench_source_key = source_key
        bench_symbol = st.sidebar.text_input("Benchmark symbol", value="MASI" if source_key == "bmce" else "SPY")

        if bench_source_key == "bmce":
            bench_bmce_file = st.sidebar.file_uploader("Upload BMCE benchmark CSV/XLSX", type=["csv", "xlsx"], key="bench_bmce")
        else:
            bench_yf_period = st.sidebar.text_input("Benchmark yfinance period", value=yf_period or "5y", key="bench_yf_period")
            bench_yf_interval = st.sidebar.selectbox("Benchmark yfinance interval", ["1d"], index=0, key="bench_yf_interval")
            bench_yf_auto_adjust = st.sidebar.checkbox("Benchmark auto_adjust", value=bool(yf_auto_adjust), key="bench_yf_auto_adjust")

    elif bench_mode == "yfinance":
        bench_source_key = "yfinance"
        bench_symbol = st.sidebar.text_input("Benchmark symbol", value="SPY")
        bench_yf_period = st.sidebar.text_input("Benchmark yfinance period", value="5y")
        bench_yf_interval = st.sidebar.selectbox("Benchmark yfinance interval", ["1d"], index=0)
        bench_yf_auto_adjust = st.sidebar.checkbox("Benchmark auto_adjust", value=False)

    else:  # bmce upload
        bench_source_key = "bmce"
        bench_symbol = st.sidebar.text_input("Benchmark symbol", value="MASI")
        bench_bmce_file = st.sidebar.file_uploader("Upload BMCE benchmark CSV/XLSX", type=["csv", "xlsx"], key="bench_bmce2")

# ============================================================
# Main preview (BMCE)
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


# ============================================================
# Tabs: Backtest / Optimize
# ============================================================

tab_backtest, tab_opt = st.tabs(["Backtest", "Optimize"])


# ============================================================
# Backtest Tab (FORM)
# ============================================================

with tab_backtest:
    st.subheader("Backtest")

    with st.form("backtest_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            use_date_range = st.checkbox("Use date range", value=False)
        with c2:
            start_date = st.date_input("Start date", value=None, disabled=(not use_date_range))
        with c3:
            end_date = st.date_input("End date", value=None, disabled=(not use_date_range))

        start_str = start_date.isoformat() if (use_date_range and start_date) else None
        end_str = end_date.isoformat() if (use_date_range and end_date) else None

        st.markdown("### Strategy parameters")
        nan_policy = "flat"
        strategy_params: Dict[str, Any] = {}

        if strategy_kind == "ma_cross":
            fast = st.number_input("Fast SMA window", min_value=2, max_value=500, value=20, step=1)
            slow = st.number_input("Slow SMA window", min_value=3, max_value=500, value=50, step=1)
            if fast >= slow:
                st.warning("Fast must be < Slow. Auto-adjusting fast.")
                fast = min(int(fast), int(slow) - 1)
            strategy_params = {"fast_window": int(fast), "slow_window": int(slow), "allow_short": bool(allow_short), "nan_policy": nan_policy}
        else:
            window = st.number_input("SMA window", min_value=2, max_value=500, value=50, step=1)
            strategy_params = {"window": int(window), "allow_short": bool(allow_short), "nan_policy": nan_policy}

        st.markdown("### Portfolio")
        initial_cash = st.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
        rebalance_policy = st.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)
        sizing_mode = st.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=1)
        cooldown_bars = st.number_input("Min bars between trades (cooldown)", min_value=0, value=0, step=1)

        st.markdown("### Sizing (percentages)")
        buy_pct_cash = st.slider("Buy % of cash per entry", 0.01, 1.00, 0.25, 0.01)
        sell_pct_shares = st.slider("Sell % of shares per exit", 0.01, 1.00, 1.00, 0.01)

        st.markdown("### Costs")
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

        run_backtest = st.form_submit_button("Run backtest")

    if run_backtest:
        # Save uploaded file to temp for engine
        tmp_path = None
        tmp_dir = None
        try:
            if source_key == "bmce":
                suffix = Path(bmce_file.name).suffix.lower()
                tmp_dir = tempfile.mkdtemp(prefix="bmce_upload_")
                tmp_path = str(Path(tmp_dir) / f"{symbol}{suffix}")
                with open(tmp_path, "wb") as f:
                    f.write(bmce_file.getbuffer())

            cost_model = CostModel(
                brokerage_bps=float(brokerage_bps),
                exchange_bps=float(exchange_bps),
                settlement_bps=float(settlement_bps),
                slippage_bps=float(slippage_bps),
                vat_rate=float(vat_rate),
            )

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

            # Cache bundle by spec signature (simple)
            key = ("bundle", str(base_spec))
            cached = st.session_state.get(key)
            if cached is None:
                with st.spinner("Running backtest..."):
                    cached = BacktestEngine(base_spec).run()
                st.session_state[key] = cached

            render_bundle(cached)

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


# ============================================================
# Optimize Tab (FORM + interval editors)
# ============================================================

with tab_opt:
    st.subheader("Optimize (rank by pnl then efficiency)")

    # Baseline fixed values (used when NOT optimized)
    with st.expander("Fixed baseline (used when NOT optimized)", expanded=False):
        initial_cash0 = st.number_input("Initial cash (baseline)", min_value=1_000.0, value=1_000_000.0, step=10_000.0, key="opt_initial_cash")
        rebalance_policy0 = st.selectbox("Rebalance policy (baseline)", ["on_change", "every_bar"], index=0, key="opt_reb_policy")
        sizing_mode0 = st.selectbox("Sizing mode (baseline)", ["target_weight", "pct_cash_shares"], index=1, key="opt_sizing_mode")
        cooldown0 = st.number_input("cooldown_bars (baseline)", min_value=0, value=0, step=1, key="opt_cd0")
        buy0 = st.slider("buy_pct_cash (baseline)", 0.01, 1.00, 0.25, 0.01, key="opt_buy0")
        sell0 = st.slider("sell_pct_shares (baseline)", 0.01, 1.00, 1.00, 0.01, key="opt_sell0")

        nan_policy0 = "flat"
        if strategy_kind == "ma_cross":
            fast0 = st.number_input("fast_window (baseline)", min_value=2, max_value=500, value=20, step=1, key="opt_fast0")
            slow0 = st.number_input("slow_window (baseline)", min_value=3, max_value=500, value=50, step=1, key="opt_slow0")
            if fast0 >= slow0:
                st.warning("Baseline fast must be < slow. Auto-adjusting.")
                fast0 = min(int(fast0), int(slow0) - 1)
            strategy_params0 = {"fast_window": int(fast0), "slow_window": int(slow0), "allow_short": bool(allow_short), "nan_policy": nan_policy0}
        else:
            w0 = st.number_input("window (baseline)", min_value=2, max_value=500, value=50, step=1, key="opt_w0")
            strategy_params0 = {"window": int(w0), "allow_short": bool(allow_short), "nan_policy": nan_policy0}

        apply_costs0 = st.checkbox("Apply costs", value=False, key="opt_apply_costs")
        if apply_costs0:
            brokerage_bps0 = st.number_input("Brokerage (bps)", value=60.0, step=1.0, key="opt_brok")
            exchange_bps0 = st.number_input("Exchange (bps)", value=10.0, step=1.0, key="opt_exch")
            settlement_bps0 = st.number_input("Settlement (bps)", value=20.0, step=1.0, key="opt_settle")
            vat_rate0 = st.number_input("VAT rate", value=0.10, step=0.01, key="opt_vat")
            slippage_bps0 = st.number_input("Slippage (bps)", value=0.0, step=1.0, key="opt_slip")
        else:
            brokerage_bps0 = exchange_bps0 = settlement_bps0 = slippage_bps0 = 0.0
            vat_rate0 = 0.0

    # Build catalog for THIS strategy_kind
    catalog = default_param_catalog(strategy_kind)
    selectable_keys = list(catalog.keys())

    active_keys = st.multiselect(
        "Select parameters to optimize",
        options=selectable_keys,
        default=["strategy.fast_window", "strategy.slow_window"] if strategy_kind == "ma_cross" else ["strategy.window"],
        key="active_keys",
    )

    # Interval editors (IMPORTANT: ParamDef is frozen -> use replace)
    st.markdown("### Intervals / choices for selected parameters")
    edited_catalog: Dict[str, ParamDef] = dict(catalog)

    for k in active_keys:
        pdef = edited_catalog[k]
        st.markdown(f"**{k}**  (`{pdef.kind}`)")

        if pdef.kind == "int":
            lo, hi, step = pdef.domain
            c1, c2, c3 = st.columns(3)
            lo2 = c1.number_input(f"{k} min", value=int(lo), step=1, key=f"{k}_min")
            hi2 = c2.number_input(f"{k} max", value=int(hi), step=1, key=f"{k}_max")
            step2 = c3.number_input(f"{k} step", value=int(step), step=1, key=f"{k}_step")
            if lo2 > hi2:
                st.error(f"{k}: min must be <= max")
                st.stop()
            edited_catalog[k] = replace(pdef, domain=(int(lo2), int(hi2), int(step2)))

        elif pdef.kind == "float":
            lo, hi, step = pdef.domain
            c1, c2, c3 = st.columns(3)
            lo2 = c1.number_input(f"{k} min", value=float(lo), step=0.01, key=f"{k}_min")
            hi2 = c2.number_input(f"{k} max", value=float(hi), step=0.01, key=f"{k}_max")
            step2 = c3.number_input(f"{k} step", value=float(step), step=0.01, key=f"{k}_step")
            if lo2 > hi2:
                st.error(f"{k}: min must be <= max")
                st.stop()
            edited_catalog[k] = replace(pdef, domain=(float(lo2), float(hi2), float(step2)))

        elif pdef.kind == "choice":
            choices = list(pdef.domain)
            picked = st.multiselect(f"{k} choices", choices, default=choices, key=f"{k}_choices")
            edited_catalog[k] = replace(pdef, domain=list(picked))

        elif pdef.kind == "date_window":
            if source_key != "bmce":
                st.warning("data.window optimization is intended for BMCE uploads.")
                edited_catalog[k] = replace(pdef, domain=[])
            else:
                min_bars = st.number_input("min bars per window", min_value=30, value=252, step=21, key="dw_min_bars")
                step_bars = st.number_input("step bars", min_value=1, value=21, step=1, key="dw_step_bars")
                max_windows = st.number_input("max windows", min_value=10, value=200, step=10, key="dw_max_windows")
                # domain filled at run time after preview_df exists

    st.markdown("### Optimization method")
    method = st.selectbox("Method", ["random", "grid"], index=0, key="opt_method")
    seed = st.number_input("Seed", min_value=0, value=42, step=1, key="opt_seed")
    top_k = st.number_input("Show top K", min_value=5, value=30, step=5, key="opt_topk")

    if method == "random":
        n_trials = st.number_input("Trials", min_value=10, value=200, step=10, key="opt_trials")
    else:
        n_trials = 0

    run_opt = st.button("Run optimization", key="run_opt_btn")

    if run_opt:
        tmp_path = None
        tmp_dir = None
        bench_tmp_path = None
        bench_tmp_dir = None
        try:
            if use_benchmark and bench_source_key == "bmce":
                if bench_bmce_file is None:
                    st.warning("Benchmark enabled but no BMCE benchmark file uploaded.")
                    st.stop()

                suffix_b = Path(bench_bmce_file.name).suffix.lower()
                bench_tmp_dir = tempfile.mkdtemp(prefix="bmce_benchmark_")
                bench_tmp_path = str(Path(bench_tmp_dir) / f"{bench_symbol}{suffix_b}")
                with open(bench_tmp_path, "wb") as f:
                    f.write(bench_bmce_file.getbuffer())
            if source_key == "bmce":
                suffix = Path(bmce_file.name).suffix.lower()
                tmp_dir = tempfile.mkdtemp(prefix="bmce_upload_")
                tmp_path = str(Path(tmp_dir) / f"{symbol}{suffix}")
                with open(tmp_path, "wb") as f:
                    f.write(bmce_file.getbuffer())

            # Fill date windows domain if selected
            if "data.window" in active_keys:
                windows = build_date_windows_from_df(
                    preview_df,
                    min_bars=int(st.session_state.get("dw_min_bars", 252)),
                    step_bars=int(st.session_state.get("dw_step_bars", 21)),
                    max_windows=int(st.session_state.get("dw_max_windows", 200)),
                )
                edited_catalog["data.window"] = replace(edited_catalog["data.window"], domain=windows)

            # Build active_params list for optimizer (this is what run_optimization expects)
            active_params: List[ParamDef] = []
            for k in active_keys:
                p = edited_catalog[k]
                if p.enabled and p.domain is not None and (p.kind != "date_window" or len(p.domain) > 0):
                    active_params.append(p)

            if not active_params:
                st.error("No active parameters to optimize (empty domains).")
                st.stop()

            cost_model = CostModel(
                brokerage_bps=float(brokerage_bps0),
                exchange_bps=float(exchange_bps0),
                settlement_bps=float(settlement_bps0),
                slippage_bps=float(slippage_bps0),
                vat_rate=float(vat_rate0),
            )

            base_spec = make_base_spec(
                source_key=source_key,
                symbol=symbol,
                timezone=timezone,
                interval=interval,
                bmce_tmp_path=tmp_path,
                start=None,
                end=None,
                yf_period=yf_period,
                yf_interval=yf_interval,
                yf_auto_adjust=yf_auto_adjust,
                strategy_kind=strategy_kind,
                strategy_params=strategy_params0,
                allow_short=bool(allow_short),
                initial_cash=float(initial_cash0),
                rebalance_policy=str(rebalance_policy0),
                sizing_mode=str(sizing_mode0),
                buy_pct_cash=float(buy0),
                sell_pct_shares=float(sell0),
                cooldown_bars=int(cooldown0),
                cost_model=cost_model,
            )

            opt_cfg = OptimizeConfig(
                method=str(method),
                seed=int(seed),
                n_trials=int(n_trials) if method == "random" else 0,
                top_k=int(top_k),
                # NOTE: no "objective" field in your OptimizeConfig
            )

            with st.spinner("Running optimization..."):
                best, top_df, best_params, best_spec = run_optimization(
                    base_spec=base_spec,
                    active_params=active_params,
                    cfg=opt_cfg,
                )

            st.success("Optimization finished (ranked by pnl then efficiency).")

            st.subheader("Best result")
            st.json({
                "pnl": best.pnl,
                "efficiency": best.efficiency,
                "traded_notional": best.traded_notional,
                "n_fills": best.n_fills,
                "params": best.params,
                "error": best.error,
            })

            st.subheader("Top candidates")
            # Ensure pnl + efficiency visible even if many params
            show_cols = [c for c in top_df.columns if c in ("pnl","efficiency","traded_notional","n_fills","error")] + \
                        [c for c in top_df.columns if c not in ("pnl","efficiency","traded_notional","n_fills","error")]
            # --- Explain optimization metrics + show top candidates ---
            table_with_info(
                "Top candidates (ranked by PnL then Efficiency)",
                "opt.top",
                top_df[[
                    # Keep only the most relevant columns first, if they exist
                    *[c for c in top_df.columns if c.startswith("strategy.")],
                    *[c for c in top_df.columns if c.startswith("portfolio.")],
                    "pnl", "efficiency", "traded_notional", "n_fills",
                    *([c for c in ["error"] if c in top_df.columns]),
                ]].copy()
            )


            st.divider()
            if st.button("Run best configuration backtest", key="run_best_backtest_btn"):
                with st.spinner("Running best backtest..."):
                    bundle = BacktestEngine(best_spec).run()
                    # ============================================================
                    # Optional: rebuild report with benchmark (no engine changes)
                    # ============================================================
                    if use_benchmark and bench_source_key and bench_symbol:
                        try:
                            bench_md = load_benchmark_market_data_cached(
                                bench_source_key=bench_source_key,
                                bench_symbol=bench_symbol,
                                timezone=timezone,
                                interval=interval,
                                bmce_path=bench_tmp_path if bench_source_key == "bmce" else None,
                                start=start_str,
                                end=end_str,
                                yf_period=bench_yf_period,
                                yf_interval=bench_yf_interval,
                                yf_auto_adjust=bench_yf_auto_adjust,
                            )

                            analyzer = ResultsAnalyzer(periods_per_year=252, rf_annual=0.0)

                            # robust attribute fetch (adjust once you confirm exact bundle field names)
                            portfolio_result = getattr(bundle, "portfolio_result", None) or getattr(bundle, "portfolio", None) or getattr(bundle, "pres", None)
                            if portfolio_result is None:
                                raise AttributeError("Bundle has no portfolio result attribute (expected portfolio_result/portfolio/pres).")

                            report = analyzer.analyze(
                                portfolio_result=portfolio_result,
                                market_data=bundle.md,
                                symbols=bundle.md.symbols(),
                                features_data=getattr(bundle, "feats", None),
                                plot_indicators=getattr(bundle.report.plots.get("price_panel", {}), "indicator_cols", None) if hasattr(bundle, "report") else None,
                                benchmark_market_data=bench_md,
                                benchmark_symbol=bench_symbol,
                            )

                            # overwrite bundle.report so render_bundle() stays unchanged
                            bundle.report = report

                        except Exception as e:
                            st.warning(f"Benchmark could not be applied: {e}")

                render_bundle(bundle)

        finally:
            if bench_tmp_path and os.path.exists(bench_tmp_path):
                try:
                    os.remove(bench_tmp_path)
                except OSError:
                    pass
            if bench_tmp_dir and os.path.isdir(bench_tmp_dir):
                try:
                    os.rmdir(bench_tmp_dir)
                except OSError:
                    pass
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
