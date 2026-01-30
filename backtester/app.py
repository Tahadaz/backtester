# app.py — Streamlit entrypoint (Backtest + Optimize)
from __future__ import annotations

import io
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plotly.graph_objects as go

# ---- project imports ----
from engine import BacktestEngine, EngineSpec, DataConfig, IndicatorsConfig, StrategyConfig
from portfolio import PortfolioConfig, CostModel

# IMPORTANT: this expects optimize.py to expose these symbols (as in the robust optimize.py we built)
from optimize import (
    OptimizeConfig,
    TrialResult,
    ParamDef,
    default_param_catalog,   # <-- (NOT default_param_catalog_for_strategy)
    run_optimization,        # <-- returns: (best, top_df, best_params, best_spec)
)

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="Backtester", layout="wide")
st.title("Backtester (TA) — Backtest / Optimize")

CACHE_UPLOAD_DIR = Path(".cache/uploads")
CACHE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utility: stable upload path (no tempdir churn)
# =============================================================================
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def persist_upload_to_cache(symbol: str, file_name: str, file_bytes: bytes) -> str:
    """
    Persist upload bytes to a deterministic cache path.
    This avoids rewriting temp files and enables stable caching across reruns.
    """
    ext = Path(file_name).suffix.lower()
    h = _sha256_bytes(file_bytes)[:24]
    safe_symbol = "".join([c for c in symbol if c.isalnum() or c in ("_", "-")])[:20] or "SYM"
    path = CACHE_UPLOAD_DIR / f"{safe_symbol}__{h}{ext}"
    if not path.exists():
        path.write_bytes(file_bytes)
    return str(path)


@st.cache_data(show_spinner=False)
def preview_bmce_bytes(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    ext = Path(file_name).suffix.lower()
    if ext == ".csv":
        # try robust parsing (sep autodetect)
        return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", sep=None, engine="python")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    raise ValueError(f"Unsupported upload extension: {ext}")


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


# =============================================================================
# Plotting helpers (fast enough; close figs to avoid leaks)
# =============================================================================
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


def plot_cum_vs_bench(cum: pd.Series, bench_cum: Optional[pd.Series]) -> plt.Figure:
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

    # --- NORMAL GRAPH: Close line ---
    if "Close" not in df.columns:
        raise KeyError("bars must contain a 'Close' column for normal price line plot.")
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"].astype(float),
            mode="lines",
            name="Close",
            line=dict(width=2),
        )
    )

    # --- Indicators (lines) ---
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

    # --- Trades markers (BUY/SELL) ---
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
        t = t.dropna(subset=["timestamp"]).sort_values("timestamp")

        if "side" not in t.columns:
            t["side"] = np.where(t["qty"].astype(float) > 0, "BUY", "SELL")

        # plot marker y = trade price if present else Close on that date
        y = pd.to_numeric(t.get("price", np.nan), errors="coerce")
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
                    text=["BUY"] * len(buys),
                    textposition="top center",
                    name="BUY",
                    marker=dict(size=9, symbol="triangle-up"),
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["y_plot"],
                    mode="markers+text",
                    text=["SELL"] * len(sells),
                    textposition="bottom center",
                    name="SELL",
                    marker=dict(size=9, symbol="triangle-down"),
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


# =============================================================================
# Styling helpers (tables)
# =============================================================================
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


def style_trade_summary(df: pd.DataFrame):
    # df is index -> metric, col "Value"
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


# =============================================================================
# Spec builder
# =============================================================================
def make_base_spec(
    source_key: str,
    symbol: str,
    timezone: str,
    interval: str,
    bmce_path: Optional[str],
    start: Optional[str],
    end: Optional[str],
    yf_period: Optional[str],
    yf_interval: Optional[str],
    yf_auto_adjust: Optional[bool],
    strategy_kind: str,
    strategy_params: Dict[str, Any],
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
        if not bmce_path:
            raise ValueError("BMCE source selected but bmce_path is None.")
        data_cfg = DataConfig(
            source="bmce",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            start=start,
            end=end,
            bmce_paths=bmce_path,
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


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# =============================================================================
# Render
# =============================================================================
def render_bundle(bundle) -> None:
    rep = bundle.report
    plots = rep.plots
    tables = rep.tables
    series = rep.series
    metrics = rep.metrics

    # headline metrics
    with st.expander("Headline metrics", expanded=True):
        st.json({k: _safe_float(v) for k, v in metrics.items()})

    # Price + indicators + trades
    st.subheader("Price + Indicators + Trades")
    pp = plots["price_panel"]
    bars = pp.get("bars", None)
    if bars is None:
        sym0 = bundle.md.symbols()[0]
        bars = bundle.md.bars[sym0]

    fig = plot_price_indicators_trades_plotly(
        bars=bars,
        indicators=pp.get("indicators"),
        trades=pp.get("trades"),
        indicator_cols=pp.get("indicator_cols"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # cum vs bench
    st.subheader("Cumulative Returns vs Benchmark")
    cvb = plots["cum_vs_bench"]
    fig2 = plot_cum_vs_bench(cvb["strategy"], cvb.get("benchmark"))
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)

    # drawdown
    st.subheader("Drawdown")
    fig3 = plot_drawdown_red(plots["drawdown"])
    st.pyplot(fig3, clear_figure=True)
    plt.close(fig3)

    # pnl / cum pnl
    st.subheader("PnL (per bar)")
    fig4 = plot_pnl_series(series["pnl"])
    st.pyplot(fig4, clear_figure=True)
    plt.close(fig4)

    st.subheader("Cumulative PnL")
    fig5 = plot_cum_pnl_series(series["cum_pnl"])
    st.pyplot(fig5, clear_figure=True)
    plt.close(fig5)

    # monthly / yearly
    st.subheader("Monthly Returns Heatmap")
    fig6 = plot_monthly_heatmap_with_values(plots["monthly_heatmap"])
    st.pyplot(fig6, clear_figure=True)
    plt.close(fig6)

    st.subheader("Yearly Returns")
    fig7 = plot_yearly_returns_bar(plots["yearly_bar"])
    st.pyplot(fig7, clear_figure=True)
    plt.close(fig7)

    # tables (your requested: trades + pnl + efficiency)
    st.subheader("Trades (fills)")
    st.dataframe(tables.get("trades", pd.DataFrame()), use_container_width=True)

    st.subheader("Trades Ledger (closed trades, FIFO) — PnL & Efficiency")
    st.dataframe(tables.get("trade_ledger", pd.DataFrame()), use_container_width=True)

    st.subheader("Trade Performance (summary)")
    st.dataframe(tables.get("trade_performance", pd.DataFrame()), use_container_width=True)

    # 3-panels like screenshot
    if "curve_vs_benchmark" in tables and "trade_summary" in tables and "time_summary" in tables:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Curve vs. Benchmark")
            st.dataframe(style_curve_vs_bench(tables["curve_vs_benchmark"]), use_container_width=True)
        with c2:
            st.markdown("### Trade")
            st.dataframe(style_trade_summary(tables["trade_summary"]), use_container_width=True)
        with c3:
            st.markdown("### Time")
            st.dataframe(style_time_table(tables["time_summary"]), use_container_width=True)

    with st.expander("Debug", expanded=False):
        sym = bundle.md.symbols()[0]
        st.write("Features columns:", list(bundle.feats.features[sym].columns))
        st.write("Signals head:")
        st.dataframe(bundle.signals.signals.head(10), use_container_width=True)


# =============================================================================
# Session state init
# =============================================================================
if "last_bundle" not in st.session_state:
    st.session_state["last_bundle"] = None
if "last_opt" not in st.session_state:
    st.session_state["last_opt"] = None  # dict with best/top_df/best_spec


# =============================================================================
# Tabs
# =============================================================================
tab_bt, tab_opt = st.tabs(["Backtest", "Optimize"])


# =============================================================================
# Shared sidebar (source + strategy)
# =============================================================================
with st.sidebar:
    st.header("Common")

    with st.form("common_form"):
        st.subheader("Data source")
        source = st.selectbox("Source", ["bmce (upload)", "yfinance"], index=0)
        source_key = "bmce" if source.startswith("bmce") else "yfinance"

        symbol = st.text_input("Symbol", value="IAM" if source_key == "bmce" else "AAPL")
        timezone = st.text_input("Timezone", value="GMT")
        interval = st.selectbox("Interval", ["1d"], index=0)

        bmce_file = None
        yf_period = yf_interval = None
        yf_auto_adjust = None
        preview_df = None
        bmce_path = None
        bmce_hash = None

        if source_key == "bmce":
            bmce_file = st.file_uploader("Upload BMCE CSV/XLSX", type=["csv", "xlsx", "xls"])
        else:
            st.caption("yfinance needs internet + yfinance in requirements.txt")
            yf_period = st.text_input("yfinance period", value="5y")
            yf_interval = st.selectbox("yfinance interval", ["1d"], index=0)
            yf_auto_adjust = st.checkbox("auto_adjust", value=False)

        st.subheader("Strategy")
        strategy_kind = st.selectbox("Strategy kind", ["ma_cross", "sma_price"], index=0)
        allow_short = st.checkbox("Allow short", value=False)

        common_submit = st.form_submit_button("Apply common settings")

    # BMCE preview outside form (doesn't force rerun loops on every widget change)
    if source_key == "bmce":
        st.subheader("BMCE preview")
        if bmce_file is None:
            st.info("Upload a BMCE CSV/XLSX to continue.")
        else:
            bmce_bytes = bmce_file.getvalue()
            bmce_hash = _sha256_bytes(bmce_bytes)
            bmce_path = persist_upload_to_cache(symbol, bmce_file.name, bmce_bytes)
            try:
                preview_df = preview_bmce_bytes(bmce_file.name, bmce_bytes)
                with st.expander("Preview upload (first 30 rows)", expanded=False):
                    st.dataframe(preview_df.head(30), use_container_width=True)
                st.caption(f"Cached file path: {bmce_path}")
            except Exception as e:
                st.error(f"Could not preview file: {e}")


# =============================================================================
# BACKTEST TAB
# =============================================================================
with tab_bt:
    st.header("Backtest")

    # Backtest controls in a form (prevents rerun spam)
    with st.sidebar:
        with st.form("backtest_form"):
            st.subheader("Backtest period")
            use_date_range = st.checkbox("Use date range", value=False, key="bt_use_date_range")
            start_str = end_str = None
            if use_date_range:
                start_date = st.date_input("Start date", value=None, key="bt_start")
                end_date = st.date_input("End date", value=None, key="bt_end")
                if start_date and end_date and start_date > end_date:
                    st.error("Start date must be <= End date.")
                start_str = start_date.isoformat() if start_date else None
                end_str = end_date.isoformat() if end_date else None

            st.subheader("Strategy parameters")
            nan_policy = "flat"
            strategy_params: Dict[str, Any] = {}

            if strategy_kind == "ma_cross":
                fast = st.number_input("Fast SMA window", min_value=2, max_value=500, value=20, step=1, key="bt_fast")
                slow = st.number_input("Slow SMA window", min_value=3, max_value=500, value=50, step=1, key="bt_slow")
                if fast >= slow:
                    st.warning("Fast must be < Slow. Fast will be clipped to slow-1.")
                    fast = min(int(fast), int(slow) - 1)
                strategy_params = {"fast_window": int(fast), "slow_window": int(slow), "allow_short": bool(allow_short), "nan_policy": nan_policy}
            else:
                window = st.number_input("SMA window", min_value=2, max_value=500, value=50, step=1, key="bt_window")
                strategy_params = {"window": int(window), "allow_short": bool(allow_short), "nan_policy": nan_policy}

            st.subheader("Portfolio")
            initial_cash = st.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0, key="bt_cash")
            rebalance_policy = st.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0, key="bt_reb")
            sizing_mode = st.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=0, key="bt_sizing")
            cooldown_bars = st.number_input("Min bars between trades (cooldown)", min_value=0, value=0, step=1, key="bt_cd")

            st.subheader("Sizing")
            buy_pct_cash = st.slider("Buy % of cash per entry", 0.01, 1.00, 0.25, 0.01, key="bt_buy")
            sell_pct_shares = st.slider("Sell % of shares per exit", 0.01, 1.00, 1.00, 0.01, key="bt_sell")

            st.subheader("Costs")
            apply_costs = st.checkbox("Apply costs", value=False, key="bt_costs")
            if apply_costs:
                brokerage_bps = st.number_input("Brokerage (bps)", value=60.0, step=1.0, key="bt_brok")
                exchange_bps = st.number_input("Exchange (bps)", value=10.0, step=1.0, key="bt_exch")
                settlement_bps = st.number_input("Settlement (bps)", value=20.0, step=1.0, key="bt_sett")
                vat_rate = st.number_input("VAT rate", value=0.10, step=0.01, key="bt_vat")
                slippage_bps = st.number_input("Slippage (bps)", value=0.0, step=1.0, key="bt_slip")
            else:
                brokerage_bps = exchange_bps = settlement_bps = slippage_bps = 0.0
                vat_rate = 0.0

            run_backtest = st.form_submit_button("Run backtest")

    if run_backtest:
        if source_key == "bmce" and not bmce_path:
            st.error("Upload BMCE file first.")
            st.stop()

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
            bmce_path=bmce_path,
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

        with st.spinner("Running backtest..."):
            bundle = BacktestEngine(base_spec).run()

        st.session_state["last_bundle"] = bundle
        render_bundle(bundle)

    else:
        if st.session_state["last_bundle"] is not None:
            st.info("Showing last backtest result (session).")
            render_bundle(st.session_state["last_bundle"])
        else:
            st.caption("Run a backtest to display results.")


# =============================================================================
# OPTIMIZE TAB
# =============================================================================
with tab_opt:
    st.header("Optimize (Objective: PnL then Efficiency)")

    with st.sidebar:
        with st.form("opt_form"):
            st.subheader("Baseline (used for non-optimized params)")
            initial_cash0 = st.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0, key="op_cash")
            rebalance_policy0 = st.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0, key="op_reb")
            sizing_mode0 = st.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=0, key="op_sizing")

            buy_pct_cash0 = st.slider("Baseline buy % cash", 0.01, 1.00, 0.25, 0.01, key="op_buy")
            sell_pct_shares0 = st.slider("Baseline sell % shares", 0.01, 1.00, 1.00, 0.01, key="op_sell")
            cooldown_bars0 = st.number_input("Baseline cooldown_bars", min_value=0, value=0, step=1, key="op_cd")

            st.subheader("Baseline strategy params")
            nan_policy0 = "flat"
            strategy_params0: Dict[str, Any] = {}
            if strategy_kind == "ma_cross":
                fast0 = st.number_input("Baseline fast_window", min_value=2, max_value=500, value=20, step=1, key="op_fast")
                slow0 = st.number_input("Baseline slow_window", min_value=3, max_value=500, value=50, step=1, key="op_slow")
                if fast0 >= slow0:
                    st.warning("Baseline fast must be < slow. Fast will be clipped to slow-1.")
                    fast0 = min(int(fast0), int(slow0) - 1)
                strategy_params0 = {"fast_window": int(fast0), "slow_window": int(slow0), "allow_short": bool(allow_short), "nan_policy": nan_policy0}
            else:
                w0 = st.number_input("Baseline window", min_value=2, max_value=500, value=50, step=1, key="op_window")
                strategy_params0 = {"window": int(w0), "allow_short": bool(allow_short), "nan_policy": nan_policy0}

            st.subheader("Costs (baseline)")
            apply_costs0 = st.checkbox("Apply costs", value=False, key="op_costs")
            if apply_costs0:
                brokerage_bps0 = st.number_input("Brokerage (bps)", value=60.0, step=1.0, key="op_brok")
                exchange_bps0 = st.number_input("Exchange (bps)", value=10.0, step=1.0, key="op_exch")
                settlement_bps0 = st.number_input("Settlement (bps)", value=20.0, step=1.0, key="op_sett")
                vat_rate0 = st.number_input("VAT rate", value=0.10, step=0.01, key="op_vat")
                slippage_bps0 = st.number_input("Slippage (bps)", value=0.0, step=1.0, key="op_slip")
            else:
                brokerage_bps0 = exchange_bps0 = settlement_bps0 = slippage_bps0 = 0.0
                vat_rate0 = 0.0

            st.subheader("Select parameters to optimize")
            catalog = default_param_catalog(strategy_kind)
            selectable_keys = list(catalog.keys())

            default_keys = ["strategy.fast_window", "strategy.slow_window"] if strategy_kind == "ma_cross" else ["strategy.window"]
            active_keys = st.multiselect("Active params", options=selectable_keys, default=default_keys, key="op_active")

            st.subheader("Intervals / domains (active)")
            # mutate local copies (don’t permanently mutate cached catalog across reruns)
            catalog2: Dict[str, ParamDef] = {k: ParamDef(**catalog[k].__dict__) for k in catalog}

            for k in active_keys:
                pdef = catalog2[k]
                st.markdown(f"**{k}** ({pdef.kind})")

                if pdef.kind == "int":
                    lo, hi, step = pdef.domain
                    c1, c2, c3 = st.columns(3)
                    lo2 = c1.number_input(f"{k} min", value=int(lo), step=1, key=f"dom_{k}_min")
                    hi2 = c2.number_input(f"{k} max", value=int(hi), step=1, key=f"dom_{k}_max")
                    step2 = c3.number_input(f"{k} step", value=int(step), step=1, key=f"dom_{k}_step")
                    if lo2 > hi2:
                        st.error(f"{k}: min must be <= max")
                    pdef.domain = (int(lo2), int(hi2), int(step2))

                elif pdef.kind == "float":
                    lo, hi, step = pdef.domain
                    c1, c2, c3 = st.columns(3)
                    lo2 = c1.number_input(f"{k} min", value=float(lo), step=0.01, key=f"dom_{k}_min")
                    hi2 = c2.number_input(f"{k} max", value=float(hi), step=0.01, key=f"dom_{k}_max")
                    step2 = c3.number_input(f"{k} step", value=float(step), step=0.01, key=f"dom_{k}_step")
                    if lo2 > hi2:
                        st.error(f"{k}: min must be <= max")
                    pdef.domain = (float(lo2), float(hi2), float(step2))

                elif pdef.kind == "choice":
                    choices = list(pdef.domain)
                    picked = st.multiselect(f"{k} choices", choices, default=choices, key=f"dom_{k}_choices")
                    pdef.domain = list(picked)

                elif pdef.kind == "date_window":
                    if source_key != "bmce" or preview_df is None:
                        st.warning("data.window is intended for BMCE uploads (needs preview).")
                        pdef.domain = []
                    else:
                        min_bars = st.number_input("min bars per window", min_value=30, value=252, step=21, key="dw_min_bars")
                        step_bars = st.number_input("step bars", min_value=1, value=21, step=1, key="dw_step_bars")
                        max_windows = st.number_input("max windows", min_value=10, value=200, step=10, key="dw_max_windows")

                        windows = build_date_windows_from_df(
                            preview_df,
                            min_bars=int(min_bars),
                            step_bars=int(step_bars),
                            max_windows=int(max_windows),
                        )
                        st.caption(f"Generated windows: {len(windows)}")
                        pdef.domain = windows

                catalog2[k] = pdef

            st.subheader("Optimization method")
            method = st.selectbox("Method", ["random", "grid"], index=0, key="op_method")
            seed = st.number_input("Seed", min_value=0, value=42, step=1, key="op_seed")
            top_k = st.number_input("Show top K", min_value=5, value=30, step=5, key="op_topk")
            n_trials = 0
            if method == "random":
                n_trials = st.number_input("Trials", min_value=10, value=200, step=10, key="op_trials")

            run_opt = st.form_submit_button("Run optimization")

    if run_opt:
        if source_key == "bmce" and not bmce_path:
            st.error("Upload BMCE file first.")
            st.stop()

        cost_model0 = CostModel(
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
            bmce_path=bmce_path,
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
            buy_pct_cash=float(buy_pct_cash0),
            sell_pct_shares=float(sell_pct_shares0),
            cooldown_bars=int(cooldown_bars0),
            cost_model=cost_model0,
        )

        # Build active ParamDef list for optimizer
        active_params: List[ParamDef] = []
        for k in active_keys:
            pdef = catalog2[k]
            # remove invalid date_window
            if pdef.kind == "date_window" and (not pdef.domain):
                continue
            active_params.append(pdef)

        opt_cfg = OptimizeConfig(
            method=method,
            seed=int(seed),
            n_trials=int(n_trials) if method == "random" else 0,
            top_k=int(top_k),
            objective="pnl_then_eff",
        )

        with st.spinner("Running optimization..."):
            best, top_df, best_params, best_spec = run_optimization(
                base_spec=base_spec,
                active_params=active_params,
                cfg=opt_cfg,
            )

        # Persist in session
        st.session_state["last_opt"] = {
            "best": best,
            "top_df": top_df,
            "best_params": best_params,
            "best_spec": best_spec,
        }

        st.subheader("Best trial (PnL then Efficiency)")
        st.json(
            {
                "pnl": float(best.pnl),
                "efficiency": float(best.efficiency),
                "traded_notional": float(best.traded_notional),
                "params": dict(best.params),
                "error": best.error,
            }
        )

        st.subheader("Top trials (sorted)")
        st.dataframe(top_df, use_container_width=True)

        st.divider()
        if st.button("Run best configuration backtest", key="run_best_bt"):
            with st.spinner("Running best backtest..."):
                bundle = BacktestEngine(best_spec).run()
            st.session_state["last_bundle"] = bundle
            render_bundle(bundle)

    else:
        # show last optimization if exists
        last = st.session_state.get("last_opt")
        if last is not None:
            best: TrialResult = last["best"]
            st.info("Showing last optimization result (session).")
            st.json(
                {
                    "pnl": float(best.pnl),
                    "efficiency": float(best.efficiency),
                    "traded_notional": float(best.traded_notional),
                    "params": dict(best.params),
                    "error": best.error,
                }
            )
            st.dataframe(last["top_df"], use_container_width=True)
        else:
            st.caption("Run an optimization to see results.")



