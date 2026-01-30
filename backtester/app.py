from __future__ import annotations

import os, tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

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
# Rendering (reuse your existing block here)
# --------------------------
def render_bundle(bundle):
    # paste your existing rendering code here, unchanged
    st.write(bundle.report.metrics)  # placeholder

# --------------------------
# Helpers: build EngineSpec from "state"
# --------------------------
def build_engine_spec(
    *,
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
    strategy_params: Dict[str, Any],
    allow_short: bool,
    initial_cash: float,
    rebalance_policy: str,
    sizing_mode: str,
    buy_pct_cash: float,
    sell_pct_shares: float,
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

    port_cfg = PortfolioConfig(
        allow_short=bool(allow_short),
        initial_cash=float(initial_cash),
        rebalance_policy=str(rebalance_policy),
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

# --------------------------
# Sidebar UI
# --------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode", ["Backtest", "Optimize"], index=0)

# --- Data source (common) ---
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

# --- Strategy (common) ---
st.sidebar.header("Strategy")
strategy_kind = st.sidebar.selectbox(
    "Choose strategy",
    options=["ma_cross", "sma_price"],
    index=0,
)
allow_short = st.sidebar.checkbox("Allow short", value=False)
nan_policy = "flat"

# Baseline strategy params ALWAYS exist (optimizer can override)
if strategy_kind == "ma_cross":
    fast = st.sidebar.number_input("Fast SMA window", min_value=2, max_value=500, value=20, step=1)
    slow = st.sidebar.number_input("Slow SMA window", min_value=3, max_value=500, value=50, step=1)
    if fast >= slow:
        st.sidebar.warning("Fast window must be < Slow window. Adjusting.")
        fast = min(int(fast), int(slow) - 1)

    strategy_params = {
        "fast_window": int(fast),
        "slow_window": int(slow),
        "allow_short": bool(allow_short),
        "nan_policy": nan_policy,
    }
else:
    window = st.sidebar.number_input("SMA window", min_value=2, max_value=500, value=50, step=1)
    strategy_params = {
        "window": int(window),
        "allow_short": bool(allow_short),
        "nan_policy": nan_policy,
    }

# --- Portfolio (common) ---
st.sidebar.header("Portfolio")
initial_cash = st.sidebar.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
rebalance_policy = st.sidebar.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)

# --- Sizing (baseline exists always; can still optimize it) ---
st.sidebar.header("Sizing (baseline)")
sizing_mode = st.sidebar.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=1)
buy_pct_cash = st.sidebar.slider("Buy % of cash per entry", 0.01, 1.00, 0.25, 0.01)
sell_pct_shares = st.sidebar.slider("Sell % of shares per exit", 0.01, 1.00, 1.00, 0.01)

# --- Costs (common) ---
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

# --- Backtest period baseline (exists always) ---
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

# --------------------------
# Optimize-specific UI
# --------------------------
opt_cfg = None
active_keys: List[str] = []
optimize_dates = False
if mode == "Optimize":
    st.sidebar.header("Optimization")
    search_mode = st.sidebar.selectbox("Search mode", ["random", "grid"], index=0)
    n_trials = st.sidebar.number_input("Trials (random)", min_value=10, value=200, step=10)
    top_k = st.sidebar.number_input("Show top K", min_value=5, value=30, step=5)

    st.sidebar.subheader("What to optimize?")
    opt_dates = st.sidebar.checkbox("Optimize backtest period", value=False)
    opt_sizing = st.sidebar.checkbox("Optimize sizing (buy/sell %)", value=True)
    opt_strategy = st.sidebar.checkbox("Optimize strategy windows", value=True)

    # Build catalog once
    cat = default_param_catalog_for_your_app()
    cat.pop("strategy.kind", None)

    # Filter selectable keys based on strategy + user choices
    candidates: List[str] = []

    if opt_strategy:
        if strategy_kind == "ma_cross":
            candidates += ["strategy.fast_window", "strategy.slow_window"]
        else:
            candidates += ["strategy.window"]

    if opt_sizing:
        candidates += ["portfolio.buy_pct_cash", "portfolio.sell_pct_shares"]

    if opt_dates:
        # We'll add "data.window" later after we have BMCE uploaded bars
        candidates += ["data.window"]

    # Only show these candidates in multiselect
    # (still allows "single param" or "multiple params" selection)
    active_keys = st.sidebar.multiselect(
        "Active parameters (choose one or many)",
        options=candidates,
        default=candidates,
    )

    optimize_dates = opt_dates

    opt_cfg = OptimizeConfig(
        mode=search_mode,
        n_trials=int(n_trials),
        top_k=int(top_k),
        seed=42,
        objective="pnl_then_efficiency",
        verbose=False,
    )

# --------------------------
# Run button
# --------------------------
st.sidebar.header("Run")
run_btn = st.sidebar.button("Run Optimization" if mode == "Optimize" else "Run Backtest")

# --------------------------
# Main page: require upload if BMCE
# --------------------------
if source_key == "bmce":
    st.subheader("BMCE upload")
    if bmce_file is None:
        st.info("Upload a BMCE CSV/XLSX then click Run.")
        st.stop()
else:
    st.subheader("yfinance mode")
    st.caption("This requires `yfinance` in requirements.txt. BMCE is recommended for desk data.")

if not run_btn:
    st.stop()

# --------------------------
# Prepare BMCE temp file if needed
# --------------------------
tmp_path = None
tmp_dir = None
try:
    if source_key == "bmce":
        suffix = Path(bmce_file.name).suffix.lower()
        tmp_dir = tempfile.mkdtemp(prefix="bmce_upload_")
        tmp_path = str(Path(tmp_dir) / f"{symbol}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(bmce_file.getbuffer())

    # Build baseline spec (used by backtest or as base for optimization)
    base_spec = build_engine_spec(
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
        allow_short=allow_short,
        initial_cash=initial_cash,
        rebalance_policy=rebalance_policy,
        sizing_mode=sizing_mode,
        buy_pct_cash=buy_pct_cash,
        sell_pct_shares=sell_pct_shares,
        brokerage_bps=brokerage_bps,
        exchange_bps=exchange_bps,
        settlement_bps=settlement_bps,
        vat_rate=vat_rate,
        slippage_bps=slippage_bps,
    )

    if mode == "Backtest":
        with st.spinner("Running backtest..."):
            bundle = BacktestEngine(base_spec).run()
        render_bundle(bundle)

    else:
        # ---- Optimize mode ----
        cat = default_param_catalog_for_your_app()
        cat.pop("strategy.kind", None)

        # Add date window param if requested (BMCE only)
        if optimize_dates and source_key == "bmce":
            windows = build_date_window_choices_from_uploaded_bmce(
                base_data_cfg=base_spec.data,
                symbol=symbol,
                min_bars=252,
                step_bars=21,
                max_windows=200,
            )
            add_date_window_param(cat, windows)

        with st.spinner("Running optimization..."):
            best, top_df, best_spec = run_optimization(
                base_spec=base_spec,
                catalog=cat,
                active_keys=active_keys,
                cfg=opt_cfg,
            )

        st.subheader("Optimization results")
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
            render_bundle(bundle)

finally:
    if tmp_path and os.path.exists(tmp_path):
        try: os.remove(tmp_path)
        except OSError: pass
    if tmp_dir and os.path.isdir(tmp_dir):
        try: os.rmdir(tmp_dir)
        except OSError: pass
