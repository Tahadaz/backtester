import io
import os
import tempfile
import hashlib
from typing import Any, List, Optional, Sequence, Tuple, Dict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---- Your project imports (adjust paths if you have a package folder) ----
from data import MarketData
from data import BMCEDataSource
import indicators as ind_mod
from indicators import IndicatorEngine, FeatureSpec
from strategy import MovingAverageCrossStrategy, MovingAverageCrossParams
from portfolio import PortfolioEngine, PortfolioConfig, CostModel
from results import ResultsAnalyzer


st.set_page_config(page_title="TA Backtester", layout="wide")
st.title("TA Backtester (BMCE/MASI)")
st.caption("data.py → indicators.py → strategy.py → portfolio.py → results.py")


# -----------------------------
# UI Controls
# -----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload BMCE CSV/XLSX", type=["csv", "xlsx"])
symbol = st.sidebar.text_input("Symbol", value="IAM")
timezone = st.sidebar.text_input("Timezone", value="UTC")

st.sidebar.header("Strategy: MA Cross")
fast = st.sidebar.number_input("Fast SMA", min_value=2, max_value=300, value=20, step=1)
slow = st.sidebar.number_input("Slow SMA", min_value=3, max_value=500, value=50, step=1)
allow_short = st.sidebar.checkbox("Allow short", value=True)

st.sidebar.header("Portfolio")
initial_cash = st.sidebar.number_input("Initial cash", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
rebalance_policy = st.sidebar.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)
max_gross = st.sidebar.number_input("Max gross exposure", min_value=0.1, value=1.0, step=0.1)
cash_buffer = st.sidebar.number_input("Cash buffer", min_value=0.0, max_value=0.5, value=0.0, step=0.01)

st.sidebar.header("Costs")
apply_costs = st.sidebar.checkbox("Apply costs", value=False)
if apply_costs:
    brokerage_bps = st.sidebar.number_input("Brokerage bps", value=60.0, step=1.0)
    exchange_bps = st.sidebar.number_input("Exchange bps", value=10.0, step=1.0)
    settlement_bps = st.sidebar.number_input("Settlement bps", value=20.0, step=1.0)
    vat_rate = st.sidebar.number_input("VAT rate", value=0.10, step=0.01)
    slippage_bps = st.sidebar.number_input("Slippage bps", value=0.0, step=1.0)
else:
    brokerage_bps = exchange_bps = settlement_bps = slippage_bps = 0.0
    vat_rate = 0.0

run_btn = st.sidebar.button("Run backtest")


# -----------------------------
# Helpers
# -----------------------------
def _bytes_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def plot_series(s: pd.Series, title: str, ylabel: str):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(s.index, s.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    return fig


def parse_upload_to_df(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


def normalize_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback if BMCEDataSource has no suitable method exposed.
    Maps common BMCE columns to canonical OHLCV.
    """
    colmap = {
        "Date": "Date",
        "Ouvt": "Open",
        "+Haut": "High",
        "+Bas": "Low",
        "Clôture": "Close",
        "Cloture": "Close",
        "Volume": "Volume",
    }
    out = df.copy()
    out = out.rename(columns={k: v for k, v in colmap.items() if k in out.columns})

    if "Date" not in out.columns:
        for cand in ["date", "DATE", "Datetime", "datetime", "Time", "time"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "Date"})
                break

    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Cannot normalize: missing {missing}. Columns={list(out.columns)}")

    out["Date"] = pd.to_datetime(out["Date"])
    out = out.set_index("Date").sort_index()

    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def build_marketdata_from_upload(symbol: str, tz: str, file_name: str, file_bytes: bytes) -> MarketData:
    """
    Preferred: call BMCEDataSource if it exposes a reader.
    Otherwise fallback to normalize_to_ohlcv and build MarketData directly.
    """
    suffix = ".csv" if file_name.lower().endswith(".csv") else ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        ds = BMCEDataSource(timezone=tz)

        # Try common method names (no guessing in user-land code)
        for meth in ["read_dataframe", "read_df", "load_dataframe", "read_file", "read_one", "load_one"]:
            if hasattr(ds, meth):
                bars_df = getattr(ds, meth)(tmp_path)
                return MarketData(bars={symbol: bars_df}, source="BMCEDataSource", timezone=tz)

        # If adapter has a method that returns MarketData
        for meth in ["load_marketdata", "load_market_data", "get_marketdata", "get_market_data"]:
            if hasattr(ds, meth):
                md = getattr(ds, meth)(tmp_path, symbols=[symbol])
                return md

        # Fallback: parse via pandas and normalize
        df_raw = parse_upload_to_df(file_name, file_bytes)
        bars_df = normalize_to_ohlcv(df_raw)
        return MarketData(bars={symbol: bars_df}, source="upload_fallback", timezone=tz)

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def build_sma_specs(fast_window: int, slow_window: int) -> List[FeatureSpec]:
    """
    Builds FeatureSpec entries that will produce columns:
      sma_{fast_window}, sma_{slow_window}
    which is exactly what MovingAverageCrossStrategy expects.

    Assumes your indicator registry key is "sma" and it reads params["window"].
    """
    fast_window = int(fast_window)
    slow_window = int(slow_window)

    return [
        FeatureSpec(
            indicator="sma",
            params={"window": fast_window},
            inputs=("Close",),
            name=f"sma_{fast_window}",
            warmup=fast_window,        # optional; OK to leave None if you prefer
            output_mode="series",
        ),
        FeatureSpec(
            indicator="sma",
            params={"window": slow_window},
            inputs=("Close",),
            name=f"sma_{slow_window}",
            warmup=slow_window,
            output_mode="series",
        ),
    ]



@st.cache_data(show_spinner=False)
def run_pipeline_cached(
    file_hash: str,
    symbol: str,
    tz: str,
    file_name: str,
    file_bytes: bytes,
    fast: int,
    slow: int,
    allow_short: bool,
    initial_cash: float,
    rebalance_policy: str,
    max_gross: float,
    cash_buffer: float,
    brokerage_bps: float,
    exchange_bps: float,
    settlement_bps: float,
    vat_rate: float,
    slippage_bps: float,
) -> Tuple[Any, Any, Any]:
    """
    Caches by file_hash + params so re-runs are instant.
    Returns: (portfolio_result, report, md)
    """
    md = build_marketdata_from_upload(symbol, tz, file_name, file_bytes)

    # Indicators
    specs = build_sma_specs(fast, slow)
    eng = IndicatorEngine(cache_dir=".cache/features", enable_disk_cache=True, enable_memory_cache=True)
    feats = eng.compute(md, specs=specs, symbols=[symbol])

    # Strategy
    params = MovingAverageCrossParams(
        fast_window=int(fast),
        slow_window=int(slow),
        allow_short=bool(allow_short),
        nan_policy="flat",
    )
    strat = MovingAverageCrossStrategy(params)
    sf = strat.generate_signals(md, feats, symbols=[symbol])

    # Portfolio
    cost_model = CostModel(
        brokerage_bps=float(brokerage_bps),
        exchange_bps=float(exchange_bps),
        settlement_bps=float(settlement_bps),
        slippage_bps=float(slippage_bps),
        vat_rate=float(vat_rate),
    )
    pcfg = PortfolioConfig(
        allow_short=bool(allow_short),
        initial_cash=float(initial_cash),
        rebalance_policy=rebalance_policy,
        max_gross=float(max_gross),
        cash_buffer=float(cash_buffer),
        cost_model=cost_model,
    )
    port = PortfolioEngine(pcfg)
    res = port.run(md, sf, symbols=[symbol])

    # Results
    an = ResultsAnalyzer(periods_per_year=252, rf_annual=0.0)
    rep = an.analyze(res, market_data=md, symbols=[symbol])

    return res, rep, md


# -----------------------------
# Main
# -----------------------------
if uploaded is None:
    st.info("Upload a BMCE CSV/XLSX to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = _bytes_hash(file_bytes)

df_preview = parse_upload_to_df(uploaded.name, file_bytes)
st.subheader("File preview")
st.dataframe(df_preview.head(30), use_container_width=True)

if not run_btn:
    st.stop()

with st.spinner("Running backtest..."):
    res, rep, md = run_pipeline_cached(
        file_hash=file_hash,
        symbol=symbol,
        tz=timezone,
        file_name=uploaded.name,
        file_bytes=file_bytes,
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

st.subheader("Headline metrics")
st.json(rep.metrics)

st.subheader("Cumulative returns")
st.pyplot(plot_series(rep.series["cum_returns"], "Cumulative Returns", "Cumulative return"))

st.subheader("Drawdown")
st.pyplot(plot_series(rep.series["drawdown"], "Drawdown", "Drawdown"))

st.subheader("Trades")
st.dataframe(rep.tables["trades"], use_container_width=True)

st.subheader("Monthly returns")
st.dataframe(rep.tables["monthly_returns"], use_container_width=True)
