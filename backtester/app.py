"""
Streamlit Backtester Application
Optimized for deployment with proper error handling and caching
"""
import io
import os
import tempfile
import hashlib
from typing import Any, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import project modules
from data import MarketData, BMCEDataSource
from indicators import IndicatorEngine, FeatureSpec
from strategy import MovingAverageCrossStrategy, MovingAverageCrossParams
from portfolio import PortfolioEngine, PortfolioConfig, CostModel
from results import ResultsAnalyzer


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="TA Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Header
# =============================================================================
st.markdown('<div class="main-header">📈 Technical Analysis Backtester</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">BMCE/MASI Market Data → Indicators → Strategy → Portfolio → Results</div>', unsafe_allow_html=True)


# =============================================================================
# Sidebar Controls
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data Section
    st.subheader("📊 Data")
    uploaded = st.file_uploader(
        "Upload BMCE CSV/XLSX",
        type=["csv", "xlsx"],
        help="Upload your market data file in BMCE format"
    )
    symbol = st.text_input("Symbol", value="IAM", help="Stock symbol identifier")
    timezone = st.text_input("Timezone", value="UTC", help="Timezone for data")
    
    st.divider()
    
    # Strategy Section
    st.subheader("🎯 Strategy: MA Cross")
    fast = st.number_input(
        "Fast SMA",
        min_value=2,
        max_value=300,
        value=20,
        step=1,
        help="Fast moving average window"
    )
    slow = st.number_input(
        "Slow SMA",
        min_value=3,
        max_value=500,
        value=50,
        step=1,
        help="Slow moving average window"
    )
    
    if fast >= slow:
        st.error("⚠️ Fast SMA must be less than Slow SMA")
    
    allow_short = st.checkbox(
        "Allow short positions",
        value=True,
        help="Enable short selling in the strategy"
    )
    
    st.divider()
    
    # Portfolio Section
    st.subheader("💼 Portfolio")
    initial_cash = st.number_input(
        "Initial cash",
        min_value=1_000.0,
        value=1_000_000.0,
        step=10_000.0,
        format="%.0f",
        help="Starting capital"
    )
    rebalance_policy = st.selectbox(
        "Rebalance policy",
        ["on_change", "every_bar"],
        index=0,
        help="When to rebalance the portfolio"
    )
    max_gross = st.number_input(
        "Max gross exposure",
        min_value=0.1,
        value=1.0,
        step=0.1,
        help="Maximum gross exposure (1.0 = 100%)"
    )
    cash_buffer = st.number_input(
        "Cash buffer",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help="Percentage of cash to keep in reserve"
    )
    
    st.divider()
    
    # Costs Section
    st.subheader("💰 Transaction Costs")
    apply_costs = st.checkbox(
        "Apply transaction costs",
        value=False,
        help="Include brokerage, exchange, and settlement fees"
    )
    
    if apply_costs:
        with st.expander("Cost Details", expanded=True):
            brokerage_bps = st.number_input("Brokerage (bps)", value=60.0, step=1.0)
            exchange_bps = st.number_input("Exchange (bps)", value=10.0, step=1.0)
            settlement_bps = st.number_input("Settlement (bps)", value=20.0, step=1.0)
            vat_rate = st.number_input("VAT rate", value=0.10, step=0.01)
            slippage_bps = st.number_input("Slippage (bps)", value=0.0, step=1.0)
    else:
        brokerage_bps = exchange_bps = settlement_bps = slippage_bps = 0.0
        vat_rate = 0.0
    
    st.divider()
    
    # Run Button
    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


# =============================================================================
# Helper Functions
# =============================================================================
def _bytes_hash(b: bytes) -> str:
    """Generate hash for file bytes for caching."""
    return hashlib.sha256(b).hexdigest()[:16]


def plot_series(s: pd.Series, title: str, ylabel: str, color: str = "#1f77b4"):
    """Create a matplotlib plot for a series."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s.values, color=color, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def parse_upload_to_df(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded file to DataFrame."""
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


def normalize_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize BMCE format to canonical OHLCV.
    Maps common BMCE columns to standard names.
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

    # Find date column
    if "Date" not in out.columns:
        for cand in ["date", "DATE", "Datetime", "datetime", "Time", "time"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "Date"})
                break

    # Validate required columns
    required = {"Date", "Open", "High", "Low", "Close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Cannot normalize: missing {missing}. Columns={list(out.columns)}")

    # Process date and set index
    out["Date"] = pd.to_datetime(out["Date"], errors='coerce')
    out = out.dropna(subset=["Date"])
    out = out.set_index("Date").sort_index()

    # Add Volume if missing
    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    # Convert to numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop rows with missing OHLC
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def build_marketdata_from_upload(
    symbol: str,
    tz: str,
    file_name: str,
    file_bytes: bytes
) -> MarketData:
    """
    Build MarketData from uploaded file.
    Tries BMCEDataSource methods first, then falls back to manual parsing.
    """
    suffix = ".csv" if file_name.lower().endswith(".csv") else ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        ds = BMCEDataSource(timezone=tz)

        # Try common method names
        for meth in ["read_dataframe", "read_df", "load_dataframe", "read_file", "read_one", "load_one"]:
            if hasattr(ds, meth):
                bars_df = getattr(ds, meth)(tmp_path)
                return MarketData(bars={symbol: bars_df}, source="BMCEDataSource", timezone=tz)

        # Try MarketData methods
        for meth in ["load_marketdata", "load_market_data", "get_marketdata", "get_market_data"]:
            if hasattr(ds, meth):
                md = getattr(ds, meth)(tmp_path, symbols=[symbol])
                return md

        # Fallback: manual parsing
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
    Build FeatureSpec entries for SMA indicators.
    Creates specs that produce columns: sma_{fast_window}, sma_{slow_window}
    """
    fast_window = int(fast_window)
    slow_window = int(slow_window)

    return [
        FeatureSpec(
            indicator="sma",
            params={"window": fast_window},
            inputs=("Close",),
            name=f"sma_{fast_window}",
            warmup=fast_window,
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
    Cached pipeline execution.
    Returns: (portfolio_result, report, market_data)
    """
    # Load data
    md = build_marketdata_from_upload(symbol, tz, file_name, file_bytes)

    # Compute indicators
    specs = build_sma_specs(fast, slow)
    eng = IndicatorEngine(
        cache_dir=".cache/features",
        enable_disk_cache=True,
        enable_memory_cache=True
    )
    feats = eng.compute(md, specs=specs, symbols=[symbol])

    # Generate strategy signals
    params = MovingAverageCrossParams(
        fast_window=int(fast),
        slow_window=int(slow),
        allow_short=bool(allow_short),
        nan_policy="flat",
    )
    strat = MovingAverageCrossStrategy(params)
    sf = strat.generate_signals(md, feats, symbols=[symbol])

    # Run portfolio simulation
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

    # Analyze results
    an = ResultsAnalyzer(periods_per_year=252, rf_annual=0.0)
    rep = an.analyze(res, market_data=md, symbols=[symbol])

    return res, rep, md


# =============================================================================
# Main Application Logic
# =============================================================================

# Check if file is uploaded
if uploaded is None:
    st.info("👆 Please upload a BMCE CSV/XLSX file to begin", icon="ℹ️")
    
    # Show example/instructions
    with st.expander("📖 How to use this app"):
        st.markdown("""
        ### Instructions:
        1. **Upload Data**: Upload your BMCE format CSV or XLSX file
        2. **Configure Strategy**: Set your moving average parameters
        3. **Set Portfolio Options**: Configure initial capital and constraints
        4. **Add Costs** (optional): Include transaction costs for realistic simulation
        5. **Run Backtest**: Click the "Run Backtest" button
        
        ### Expected Data Format:
        Your file should contain columns like:
        - `Date` (or similar date column)
        - `Ouvt` (Open)
        - `+Haut` (High)
        - `+Bas` (Low)
        - `Clôture` (Close)
        - `Volume` (optional)
        """)
    
    st.stop()

# Validate fast < slow
if fast >= slow:
    st.error("❌ Fast SMA must be less than Slow SMA. Please adjust the parameters.", icon="🚫")
    st.stop()

# Get file bytes and hash
file_bytes = uploaded.getvalue()
file_hash = _bytes_hash(file_bytes)

# Show file preview
st.subheader("📄 Data Preview")
try:
    df_preview = parse_upload_to_df(uploaded.name, file_bytes)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df_preview))
    with col2:
        st.metric("Columns", len(df_preview.columns))
    with col3:
        if "Date" in df_preview.columns or any(c.lower() == "date" for c in df_preview.columns):
            st.metric("Status", "✅ Valid")
        else:
            st.metric("Status", "⚠️ Check Format")
    
    st.dataframe(df_preview.head(30), use_container_width=True, height=300)
    
except Exception as e:
    st.error(f"❌ Error reading file: {str(e)}", icon="🚫")
    st.stop()

# Wait for run button
if not run_btn:
    st.info("👈 Click 'Run Backtest' in the sidebar to start the analysis", icon="ℹ️")
    st.stop()

# Run backtest
try:
    with st.spinner("🔄 Running backtest... This may take a moment."):
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
    
    st.success("✅ Backtest completed successfully!", icon="🎉")
    
except Exception as e:
    st.error(f"❌ Error running backtest: {str(e)}", icon="🚫")
    with st.expander("🔍 Error Details"):
        st.exception(e)
    st.stop()


# =============================================================================
# Display Results
# =============================================================================

# Key Metrics
st.subheader("📊 Performance Metrics")

# Create metrics grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Return",
        f"{rep.metrics.get('total_return', 0)*100:.2f}%",
        help="Total return over the backtest period"
    )
    st.metric(
        "CAGR",
        f"{rep.metrics.get('CAGR', 0)*100:.2f}%",
        help="Compound Annual Growth Rate"
    )

with col2:
    st.metric(
        "Sharpe Ratio",
        f"{rep.metrics.get('sharpe', 0):.2f}",
        help="Risk-adjusted return (higher is better)"
    )
    st.metric(
        "Sortino Ratio",
        f"{rep.metrics.get('sortino', 0):.2f}",
        help="Downside risk-adjusted return"
    )

with col3:
    st.metric(
        "Max Drawdown",
        f"{rep.metrics.get('max_drawdown', 0)*100:.2f}%",
        help="Maximum peak-to-trough decline"
    )
    st.metric(
        "Calmar Ratio",
        f"{rep.metrics.get('calmar', 0):.2f}",
        help="CAGR / Max Drawdown"
    )

with col4:
    st.metric(
        "Annual Volatility",
        f"{rep.metrics.get('vol_annual', 0)*100:.2f}%",
        help="Annualized standard deviation of returns"
    )
    st.metric(
        "Total Trades",
        f"{int(rep.metrics.get('num_fills', 0))}",
        help="Number of executed trades"
    )

# Additional metrics in expander
with st.expander("📈 Additional Metrics"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Turnover", f"{rep.metrics.get('turnover', 0):.2f}")
        st.metric("Total Costs", f"{rep.metrics.get('total_cost', 0):,.2f}")
    
    with col2:
        st.metric("Avg Gross Exposure", f"{rep.metrics.get('avg_gross_exposure', 0)*100:.1f}%")
        st.metric("Avg Net Exposure", f"{rep.metrics.get('avg_net_exposure', 0)*100:.1f}%")
    
    with col3:
        st.metric("Time in Market", f"{rep.metrics.get('pct_time_in_market', 0)*100:.1f}%")
        st.metric("Cost % of Initial", f"{rep.metrics.get('cost_as_pct_initial_equity', 0)*100:.2f}%")

st.divider()

# Charts
st.subheader("📈 Performance Charts")

# Cumulative Returns
st.markdown("#### Cumulative Returns")
fig_cum = plot_series(
    rep.series["cum_returns"],
    "Cumulative Returns Over Time",
    "Cumulative Return",
    color="#2ecc71"
)
st.pyplot(fig_cum)

# Drawdown
st.markdown("#### Drawdown")
fig_dd = plot_series(
    rep.series["drawdown"],
    "Drawdown Over Time",
    "Drawdown",
    color="#e74c3c"
)
st.pyplot(fig_dd)

# Equity Curve
with st.expander("💰 Equity Curve"):
    fig_eq = plot_series(
        rep.series["equity"],
        "Portfolio Equity Over Time",
        "Equity (NAV)",
        color="#3498db"
    )
    st.pyplot(fig_eq)

st.divider()

# Tables
st.subheader("📋 Detailed Analysis")

# Trades
st.markdown("#### Trade History")
if not rep.tables["trades"].empty:
    st.dataframe(
        rep.tables["trades"],
        use_container_width=True,
        height=400
    )
    
    # Download button for trades
    csv = rep.tables["trades"].to_csv(index=False)
    st.download_button(
        label="📥 Download Trades CSV",
        data=csv,
        file_name=f"trades_{symbol}_{file_hash}.csv",
        mime="text/csv"
    )
else:
    st.info("No trades executed during the backtest period.")

# Monthly Returns
st.markdown("#### Monthly Returns")
if not rep.tables["monthly_returns"].empty:
    # Format as percentage
    monthly_pct = rep.tables["monthly_returns"] * 100
    st.dataframe(
        monthly_pct.style.format("{:.2f}%").background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True
    )
else:
    st.info("Insufficient data for monthly returns breakdown.")

st.divider()

# Full metrics JSON
with st.expander("🔍 Raw Metrics (JSON)"):
    st.json(rep.metrics)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Data → Indicators → Strategy → Portfolio → Results")
