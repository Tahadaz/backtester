from __future__ import annotations

import streamlit as st
st.set_page_config(page_title="Backtester", layout="wide")

from plotly.subplots import make_subplots
import copy

import os
import tempfile
from pathlib import Path
from dataclasses import replace
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from optimize import build_spec_from_result_row, batch_optimize_by_period


import plotly.graph_objects as go

# ---- Project imports ----
from engine import BacktestEngine, BenchmarkConfig, EngineSpec, DataConfig, IndicatorsConfig, StrategyConfig
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


# -----------------------------
# Period catalogs (edit freely)
# -----------------------------
MASI_PERIODS: dict[str, tuple[str, str]] = {
    "Pre-2008 (2005-2007)": ("2005-01-01", "2007-12-31"),
    "Crise (2008-2009)": ("2008-01-01", "2009-12-31"),
    "Mid-crisis recovery (2010)": ("2010-01-01", "2010-12-31"),
    "Primtemps arabe / Eurozone (2011-2013)": ("2011-01-01", "2013-12-31"),
    "Normalization (2014-2019)": ("2014-01-01", "2019-12-31"),
    "COVID shock (2020)": ("2020-01-01", "2020-12-31"),
    "Inflation / rates shock (2022-mid2023)": ("2022-01-01", "2023-06-01"),
    "Post-2024 (2024-06-25+)": ("2023-06-02", "2025-12-31"),
}

IAM_PERIODS: dict[str, tuple[str, str]] = {
   "Etisalat control transition (2013-2015)": ("2013-01-01", "2015-12-31"),
    "Affaire Inwi (2024-01-29 to 2025-03-01)": ("2024-01-29", "2025-03-01"),
    "Changement Leadership (2025-03-01+)": ("2025-03-01", "2026-12-31"),
}


import optimize as _opt_mod
st.title("Backtester (TA) — Backtest / Optimize")
st.sidebar.caption(f"optimize.py loaded from: {_opt_mod.__file__}")
st.sidebar.caption(f"run_optimization: {_opt_mod.run_optimization.__module__}.{_opt_mod.run_optimization.__name__}")
# ============================================================
# Upload persistence (FIXES caching + speed)
#   - Streamlit reruns were writing uploads to a NEW temp path each run,
#     which breaks caching keys and forces full recompute.
#   - We persist uploads to a deterministic path based on file content hash.
# ============================================================

from dataclasses import replace
def render_strategy_params_ui(kind: str, *, prefix: str, allow_short: bool, nan_policy: str = "flat") -> dict:
    k = (kind or "").lower()
    p = {}

    if k == "buy_hold":
        p["buy_pct_cash"] = st.slider(f"{prefix} buy_pct_cash", 0.01, 1.00, 1.0, 0.01, key=f"{prefix}_bh_buy")
        p["nan_policy"] = nan_policy
        return p

    if k == "ma_cross":
        fast = st.number_input(f"{prefix} fast_window", 2, 500, 20, 1, key=f"{prefix}_fast")
        slow = st.number_input(f"{prefix} slow_window", 3, 500, 50, 1, key=f"{prefix}_slow")
        if fast >= slow:
            st.warning("fast must be < slow; auto-adjusting")
            fast = min(int(fast), int(slow) - 1)
        p.update({"sma_fast_window": int(fast), "sma_slow_window": int(slow)})

    elif k == "sma_price":
        w = st.number_input(f"{prefix} sma_window", 2, 500, 50, 1, key=f"{prefix}_w")
        p["sma_window"] = int(w)

    elif k == "rsi":
        w = st.number_input(f"{prefix} rsi_window", 2, 500, 14, 1, key=f"{prefix}_rsiw")
        lo = st.number_input(f"{prefix} oversold", 1, 49, 30, 1, key=f"{prefix}_rsilo")
        hi = st.number_input(f"{prefix} overbought", 51, 99, 70, 1, key=f"{prefix}_rsihi")
        if lo >= hi:
            st.warning("oversold must be < overbought; auto-adjusting")
            lo = min(int(lo), int(hi) - 1)
        p.update({"rsi_window": int(w), "rsi_oversold": float(lo), "rsi_overbought": float(hi)})

    elif k == "macd":
        fast = st.number_input(f"{prefix} macd_fast", 2, 500, 12, 1, key=f"{prefix}_macdf")
        slow = st.number_input(f"{prefix} macd_slow", 3, 500, 26, 1, key=f"{prefix}_macds")
        sig  = st.number_input(f"{prefix} macd_signal", 2, 500, 9, 1, key=f"{prefix}_macdsg")
        if fast >= slow:
            st.warning("fast must be < slow; auto-adjusting")
            fast = min(int(fast), int(slow) - 1)
        p.update({"macd_fast_window": int(fast), "macd_slow_window": int(slow), "macd_signal_window": int(sig)})

    elif k == "bollinger":
        w = st.number_input(f"{prefix} bb_window", 2, 500, 20, 1, key=f"{prefix}_bbw")
        kk = st.number_input(f"{prefix} bb_k", 0.1, 5.0, 2.0, 0.1, key=f"{prefix}_bbk")
        p.update({"bb_window": int(w), "bb_k": float(kk)})

    # common
    p["allow_short"] = bool(allow_short)
    p["nan_policy"] = nan_policy
    return p

def attach_comparator_to_bundle(
    bundle_a,
    bundle_b,
    *,
    label_a: str,
    label_b: str,
    periods_per_year: int,
    rf_annual: float,
):
    analyzer = ResultsAnalyzer(periods_per_year=periods_per_year, rf_annual=rf_annual)
    comp = analyzer.compare(
        report_a=bundle_a.report,
        report_b=bundle_b.report,
        label_a=label_a,
        label_b=label_b,
    )

    rep0 = bundle_a.report

    new_meta = dict(getattr(rep0, "meta", None) or {})
    new_meta["cum_compare_labels"] = {"a": label_a, "b": label_b}

    new_plots = dict(getattr(rep0, "plots", None) or {})
    new_plots["cum_vs_bench"] = comp["cum_plot"]

    new_tables = dict(getattr(rep0, "tables", None) or {})
    new_tables["curve_vs_comparator"] = comp["curve_table"]

    rep1 = replace(rep0, meta=new_meta, plots=new_plots, tables=new_tables)
    return replace(bundle_a, report=rep1)


import hashlib
import json, hashlib
def _spec_key(spec: EngineSpec) -> str:
    d = {
        "data": spec.data.__dict__,
        "strategy": {"kind": spec.strategy.kind, "params": spec.strategy.params},
        "portfolio": spec.portfolio.__dict__,
        "interval": spec.data.interval,
    }
    s = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha1(s).hexdigest()

def parse_int_list(s: str) -> list[int]:
    """
    Parse '5,10, 20 50' into [5,10,20,50]. Ignores blanks.
    Raises ValueError on invalid tokens.
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    # allow commas or spaces
    tokens = [t.strip() for t in s.replace(",", " ").split()]
    out = []
    for t in tokens:
        if not t:
            continue
        out.append(int(float(t)))  # lets user type "20.0" too
    # dedupe while preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def add_volume_to_trades_table(
    trades: pd.DataFrame,
    md,  # MarketData
    *,
    volume_col: str = "Volume",
    out_col: str = "volume",
) -> pd.DataFrame:
    """
    Adds bar volume to each fill row using (timestamp, symbol) lookup.
    trades must have columns: ['timestamp','symbol'].
    """
    if trades is None or trades.empty:
        return trades
    if "timestamp" not in trades.columns or "symbol" not in trades.columns:
        return trades

    tdf = trades.copy()
    tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
    tdf = tdf.dropna(subset=["timestamp", "symbol"])

    # Build a lookup table: (timestamp, symbol) -> volume
    parts = []
    for sym, bars in md.bars.items():
        if volume_col not in bars.columns:
            continue
        b = bars[[volume_col]].copy()
        b = b.sort_index()
        b = b.reset_index().rename(columns={b.index.name or "index": "timestamp", volume_col: out_col})
        b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce")
        b["symbol"] = sym
        parts.append(b[["timestamp", "symbol", out_col]])

    if not parts:
        return tdf

    vol_df = pd.concat(parts, ignore_index=True)
    # Exact join on timestamp+symbol (your fills timestamp should match bars index)
    tdf = tdf.merge(vol_df, on=["timestamp", "symbol"], how="left")

    return tdf

def _persist_upload_to_cache(uploaded_file, tag: str, symbol: str) -> tuple[str, str]:
    """
    Persist an UploadedFile to a stable on-disk path based on content hash.
    Returns (path, sha1_hex).
    """
    data = uploaded_file.getvalue()
    h = hashlib.sha1(data).hexdigest()
    suffix = Path(uploaded_file.name).suffix.lower() or ".dat"
    safe_sym = "".join(ch for ch in (symbol or "SYM") if ch.isalnum() or ch in ("_", "-"))[:32]
    cache_root = Path.home() / ".backtester_cache" / "uploads"
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / f"{tag}_{safe_sym}_{h}{suffix}"
    if (not out_path.exists()) or (out_path.stat().st_size != len(data)):
        out_path.write_bytes(data)
    return str(out_path), h





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



from plotly.subplots import make_subplots

def plot_price_indicators_trades_line(
    bars: pd.DataFrame,
    strategy_params: dict | None = None,
    indicators: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    indicator_cols: list[str] | None = None,
    *,
    port_cfg: PortfolioConfig | None = None,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
) -> go.Figure:
    """
    Price (row 1), RSI and/or MACD panels (middle rows), Volume (last row).

    - RSI is plotted in its own panel with threshold lines + shaded regions.
    - MACD panel plots EMA_fast, EMA_slow + histogram (EMA_fast-EMA_slow) as bars
      with per-bar green/red intensity (stronger color as magnitude increases).
    """

    df = bars.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    ind = None
    if indicators is not None and not indicators.empty:
        ind = indicators.copy()
        if not isinstance(ind.index, pd.DatetimeIndex):
            ind.index = pd.to_datetime(ind.index)
        ind = ind.reindex(df.index)

        if indicator_cols is not None:
            keep = [c for c in indicator_cols if c in ind.columns]
            if keep:
                ind = ind[keep]

    def _maybe_add_bollinger_overlay():
        nonlocal fig, df, ind, indicator_cols

        if ind is None or ind.empty:
            return

        cols = list(ind.columns)

        # --- HARD GATE: only plot BB if explicitly requested ---
        is_bollinger_strategy = bool(strategy_params) and (
            ("bb_k" in strategy_params) or ("bb_window" in strategy_params)
        )

        has_std_col = any(c.startswith("std_") for c in cols)
        has_std_requested = bool(indicator_cols) and any(c.startswith("std_") for c in indicator_cols)

        # If it's not a bollinger strategy and no std column is present/requested, do nothing
        if not (is_bollinger_strategy or has_std_col or has_std_requested):
            return

        # From here on, infer w ONLY from std_ first (preferred), then sma_ if needed
        w = None

        # Prefer explicit requested std_
        if indicator_cols:
            for c in indicator_cols:
                if c.startswith("std_"):
                    try:
                        w = int(c.split("_", 1)[1])
                        break
                    except Exception:
                        pass

        # Else infer from actual std_ columns present
        if w is None:
            for c in cols:
                if c.startswith("std_"):
                    try:
                        w = int(c.split("_", 1)[1])
                        break
                    except Exception:
                        pass

        # If still none, fall back to SMA window (only if bollinger strategy)
        if w is None and is_bollinger_strategy:
            for c in cols:
                if c.startswith("sma_"):
                    try:
                        w = int(c.split("_", 1)[1])
                        break
                    except Exception:
                        pass

        if w is None:
            return

        k = float((strategy_params or {}).get("bb_k", 2.0))
        col_mid = f"sma_{w}"
        col_std = f"std_{w}"

        if col_mid not in ind.columns:
            return

        mid = pd.to_numeric(ind[col_mid], errors="coerce")

        if col_std in ind.columns:
            std = pd.to_numeric(ind[col_std], errors="coerce")
        else:
            # Only allow computing std fallback if this is explicitly bollinger strategy
            if not is_bollinger_strategy:
                return
            std = pd.to_numeric(df["Close"], errors="coerce").rolling(int(w), min_periods=int(w)).std()

        upper = mid + k * std
        lower = mid - k * std

        # Add mid/upper/lower on PRICE row (row=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=mid, mode="lines", name=f"BB mid (SMA{w})", line=dict(width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=upper, mode="lines", name=f"BB upper (k={k})", line=dict(width=1, dash="dash")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=lower, mode="lines", name=f"BB lower (k={k})", line=dict(width=1, dash="dash")),
            row=1, col=1
        )

        # Optional fill between upper and lower (nice visual)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=lower,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name="BB band",
                opacity=0.12,
                hoverinfo="skip",
            ),
            row=1, col=1
        )

    # ----------------------------
    # Detect RSI / MACD columns
    # ----------------------------
    rsi_cols: list[str] = []
    macd_bases: list[str] = []

    if ind is not None and not ind.empty:
        cols = list(ind.columns)

        # RSI columns: rsi_{n}
        rsi_cols = [c for c in cols if c.startswith("rsi_")]

        # MACD base: macd_{fast}_{slow}_{sig} (but in DF you likely have __line/__signal/__hist)
        # We'll infer unique "macd_{fast}_{slow}_{sig}" bases from columns.
        for c in cols:
            if c.startswith("macd_") and "__" in c:
                base = c.split("__", 1)[0]
                if base not in macd_bases:
                    macd_bases.append(base)

    has_rsi = len(rsi_cols) > 0
    has_macd = len(macd_bases) > 0

    # If multiple RSI/MACD exist, we plot the first by default (you can drive this via indicator_cols).
    rsi_col = rsi_cols[0] if has_rsi else None
    macd_base = macd_bases[0] if has_macd else None

    # ----------------------------
    # Layout: rows depend on panels
    # ----------------------------
    # row 1: price
    # optional row 2: RSI
    # optional row 3: MACD
    # last row: volume
    n_mid = int(has_rsi) + int(has_macd)
    n_rows = 2 + n_mid  # price + volume + middle panels

    # heights: price biggest, then mid panels, then volume
    # normalize later by relative weights
    row_heights = []
    row_heights.append(0.58)  # price
    if has_rsi:
        row_heights.append(0.20)
    if has_macd:
        row_heights.append(0.20)
    row_heights.append(0.22)  # volume

    # normalize to sum=1
    s = sum(row_heights)
    row_heights = [h / s for h in row_heights]

    # Determine which row is MACD row (if present)
    macd_row = None
    tmp_row = 2
    if has_rsi:
        tmp_row += 1
    if has_macd:
        macd_row = tmp_row  # the current row where MACD panel is plotted

    specs = [[{}] for _ in range(n_rows)]
    if macd_row is not None:
        specs[macd_row - 1][0] = {"secondary_y": True}  # plotly is 1-indexed rows

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        specs=specs,
    )


    # =========================
    # Row 1: Price + Trades
    # =========================
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"].astype(float), mode="lines", name="Close"),
        row=1, col=1
    )
    _maybe_add_bollinger_overlay()
    if ind is not None and not ind.empty:
        # Keep it conservative: only draw moving averages on the price panel
        price_level_cols = [c for c in ind.columns if c.startswith(("sma_", "ema_"))]

        for c in price_level_cols:
            y = pd.to_numeric(ind[c], errors="coerce")
            if y.notna().any():  # avoid adding empty traces
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=y,
                        mode="lines",
                        name=c,
                        line=dict(width=1),
                    ),
                    row=1, col=1
                )

    # Trades markers on row 1
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
        t = t.dropna(subset=["timestamp"]).sort_values("timestamp")

        if "side" not in t.columns:
            t["side"] = np.where(pd.to_numeric(t["qty"], errors="coerce").fillna(0) > 0, "BUY", "SELL")

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
                ),
                row=1, col=1
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
                ),
                row=1, col=1
            )

    # =========================
    # Middle panels: RSI / MACD
    # =========================
    cur_row = 2  # first middle row

    # ---- RSI panel ----
    if has_rsi and rsi_col is not None:
        rsi = pd.to_numeric(ind[rsi_col], errors="coerce")

        # main RSI line
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, mode="lines", name=rsi_col),
            row=cur_row, col=1
        )

        # threshold lines
        fig.add_hline(y=rsi_low, line_width=1, line_dash="dash", row=cur_row, col=1)
        fig.add_hline(y=rsi_high, line_width=1, line_dash="dash", row=cur_row, col=1)

        # shaded regions:
        # - oversold region: [0, rsi_low]
        # - overbought region: [rsi_high, 100]
        # Plotly: rectangle shapes in that subplot's y-domain
        fig.add_hrect(y0=0, y1=rsi_low, row=cur_row, col=1, opacity=0.12, line_width=0)
        fig.add_hrect(y0=rsi_high, y1=100, row=cur_row, col=1, opacity=0.12, line_width=0)

        # keep RSI scale consistent
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=cur_row, col=1)

        cur_row += 1

    # ---- MACD panel ----
    if has_macd and macd_base is not None:
        line_col = f"{macd_base}__line"
        sig_col  = f"{macd_base}__signal"
        hist_col = f"{macd_base}__hist"

        line = pd.to_numeric(ind.get(line_col), errors="coerce")
        sigl = pd.to_numeric(ind.get(sig_col), errors="coerce")

        # histogram: prefer __hist if you have it, else line - signal
        if hist_col in ind.columns:
            hist = pd.to_numeric(ind[hist_col], errors="coerce")
        else:
            hist = line - sigl

        # MACD line + signal line (PRIMARY y)
        fig.add_trace(
            go.Scatter(x=df.index, y=line, mode="lines", name=f"{macd_base} line"),
            row=cur_row, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=sigl, mode="lines", name=f"{macd_base} signal"),
            row=cur_row, col=1, secondary_y=False
        )

        # Histogram bars (SECONDARY y)
        h = hist.fillna(0.0).astype(float)
        bar_colors = np.where(h >= 0, "rgba(0,180,0,0.6)", "rgba(200,0,0,0.6)")

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=h.values,
                name=f"{macd_base} hist",
                marker=dict(color=bar_colors),
            ),
            row=cur_row, col=1, secondary_y=True
        )

        # Zero line on histogram axis (SECONDARY y)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=np.zeros(len(df.index)),
                mode="lines",
                showlegend=False,
            ),
            row=cur_row, col=1, secondary_y=True
        )

        fig.update_yaxes(title_text="MACD", row=cur_row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Hist", row=cur_row, col=1, secondary_y=True)

        cur_row += 1


    # =========================
    # Last Row: Volume + Gate/Cap
    # =========================
    vol_row = n_rows

    if port_cfg is not None:
        vcol = getattr(port_cfg, "volume_col", "Volume")
    else:
        vcol = "Volume"

    if vcol in df.columns:
        vol = pd.to_numeric(df[vcol], errors="coerce").fillna(0.0).astype(float)

        fig.add_trace(
            go.Bar(x=df.index, y=vol.values, name="Volume"),
            row=vol_row, col=1
        )

        def _adv(series: pd.Series, w: int) -> pd.Series:
            w = int(max(1, w))
            return series.rolling(w, min_periods=1).mean()

        if port_cfg is not None:
            if bool(getattr(port_cfg, "use_volume_gate", False)):
                kind = str(getattr(port_cfg, "volume_gate_kind", "min_abs"))
                if kind == "min_abs":
                    gate_val = float(getattr(port_cfg, "min_volume_abs", 0.0))
                    gate_line = pd.Series(gate_val, index=df.index)
                else:
                    ratio = float(getattr(port_cfg, "min_volume_ratio_adv", 0.0))
                    w = int(getattr(port_cfg, "volume_gate_adv_window", 20))
                    gate_line = ratio * _adv(vol, w)

                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=gate_line.values,
                        mode="lines",
                        name="Gate",
                        line=dict(width=3),
                        showlegend=True,
                    ),
                    row=vol_row, col=1
                )

            if bool(getattr(port_cfg, "use_participation_cap", False)):
                pr = float(getattr(port_cfg, "participation_rate", 0.05))
                basis = str(getattr(port_cfg, "participation_basis", "bar"))
                if basis == "bar":
                    liq = vol
                else:
                    w = int(getattr(port_cfg, "adv_window", 20))
                    liq = _adv(vol, w)

                cap_line = pr * liq

                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=cap_line.values,
                        mode="lines",
                        name="Cap",
                        line=dict(width=3, dash="dash"),
                        showlegend=True,
                    ),
                    row=vol_row, col=1
                )

    # ----------------------------
    # Layout polish
    # ----------------------------
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
        bargap=0.0,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=vol_row, col=1)

    return fig



def plot_cum_vs_bench(
    cum: pd.Series,
    bench_cum: pd.Series | None,
    *,
    label_a: str = "Strategy",
    label_b: str = "Comparator",
    title: str = "Cumulative Returns",
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()

    ax.plot(cum.index, cum.values, label=label_a)
    if bench_cum is not None:
        ax.plot(bench_cum.index, bench_cum.values, label=label_b)

    ax.set_title(title)
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





def render_comparison(
    bundle_a,
    bundle_b,
    *,
    label_a: str = "Strategy",
    label_b: str = "Benchmark",
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
):
    analyzer = ResultsAnalyzer(periods_per_year=periods_per_year, rf_annual=rf_annual)

    # You will implement analyzer.compare(...) in results.py
    comp = analyzer.compare(
        report_a=bundle_a.report,
        report_b=bundle_b.report,
        label_a=label_a,
        label_b=label_b,
    )

    st.subheader("Cumulative Returns vs Comparator")
    cvb = comp["cum_plot"]
    st.pyplot(plot_cum_vs_bench(cvb["strategy"], cvb["benchmark"]))

    st.subheader("Curve vs Comparator")
    st.dataframe(comp["curve_table"], use_container_width=True)


# ============================================================
# Render bundle
# ============================================================

def render_bundle(bundle, *, port_cfg: PortfolioConfig | None = None, label : str| None = None):
    rep = bundle.report
    plots = rep.plots
    tables = rep.tables
    st.subheader("Price + Indicators + Trades")
    pp = plots["price_panel"]
    sym0 = bundle.md.symbols()[0]
    bars = bundle.md.bars[sym0]

    fig = plot_price_indicators_trades_line(
        bars=bars,
        strategy_params=getattr(bundle, "spec", None).strategy.params if hasattr(bundle, "spec") else None,
        indicators=pp.get("indicators"),
        trades=pp.get("trades"),
        indicator_cols=pp.get("indicator_cols"),
        port_cfg=port_cfg,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Returns vs Comparator")
    cvb = plots["cum_vs_bench"]

    sym0 = bundle.md.symbols()[0] if hasattr(bundle, "md") and bundle.md.symbols() else ""
    kind = getattr(getattr(bundle, "spec", None), "strategy", None)
    kind = getattr(kind, "kind", "strategy")

    labels = rep.meta.get("cum_compare_labels", {}) if hasattr(rep, "meta") else {}
    label_a = labels.get("a", f"{kind} | {sym0}")
    label_b = labels.get("b", "Comparator")

    st.pyplot(
        plot_cum_vs_bench(
            cvb["strategy"],
            cvb.get("benchmark"),
            label_a=label_a,
            label_b=label_b,
        )
    )


    st.subheader("Drawdown")
    st.pyplot(plot_drawdown_red(plots["drawdown"]))


    st.subheader("Monthly Returns Heatmap")
    st.pyplot(plot_monthly_heatmap_with_values(plots["monthly_heatmap"]))

    st.subheader("Yearly Returns")
    st.pyplot(plot_yearly_returns_bar(plots["yearly_bar"]))

    st.subheader("Trades (fills)")
    if "trades" in tables:
        trades_df = tables["trades"]

        # pick the right volume column name
        vcol = getattr(port_cfg, "volume_col", "Volume") if port_cfg is not None else "Volume"

        trades_df = add_volume_to_trades_table(trades_df, bundle.md, volume_col=vcol, out_col="volume")

        # optional: choose visible columns order
        cols_first = [c for c in ["timestamp","symbol","side","qty","price","notional","cost","volume"] if c in trades_df.columns]
        cols_rest = [c for c in trades_df.columns if c not in cols_first]
        trades_df = trades_df[cols_first + cols_rest]

        st.dataframe(trades_df, use_container_width=True)


    st.subheader("Trade Ledger (PnL per closed trade)")
    if "trade_ledger" in tables:
        st.dataframe(tables["trade_ledger"], use_container_width=True)

    st.subheader("Trade Performance (summary)")
    if "trade_performance" in tables:
        st.dataframe(tables["trade_performance"], use_container_width=True)
    if "curve_vs_comparator" in tables:
        st.subheader("Curve vs Comparator")
        st.dataframe(tables["curve_vs_comparator"], use_container_width=True)




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
    include_windows: Optional[list[tuple[str, str]]],
    exclude_windows: Optional[list[tuple[str, str]]],
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
    use_volume_gate: bool,
    volume_gate_kind: str,
    min_volume_abs: float,
    min_volume_ratio_adv: float,
    volume_gate_adv_window: int,
    use_participation_cap: bool,
    participation_rate: float,
    participation_basis: str,
    adv_window: int,
) -> EngineSpec:

    if source_key == "bmce":
        if not bmce_tmp_path:
            raise ValueError("BMCE selected but bmce_tmp_path is None/empty.")
        data_cfg = DataConfig(
            source="bmce",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            start=start,
            end=end,
            include_windows=include_windows,
            exclude_windows=exclude_windows,
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
            include_windows=include_windows,
            exclude_windows=exclude_windows,
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
        min_return_before_sell=float(min_return_before_sell),
        cooldown_bars=int(cooldown_bars),
        use_volume_gate=bool(use_volume_gate),
        volume_gate_kind=str(volume_gate_kind),
        min_volume_abs=float(min_volume_abs),
        min_volume_ratio_adv=float(min_volume_ratio_adv),
        volume_gate_adv_window=int(volume_gate_adv_window),

        use_participation_cap=bool(use_participation_cap),
        participation_rate=float(participation_rate),
        participation_basis=str(participation_basis),
        adv_window=int(adv_window),

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
source = st.sidebar.selectbox("Source", ["(upload)", "yfinance","(local file)"], index=0)
source_key = "bmce" if source.startswith("(") else "yfinance"

symbol = st.sidebar.text_input("Symbol", value="IAM" if source_key == "bmce" else "AAPL")
timezone = st.sidebar.text_input("Timezone", value="GMT")
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)

bmce_file = None
yf_period = yf_interval = None
yf_auto_adjust = None

if source_key == "bmce":
    bmce_file = None
    bmce_path = None  # <- this is what we will pass into DataConfig.bmce_paths

    if source == "(local file)":
        # Put your default IAM file somewhere in your project, e.g. ./data/IAM.xlsx
        default_path = str((Path(__file__).parent / "data" / "Data IAM.xlsx").resolve())
        bmce_path = st.sidebar.text_input("Local BMCE file path (CSV/XLSX)", value=default_path)

    else:  # bmce (upload)
        bmce_file = st.sidebar.file_uploader("Upload BMCE CSV/XLSX", type=["csv", "xlsx"])
        if bmce_file is not None:
            bmce_cached_path, bmce_file_hash = _persist_upload_to_cache(bmce_file, tag="bmce", symbol=symbol)
            st.session_state["bmce_cached_path"] = bmce_cached_path
            st.session_state["bmce_file_hash"] = bmce_file_hash
            bmce_path = bmce_cached_path

else:
    st.sidebar.caption("yfinance needs internet + yfinance in requirements.txt")
    yf_period = st.sidebar.text_input("yfinance period", value="5y")
    yf_interval = st.sidebar.selectbox("yfinance interval", ["1d"], index=0)
    yf_auto_adjust = st.sidebar.checkbox("auto_adjust", value=False)

st.sidebar.header("Strategy")
strategy_kind = st.sidebar.selectbox("Strategy kind", ["ma_cross", "sma_price", "rsi", "macd", "bollinger"], index=0)
allow_short = st.sidebar.checkbox("Allow short", value=False)

# Reset optimization UI on strategy change (prevents stale widget keys)
prev_kind = ss_get("prev_strategy_kind", strategy_kind)
if prev_kind != strategy_kind:
    for k in list(st.session_state.keys()):
        if k.endswith("_min") or k.endswith("_max") or k.endswith("_step") or k.endswith("_choices"):
            del st.session_state[k]
    st.session_state["prev_strategy_kind"] = strategy_kind

st.sidebar.header("Comparison")

compare_mode = st.sidebar.selectbox(
    "Compare against",
    ["None", "Buy & Hold (same data)", "Another strategy (same data)"],
    index=0,
)

compare_strategy_kind = None
compare_params = {}

if compare_mode == "Another strategy (same data)":
    compare_strategy_kind = st.sidebar.selectbox(
        "Comparator strategy kind",
        ["buy_hold", "ma_cross", "sma_price", "rsi", "macd", "bollinger"],
        index=0,
        key="cmp_kind",
    )
    with st.expander("Comparator parameters", expanded=True):
        compare_params = render_strategy_params_ui(compare_strategy_kind, prefix="cmp", allow_short=allow_short)



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

        if bench_bmce_file is not None:
            bench_cached_path, bench_file_hash = _persist_upload_to_cache(bench_bmce_file, tag="bmce_bench", symbol=bench_symbol)
            st.session_state["bench_cached_path"] = bench_cached_path
            st.session_state["bench_file_hash"] = bench_file_hash

st.sidebar.header("Periods (include/exclude)")

use_period_filters = st.sidebar.checkbox("Enable period filters", value=False)

include_windows: list[tuple[str, str]] | None = None
exclude_windows: list[tuple[str, str]] | None = None

if use_period_filters:
    st.sidebar.subheader("Market periods (MASI-level)")

    market_include = st.sidebar.multiselect(
        "Include market periods",
        options=list(MASI_PERIODS.keys()),
        default=list(MASI_PERIODS.keys()),
    )
    market_exclude = st.sidebar.multiselect(
        "Exclude market periods",
        options=list(MASI_PERIODS.keys()),
        default=[],
        key="market_exclude",
    )

    # Build include windows from selection (if user unselects all, treat as "no include restriction")
    inc = [MASI_PERIODS[k] for k in market_include] if market_include else []
    exc = [MASI_PERIODS[k] for k in market_exclude] if market_exclude else []

    st.sidebar.subheader("Stock-specific overlays")

    if symbol.upper() == "IAM":
        iam_include = st.sidebar.multiselect(
            "Include IAM periods (optional)",
            options=list(IAM_PERIODS.keys()),
            default=[],
            key="iam_include",
        )
        iam_exclude = st.sidebar.multiselect(
            "Exclude IAM periods",
            options=list(IAM_PERIODS.keys()),
            default=[],
            key="iam_exclude",
        )

        inc_i = [IAM_PERIODS[k] for k in iam_include] if iam_include else []
        exc_i = [IAM_PERIODS[k] for k in iam_exclude] if iam_exclude else []

        # Methodology:
        # 1) include = market includes (if any were selected)
        # 2) if IAM include selected, we further restrict by intersecting -> easiest way is to add to include_windows
        #    and let engine do union-of-includes; to get true intersection you either:
        #      - do it in engine (more complex), or
        #      - choose to interpret "IAM include" as an additional allowed window set.
        # Here we implement "IAM include" as additional includes; if you want strict intersection later, we can upgrade.
        inc = inc + inc_i
        exc = exc + exc_i

    include_windows = inc if inc else None
    exclude_windows = exc if exc else None


# ============================================================
# Main preview (BMCE)
# ============================================================

preview_df = None
if source_key == "bmce":
    st.subheader("BMCE data")

    if source == "(upload)" and bmce_file is None:
        st.info("Upload a BMCE CSV/XLSX.")
        st.stop()

    if bmce_path is None:
        st.error("No BMCE path provided.")
        st.stop()

    try:
        p = Path(bmce_path)
        if not p.exists():
            st.error(f"Local file not found: {bmce_path}")
            st.stop()

        if p.suffix.lower() == ".csv":
            preview_df = pd.read_csv(bmce_path)
        else:
            preview_df = pd.read_excel(bmce_path, engine="openpyxl")

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

# ---- Global date range used by BOTH Backtest and Optimize ----
use_date_range = st.sidebar.checkbox("Use date range", value=False)

c1, c2 = st.sidebar.columns(2)
with c1:
    start_date = st.date_input("Start date", value=None, disabled=not use_date_range)
with c2:
    end_date = st.date_input("End date", value=None, disabled=not use_date_range)

start_str = start_date.isoformat() if (use_date_range and start_date) else None
end_str = end_date.isoformat() if (use_date_range and end_date) else None

if use_date_range and start_str and end_str and start_str > end_str:
    st.sidebar.error("Start date must be <= End date")
    st.stop()


# ============================================================
# Backtest Tab (FORM)
# ============================================================

with tab_backtest:
    st.subheader("Backtest")

    with st.form("backtest_form", clear_on_submit=False):

        st.markdown("### Strategy parameters")
        nan_policy = "flat"
        strategy_params: Dict[str, Any] = {}

        if strategy_kind == "ma_cross":
            fast = st.number_input("Fast SMA window", min_value=2, max_value=500, value=20, step=1)
            slow = st.number_input("Slow SMA window", min_value=3, max_value=500, value=50, step=1)
            if fast >= slow:
                st.warning("Fast must be < Slow. Auto-adjusting fast.")
                fast = min(int(fast), int(slow) - 1)
            strategy_params = {"sma_fast_window": int(fast), "sma_slow_window": int(slow), "allow_short": bool(allow_short), "nan_policy": nan_policy}
        elif strategy_kind == "sma_price":
            window = st.number_input("SMA window", min_value=2, max_value=500, value=50, step=1)
            strategy_params = {"sma_window": int(window), "allow_short": bool(allow_short), "nan_policy": nan_policy}
        elif strategy_kind == "rsi":
            window = st.number_input("RSI window", min_value=2, max_value=500, value=14, step=1)
            oversold = st.number_input("Oversold threshold", min_value=1, max_value=49, value=30, step=1)
            overbought = st.number_input("Overbought threshold", min_value=51, max_value=99, value=70, step=1)
            if oversold >= overbought:
                st.warning("Oversold must be < Overbought. Auto-adjusting oversold.")
                oversold = min(int(oversold), int(overbought) - 1)
            strategy_params = {
                "rsi_window": int(window),
                "rsi_oversold": int(oversold),
                "rsi_overbought": int(overbought),
                "allow_short": bool(allow_short),
                "nan_policy": nan_policy,
            }
        elif strategy_kind == "macd":
            fast = st.number_input("Fast EMA window", min_value=2, max_value=500, value=12, step=1)
            slow = st.number_input("Slow EMA window", min_value=3, max_value=500, value=26, step=1)
            signal = st.number_input("Signal EMA window", min_value=2, max_value=500, value=9, step=1)
            if fast >= slow:
                st.warning("Fast must be < Slow. Auto-adjusting fast.")
                fast = min(int(fast), int(slow) - 1)
            strategy_params = {
                "macd_fast_window": int(fast),
                "macd_slow_window": int(slow),
                "macd_signal_window": int(signal),
                "allow_short": bool(allow_short),
                "nan_policy": nan_policy,
            }
        elif strategy_kind == "bollinger":
            window = st.number_input("BB window", min_value=2, max_value=500, value=20, step=1)
            k = st.number_input("BB k (std dev multiplier)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
            strategy_params = {
                "bb_window": int(window),
                "bb_k": float(k),
                "allow_short": bool(allow_short),
                "nan_policy": nan_policy,
            }

        
        st.markdown("### Portfolio")
        initial_cash = st.number_input("Initial cash", min_value=1_000.0, value=100_000.0, step=10_000.0)
        rebalance_policy = st.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)
        sizing_mode = st.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=1)
        cooldown_bars = st.number_input("Min bars between trades (cooldown)", min_value=0, value=0, step=1)

        st.markdown("### Sizing (percentages)")
        buy_pct_cash = st.slider("Buy % of cash per entry", 0.01, 1.00, 0.25, 0.01)
        sell_pct_shares = st.slider("Sell % of shares per exit", 0.01, 1.00, 1.00, 0.01)
        min_return_before_sell = st.slider(
            "Min return before selling (take-profit, %)",
            0.0, 50.0, 0.0, 0.1
        ) / 100.0


        st.markdown("### Liquidity / Volume")

        use_volume_gate = st.checkbox("Enable volume gate (Layer 1)", value=False)
        volume_gate_kind = st.selectbox("Gate type", ["min_abs", "min_ratio_adv"], index=0, disabled=not use_volume_gate)

        min_volume_abs = st.number_input("Min Volume (abs shares)", min_value=0.0, value=0.0, step=10_000.0, disabled=(not use_volume_gate or volume_gate_kind != "min_abs"))
        min_volume_ratio_adv = st.slider("Min Volume / ADV ratio", 0.0, 2.0, 0.3, 0.05, disabled=(not use_volume_gate or volume_gate_kind != "min_ratio_adv"))
        volume_gate_adv_window = st.number_input("ADV window for gate", min_value=1, value=20, step=1, disabled=(not use_volume_gate or volume_gate_kind != "min_ratio_adv"))

        use_participation_cap = st.checkbox("Enable participation cap (Layer 3)", value=False)
        participation_rate = st.slider("Participation rate", 0.001, 0.50, 0.05, 0.001, disabled=not use_participation_cap)
        participation_basis = st.selectbox("Participation basis", ["bar", "adv"], index=0, disabled=not use_participation_cap)
        adv_window = st.number_input("ADV window (for cap)", min_value=1, value=20, step=1, disabled=(not use_participation_cap or participation_basis != "adv"))


        st.markdown("### Costs")
        apply_costs = st.checkbox("Apply costs", value=False)
        if apply_costs:
            brokerage_bps = st.number_input("Brokerage (bps)", value=60.0, step=1.0)
            comm_bourse_bps = st.number_input("Commission de la BdC (bps)", value=10.0, step=1.0)
            reg_liv_bps = st.number_input("Frais Règlement-Livraison (bps)", value=20.0, step=1.0)
            tva_rate = st.number_input("tva", value=0.10, step=0.01)
            slippage_bps = st.number_input("Slippage (bps)", value=0.0, step=1.0)
        else:
            brokerage_bps = comm_bourse_bps = reg_liv_bps = slippage_bps = 0.0
            tva_rate = 0.0

        run_backtest = st.form_submit_button("Run backtest")

    if run_backtest:
        # Save uploaded file to temp for engine
        tmp_path = None
        tmp_dir = None
        tmp_is_temp = True
        try:
            if source_key == "bmce":
                # Use stable persisted path (content-hash) to enable caching / speed
                tmp_is_temp = False
                tmp_path = st.session_state.get("bmce_cached_path")
                if not tmp_path:
                    st.error("BMCE file not persisted. Re-upload the file.")
                    st.stop()

            cost_model = CostModel(
                brokerage_bps=float(brokerage_bps),
                comm_bourse_bps=float(comm_bourse_bps),
                reg_liv_bps=float(reg_liv_bps),
                slippage_bps=float(slippage_bps),
                tva_rate=float(tva_rate),
            )

            base_spec = make_base_spec(
                source_key=source_key,
                symbol=symbol,
                timezone=timezone,
                interval=interval,
                bmce_tmp_path=tmp_path,
                start=start_str,
                end=end_str,
                include_windows=include_windows,
                exclude_windows=exclude_windows,
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
                use_volume_gate=use_volume_gate,
                volume_gate_kind=volume_gate_kind,
                min_volume_abs=min_volume_abs,
                min_volume_ratio_adv=min_volume_ratio_adv,
                volume_gate_adv_window=volume_gate_adv_window,
                use_participation_cap=use_participation_cap,
                participation_rate=participation_rate,
                participation_basis=participation_basis,
                adv_window=adv_window,
            )

            cmp_spec = None
            if compare_mode == "Buy & Hold (same data)":
                label_b = f"buy_hold | {symbol}"

                cmp_spec = EngineSpec(
                    data=base_spec.data,
                    indicators=base_spec.indicators,
                    strategy=StrategyConfig(kind="buy_hold", params={"buy_pct_cash": 1.0}),
                    portfolio=base_spec.portfolio,
                    benchmark=BenchmarkConfig(enabled=False),  # keep off
                    plot_indicators=[],
                    periods_per_year=base_spec.periods_per_year,
                    rf_annual=base_spec.rf_annual,
                )

            elif compare_mode == "Another strategy (same data)":
                cmp_spec = EngineSpec(
                    data=base_spec.data,
                    indicators=base_spec.indicators,
                    strategy=StrategyConfig(kind=compare_strategy_kind, params=compare_params),
                    portfolio=base_spec.portfolio,
                    benchmark=BenchmarkConfig(enabled=False),
                    plot_indicators=[],
                    periods_per_year=base_spec.periods_per_year,
                    rf_annual=base_spec.rf_annual,
                )

            # --- Run A (main strategy) once ---
            key_a = ("bundle", _spec_key(base_spec))
            bundle_a = st.session_state.get(key_a)
            if bundle_a is None:
                with st.spinner("Running backtest..."):
                    bundle_a = BacktestEngine(base_spec).run()
                st.session_state[key_a] = bundle_a

            # --- Run B (comparator) if enabled ---
            bundle_b = None
            if cmp_spec is not None:
                key_b = ("bundle", _spec_key(cmp_spec))
                bundle_b = st.session_state.get(key_b)
                if bundle_b is None:
                    with st.spinner("Running comparator..."):
                        bundle_b = BacktestEngine(cmp_spec).run()
                    st.session_state[key_b] = bundle_b

            # --- Attach comparator artifacts for display WITHOUT mutating cached bundle_a ---
            display_bundle_a = bundle_a  # default

            if bundle_b is not None:
                sym = symbol  # same data in your compare mode
                label_a = f"{base_spec.strategy.kind} | {sym}"
                label_b = f"{cmp_spec.strategy.kind} | {sym}"

                analyzer = ResultsAnalyzer(periods_per_year=base_spec.periods_per_year, rf_annual=base_spec.rf_annual)
                comp = analyzer.compare(
                    report_a=bundle_a.report,
                    report_b=bundle_b.report,
                    label_a=label_a,
                    label_b=label_b,
                )

                # Make a deep copy of the WHOLE bundle for display
                # (so we can mutate nested dicts safely)
                display_key = ("bundle_display", _spec_key(base_spec), _spec_key(cmp_spec))
                display_bundle_a = bundle_a
                st.session_state[display_key] = display_bundle_a

                # Now mutate the COPY's nested objects (allowed)
                rep0 = display_bundle_a.report

                # build new dicts without mutating frozen fields
                new_meta = dict(getattr(rep0, "meta", None) or {})
                new_meta["cum_compare_labels"] = {"a": label_a, "b": label_b}

                new_plots = dict(getattr(rep0, "plots", None) or {})
                new_plots["cum_vs_bench"] = comp["cum_plot"]

                new_tables = dict(getattr(rep0, "tables", None) or {})
                new_tables["curve_vs_comparator"] = comp["curve_table"]

                # create a NEW report object (works even if frozen)
                rep1 = replace(rep0, meta=new_meta, plots=new_plots, tables=new_tables)

                # now update the BUNDLE with the new report (bundle may be frozen too)
                try:
                    display_bundle_a = replace(display_bundle_a, report=rep1)
                except Exception:
                    # if bundle is not a dataclass but a namedtuple or custom object:
                    if hasattr(display_bundle_a, "_replace"):
                        display_bundle_a = display_bundle_a._replace(report=rep1)
                    else:
                        # last resort: keep it in session and render using rep1 directly
                        # (but your render_bundle expects bundle.report, so this is rarely needed)
                        raise

                st.session_state[display_key] = display_bundle_a


            # --- Render ONLY ONCE: render the display bundle (copy if comparator on) ---
            render_bundle(display_bundle_a, port_cfg=base_spec.portfolio, label="")



        finally:
            if tmp_is_temp and tmp_path and os.path.exists(tmp_path):
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
    st.subheader("Optimize (rank by pnl then cagr)")
    # Baseline fixed values (used when NOT optimized)
    with st.expander("Fixed baseline (used when NOT optimized)", expanded=False):
        initial_cash0 = st.number_input("Initial cash (baseline)", min_value=1_000.0, value=100_000.0, step=10_000.0, key="opt_initial_cash")
        rebalance_policy0 = st.selectbox("Rebalance policy (baseline)", ["on_change", "every_bar"], index=0, key="opt_reb_policy")
        sizing_mode0 = st.selectbox("Sizing mode (baseline)", ["target_weight", "pct_cash_shares"], index=1, key="opt_sizing_mode")
        cooldown0 = st.number_input("cooldown_bars (baseline)", min_value=0, value=0, step=1, key="opt_cd0")
        buy0 = st.slider("buy_pct_cash (baseline)", 0.01, 1.00, 0.25, 0.01, key="opt_buy0")
        sell0 = st.slider("sell_pct_shares (baseline)", 0.01, 1.00, 1.00, 0.01, key="opt_sell0")

        nan_policy0 = "flat"
        st.markdown("### Strategy parameters (baseline)")
        if strategy_kind == "ma_cross":
            fast0 = st.number_input("fast_window (baseline)", min_value=2, max_value=500, value=20, step=1, key="opt_fast0")
            slow0 = st.number_input("slow_window (baseline)", min_value=3, max_value=500, value=50, step=1, key="opt_slow0")
            if fast0 >= slow0:
                st.warning("Baseline fast must be < slow. Auto-adjusting.")
                fast0 = min(int(fast0), int(slow0) - 1)
            strategy_params0 = {"sma_fast_window": int(fast0), "sma_slow_window": int(slow0), "allow_short": bool(allow_short), "nan_policy": nan_policy0}
        elif strategy_kind == "sma_price":
            w0 = st.number_input("window (baseline)", min_value=2, max_value=500, value=50, step=1, key="opt_w0")
            strategy_params0 = {"sma_window": int(w0), "allow_short": bool(allow_short), "nan_policy": nan_policy0}
        elif strategy_kind == "rsi":
            period = st.number_input("RSI period", min_value=2, max_value=200, value=14, step=1)
            low = st.number_input("RSI low", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
            high = st.number_input("RSI high", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            mode = st.selectbox("RSI mode", ["reversal", "momentum"], index=0)
            strategy_params0 = {"rsi_window": int(period), "rsi_oversold": float(low), "rsi_overbought": float(high), "mode": mode, "allow_short": bool(allow_short), "nan_policy": nan_policy0}

        elif strategy_kind == "macd":
            fast = st.number_input("MACD fast", min_value=2, max_value=200, value=12, step=1)
            slow = st.number_input("MACD slow", min_value=3, max_value=400, value=26, step=1)
            if fast >= slow:
                st.warning("Fast must be < Slow. Auto-adjusting fast.")
                fast = min(int(fast), int(slow) - 1)
            sig = st.number_input("MACD signal", min_value=2, max_value=200, value=9, step=1)
            trigger = st.selectbox("MACD trigger", ["cross", "zero"], index=0)
            strategy_params0 = {"macd_fast_window": int(fast), "macd_slow_window": int(slow), "macd_signal_window": int(sig), "trigger": trigger, "allow_short": bool(allow_short), "nan_policy": nan_policy0}
        elif strategy_kind == "bollinger":
            window = st.number_input("BB window (baseline)", min_value=2, max_value=500, value=20, step=1, key="opt_bb_window0")
            k = st.number_input("BB k (std dev multiplier) (baseline)", min_value=0.1, max_value=5.0, value=2.0, step=0.1, key="opt_bb_k0")
            strategy_params0 = {"bb_window": int(window), "bb_k": float(k), "allow_short": bool(allow_short), "nan_policy": nan_policy0}
        st.markdown("### Liquidity / Volume (baseline)")
        use_volume_gate0 = st.checkbox("Enable volume gate (baseline)", value=False, key="opt_use_volume_gate")
        volume_gate_kind0 = st.selectbox(
            "Gate type (baseline)",
            ["min_abs", "min_ratio_adv"],
            index=0,
            disabled=not use_volume_gate0,
            key="opt_volume_gate_kind",
        )

        min_volume_abs0 = st.number_input(
            "Min Volume (abs shares) baseline",
            min_value=0.0,
            value=0.0,
            step=10_000.0,
            disabled=(not use_volume_gate0 or volume_gate_kind0 != "min_abs"),
            key="opt_min_volume_abs",
        )
        min_volume_ratio_adv0 = st.slider(
            "Min Volume / ADV ratio baseline",
            0.0, 2.0, 0.3, 0.05,
            disabled=(not use_volume_gate0 or volume_gate_kind0 != "min_ratio_adv"),
            key="opt_min_volume_ratio_adv",
        )
        volume_gate_adv_window0 = st.number_input(
            "ADV window for gate baseline",
            min_value=1,
            value=20,
            step=1,
            disabled=(not use_volume_gate0 or volume_gate_kind0 != "min_ratio_adv"),
            key="opt_volume_gate_adv_window",
        )
        use_participation_cap0 = st.checkbox("Enable participation cap (baseline)", value=False, key="opt_use_participation_cap")
        participation_rate0 = st.slider(
            "Participation rate baseline",
            0.001, 0.50, 0.05, 0.001,
            disabled=not use_participation_cap0,
            key="opt_participation_rate",
        )
        participation_basis0 = st.selectbox(
            "Participation basis baseline",
            ["bar", "adv"],
            index=0,
            disabled=not use_participation_cap0,
            key="opt_participation_basis",
        )
        adv_window0 = st.number_input(
            "ADV window (for cap) baseline",
            min_value=1,
            value=20,
            step=1,
            disabled=(not use_participation_cap0 or participation_basis0 != "adv"),
            key="opt_adv_window",
        )


        st.markdown("### Costs (baseline)")
        apply_costs0 = st.checkbox("Apply costs", value=False, key="opt_apply_costs")
        if apply_costs0:
            brokerage_bps0 = st.number_input("Brokerage (bps)", value=60.0, step=1.0, key="opt_brok")
            comm_bourse_bps0 = st.number_input("Commission de la BdC (bps)", value=10.0, step=1.0, key="opt_exch")
            reg_liv_bps0 = st.number_input("Frais Règlement-Livraison (bps)", value=20.0, step=1.0, key="opt_settle")
            tva_rate0 = st.number_input("tva", value=0.10, step=0.01, key="opt_tva")
            slippage_bps0 = st.number_input("Slippage (bps)", value=0.0, step=1.0, key="opt_slip")
        else:
            brokerage_bps0 = comm_bourse_bps0 = reg_liv_bps0 = slippage_bps0 = 0.0
            tva_rate0 = 0.0

    # Build catalog for THIS strategy_kind
    catalog = default_param_catalog(strategy_kind)
    selectable_keys = list(catalog.keys())

    default_active = (
        ["strategy.sma_fast_window", "strategy.sma_slow_window"] if strategy_kind == "ma_cross"
        else ["strategy.sma_window"] if strategy_kind == "sma_price"
        else ["strategy.rsi_window", "strategy.rsi_oversold", "strategy.rsi_overbought"] if strategy_kind == "rsi"
        else ["strategy.macd_fast_window", "strategy.macd_slow_window", "strategy.macd_signal_window"] if strategy_kind == "macd"
        else ["strategy.bb_window", "strategy.bb_k"] if strategy_kind == "bollinger"
        else []
    )

    active_keys = st.multiselect(
        "Select parameters to optimize",
        options=selectable_keys,
        default=default_active,
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

            mode = st.radio(
                "Domain mode",
                ["range", "manual list"],
                index=0,
                horizontal=True,
                key=f"{k}_mode",
            )

            if mode == "range":
                c1, c2, c3 = st.columns(3)
                lo2 = c1.number_input(f"{k} min", value=int(lo), step=1, key=f"{k}_min")
                hi2 = c2.number_input(f"{k} max", value=int(hi), step=1, key=f"{k}_max")
                step2 = c3.number_input(f"{k} step", value=int(step), step=1, key=f"{k}_step")
                if lo2 > hi2:
                    st.error(f"{k}: min must be <= max")
                    st.stop()
                edited_catalog[k] = replace(pdef, domain=(int(lo2), int(hi2), int(step2)))

            else:
                # manual list -> we convert to a CHOICE domain so optimizer uses exactly those values
                default_txt = f"{int(lo)},{int((lo+hi)//2)},{int(hi)}"
                txt = st.text_input(
                    f"{k} values (comma/space separated)",
                    value=st.session_state.get(f"{k}_manual", default_txt),
                    key=f"{k}_manual",
                )
                try:
                    vals = parse_int_list(txt)
                except Exception as e:
                    st.error(f"{k}: invalid list: {e}")
                    st.stop()

                if not vals:
                    st.error(f"{k}: please provide at least one value.")
                    st.stop()

                edited_catalog[k] = replace(pdef, kind="choice", domain=vals)


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
    top_k = st.number_input("Show top K", min_value=5, value=30, step=5, key="opt_topk")

    if method == "random":
        n_trials = st.number_input("Trials", min_value=10, value=200, step=10, key="opt_trials")
    else:
        n_trials = 0

    BATCH_PERIODS = dict(MASI_PERIODS)  # start with MASI
    if symbol.upper() == "IAM":
        # namespace IAM keys to prevent accidental collisions
        BATCH_PERIODS.update({f"IAM — {k}": v for k, v in IAM_PERIODS.items()})


    st.subheader("Batch tests")

    do_batch = st.checkbox("Batch test by period", value=False)

    selected_periods = []
    if do_batch:
        period_options = list(BATCH_PERIODS.keys())
        selected_periods = st.multiselect(
            "Select periods to run",
            options=period_options,
            default=period_options,
        )

        batch_objective = st.selectbox(
            "Objective for comparison",
            ["pnl", "cagr"],  # match what stats dict returns
            index=0
        )

        optimize_within_each = st.checkbox("Optimize within each period", value=True)


    run_opt = st.button("Run optimization", key="run_opt_btn")


    if run_opt:
        tmp_path = None
        tmp_dir = None
        tmp_is_temp = True
        bench_tmp_path = None
        bench_tmp_dir = None
        bench_is_temp = True
        try:
            if use_benchmark and bench_source_key == "bmce":
                if bench_bmce_file is None:
                    st.warning("Benchmark enabled but no BMCE benchmark file uploaded.")
                    st.stop()

                suffix_b = Path(bench_bmce_file.name).suffix.lower()
                bench_is_temp = False
                bench_tmp_path = st.session_state.get("bench_cached_path")
                if not bench_tmp_path:
                    st.error("Benchmark BMCE file not persisted. Re-upload the benchmark file.")
                    st.stop()
            if source_key == "bmce":
                # Use stable persisted path (content-hash) to enable caching / speed
                tmp_is_temp = False
                tmp_path = st.session_state.get("bmce_cached_path")
                if not tmp_path:
                    st.error("BMCE file not persisted. Re-upload the file.")
                    st.stop()

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
                comm_bourse_bps=float(comm_bourse_bps0),
                reg_liv_bps=float(reg_liv_bps0),
                slippage_bps=float(slippage_bps0),
                tva_rate=float(tva_rate0),
            )

            base_spec = make_base_spec(
                source_key=source_key,
                symbol=symbol,
                timezone=timezone,
                interval=interval,
                bmce_tmp_path=tmp_path,
                start=start_str,
                end=end_str,
                include_windows=include_windows,
                exclude_windows=exclude_windows,
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
                use_volume_gate=bool(use_volume_gate0),
                volume_gate_kind=str(volume_gate_kind0),
                min_volume_abs=float(min_volume_abs0),
                min_volume_ratio_adv=float(min_volume_ratio_adv0),
                volume_gate_adv_window=int(volume_gate_adv_window0),

                use_participation_cap=bool(use_participation_cap0),
                participation_rate=float(participation_rate0),
                participation_basis=str(participation_basis0),
                adv_window=int(adv_window0),
                cost_model=cost_model,
            )

            opt_cfg = OptimizeConfig(
                method=str(method),
                seed=42,
                n_trials=int(n_trials) if method == "random" else 0,
                top_k=int(top_k),
                # NOTE: no "objective" field in your OptimizeConfig
            )
            if do_batch and selected_periods:
                df_batch = batch_optimize_by_period(
                    base_spec=base_spec,
                    active_params=active_params,
                    cfg=opt_cfg,
                    periods=BATCH_PERIODS,
                    selected_period_labels=selected_periods,
                    objective=batch_objective,
                )

                df_batch_sorted = df_batch.sort_values("objective_value", ascending=False).reset_index(drop=True)
                st.dataframe(df_batch_sorted, use_container_width=True)

                if df_batch_sorted.empty:
                    st.error("Batch optimization returned no results.")
                    st.stop()

                # ---- winner row ----
                winner = df_batch_sorted.iloc[0]
                st.subheader("Batch winner (best period + best params)")
                st.json({
                    "period": winner.get("period"),
                    "start": winner.get("start"),
                    "end": winner.get("end"),
                    "objective": winner.get("objective"),
                    "objective_value": float(winner.get("objective_value", np.nan)),
                })

                # ---- build winner spec (params + forced period) ----
                # 1) apply params from row
                winner_spec = build_spec_from_result_row(base_spec, winner)

                # 2) FORCE the period start/end from the batch row (don’t rely on params)
                winner_spec = replace(
                    winner_spec,
                    data=replace(
                        winner_spec.data,
                        start=str(winner.get("start")),
                        end=str(winner.get("end")),
                        include_windows=None,
                        exclude_windows=None,
                    )
                )

                # ---- backtest winner spec ----
                st.divider()
                with st.spinner("Backtesting batch winner..."):
                    winner_bundle = BacktestEngine(winner_spec).run()

                # Persist so the “loaded from session cache” section works
                st.session_state["opt_best_spec"] = winner_spec
                st.session_state["opt_best_bundle"] = winner_bundle
                st.session_state["opt_top_df"] = df_batch_sorted
                st.session_state["opt_best"] = None

                render_bundle(winner_bundle, port_cfg=winner_spec.portfolio)

                # CRITICAL: stop so the global run_optimization below doesn't overwrite the batch winner
                st.stop()

            with st.spinner("Running optimization..."):
                best, top_df, best_params, best_spec, ranked_df = run_optimization(
                    base_spec=base_spec,
                    active_params=active_params,
                    cfg=opt_cfg,
                )

            # ✅ Run best main backtest NOW so we have `best_bundle` available for comparison
            with st.spinner("Backtesting best main strategy..."):
                best_bundle = BacktestEngine(best_spec).run()

            st.session_state["opt_best_bundle"] = best_bundle  # keep your persistence
            # =============================
            # Best-vs-best comparator in OPTIMIZE
            # =============================
            cmp_best_spec = None
            cmp_bundle = None

            if compare_mode == "Another strategy (same data)" and compare_strategy_kind:
                # 1) build comparator base spec (same data/portfolio, comparator strategy)
                # You need baseline params for comparator; simplest: empty dict -> strategy defaults inside engine
                cmp_strategy_params0 = {}  # TODO optionally build baseline UI per strategy kind

                cmp_base_spec = make_base_spec(
                    source_key=source_key,
                    symbol=symbol,
                    timezone=timezone,
                    interval=interval,
                    bmce_tmp_path=tmp_path,
                    start=start_str,
                    end=end_str,
                    include_windows=include_windows,
                    exclude_windows=exclude_windows,
                    yf_period=yf_period,
                    yf_interval=yf_interval,
                    yf_auto_adjust=yf_auto_adjust,
                    strategy_kind=str(compare_strategy_kind),
                    strategy_params=cmp_strategy_params0,
                    allow_short=bool(allow_short),
                    initial_cash=float(initial_cash0),
                    rebalance_policy=str(rebalance_policy0),
                    sizing_mode=str(sizing_mode0),
                    buy_pct_cash=float(buy0),
                    sell_pct_shares=float(sell0),
                    cooldown_bars=int(cooldown0),
                    use_volume_gate=bool(use_volume_gate0),
                    volume_gate_kind=str(volume_gate_kind0),
                    min_volume_abs=float(min_volume_abs0),
                    min_volume_ratio_adv=float(min_volume_ratio_adv0),
                    volume_gate_adv_window=int(volume_gate_adv_window0),
                    use_participation_cap=bool(use_participation_cap0),
                    participation_rate=float(participation_rate0),
                    participation_basis=str(participation_basis0),
                    adv_window=int(adv_window0),
                    cost_model=cost_model,
                )

                # 2) define which parameters to optimize for comparator strategy
                cmp_catalog = default_param_catalog(str(compare_strategy_kind))

                # simplest: optimize comparator "default_active" like you did for main
                cmp_default_active = (
                    ["strategy.sma_fast_window", "strategy.sma_slow_window"] if compare_strategy_kind == "ma_cross"
                    else ["strategy.sma_window"] if compare_strategy_kind == "sma_price"
                    else ["strategy.rsi_window", "strategy.rsi_oversold", "strategy.rsi_overbought"] if compare_strategy_kind == "rsi"
                    else ["strategy.macd_fast_window", "strategy.macd_slow_window", "strategy.macd_signal_window"] if compare_strategy_kind == "macd"
                    else ["strategy.bb_window", "strategy.bb_k"] if compare_strategy_kind == "bollinger"
                    else []
                )

                cmp_active_params = []
                for k in cmp_default_active:
                    if k in cmp_catalog:
                        p = cmp_catalog[k]
                        if p.enabled and p.domain is not None and (p.kind != "date_window" or len(p.domain) > 0):
                            cmp_active_params.append(p)

                # 3) optimize comparator
                with st.spinner("Optimizing comparator strategy (best-vs-best)..."):
                    cmp_best, cmp_top_df, cmp_best_params, cmp_best_spec, cmp_ranked_df = run_optimization(
                        base_spec=cmp_base_spec,
                        active_params=cmp_active_params,
                        cfg=opt_cfg,
                    )

                # 4) run backtest on BOTH best specs
                with st.spinner("Backtesting best comparator..."):
                    cmp_bundle = BacktestEngine(cmp_best_spec).run()

                # 5) compare best-vs-best (same analyzer.compare you already wrote)
                analyzer = ResultsAnalyzer(periods_per_year=base_spec.periods_per_year, rf_annual=base_spec.rf_annual)
                comp = analyzer.compare(
                    report_a=best_bundle.report,  # ✅ now defined
                    report_b=cmp_bundle.report,
                    label_a=f"{best_spec.strategy.kind} | {symbol}",
                    label_b=f"{cmp_best_spec.strategy.kind} | {symbol}",
                )

                # inject into display bundle using replace() (frozen-safe)
                rep0 = best_bundle.report
                new_meta = dict(getattr(rep0, "meta", None) or {})
                new_meta["cum_compare_labels"] = {"a": f"{best_spec.strategy.kind} | {symbol}", "b": f"{cmp_best_spec.strategy.kind} | {symbol}"}
                new_plots = dict(getattr(rep0, "plots", None) or {})
                new_plots["cum_vs_bench"] = comp["cum_plot"]
                new_tables = dict(getattr(rep0, "tables", None) or {})
                new_tables["curve_vs_comparator"] = comp["curve_table"]
                best_bundle = replace(best_bundle, report=replace(rep0, meta=new_meta, plots=new_plots, tables=new_tables))


            st.subheader("Best result")
            st.json({
                "pnl": best.pnl,
                "cagr": best.cagr,
                "efficiency": best.efficiency,
                "n_fills": best.n_fills,
                "params": best.params,
                "error": best.error,
            })

            st.subheader("Top candidates")

            # Always hide traded_notional if present
            if "traded_notional" in top_df.columns:
                top_df = top_df.drop(columns=["traded_notional"])

            # Build show_cols AFTER dropping
            core = [c for c in ["cagr", "pnl", "n_fills", "error"] if c in top_df.columns]
            rest = [c for c in top_df.columns if c not in set(core + ["pnl"])]  # optionally hide pnl too
            show_cols = core + rest

            st.dataframe(top_df[show_cols], use_container_width=True)


            if ranked_df is not None and isinstance(ranked_df, pd.DataFrame) and (not ranked_df.empty):
                best5_df = ranked_df.head(5)
                worst5_df = ranked_df.tail(5).sort_values(["pnl","cagr"], ascending=[True, True]).reset_index(drop=True)
                mid_start = max(0, (len(ranked_df) // 2) - 2)
                mid5_df = ranked_df.iloc[mid_start: mid_start + 5].reset_index(drop=True)

                tab_best, tab_best5, tab_mid5, tab_worst5 = st.tabs(["Best", "Best 5", "Mid 5", "Worst 5"])

                def _candidate_selector(df_in: pd.DataFrame, label: str, key_prefix: str):
                    st.dataframe(df_in, use_container_width=True)
                    picked = st.selectbox(
                        f"Select candidate row ({label})",
                        options=list(range(len(df_in))),
                        index=0,
                        key=f"{key_prefix}_pick",
                    )
                    row = df_in.iloc[int(picked)]
                    spec_i = build_spec_from_result_row(base_spec, row)

                    if st.button(f"Run backtest + ledger for {label} #{int(picked)+1}", key=f"{key_prefix}_run"):
                        bundle_a = BacktestEngine(spec_i).run()
                        bundle_b = None
                        if cmp_spec is not None:
                            bundle_b = BacktestEngine(cmp_spec).run()

                        st.dataframe(bundle_a.report.tables.get("trade_ledger", pd.DataFrame()), use_container_width=True)
                        st.dataframe(bundle_a.report.tables.get("trades", pd.DataFrame()), use_container_width=True)
                        st.dataframe(bundle_a.report.tables.get("trade_performance", pd.DataFrame()), use_container_width=True)

                        if bundle_b is not None:
                            st.subheader(f"Compare with {cmp_spec.strategy.kind}")
                            st.dataframe(bundle_b.report.tables.get("trade_ledger", pd.DataFrame()), use_container_width=True)
                            st.dataframe(bundle_b.report.tables.get("trades", pd.DataFrame()), use_container_width=True)
                            st.dataframe(bundle_b.report.tables.get("trade_performance", pd.DataFrame()), use_container_width=True)

                with tab_best:
                    _candidate_selector(ranked_df.head(1).reset_index(drop=True), "Best", "opt_best")
                with tab_best5:
                    _candidate_selector(best5_df, "Best 5", "opt_best5")
                with tab_mid5:
                    _candidate_selector(mid5_df, "Mid 5", "opt_mid5")
                with tab_worst5:
                    _candidate_selector(worst5_df, "Worst 5", "opt_worst5")

            
            st.divider()
            # Auto-run the best configuration backtest (no extra button)
            with st.spinner("Running best backtest..."):
                bundle = BacktestEngine(best_spec).run()
                # ------------------------------------------------------------
                # Attach comparator (Optimize tab)
                # ------------------------------------------------------------
                if compare_mode != "None":
                    label_a = f"{best_spec.strategy.kind} | {symbol}"

                    if compare_mode == "Buy & Hold (same data)":
                        cmp_spec = EngineSpec(
                            data=best_spec.data,
                            indicators=best_spec.indicators,
                            strategy=StrategyConfig(kind="buy_hold", params={"buy_pct_cash": 1.0}),
                            portfolio=best_spec.portfolio,
                            benchmark=BenchmarkConfig(enabled=False),
                            plot_indicators=[],
                            periods_per_year=best_spec.periods_per_year,
                            rf_annual=best_spec.rf_annual,
                        )
                        with st.spinner("Backtesting buy&hold comparator..."):
                            cmp_bundle = BacktestEngine(cmp_spec).run()

                        bundle = attach_comparator_to_bundle(
                            bundle, cmp_bundle,
                            label_a=label_a,
                            label_b=f"buy_hold | {symbol}",
                            periods_per_year=best_spec.periods_per_year,
                            rf_annual=best_spec.rf_annual,
                        )

                    elif compare_mode == "Another strategy (same data)":
                        if (cmp_bundle is not None) and (cmp_best_spec is not None):
                            bundle = attach_comparator_to_bundle(
                                bundle, cmp_bundle,
                                label_a=label_a,
                                label_b=f"{cmp_best_spec.strategy.kind} | {symbol}",
                                periods_per_year=best_spec.periods_per_year,
                                rf_annual=best_spec.rf_annual,
                            )

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
                        bundle = replace(bundle, report=report)


                    except Exception as e:
                        st.warning(f"Benchmark could not be applied: {e}")

            # Persist bundle so UI doesn't 'lose' it on rerun
            st.session_state["opt_best_bundle"] = bundle

            # Show the fills ledger (one row per execution) with net_invested
            fills_df = getattr(bundle, "report", None).tables.get("trades", pd.DataFrame()) if getattr(bundle, "report", None) is not None else pd.DataFrame()
            if fills_df is not None and not fills_df.empty and "net_invested" in fills_df.columns:
                st.subheader("Best strategy fills ledger (net_invested)")
                show_cols = [c for c in ["timestamp","symbol","side","qty","price","notional","cost","net_invested","cash_after"] if c in fills_df.columns]
                st.dataframe(fills_df[show_cols], use_container_width=True)

            render_bundle(best_bundle, port_cfg=best_spec.portfolio)

        finally:
            if bench_is_temp and bench_tmp_path and os.path.exists(bench_tmp_path):
                try:
                    os.remove(bench_tmp_path)
                except OSError:
                    pass
            if bench_tmp_dir and os.path.isdir(bench_tmp_dir):
                try:
                    os.rmdir(bench_tmp_dir)
                except OSError:
                    pass
            if tmp_is_temp and tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if tmp_dir and os.path.isdir(tmp_dir):
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
    # ------------------------------------------------------------
    # Persisted optimization display (survives Streamlit reruns)
    # ------------------------------------------------------------
    if (not run_opt) and (st.session_state.get("opt_best_bundle") is not None):
        best = st.session_state.get("opt_best", None)
        best_spec = st.session_state.get("opt_best_spec", None)
        top_df = st.session_state.get("opt_top_df", None)
        bundle = st.session_state.get("opt_best_bundle")

        st.success("Optimization finished (loaded from session cache).")

        if best is not None:
            st.subheader("Best result")
            st.json({
                "pnl": getattr(best, "pnl", None),
                "n_fills": getattr(best, "n_fills", None),
                "params": getattr(best, "params", None),
                "error": getattr(best, "error", None),
            })

        if isinstance(top_df, pd.DataFrame):
            st.subheader("Top candidates")
            show_cols = [c for c in ["pnl","cagr","n_fills","error"] if c in top_df.columns] + \
                        [c for c in top_df.columns if c not in ("pnl","cagr","n_fills","error","traded_notional")]
            # Hide traded_notional from UI if present
            if "traded_notional" in top_df.columns:
                top_df = top_df.drop(columns=["traded_notional"])
            st.dataframe(top_df[show_cols], use_container_width=True)
            fills_df = getattr(bundle, "report", None).tables.get("trades", pd.DataFrame()) if getattr(bundle, "report", None) is not None else pd.DataFrame()
            if fills_df is not None and not fills_df.empty and "net_invested" in fills_df.columns:
                st.subheader("Best strategy fills ledger (net_invested)")
                show_cols = [c for c in ["timestamp","symbol","side","qty","price","notional","cost","net_invested","cash_after"] if c in fills_df.columns]
                st.dataframe(fills_df[show_cols], use_container_width=True)

            render_bundle(bundle, port_cfg=(best_spec.portfolio if best_spec is not None else None))

