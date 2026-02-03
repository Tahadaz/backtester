# optimize.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable
import itertools
import math
import random

import numpy as np
import pandas as pd

from data import BMCEDataSource, YahooFinanceDataSource, MarketData
from indicators import IndicatorEngine, FeatureSpec, FeaturesData
from strategy import SignalFrame
from engine import EngineSpec, DataConfig, StrategyConfig
from portfolio import PortfolioEngine, PortfolioConfig


# ============================================================
# Public dataclasses
# ============================================================

@dataclass(frozen=True)
class OptimizeConfig:
    method: str = "random"        # "random" | "grid"
    seed: int = 42
    n_trials: int = 300           # for random
    top_k: int = 30

    # indicator engine cache options (optimization usually wants memory cache only)
    feature_cache_dir: str = ".cache/features"
    enable_disk_cache: bool = False
    enable_memory_cache: bool = True


@dataclass(frozen=True)
class ParamDef:
    """
    key:
      - "strategy.fast_window", "strategy.slow_window", "strategy.window"
      - "portfolio.cooldown_bars", "portfolio.buy_pct_cash", "portfolio.sell_pct_shares"
      - "data.window" -> tuple(start,end) strings

    kind:
      - "int" | "float" | "choice" | "date_window"

    domain:
      - int:   (lo, hi, step)
      - float: (lo, hi, step)
      - choice: [v1, v2, ...]
      - date_window: [(start, end), ...]
    """
    key: str
    kind: str
    domain: Any
    cast: Callable[[Any], Any] = lambda x: x
    enabled: bool = True


@dataclass(frozen=True)
class TrialResult:
    params: Dict[str, Any]
    pnl: float
    traded_notional: float
    efficiency: float
    n_fills: int
    error: Optional[str] = None


# ============================================================
# Strategy adapter registry (fast + strategy-agnostic optimization)
# ============================================================

@dataclass(frozen=True)
class StrategyAdapter:
    kind: str

    def required_sma_windows(
        self,
        base_spec: EngineSpec,
        active_params: List[ParamDef],
    ) -> List[int]:
        raise NotImplementedError

    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        raise NotImplementedError

    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        return True, None


class MACrossAdapter(StrategyAdapter):
    def __init__(self):
        super().__init__(kind="ma_cross")

    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        f = int(params.get("strategy.fast_window", base_spec.strategy.params.get("fast_window", 15)))
        s = int(params.get("strategy.slow_window", base_spec.strategy.params.get("slow_window", 50)))
        if f >= s:
            return False, "fast_window must be < slow_window"
        if f <= 0 or s <= 0:
            return False, "windows must be positive"
        return True, None

    def required_sma_windows(self, base_spec: EngineSpec, active_params: List[ParamDef]) -> List[int]:
        # union of domains + base values (so we can evaluate even if a param isn't optimized)
        f0 = int(base_spec.strategy.params.get("fast_window", 15))
        s0 = int(base_spec.strategy.params.get("slow_window", 50))
        ws = {f0, s0}

        dom_fast = _domain_values_int(active_params, "strategy.fast_window")
        dom_slow = _domain_values_int(active_params, "strategy.slow_window")
        ws.update(dom_fast)
        ws.update(dom_slow)

        return sorted(w for w in ws if w is not None)

    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        f = int(params.get("strategy.fast_window", base_spec.strategy.params.get("fast_window", 15)))
        s = int(params.get("strategy.slow_window", base_spec.strategy.params.get("slow_window", 50)))

        allow_short = bool(base_spec.strategy.params.get("allow_short", False))
        # allow overriding allow_short if the app exposes it as a choice param
        if "strategy.allow_short" in params:
            allow_short = bool(params["strategy.allow_short"])

        nan_policy = str(base_spec.strategy.params.get("nan_policy", "flat"))
        if "strategy.nan_policy" in params:
            nan_policy = str(params["strategy.nan_policy"])

        col_fast = f"sma_{f}"
        col_slow = f"sma_{s}"

        sig = pd.DataFrame(index=index, columns=symbols, dtype="float64")
        valid = pd.DataFrame(index=index, columns=symbols, dtype="bool")

        for sym in symbols:
            fast = bank[sym][col_fast]
            slow = bank[sym][col_slow]

            v = (~np.isnan(fast)) & (~np.isnan(slow))
            long_mask = fast > slow

            if allow_short:
                short_mask = fast < slow
                out = np.zeros(len(index), dtype=np.float64)
                out[long_mask] = 1.0
                out[short_mask] = -1.0
            else:
                out = long_mask.astype(np.float64)

            if nan_policy == "flat":
                out = np.where(v, out, 0.0)
            else:
                out = np.where(v, out, np.nan)

            sig[sym] = out
            valid[sym] = v

        return SignalFrame(
            signals=sig,
            validity=valid,
            meta={"adapter": "ma_cross", "fast_window": f, "slow_window": s, "allow_short": allow_short, "nan_policy": nan_policy},
        )


class PriceAboveSMAAdapter(StrategyAdapter):
    def __init__(self):
        super().__init__(kind="sma_price")

    def required_sma_windows(self, base_spec: EngineSpec, active_params: List[ParamDef]) -> List[int]:
        w0 = int(base_spec.strategy.params.get("window", 50))
        ws = {w0}
        dom_w = _domain_values_int(active_params, "strategy.window")
        ws.update(dom_w)
        return sorted(w for w in ws if w is not None)

    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        w = int(params.get("strategy.window", base_spec.strategy.params.get("window", 50)))
        if w <= 0:
            return False, "window must be positive"
        return True, None

    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        w = int(params.get("strategy.window", base_spec.strategy.params.get("window", 50)))

        allow_short = bool(base_spec.strategy.params.get("allow_short", False))
        if "strategy.allow_short" in params:
            allow_short = bool(params["strategy.allow_short"])

        nan_policy = str(base_spec.strategy.params.get("nan_policy", "flat"))
        if "strategy.nan_policy" in params:
            nan_policy = str(params["strategy.nan_policy"])

        col_sma = f"sma_{w}"

        sig = pd.DataFrame(index=index, columns=symbols, dtype="float64")
        valid = pd.DataFrame(index=index, columns=symbols, dtype="bool")

        for sym in symbols:
            close = bars_close[sym]
            sma = bank[sym][col_sma]
            v = (~np.isnan(close)) & (~np.isnan(sma))

            long_mask = close > sma
            if allow_short:
                short_mask = close < sma
                out = np.zeros(len(index), dtype=np.float64)
                out[long_mask] = 1.0
                out[short_mask] = -1.0
            else:
                out = long_mask.astype(np.float64)

            if nan_policy == "flat":
                out = np.where(v, out, 0.0)
            else:
                out = np.where(v, out, np.nan)

            sig[sym] = out
            valid[sym] = v

        return SignalFrame(
            signals=sig,
            validity=valid,
            meta={"adapter": "sma_price", "window": w, "allow_short": allow_short, "nan_policy": nan_policy},
        )


STRATEGY_ADAPTERS: Dict[str, StrategyAdapter] = {
    "ma_cross": MACrossAdapter(),
    "sma_price": PriceAboveSMAAdapter(),
}


# ============================================================
# Catalog helpers (optional but useful for Streamlit UI)
# ============================================================

def default_param_catalog(strategy_kind: str) -> Dict[str, ParamDef]:
    cat: Dict[str, ParamDef] = {}

    if strategy_kind == "ma_cross":
        cat["strategy.fast_window"] = ParamDef("strategy.fast_window", "int", (5, 60, 1), int)
        cat["strategy.slow_window"] = ParamDef("strategy.slow_window", "int", (20, 250, 1), int)
      

    elif strategy_kind == "sma_price":
        cat["strategy.window"] = ParamDef("strategy.window", "int", (10, 250, 1), int)
       
    # portfolio knobs you mentioned
    cat["portfolio.cooldown_bars"] = ParamDef("portfolio.cooldown_bars", "int", (0, 30, 1), int)
    cat["portfolio.buy_pct_cash"] = ParamDef("portfolio.buy_pct_cash", "float", (0.05, 1.0, 0.05), float)
    cat["portfolio.sell_pct_shares"] = ParamDef("portfolio.sell_pct_shares", "float", (0.05, 1.0, 0.05), float)

    return cat


# ============================================================
# Core optimization
# ============================================================

def run_optimization(
    base_spec: EngineSpec,
    active_params: List[ParamDef],
    cfg: OptimizeConfig,
) -> Tuple[TrialResult, pd.DataFrame, Dict[str, Any], EngineSpec, pd.DataFrame]:
    """
    Fast optimizer:
      - load MarketData once
      - precompute required features once (union of needed SMA windows)
      - per trial: generate signals from precomputed arrays (adapter), apply cooldown via PortfolioConfig, run portfolio stats fast

    Returns:
      best_result, top_df, best_params, best_spec
    """
    if not active_params:
        raise ValueError("active_params is empty; nothing to optimize.")

    strategy_kind = base_spec.strategy.kind.lower()
    if strategy_kind not in STRATEGY_ADAPTERS:
        raise ValueError(f"No StrategyAdapter registered for strategy kind '{strategy_kind}'")

    adapter = STRATEGY_ADAPTERS[strategy_kind]

    # 1) Load data once (respect base_spec.data start/end to define the available universe)
    md_full = _load_market_data_from_spec(base_spec.data)

    symbols = list(base_spec.data.symbols)
    if not symbols:
        raise ValueError("No symbols in base_spec.data.symbols")

    # 2) Build a common index once (inner intersection across symbols for robustness)
    #    This ensures arrays are aligned and portfolio doesn't hit missing timestamps.
    md = _align_marketdata_inner(md_full, symbols)

    common_index = md.bars[symbols[0]].index
    if len(common_index) < 2:
        raise ValueError("Not enough bars after alignment; need at least 2 timestamps.")

    # 3) Precompute arrays once (ALWAYS) + SMA bank (IF NEEDED)
    sma_windows = adapter.required_sma_windows(base_spec, active_params)

    def _sma_bank_numpy(close: np.ndarray, windows: List[int]) -> Dict[str, np.ndarray]:
        n = close.size
        csum = np.cumsum(np.insert(close.astype(np.float64, copy=False), 0, 0.0))
        out: Dict[str, np.ndarray] = {}
        for w0 in windows:
            w = int(w0)
            sma = np.full(n, np.nan, dtype=np.float64)
            if 0 < w <= n:
                sma[w - 1 :] = (csum[w:] - csum[:-w]) / w
            out[f"sma_{w}"] = sma
        return out

    # --- aligned price arrays ONCE ---
    bars_open: Dict[str, np.ndarray] = {}
    bars_close: Dict[str, np.ndarray] = {}

    for s in symbols:
        b = md.bars[s].reindex(common_index)  # aligned already; reindex ok & explicit
        bars_open[s] = b["Open"].to_numpy(dtype=np.float64, copy=False)
        bars_close[s] = b["Close"].to_numpy(dtype=np.float64, copy=False)

    # --- SMA bank (optional) ---
    bank: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in symbols}
    if sma_windows:
        uniq = sorted(set(int(w) for w in sma_windows))
        bank = {s: _sma_bank_numpy(bars_close[s], uniq) for s in symbols}


        



    # bars close arrays (needed for sma_price; also convenient)
    bars_close: Dict[str, np.ndarray] = {}
    for s in symbols:
        bars_close[s] = md.bars[s]["Close"].reindex(common_index).to_numpy(dtype=np.float64)

    # 4) Candidate iterator
    method = cfg.method.lower()
    if method == "grid":
        candidates = _iter_grid([p for p in active_params if p.enabled])
    elif method == "random":
        candidates = _iter_random([p for p in active_params if p.enabled], n_trials=int(cfg.n_trials), seed=int(cfg.seed))
    else:
        raise ValueError(f"Unknown optimization method: {cfg.method}")

    # 5) Evaluate
    results: List[TrialResult] = []
    for params in candidates:
        r = _eval_one_trial(
            base_spec=base_spec,
            md=md,
            common_index=common_index,
            bank=bank,
            bars_open=bars_open,      # NEW
            bars_close=bars_close,
            adapter=adapter,
            params=params,
        )

        results.append(r)

    df = pd.DataFrame([{
        **r.params,
        "pnl": r.pnl,
        "traded_notional": r.traded_notional,
        "efficiency": r.efficiency,
        "n_fills": r.n_fills,
        "error": r.error,
    } for r in results])

    # rank valid rows by (pnl desc, efficiency desc)
    df_valid = df[df["error"].isna()].copy()
    if df_valid.empty:
        ranked_df = df.sort_values(["pnl", "efficiency"], ascending=[False, False]).reset_index(drop=True)
        top_df = ranked_df.head(int(cfg.top_k)).reset_index(drop=True)
        best = ranked_df.iloc[0]
        best_spec = _apply_params_to_spec(base_spec, best.params)
        return best, top_df, best.params, best_spec

    ranked_df = df_valid.sort_values(["pnl", "efficiency"], ascending=[False, False]).reset_index(drop=True)
    df_valid = df_valid.sort_values(["pnl", "efficiency"], ascending=[False, False])
    top_df = ranked_df.head(int(cfg.top_k)).reset_index(drop=True)

    best_row = top_df.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in best_row.keys() if k not in ("pnl", "traded_notional", "efficiency", "n_fills", "error")}
    best = TrialResult(
        params=best_params,
        pnl=float(best_row["pnl"]),
        traded_notional=float(best_row["traded_notional"]),
        efficiency=float(best_row["efficiency"]),
        n_fills=int(best_row["n_fills"]),
        error=None,
    )
    best_spec = _apply_params_to_spec(base_spec, best.params)
    return best, top_df, best_params, best_spec, ranked_df


def build_spec_from_result_row(base_spec: EngineSpec, row: Any) -> EngineSpec:
    if isinstance(row, pd.Series):
        d = row.to_dict()
    else:
        d = dict(row)

    metric_cols = {"pnl", "traded_notional", "efficiency", "n_fills", "error"}
    params = {k: v for k, v in d.items() if k not in metric_cols}
    return _apply_params_to_spec(base_spec, params)
# ============================================================
# Trial evaluation
# ============================================================
def _eval_one_trial(
    base_spec: EngineSpec,
    md: MarketData,
    common_index: pd.DatetimeIndex,
    bank: Dict[str, Dict[str, np.ndarray]],
    bars_open: Dict[str, np.ndarray],
    bars_close: Dict[str, np.ndarray],
    adapter: StrategyAdapter,
    params: Dict[str, Any],
) -> TrialResult:
    try:
        symbols = list(base_spec.data.symbols)
        if len(symbols) != 1:
            # Fallback to your old logic (multi-symbol) for now
            return _eval_one_trial_slow_pandas(
                base_spec=base_spec,
                md=md,
                common_index=common_index,
                bank=bank,
                bars_close=bars_close,
                adapter=adapter,
                params=params,
            )

        sym = symbols[0]

        # ----------------------------
        # A) Select the backtest window (optional)
        # ----------------------------
        # If you have a param like params["data.window"] = ("2020-01-01","2022-12-31")
        # Otherwise just use the full aligned arrays.
        idx_pos = None
        if "data.window" in params and params["data.window"] is not None:
            start, end = params["data.window"]
            idx_slice = common_index
            if start is not None:
                idx_slice = idx_slice[idx_slice >= pd.to_datetime(start)]
            if end is not None:
                idx_slice = idx_slice[idx_slice <= pd.to_datetime(end)]
            if len(idx_slice) < 2:
                return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0, error="window too small")
            idx_pos = common_index.get_indexer_for(idx_slice)

        open_px = bars_open[sym] if idx_pos is None else bars_open[sym][idx_pos]
        close_px = bars_close[sym] if idx_pos is None else bars_close[sym][idx_pos]

        # ----------------------------
        # B) Build numpy signals from SMA bank (no pandas)
        # ----------------------------
        sk = base_spec.strategy.kind.lower()

        # Pull allow_short / etc from params (or base_spec)
        allow_short = bool(params.get("strategy.allow_short", base_spec.strategy.params.get("allow_short", False)))

        if sk == "sma_price":
            w = int(params.get("strategy.window", base_spec.strategy.params.get("window", 50)))
            sma = bank[sym].get(f"sma_{w}")
            if sma is None:
                return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0, error=f"missing sma_{w} in bank")
            if idx_pos is not None:
                sma = sma[idx_pos]

            valid = np.isfinite(close_px) & np.isfinite(sma)
            if allow_short:
                sig = np.zeros(close_px.size, dtype=np.float64)
                sig[valid & (close_px > sma)] = 1.0
                sig[valid & (close_px < sma)] = -1.0
                sig[~valid] = 0.0
            else:
                # long/flat
                sig = np.where(valid & (close_px > sma), 1.0, 0.0).astype(np.float64)

        elif sk == "ma_cross":
            f = int(params.get("strategy.fast_window", base_spec.strategy.params.get("fast_window", 15)))
            s = int(params.get("strategy.slow_window", base_spec.strategy.params.get("slow_window", 50)))
            sma_f = bank[sym].get(f"sma_{f}")
            sma_s = bank[sym].get(f"sma_{s}")
            if sma_f is None or sma_s is None:
                return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0, error=f"missing sma_{f} or sma_{s} in bank")
            if idx_pos is not None:
                sma_f = sma_f[idx_pos]
                sma_s = sma_s[idx_pos]

            valid = np.isfinite(sma_f) & np.isfinite(sma_s)
            if allow_short:
                sig = np.zeros(sma_f.size, dtype=np.float64)
                sig[valid & (sma_f > sma_s)] = 1.0
                sig[valid & (sma_f < sma_s)] = -1.0
                sig[~valid] = 0.0
            else:
                sig = np.where(valid & (sma_f > sma_s), 1.0, 0.0).astype(np.float64)

        else:
            return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0, error=f"unknown strategy kind {sk}")

        # ----------------------------
        # C) Build portfolio config for this trial (apply params)
        # ----------------------------
        port_cfg = _apply_portfolio_params(base_spec.portfolio, params)
        port = PortfolioEngine(port_cfg)

        # IMPORTANT: call your NEW arrays fast path
        stats = port.run_stats_only_arrays(open_px=open_px, close_px=close_px, sig=sig)

        pnl = float(stats.pnl)
        traded = float(stats.traded_notional)
        n_fills = int(stats.n_fills)
        eff = pnl / traded if traded > 0 else float("-inf")

        return TrialResult(
            params=params,
            pnl=pnl,
            traded_notional=traded,
            efficiency=eff,
            n_fills=n_fills,
            error=None,
        )

    except Exception as e:
        return TrialResult(
            params=params,
            pnl=float("-inf"),
            traded_notional=0.0,
            efficiency=float("-inf"),
            n_fills=0,
            error=str(e),
        )




# ============================================================
# Candidate generation
# ============================================================

def _expand_domain(p: ParamDef) -> List[Any]:
    if p.kind in ("choice", "date_window"):
        return list(p.domain)
    if p.kind == "int":
        lo, hi, step = p.domain
        return list(range(int(lo), int(hi) + 1, int(step)))
    if p.kind == "float":
        lo, hi, step = map(float, p.domain)
        n = int(math.floor((hi - lo) / step)) + 1
        return [lo + i * step for i in range(n)]
    raise ValueError(f"Unknown ParamDef kind: {p.kind}")

def _iter_grid(active: List[ParamDef]) -> Iterable[Dict[str, Any]]:
    grids = [_expand_domain(p) for p in active]
    keys = [p.key for p in active]
    casts = [p.cast for p in active]
    for combo in itertools.product(*grids):
        out: Dict[str, Any] = {}
        for k, v, c in zip(keys, combo, casts):
            out[k] = c(v)
        yield out

def _iter_random(active: List[ParamDef], n_trials: int, seed: int) -> Iterable[Dict[str, Any]]:
    rng = random.Random(seed)
    grids = {p.key: _expand_domain(p) for p in active}
    casts = {p.key: p.cast for p in active}
    keys = [p.key for p in active]
    for _ in range(n_trials):
        out: Dict[str, Any] = {}
        for k in keys:
            out[k] = casts[k](rng.choice(grids[k]))
        yield out


def _domain_values_int(active: List[ParamDef], key: str) -> List[int]:
    p = next((x for x in active if x.key == key and x.enabled), None)
    if p is None:
        return []
    if p.kind != "int":
        return []
    return [int(v) for v in _expand_domain(p)]


# ============================================================
# Apply params to spec/configs (for returning best_spec)
# ============================================================

def _apply_params_to_spec(base_spec: EngineSpec, params: Dict[str, Any]) -> EngineSpec:
    # DataConfig: apply date window if optimized
    data_cfg = base_spec.data
    if "data.window" in params and params["data.window"] is not None:
        start, end = params["data.window"]
        data_cfg = replace(data_cfg, start=str(start), end=str(end))

    # StrategyConfig: apply strategy params
    strat_cfg = base_spec.strategy
    sp = dict(strat_cfg.params or {})
    if "strategy.fast_window" in params:
        sp["fast_window"] = int(params["strategy.fast_window"])
    if "strategy.slow_window" in params:
        sp["slow_window"] = int(params["strategy.slow_window"])
    if "strategy.window" in params:
        sp["window"] = int(params["strategy.window"])
    if "strategy.allow_short" in params:
        sp["allow_short"] = bool(params["strategy.allow_short"])
    if "strategy.nan_policy" in params:
        sp["nan_policy"] = str(params["strategy.nan_policy"])
    strat_cfg = replace(strat_cfg, params=sp)

    # PortfolioConfig
    port_cfg = _apply_portfolio_params(base_spec.portfolio, params)

    return replace(base_spec, data=data_cfg, strategy=strat_cfg, portfolio=port_cfg)


def _apply_portfolio_params(port_cfg: PortfolioConfig, params: Dict[str, Any]) -> PortfolioConfig:
    upd = port_cfg
    if "portfolio.cooldown_bars" in params:
        upd = replace(upd, cooldown_bars=int(params["portfolio.cooldown_bars"]))
    if "portfolio.buy_pct_cash" in params:
        upd = replace(upd, buy_pct_cash=float(params["portfolio.buy_pct_cash"]))
    if "portfolio.sell_pct_shares" in params:
        upd = replace(upd, sell_pct_shares=float(params["portfolio.sell_pct_shares"]))
    return upd


# ============================================================
# Data loading + alignment + slicing
# ============================================================

def _load_market_data_from_spec(cfg: DataConfig) -> MarketData:
    if cfg.source == "bmce":
        if cfg.bmce_paths is None:
            raise ValueError("BMCE source selected but bmce_paths is None.")
        ds = BMCEDataSource(timezone=cfg.timezone)
        return ds.load(
            symbols=cfg.symbols,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
            paths=cfg.bmce_paths,
        )

    if cfg.source == "yfinance":
        ds = YahooFinanceDataSource(timezone=cfg.timezone)
        # Note: your YahooFinanceDataSource.load supports start/end/interval + kwargs
        return ds.load(
            symbols=cfg.symbols,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
            auto_adjust=cfg.yf_auto_adjust,
            progress=False,
        )

    raise ValueError(f"Unknown data source: {cfg.source}")


def _align_marketdata_inner(md: MarketData, symbols: List[str]) -> MarketData:
    # inner intersection index across symbols (robust multi-asset)
    idx = md.bars[symbols[0]].index
    for s in symbols[1:]:
        idx = idx.intersection(md.bars[s].index)
    idx = idx.sort_values()

    bars_aligned: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        bars_aligned[s] = md.bars[s].reindex(idx)

    return MarketData(
        bars=bars_aligned,
        source=md.source,
        timezone=md.timezone,
        interval=md.interval,
        meta=dict(md.meta),
    )


def _slice_marketdata(md: MarketData, symbols: List[str], index: pd.DatetimeIndex) -> MarketData:
    """Slice MarketData to a given DatetimeIndex.

    Optimization hot loop calls this only when you optimize data.window.
    Prefer a cheap .loc[start:end] slice when possible; fall back to reindex
    only if the slice doesn't exactly match the requested index.
    """
    bars: Dict[str, pd.DataFrame] = {}

    if len(index) == 0:
        for s in symbols:
            bars[s] = md.bars[s].iloc[0:0].reindex(index)
        return MarketData(
            bars=bars,
            source=md.source,
            timezone=md.timezone,
            interval=md.interval,
            meta=dict(md.meta),
        )

    start = index[0]
    end = index[-1]

    for s in symbols:
        df = md.bars[s]
        df_span = df.loc[start:end]
        # If df_span already has the exact index, keep it; else reindex
        if df_span.index.equals(index):
            bars[s] = df_span
        else:
            bars[s] = df.reindex(index)

    return MarketData(
        bars=bars,
        source=md.source,
        timezone=md.timezone,
        interval=md.interval,
        meta=dict(md.meta),
    )

def _slice_index(index: pd.DatetimeIndex, start: Optional[str], end: Optional[str]) -> Optional[pd.DatetimeIndex]:
    if start is None and end is None:
        return index
    tz = index.tz
    s = pd.Timestamp(start, tz=tz) if start is not None else None
    e = pd.Timestamp(end, tz=tz) if end is not None else None
    out = index
    if s is not None:
        out = out[out >= s]
    if e is not None:
        out = out[out <= e]
    return out if len(out) > 0 else None


# ============================================================
# TEXT EXPLANATION (for Cursor review)
# ============================================================
"""
What this optimize.py does (fast + robust):

1) Strategy-agnostic optimization via StrategyAdapters:
   - Each strategy kind has an adapter that:
     (a) declares which SMA windows must be precomputed (union over parameter ranges)
     (b) validates parameters (e.g., fast_window < slow_window)
     (c) generates signals from precomputed arrays (no indicator recompute per trial)

2) Compute-once:
   - Load MarketData once using the correct instance method:
       BMCEDataSource(...).load(...)
       YahooFinanceDataSource(...).load(...)
   - Align all symbols to a common index via inner intersection (robust multi-asset)
   - Precompute required SMA features once with IndicatorEngine.compute(...)
   - Convert features to NumPy arrays in a FeatureBank dict: bank[symbol]["sma_50"] -> np.ndarray

3) Evaluate candidates:
   - Candidate params come from ParamDef domains (grid or random)
   - Optional date window optimization: "data.window" provides (start,end) strings.
     We slice the common index and reindex MarketData to that slice (no mutation).
   - Signals are generated by adapter.make_signals_from_bank(...) into a minimal SignalFrame.

4) Portfolio speed:
   - If you implemented PortfolioEngine.run_stats_only(), optimizer uses it (fastest).
   - Otherwise it falls back to PortfolioEngine.run() and computes traded_notional from pres.trades["notional"].

5) Objective:
   - Primary: pnl = final_equity - initial_cash
   - Secondary: efficiency = pnl / net_inv (if net_inv>0 else -inf)
   - Ranking: sort by pnl desc, efficiency desc.

Return values:
   best_result (TrialResult), top_df (DataFrame), best_params (dict), best_spec (EngineSpec)
   best_spec is base_spec with best params applied (DataConfig start/end, StrategyConfig params, PortfolioConfig knobs).

How to integrate with Streamlit:
   - Build active_params from user-selected keys and ranges.
   - Call run_optimization(base_spec, active_params, cfg).
   - Use best_spec to run a full BacktestEngine for plots in "Optimize mode".
"""
