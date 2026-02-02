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
            # validity omitted for speed

        return SignalFrame(
            signals=sig,
            validity=None,
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
            # validity omitted for speed

        return SignalFrame(
            signals=sig,
            validity=None,
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
        # optional strategy flags if you expose them
        cat["strategy.allow_short"] = ParamDef("strategy.allow_short", "choice", [False, True], bool)
        cat["strategy.nan_policy"] = ParamDef("strategy.nan_policy", "choice", ["flat", "nan"], str)

    elif strategy_kind == "sma_price":
        cat["strategy.window"] = ParamDef("strategy.window", "int", (10, 250, 1), int)
        cat["strategy.allow_short"] = ParamDef("strategy.allow_short", "choice", [False, True], bool)
        cat["strategy.nan_policy"] = ParamDef("strategy.nan_policy", "choice", ["flat", "nan"], str)

    # portfolio knobs you mentioned
    cat["portfolio.cooldown_bars"] = ParamDef("portfolio.cooldown_bars", "int", (0, 30, 1), int)
    cat["portfolio.buy_pct_cash"] = ParamDef("portfolio.buy_pct_cash", "float", (0.05, 1.0, 0.05), float)
    cat["portfolio.sell_pct_shares"] = ParamDef("portfolio.sell_pct_shares", "float", (0.05, 1.0, 0.05), float)

    # date-window optimization
    cat["data.window"] = ParamDef("data.window", "date_window", [], lambda x: x)

    return cat


# ============================================================
# Core optimization
# ============================================================

def run_optimization(
    base_spec: EngineSpec,
    active_params: List[ParamDef],
    cfg: OptimizeConfig,
) -> Tuple[TrialResult, pd.DataFrame, Dict[str, Any], EngineSpec]:
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

    # 3) Precompute union features once
    sma_windows = adapter.required_sma_windows(base_spec, active_params)

    feats: Optional[FeaturesData] = None
    bank: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in symbols}

    if sma_windows:
        specs = [
            FeatureSpec(indicator="sma", params={"window": int(w)}, inputs=("Close",), warmup=int(w))
            for w in sma_windows
        ]
        ind = IndicatorEngine(
            cache_dir=cfg.feature_cache_dir,
            enable_disk_cache=cfg.enable_disk_cache,
            enable_memory_cache=cfg.enable_memory_cache,
            engine_version="v1",
        )
        feats = ind.compute(md, specs=specs, symbols=symbols)

        # Convert features to numpy arrays (hot loop wants arrays)
        for s in symbols:
            fdf = feats.features[s].reindex(common_index)  # ensure aligned
            for w in sma_windows:
                col = f"sma_{int(w)}"
                if col not in fdf.columns:
                    raise KeyError(f"Missing feature column '{col}' for symbol '{s}'. Have: {list(fdf.columns)[:10]} ...")
                bank[s][col] = fdf[col].to_numpy(dtype=np.float64)

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
        top_df = df.sort_values(["pnl", "efficiency"], ascending=[False, False]).head(int(cfg.top_k)).reset_index(drop=True)
        best = results[0]
        best_spec = _apply_params_to_spec(base_spec, best.params)
        return best, top_df, best.params, best_spec

    df_valid = df_valid.sort_values(["pnl", "efficiency"], ascending=[False, False])
    top_df = df_valid.head(int(cfg.top_k)).reset_index(drop=True)

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
    return best, top_df, best_params, best_spec


# ============================================================
# Trial evaluation
# ============================================================

def _eval_one_trial(
    base_spec: EngineSpec,
    md: MarketData,
    common_index: pd.DatetimeIndex,
    bank: Dict[str, Dict[str, np.ndarray]],
    bars_close: Dict[str, np.ndarray],
    adapter: StrategyAdapter,
    params: Dict[str, Any],
) -> TrialResult:


    # Validate strategy constraints early (fast<slow, etc.)
    ok, err = adapter.validate_params(params, base_spec)
    if not ok:
        return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0, error=err)

    try:
        # Optional date-window slicing
        win = params.get("data.window", None)
        if win is not None:
            start, end = win
            idx_slice = _slice_index(common_index, start, end)
            if idx_slice is None or len(idx_slice) < 2:
                return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0, error="date window too small/empty")
            index = idx_slice
        else:
            index = common_index

        symbols = list(base_spec.data.symbols)

        # Signals from feature bank (fast, no indicator recompute)
        sf = adapter.make_signals_from_bank(
            symbols=symbols,
            index=index,
            bank=bank,
            bars_close={s: bars_close[s][(common_index.get_indexer_for(index))] for s in symbols},
            params=params,
            base_spec=base_spec,
        )

        # Portfolio config patched per trial (cooldown + sizing knobs)
        port_cfg = base_spec.portfolio
        port_cfg = _apply_portfolio_params(port_cfg, params)

        # Run portfolio fast stats if available, else full run
        port = PortfolioEngine(port_cfg)

        # FAST path: for single-symbol optimization, avoid slicing/reindexing MarketData per trial.
        used_fast = False
        if hasattr(port, "run_stats_only"):
            try:
                if len(symbols) == 1:
                    stats = port.run_stats_only(md, sf, symbols=symbols)  # type: ignore[attr-defined]
                else:
                    md_slice = _slice_marketdata(md, symbols, index)
                    stats = port.run_stats_only(md_slice, sf, symbols=symbols)  # type: ignore[attr-defined]

                pnl = float(stats.pnl)
                traded = float(stats.traded_notional)
                n_fills = int(stats.n_fills)
                used_fast = True
            except Exception:
                used_fast = False

        # Fallback to full portfolio run (always works)
        if not used_fast:
            md_slice = _slice_marketdata(md, symbols, index)
            pres = port.run(md_slice, sf, symbols=symbols)
            pnl = float(pres.equity_curve.iloc[-1]) - float(port_cfg.initial_cash) if len(pres.equity_curve) else 0.0
            traded = float(pres.trades["notional"].sum()) if (not pres.trades.empty and "notional" in pres.trades.columns) else 0.0
            n_fills = int(len(pres.trades))


        eff = pnl / traded if traded > 0 else float("-inf")
        return TrialResult(params=params, pnl=pnl, traded_notional=traded, efficiency=eff, n_fills=n_fills)

    except Exception as e:
        return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0, error=str(e))


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
    bars: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        bars[s] = md.bars[s].reindex(index)
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
   - Secondary: efficiency = pnl / traded_notional (if traded_notional>0 else -inf)
   - Ranking: sort by pnl desc, efficiency desc.

Return values:
   best_result (TrialResult), top_df (DataFrame), best_params (dict), best_spec (EngineSpec)
   best_spec is base_spec with best params applied (DataConfig start/end, StrategyConfig params, PortfolioConfig knobs).

How to integrate with Streamlit:
   - Build active_params from user-selected keys and ranges.
   - Call run_optimization(base_spec, active_params, cfg).
   - Use best_spec to run a full BacktestEngine for plots in "Optimize mode".
"""
