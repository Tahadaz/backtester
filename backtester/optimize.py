# optimize.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable, Callable
import itertools
import random
import math
import pandas as pd

# ---- your project imports ----
from data import BMCEDataSource                       # must exist (or adjust)
from indicators import FeatureSpec, IndicatorEngine     # must exist (or adjust)
from strategy import BaseStrategy                      # must exist (or adjust)
from portfolio import PortfolioEngine                    # must exist (or adjust)
from engine import EngineSpec, DataConfig, StrategyConfig # used for configs/dataclasses


# =========================
# Core dataclasses
# =========================

@dataclass
class OptimizeConfig:
    method: str  # "grid" | "random" 
    seed: int = 42

    # random search
    n_trials: int = 200

    # display
    top_k: int = 30

    # objective
    objective: str = "pnl_then_eff"  # pnl_then_eff




@dataclass
class TrialResult:
    params: Dict[str, Any]
    pnl: float
    traded_notional: float
    efficiency: float
    error: Optional[str] = None




# Parameter definition the UI can build
@dataclass
class ParamDef:
    key: str                      # e.g. "strategy.fast_window"
    kind: str                     # "int" | "float" | "choice" | "date_window"
    domain: Any                   # list for choice, or (min,max,step) for numeric, or list of (start,end)
    cast: Callable[[Any], Any]    # int/float/identity
    enabled: bool = True


# =========================
# Strategy-aware catalog
# =========================

def default_param_catalog_for_strategy(strategy_kind: str) -> Dict[str, ParamDef]:
    """
    Returns a catalog of possible parameters. The Streamlit UI should:
      - choose active keys from this catalog
      - for each chosen key, override its domain (min/max/step or choices)
    """
    cat: Dict[str, ParamDef] = {}

    if strategy_kind == "ma_cross":
        cat["strategy.fast_window"] = ParamDef("strategy.fast_window", "int", (5, 50, 5), int)
        cat["strategy.slow_window"] = ParamDef("strategy.slow_window", "int", (20, 200, 10), int)

    elif strategy_kind == "sma_price":
        cat["strategy.window"] = ParamDef("strategy.window", "int", (10, 200, 10), int)

    # Portfolio sizing params
    cat["portfolio.buy_pct_cash"] = ParamDef("portfolio.buy_pct_cash", "float", (0.05, 1.0, 0.05), float)
    cat["portfolio.sell_pct_shares"] = ParamDef("portfolio.sell_pct_shares", "float", (0.05, 1.0, 0.05), float)

    # Optional: min bars between trades (cooldown)
    cat["portfolio.cooldown_bars"] = ParamDef("portfolio.cooldown_bars", "int", (0, 30, 1), int)

    # Optional: date window optimization (list of (start_iso, end_iso))
    cat["data.window"] = ParamDef("data.window", "date_window", [], lambda x: x)

    return cat


# =========================
# Candidate generation
# =========================

def _expand_domain(p: ParamDef) -> List[Any]:
    if p.kind in ("choice", "date_window"):
        return list(p.domain)
    if p.kind == "int":
        lo, hi, step = p.domain
        return list(range(int(lo), int(hi) + 1, int(step)))
    if p.kind == "float":
        lo, hi, step = p.domain
        lo = float(lo); hi = float(hi); step = float(step)
        n = int(math.floor((hi - lo) / step)) + 1
        return [lo + i * step for i in range(n)]
    raise ValueError(f"Unknown ParamDef kind: {p.kind}")

def _iter_grid(active: List[ParamDef]) -> Iterable[Dict[str, Any]]:
    grids = [_expand_domain(p) for p in active]
    for combo in itertools.product(*grids):
        out = {}
        for p, v in zip(active, combo):
            out[p.key] = p.cast(v)
        yield out

def _iter_random(active: List[ParamDef], n_trials: int, seed: int) -> Iterable[Dict[str, Any]]:
    rng = random.Random(seed)
    grids = {p.key: _expand_domain(p) for p in active}
    for _ in range(n_trials):
        out = {}
        for p in active:
            out[p.key] = p.cast(rng.choice(grids[p.key]))
        yield out


# =========================
# Constraints / validity
# =========================

def _validate_params(strategy_kind: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if strategy_kind == "ma_cross":
        f = int(params.get("strategy.fast_window", 0))
        s = int(params.get("strategy.slow_window", 0))
        if f >= s:
            return False, "fast_window must be < slow_window"
    return True, None


# =========================
# Precompute indicators once
# =========================

def _required_sma_windows(strategy_kind: str, active: List[ParamDef]) -> List[int]:
    keys = {p.key for p in active}
    ws: List[int] = []

    def domain_int(key: str) -> List[int]:
        p = next((x for x in active if x.key == key), None)
        if p is None:
            return []
        return [int(v) for v in _expand_domain(p)]

    if strategy_kind == "ma_cross":
        if "strategy.fast_window" in keys:
            ws += domain_int("strategy.fast_window")
        if "strategy.slow_window" in keys:
            ws += domain_int("strategy.slow_window")
    elif strategy_kind == "sma_price":
        if "strategy.window" in keys:
            ws += domain_int("strategy.window")

    return sorted(set(ws))

def _build_sma_specs(windows: Sequence[int]) -> List[FeatureSpec]:
    # IMPORTANT: ensure this matches your indicators.py FeatureSpec definition.
    # If you already have SMAFeatureSpec, use it here instead.
    specs: List[FeatureSpec] = []
    for w in sorted(set(int(x) for x in windows)):
        specs.append(
            FeatureSpec(
                indicator="sma",
                params={"window": int(w)},
                inputs=("Close",),
                name=f"sma_{int(w)}",
                warmup=int(w),
                output_mode="series",
            )
        )
    return specs


# =========================
# Scoring (PnL + profit/volume)
# =========================

def _score(port_res, initial_cash: float) -> Tuple[float, float, float]:
    # pnl
    if port_res.equity_curve is None or len(port_res.equity_curve) == 0:
        pnl = 0.0
    else:
        pnl = float(port_res.equity_curve.iloc[-1]) - float(initial_cash)

    # traded notional (must be stored in meta during portfolio.run)
    traded = 0.0
    if hasattr(port_res, "meta") and isinstance(port_res.meta, dict):
        traded = float(port_res.meta.get("traded_notional", 0.0))

    eff = pnl / traded if traded > 0 else 0.0
    return pnl, traded, eff


# =========================
# Apply params to configs
# =========================

def _apply_params_to_cfgs(
    base_spec: EngineSpec,
    params: Dict[str, Any],
) -> Tuple[DataConfig, StrategyConfig, Any]:
    """
    Returns (data_cfg, strategy_cfg, portfolio_cfg) derived from base_spec + params.
    Portfolio cfg is your PortfolioConfig instance.
    """
    # --- data cfg ---
    data_cfg = base_spec.data
    if "data.window" in params and params["data.window"] is not None:
        start, end = params["data.window"]
        data_cfg = type(data_cfg)(**{**data_cfg.__dict__, "start": start, "end": end})

    # --- strategy cfg ---
    strat_cfg = base_spec.strategy
    strat_params = dict(strat_cfg.params or {})

    # strategy params
    if "strategy.fast_window" in params:
        strat_params["fast_window"] = int(params["strategy.fast_window"])
    if "strategy.slow_window" in params:
        strat_params["slow_window"] = int(params["strategy.slow_window"])
    if "strategy.window" in params:
        strat_params["window"] = int(params["strategy.window"])

    strategy_cfg = StrategyConfig(kind=strat_cfg.kind, params=strat_params)

    # --- portfolio cfg ---
    port_cfg = base_spec.portfolio
    port_dict = dict(port_cfg.__dict__)

    if "portfolio.buy_pct_cash" in params:
        port_dict["buy_pct_cash"] = float(params["portfolio.buy_pct_cash"])
    if "portfolio.sell_pct_shares" in params:
        port_dict["sell_pct_shares"] = float(params["portfolio.sell_pct_shares"])
    if "portfolio.cooldown_bars" in params:
        port_dict["cooldown_bars"] = int(params["portfolio.cooldown_bars"])

    portfolio_cfg = type(port_cfg)(**port_dict)

    return data_cfg, strategy_cfg, portfolio_cfg


# =========================
# Data/feature slicing helper
# =========================

def _slice_by_data_cfg(md, feats, data_cfg: DataConfig, symbols: Sequence[str]):
    """
    If your MarketData and FeaturesData are already sliced in load_market_data,
    you can no-op here. Otherwise slice by start/end.
    """
    start = getattr(data_cfg, "start", None)
    end = getattr(data_cfg, "end", None)
    if start is None and end is None:
        return md, feats

    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None

    # shallow copy style: keep same object types if possible
    for s in symbols:
        bars = md.bars[s]
        md.bars[s] = bars.loc[start_ts:end_ts] if (start_ts or end_ts) else bars

        if feats is not None and hasattr(feats, "features") and s in feats.features:
            fdf = feats.features[s]
            feats.features[s] = fdf.loc[start_ts:end_ts] if (start_ts or end_ts) else fdf

    return md, feats


# =========================
# Evaluation (one candidate)
# =========================

def _eval_one(
    base_spec: EngineSpec,
    md_full,
    feats_full,
    params: Dict[str, Any],
) -> TrialResult:
    ok, err = _validate_params(base_spec.strategy.kind, params)
    if not ok:
        return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=0.0, error=err)

    try:
        data_cfg, strategy_cfg, portfolio_cfg = _apply_params_to_cfgs(base_spec, params)

        # slice md + feats to the date window if needed
        # IMPORTANT: to avoid mutating shared md_full across trials, we shallow-copy bars dict
        md = type(md_full)(**md_full.__dict__)
        md.bars = {k: v.copy() for k, v in md_full.bars.items()}

        feats = None
        if feats_full is not None:
            feats = type(feats_full)(**feats_full.__dict__)
            feats.features = {k: v.copy() for k, v in feats_full.features.items()}

        md, feats = _slice_by_data_cfg(md, feats, data_cfg, base_spec.data.symbols)

        # strategy -> signals
        strat = BaseStrategy(strategy_cfg)
        sf = strat.generate_signals(md, feats, symbols=list(base_spec.data.symbols))

        # portfolio -> result
        port = PortfolioEngine(portfolio_cfg)
        pres = port.run(md, sf, symbols=list(base_spec.data.symbols))

        pnl, traded, eff = _score(pres, initial_cash=float(base_spec.portfolio.initial_cash))
        return TrialResult(params=params, pnl=pnl, traded_notional=traded, efficiency=eff)

    except Exception as e:
        return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=0.0, error=str(e))


#
# =========================
# Public API
# =========================

def run_optimization(
    base_spec: EngineSpec,
    active_keys: List[str],
    catalog: Dict[str, ParamDef],
    cfg: OptimizeConfig,
) -> Tuple[TrialResult, pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      best_result, top_df, best_params
    (App can apply best_params to spec and run the final backtest for plots.)
    """
    method = cfg.method.lower()
    rng = random.Random(cfg.seed)

    # Build active ParamDef list
    active = [catalog[k] for k in active_keys if k in catalog]

    # Load data ONCE
    md = BMCEDataSource.load(base_spec.data)

    # Precompute indicators ONCE (SMA windows union)
    windows = _required_sma_windows(base_spec.strategy.kind, active)
    feats = None
    if windows:
        specs = _build_sma_specs(windows)
        feats = compute_features(md, specs=specs, symbols=list(base_spec.data.symbols))

    
    # Candidates
    if method == "grid":
        candidates = _iter_grid(active)
    elif method == "random":
        candidates = _iter_random(active, n_trials=int(cfg.n_trials), seed=int(cfg.seed))
    else:
        raise ValueError(f"Unknown optimization method: {cfg.method}")

    # Evaluate
    results: List[TrialResult] = []
    for params in candidates:
        r = _eval_one(base_spec, md, feats, params)
        results.append(r)

    df = pd.DataFrame([{
        **r.params,
        "pnl": r.pnl,
        "traded_notional": r.traded_notional,
        "efficiency": r.efficiency,
        "error": r.error,
    } for r in results])

    df_valid = df[df["error"].isna()].copy()
    if df_valid.empty:
        # show failures to help debug
        top_df = df.sort_values(["pnl", "efficiency"], ascending=[False, False]).head(int(cfg.top_k))
        best = results[0]
        return best, top_df.reset_index(drop=True), best.params

    df_valid = df_valid.sort_values(["pnl", "efficiency"], ascending=[False, False])
    top_df = df_valid.head(int(cfg.top_k)).reset_index(drop=True)
    best_row = top_df.iloc[0].to_dict()

    # best params are only active keys
    best_params = {k: best_row[k] for k in active_keys if k in best_row}

    best = TrialResult(
        params=best_params,
        pnl=float(best_row["pnl"]),
        traded_notional=float(best_row["traded_notional"]),
        efficiency=float(best_row["efficiency"]),
        error=None,
    )
    return best, top_df, best_params
