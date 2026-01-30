# optimize.py
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine import BacktestEngine, EngineSpec


# ============================================================
# Parameter specification
# ============================================================
@dataclass(frozen=True)
class ParamSpec:
    """
    A spec describing how to sample a parameter.

    kind:
      - "int": integer range [low, high], optional step for grid
      - "float": float range [low, high], optional step for grid
      - "categorical": list of choices
    """
    kind: str  # "int" | "float" | "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None


# ============================================================
# Optimization config + result
# ============================================================
@dataclass(frozen=True)
class OptimizeConfig:
    """
    mode:
      - "random": random search over active parameters
      - "grid": cartesian grid search over active parameters
      - "wfo": walk-forward evaluation (candidate params evaluated over rolling test windows)

    objective:
      - "pnl_then_efficiency"  (primary: pnl, secondary: profit/volume)
    """
    mode: str  # "random" | "grid" | "wfo"
    n_trials: Optional[int] = 200               # random only
    top_k: int = 30
    seed: int = 42
    objective: str = "pnl_then_efficiency"
    verbose: bool = False

    # grid only
    grid_max_combos: Optional[int] = 500

    # wfo only (bars count; daily data => 252 ~ 1y)
    wfo_train_bars: Optional[int] = 252
    wfo_test_bars: Optional[int] = 63
    wfo_step_bars: Optional[int] = 21


@dataclass
class TrialResult:
    params: Dict[str, Any]
    pnl: float
    traded_notional: float
    efficiency: float  # pnl / traded_notional (0 if traded_notional=0)
    spec: EngineSpec


# ============================================================
# Public API
# ============================================================
def default_param_catalog_for_your_app(strategy_kind: Optional[str] = None) -> Dict[str, ParamSpec]:
    """
    Catalog keys use a dotted-path convention. Supported keys in this file:
      - strategy.fast_window, strategy.slow_window, strategy.window
      - portfolio.buy_pct_cash, portfolio.sell_pct_shares, portfolio.cooldown_bars
      - data.window  (categorical: {"start": "...", "end": "..."} dicts)  -> added via add_date_window_param()

    If strategy_kind is provided, strategy keys are filtered accordingly.
    """
    cat: Dict[str, ParamSpec] = {
        # ---------- Strategy ----------
        "strategy.fast_window": ParamSpec(kind="int", low=2, high=200, step=1, default=20),
        "strategy.slow_window": ParamSpec(kind="int", low=3, high=400, step=1, default=50),
        "strategy.window":      ParamSpec(kind="int", low=2, high=400, step=1, default=50),

        # ---------- Portfolio / sizing ----------
        "portfolio.buy_pct_cash":   ParamSpec(kind="float", low=0.01, high=1.00, step=0.01, default=0.25),
        "portfolio.sell_pct_shares":ParamSpec(kind="float", low=0.01, high=1.00, step=0.01, default=1.00),

        # Minimum period between trades (requires you to add and enforce cooldown_bars in PortfolioConfig + PortfolioEngine)
        "portfolio.cooldown_bars":  ParamSpec(kind="int", low=0, high=60, step=1, default=0),
    }

    if strategy_kind is None:
        return cat

    return filter_catalog_for_strategy(cat, strategy_kind=strategy_kind)

def _is_valid_params_for_strategy(strategy_kind: str, params: dict) -> bool:
    kind = (strategy_kind or "").lower()
    if kind in ("ma_cross", "moving_average_cross"):
        fw = params.get("strategy.fast_window")
        sw = params.get("strategy.slow_window")
        if fw is None or sw is None:
            return True  # if you're not optimizing both, it's fine
        return int(fw) < int(sw)
    return True

def filter_catalog_for_strategy(catalog: Dict[str, ParamSpec], strategy_kind: str) -> Dict[str, ParamSpec]:
    """
    Keep only strategy keys relevant to the selected strategy.
    Non-strategy keys are always preserved.
    """
    out: Dict[str, ParamSpec] = {}
    for k, spec in catalog.items():
        if not k.startswith("strategy."):
            out[k] = spec
            continue

        if strategy_kind == "ma_cross" and k in ("strategy.fast_window", "strategy.slow_window"):
            out[k] = spec
        elif strategy_kind == "sma_price" and k in ("strategy.window",):
            out[k] = spec

    return out


def build_date_window_choices_from_uploaded_bmce(
    base_data_cfg: Any,
    symbol: str,
    min_bars: int = 252,
    step_bars: int = 21,
    max_windows: int = 200,
) -> List[Dict[str, str]]:
    """
    Builds rolling start/end choices (ISO date strings) from a BMCE uploaded file.

    Assumptions:
      - base_data_cfg.source == "bmce"
      - base_data_cfg.bmce_paths points to the uploaded CSV/XLSX (single path)
    """
    path = getattr(base_data_cfg, "bmce_paths", None)
    if not path:
        return []
    # In your app you pass bmce_paths=tmp_path (string). If it's a list, take first.
    if isinstance(path, (list, tuple)):
        path = path[0]

    idx = _load_datetime_index_from_bmce_file(path)
    if idx is None or len(idx) < min_bars:
        return []

    idx = idx.sort_values()
    windows: List[Dict[str, str]] = []

    # rolling windows over the entire history
    # Window length grows? Here we use fixed length = min_bars and slide by step_bars.
    # You can change to variable length by changing end_i.
    start_i = 0
    while start_i + min_bars <= len(idx):
        end_i = start_i + min_bars - 1
        start_dt = idx[start_i]
        end_dt = idx[end_i]
        windows.append({"start": start_dt.date().isoformat(), "end": end_dt.date().isoformat()})
        start_i += step_bars
        if len(windows) >= max_windows:
            break

    return windows


def add_date_window_param(catalog: Dict[str, ParamSpec], windows: List[Dict[str, str]]) -> None:
    """
    Adds a categorical param: data.window, choices=[{"start": "...", "end": "..."}, ...]
    """
    if not windows:
        return
    catalog["data.window"] = ParamSpec(kind="categorical", choices=list(windows), default=windows[0])


def run_optimization(
    base_spec: EngineSpec,
    catalog: Dict[str, ParamSpec],
    active_keys: List[str],
    cfg: OptimizeConfig,
) -> Tuple[TrialResult, pd.DataFrame, EngineSpec]:
    """
    Returns:
      - best TrialResult
      - top_df (top K)
      - best_spec (EngineSpec for best)
    """
    rng = random.Random(int(cfg.seed))
    np_rng = np.random.default_rng(int(cfg.seed))

    # Validate active keys
    missing = [k for k in active_keys if k not in catalog]
    if missing:
        raise KeyError(f"Active keys not found in catalog: {missing}")

    # Generate candidate parameter sets
    if cfg.mode == "random":
        if cfg.n_trials is None:
            raise ValueError("OptimizeConfig.n_trials must be set for random mode.")
        candidates = _iter_random_candidates(catalog, active_keys, int(cfg.n_trials), rng, np_rng)

    elif cfg.mode == "grid":
        candidates = _iter_grid_candidates(
            catalog=catalog,
            active_keys=active_keys,
            max_combos=cfg.grid_max_combos,
            rng=rng,
        )

    elif cfg.mode == "wfo":
        candidates = _iter_candidates_for_wfo(
            catalog=catalog,
            active_keys=active_keys,
            cfg=cfg,
            rng=rng,
            np_rng=np_rng,
        )

    else:
        raise ValueError(f"Unknown optimization mode: {cfg.mode}")

    # Evaluate candidates
    results: List[TrialResult] = []
    for i, params in enumerate(candidates, start=1):
        if cfg.mode == "wfo":
            res = _eval_candidate_wfo(base_spec, params, cfg)
        else:
            res = _eval_candidate_once(base_spec, params)

        results.append(res)

        if cfg.verbose and (i % 25 == 0):
            print(f"[opt] evaluated {i} candidates; last pnl={res.pnl:.2f}, eff={res.efficiency:.6f}")

    # Rank results
    results_sorted = sorted(results, key=lambda r: _objective_key(r, cfg.objective), reverse=True)
    best = results_sorted[0]

    # Build top dataframe
    top = results_sorted[: int(cfg.top_k)]
    top_df = pd.DataFrame([{
        "rank": j + 1,
        "pnl": r.pnl,
        "traded_notional": r.traded_notional,
        "efficiency": r.efficiency,
        **{f"param.{k}": v for k, v in r.params.items()},
    } for j, r in enumerate(top)])

    return best, top_df, best.spec


# ============================================================
# Candidate generation
# ============================================================
def _iter_random_candidates(
    catalog: Dict[str, ParamSpec],
    active_keys: List[str],
    n_trials: int,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Iterable[Dict[str, Any]]:
    for _ in range(n_trials):
        p: Dict[str, Any] = {}
        for k in active_keys:
            p[k] = _sample_one(catalog[k], rng, np_rng)
        yield p


def _iter_grid_candidates(
    catalog: Dict[str, ParamSpec],
    active_keys: List[str],
    max_combos: Optional[int],
    rng: random.Random,
) -> Iterable[Dict[str, Any]]:
    grids: List[Tuple[str, List[Any]]] = []
    for k in active_keys:
        grids.append((k, _grid_values(catalog[k])))

    # cartesian product
    combos = 1
    for _, vals in grids:
        combos *= max(1, len(vals))

    # If too many combos, we will randomly sample a subset of combos (without replacement approximation)
    # to keep runtime bounded.
    if max_combos is not None and combos > int(max_combos):
        # Create an index-based sampler
        # We approximate without replacement by sampling random tuples directly.
        # This is simpler than materializing all combos.
        n = int(max_combos)
        for _ in range(n):
            p = {k: rng.choice(vals) for k, vals in grids}
            yield p
        return

    # full cartesian
    def rec(i: int, cur: Dict[str, Any]):
        if i == len(grids):
            yield dict(cur)
            return
        k, vals = grids[i]
        for v in vals:
            cur[k] = v
            yield from rec(i + 1, cur)

    yield from rec(0, {})


def _iter_candidates_for_wfo(
    catalog: Dict[str, ParamSpec],
    active_keys: List[str],
    cfg: OptimizeConfig,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Iterable[Dict[str, Any]]:
    """
    WFO can be run as either:
      - grid (if all active keys have reasonable step/choices)
      - random (fallback)
    Here: if grid_max_combos is provided we use grid sampling, else random with n_trials.
    """
    # If the user set n_trials, use random; else use grid sampling capped by grid_max_combos.
    if cfg.n_trials is not None:
        return _iter_random_candidates(catalog, active_keys, int(cfg.n_trials), rng, np_rng)
    return _iter_grid_candidates(catalog, active_keys, cfg.grid_max_combos, rng)


def _sample_one(spec: ParamSpec, rng: random.Random, np_rng: np.random.Generator) -> Any:
    if spec.kind == "categorical":
        if not spec.choices:
            return spec.default
        return rng.choice(spec.choices)

    if spec.kind == "int":
        lo = int(spec.low) if spec.low is not None else 0
        hi = int(spec.high) if spec.high is not None else lo
        return rng.randint(lo, hi)

    if spec.kind == "float":
        lo = float(spec.low) if spec.low is not None else 0.0
        hi = float(spec.high) if spec.high is not None else lo
        return float(np_rng.uniform(lo, hi))

    raise ValueError(f"Unknown ParamSpec.kind: {spec.kind}")


def _grid_values(spec: ParamSpec) -> List[Any]:
    if spec.kind == "categorical":
        return list(spec.choices or [])

    if spec.kind in ("int", "float"):
        if spec.low is None or spec.high is None:
            return [spec.default]
        step = spec.step
        if step is None or step <= 0:
            # fallback: just low/high/default unique
            vals = [spec.default, spec.low, spec.high]
            # unique, stable
            out = []
            for v in vals:
                if v is None:
                    continue
                if v not in out:
                    out.append(v)
            return out

        lo = float(spec.low)
        hi = float(spec.high)
        n = int(math.floor((hi - lo) / float(step))) + 1
        arr = [lo + i * float(step) for i in range(max(1, n))]

        if spec.kind == "int":
            arr = [int(round(x)) for x in arr]
            # remove duplicates due to rounding
            out = []
            for v in arr:
                if v not in out:
                    out.append(v)
            return out
        else:
            return [float(x) for x in arr]

    raise ValueError(f"Unknown ParamSpec.kind: {spec.kind}")


# ============================================================
# Evaluation
# ============================================================
def _eval_candidate_once(base_spec: EngineSpec, params: Dict[str, Any]) -> TrialResult:
    if not _is_valid_params_for_strategy(base_spec.strategy.kind, params):
        return TrialResult(
            pnl=float("-inf"),
            traded_notional=float("inf"),
            efficiency=float("-inf"),
            params=dict(params),
            error="invalid_params: fast_window must be < slow_window",
        )

    spec = apply_params_to_spec(base_spec, params)
    bundle = BacktestEngine(spec).run()
    pnl, traded_notional = extract_pnl_and_traded_notional(bundle)
    eff = pnl / traded_notional if traded_notional > 0 else 0.0
    return TrialResult(params=dict(params), pnl=float(pnl), traded_notional=float(traded_notional), efficiency=float(eff), spec=spec)


def _eval_candidate_wfo(base_spec: EngineSpec, params: Dict[str, Any], cfg: OptimizeConfig) -> TrialResult:
    """
    Walk-forward evaluation:
      - We build rolling windows from BMCE file index
      - For each fold we evaluate candidate on TEST window only
      - Aggregate pnl and traded_notional across folds
    """
    # WFO requires BMCE upload (so we can derive datetime index reliably without calling yfinance)
    data_cfg = getattr(base_spec, "data", None)
    if data_cfg is None or getattr(data_cfg, "source", None) != "bmce":
        raise ValueError("WFO mode currently supported for BMCE uploads only.")

    path = getattr(data_cfg, "bmce_paths", None)
    if isinstance(path, (list, tuple)):
        path = path[0]
    if not path:
        raise ValueError("WFO needs base_spec.data.bmce_paths to point to the uploaded file.")

    idx = _load_datetime_index_from_bmce_file(path)
    if idx is None:
        raise ValueError("Could not load BMCE datetime index for WFO.")
    idx = idx.sort_values()

    train_bars = int(cfg.wfo_train_bars or 252)
    test_bars = int(cfg.wfo_test_bars or 63)
    step_bars = int(cfg.wfo_step_bars or 21)

    min_needed = train_bars + test_bars
    if len(idx) < min_needed:
        # fallback: run once on full range
        return _eval_candidate_once(base_spec, params)

    total_pnl = 0.0
    total_notional = 0.0
    folds = 0

    start_i = 0
    while start_i + min_needed <= len(idx):
        # Train: [start_i, start_i+train_bars-1]   (not directly used here)
        # Test:  [start_i+train_bars, start_i+train_bars+test_bars-1]
        test_start_i = start_i + train_bars
        test_end_i = test_start_i + test_bars - 1

        test_start = idx[test_start_i].date().isoformat()
        test_end = idx[test_end_i].date().isoformat()

        fold_params = dict(params)
        # If user also optimizes data.window, we respect it and skip WFO slicing.
        # Otherwise, we set the fold window.
        if "data.window" not in fold_params:
            fold_params["data.window"] = {"start": test_start, "end": test_end}

        spec = apply_params_to_spec(base_spec, fold_params)
        bundle = BacktestEngine(spec).run()
        pnl, notional = extract_pnl_and_traded_notional(bundle)
        total_pnl += float(pnl)
        total_notional += float(notional)
        folds += 1

        start_i += step_bars

    eff = total_pnl / total_notional if total_notional > 0 else 0.0
    # Return a spec that corresponds to "full range" with chosen params (not fold-specific),
    # so the app can "Run best configuration" as a single backtest.
    final_spec = apply_params_to_spec(base_spec, params)

    return TrialResult(
        params=dict(params),
        pnl=float(total_pnl),
        traded_notional=float(total_notional),
        efficiency=float(eff),
        spec=final_spec,
    )


def _objective_key(r: TrialResult, objective: str) -> Tuple[float, float]:
    if objective == "pnl_then_efficiency":
        return (float(r.pnl), float(r.efficiency))
    # fallback
    return (float(r.pnl), float(r.efficiency))


# ============================================================
# Spec mutation / param application
# ============================================================
def apply_params_to_spec(base_spec: EngineSpec, params: Dict[str, Any]) -> EngineSpec:
    """
    Returns a deep-copied spec with params applied.

    Supported keys:
      - strategy.fast_window, strategy.slow_window, strategy.window
      - portfolio.buy_pct_cash, portfolio.sell_pct_shares, portfolio.cooldown_bars
      - data.window: {"start": "...", "end": "..."}
    """
    spec: EngineSpec = copy.deepcopy(base_spec)

    # Strategy
    strat = getattr(spec, "strategy", None)
    if strat is None:
        raise ValueError("EngineSpec.strategy missing")

    if "strategy.fast_window" in params:
        strat.params["fast_window"] = int(params["strategy.fast_window"])
    if "strategy.slow_window" in params:
        strat.params["slow_window"] = int(params["strategy.slow_window"])
    if "strategy.window" in params:
        strat.params["window"] = int(params["strategy.window"])

    # Always ensure allow_short remains consistent if present in base params
    # (We don't optimize allow_short here by default)
    # If you want it optimizable, add it to the catalog and handle it here.

    # Portfolio sizing / cooldown
    port = getattr(spec, "portfolio", None)
    if port is None:
        raise ValueError("EngineSpec.portfolio missing")

    if "portfolio.buy_pct_cash" in params:
        port.buy_pct_cash = float(params["portfolio.buy_pct_cash"])
    if "portfolio.sell_pct_shares" in params:
        port.sell_pct_shares = float(params["portfolio.sell_pct_shares"])
    if "portfolio.cooldown_bars" in params:
        # requires PortfolioConfig.cooldown_bars to exist
        port.cooldown_bars = int(params["portfolio.cooldown_bars"])

    # Data window override
    data = getattr(spec, "data", None)
    if data is None:
        raise ValueError("EngineSpec.data missing")

    if "data.window" in params and params["data.window"] is not None:
        w = params["data.window"]
        if isinstance(w, dict) and "start" in w and "end" in w:
            data.start = w["start"]
            data.end = w["end"]

    return spec


# ============================================================
# Metrics extraction
# ============================================================
def extract_pnl_and_traded_notional(bundle: Any) -> Tuple[float, float]:
    """
    Robustly extract:
      - pnl (absolute profit) for the backtest
      - traded_notional (proxy for "volume" / activity)

    Priority:
      1) report.tables["trade_ledger"] if present (net_pnl and qty*price)
      2) report.series["pnl"] sum if present (fallback)
      3) else 0
    """
    rep = getattr(bundle, "report", None)
    if rep is None:
        return 0.0, 0.0

    tables = getattr(rep, "tables", {}) or {}
    series = getattr(rep, "series", {}) or {}

    # Trade ledger approach (best)
    ledger = tables.get("trade_ledger", None)
    if isinstance(ledger, pd.DataFrame) and not ledger.empty:
        pnl = 0.0
        if "net_pnl" in ledger.columns:
            pnl = float(pd.to_numeric(ledger["net_pnl"], errors="coerce").fillna(0.0).sum())
        elif "gross_pnl" in ledger.columns:
            pnl = float(pd.to_numeric(ledger["gross_pnl"], errors="coerce").fillna(0.0).sum())

        # traded notional = sum(|qty| * entry_price) as a proxy
        qty_col = "qty" if "qty" in ledger.columns else None
        price_col = "entry_price" if "entry_price" in ledger.columns else ("price" if "price" in ledger.columns else None)

        traded = 0.0
        if qty_col and price_col:
            q = pd.to_numeric(ledger[qty_col], errors="coerce").fillna(0.0).abs()
            p = pd.to_numeric(ledger[price_col], errors="coerce").fillna(0.0).abs()
            traded = float((q * p).sum())

        return pnl, traded

    # PnL series fallback
    pnl_series = series.get("pnl", None)
    if isinstance(pnl_series, pd.Series) and not pnl_series.empty:
        pnl = float(pd.to_numeric(pnl_series, errors="coerce").fillna(0.0).sum())
        # no traded notional available -> 0
        return pnl, 0.0

    # Last resort: try curve vs benchmark table total return * initial cash? (not recommended)
    return 0.0, 0.0


# ============================================================
# BMCE datetime parsing
# ============================================================
def _load_datetime_index_from_bmce_file(path: str) -> Optional[pd.DatetimeIndex]:
    """
    Tries to load a BMCE CSV/XLSX and extract a datetime index.
    Handles common column names: Date, date, Datetime, timestamp.
    """
    p = str(path).lower()
    try:
        if p.endswith(".csv"):
            df = pd.read_csv(path)
        elif p.endswith(".xlsx") or p.endswith(".xls"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            # try csv as fallback
            df = pd.read_csv(path)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Find a date-like column
    candidates = ["Date", "date", "Datetime", "datetime", "Timestamp", "timestamp", "Time", "time"]
    date_col = None
    for c in candidates:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        # fallback to first column
        date_col = df.columns[0]

    dt = pd.to_datetime(df[date_col], errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        return None

    return pd.DatetimeIndex(dt.values)
