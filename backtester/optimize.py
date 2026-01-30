# optimize.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal
import itertools
import random
import math
import pandas as pd
import numpy as np

# ---- project imports ----
# adjust import paths if you later package your code under a module folder
from engine import BacktestEngine, EngineSpec, DataConfig, StrategyConfig, IndicatorsConfig, load_marketdata

StrategyKind = Literal["ma_cross","sma_price", ...]

# ============================================================
# Types
# ============================================================
Params = Dict[str, Any]

SearchMode = Literal["grid", "random"]
ObjectiveMode = Literal["pnl_then_efficiency"]


# ============================================================
# Parameter Specs (simple, readable)
# ============================================================
@dataclass(frozen=True)
class ChoiceParam:
    key: str
    choices: Sequence[Any]

    def grid(self) -> List[Any]:
        return list(self.choices)

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(list(self.choices))


@dataclass(frozen=True)
class IntRangeParam:
    key: str
    low: int
    high: int
    step: int = 1

    def grid(self) -> List[int]:
        return list(range(int(self.low), int(self.high) + 1, int(self.step)))

    def sample(self, rng: random.Random) -> int:
        return rng.randrange(int(self.low), int(self.high) + 1, int(self.step))


@dataclass(frozen=True)
class FloatRangeParam:
    key: str
    low: float
    high: float
    step: Optional[float] = None  # if None => random only

    def grid(self) -> List[float]:
        if self.step is None:
            raise ValueError(f"{self.key}: step=None => cannot grid.")
        n = int(math.floor((self.high - self.low) / self.step)) + 1
        return [float(self.low + i * self.step) for i in range(n)]

    def sample(self, rng: random.Random) -> float:
        return float(rng.uniform(float(self.low), float(self.high)))


ParamSpec = Union[ChoiceParam, IntRangeParam, FloatRangeParam]


@dataclass
class SearchSpace:
    specs: List[ParamSpec]

    def grid(self) -> Iterable[Params]:
        keys = [s.key for s in self.specs]
        grids = [s.grid() for s in self.specs]
        for vals in itertools.product(*grids):
            yield dict(zip(keys, vals))

    def sample(self, rng: random.Random) -> Params:
        return {s.key: s.sample(rng) for s in self.specs}


def build_space(catalog: Dict[str, ParamSpec], active_keys: Sequence[str]) -> SearchSpace:
    missing = [k for k in active_keys if k not in catalog]
    if missing:
        raise KeyError(f"Unknown parameter keys: {missing}. Available: {list(catalog.keys())}")
    return SearchSpace([catalog[k] for k in active_keys])


# ============================================================
# Trial output
# ============================================================
@dataclass
class TrialResult:
    params: Params
    pnl: float
    traded_notional: float
    efficiency: float  # pnl / traded_notional
    score: Tuple[float, float]  # (pnl, efficiency) for lexicographic ranking
    metrics: Dict[str, float]
    notes: Dict[str, Any]


# ============================================================
# Objective
# ============================================================
def score_pnl_then_efficiency(pnl: float, efficiency: float) -> Tuple[float, float]:
    # lexicographic: primary pnl, tie-breaker efficiency
    return (float(pnl), float(efficiency))


# ============================================================
# Parameter application: Params dict -> EngineSpec
# Keys convention:
#   data.start, data.end
#   strategy.kind, strategy.fast_window, strategy.slow_window, strategy.window
#   portfolio.sizing_mode, portfolio.buy_pct_cash, portfolio.sell_pct_shares, etc.
# ============================================================
def _set_in_dict(d: Dict[str, Any], k: str, v: Any) -> None:
    d[k] = v


def apply_params_to_spec(base: EngineSpec, params: Params) -> EngineSpec:
    """
    Returns a NEW EngineSpec with params applied.
    Uses dataclasses.replace for frozen dataclasses.
    """
    spec = base

    # ---- DataConfig ----
    data_updates = {}
    if "data.start" in params:
        data_updates["start"] = params["data.start"]
    if "data.end" in params:
        data_updates["end"] = params["data.end"]
    if "data.symbol" in params:
        data_updates["symbols"] = [str(params["data.symbol"])]

    if data_updates:
        spec = replace(spec, data=replace(spec.data, **data_updates))

    # ---- StrategyConfig ----
    strat_updates = {}
    strat_params = dict(spec.strategy.params or {})

    if "strategy.kind" in params:
        strat_updates["kind"] = str(params["strategy.kind"])

    # MA cross
    if "strategy.fast_window" in params:
        strat_params["fast_window"] = int(params["strategy.fast_window"])
    if "strategy.slow_window" in params:
        strat_params["slow_window"] = int(params["strategy.slow_window"])

    # SMA price
    if "strategy.window" in params:
        strat_params["window"] = int(params["strategy.window"])

    # allow_short / nan_policy if you ever decide to tune them
    if "strategy.allow_short" in params:
        strat_params["allow_short"] = bool(params["strategy.allow_short"])
    if "strategy.nan_policy" in params:
        strat_params["nan_policy"] = str(params["strategy.nan_policy"])

    strat_updates["params"] = strat_params
    spec = replace(spec, strategy=replace(spec.strategy, **strat_updates))

    # ---- PortfolioConfig ----
    port = spec.portfolio
    port_updates = {}

    # common mechanics
    for key, field_name, cast in [
        ("portfolio.initial_cash", "initial_cash", float),
        ("portfolio.rebalance_policy", "rebalance_policy", str),
        ("portfolio.max_gross", "max_gross", float),
        ("portfolio.cash_buffer", "cash_buffer", float),
        ("portfolio.sizing_mode", "sizing_mode", str),
        ("portfolio.buy_pct_cash", "buy_pct_cash", float),
        ("portfolio.sell_pct_shares", "sell_pct_shares", float),
    ]:
        if key in params:
            port_updates[field_name] = cast(params[key])

    # costs
    # Note: cost_model is a dataclass in PortfolioConfig; we keep it unchanged unless you tune it.
    if port_updates:
        spec = replace(spec, portfolio=replace(port, **port_updates))

    # ---- IndicatorsConfig (usually inferred; keep as is) ----
    # If you later want to tune indicator engine config, you can add keys here.

    return spec


# ============================================================
# Constraints / validation to avoid wasting trials
# ============================================================
def validate_params(spec: EngineSpec) -> Tuple[bool, str]:
    """
    Return (ok, reason). Keep it minimal and explicit.
    """
    kind = str(spec.strategy.kind).lower()
    p = spec.strategy.params or {}

    if kind in ("ma_cross", "moving_average_cross"):
        fast = int(p.get("fast_window", 0))
        slow = int(p.get("slow_window", 0))
        if fast < 2 or slow < 3:
            return False, "MA cross windows too small"
        if fast >= slow:
            return False, "MA cross constraint violated: fast_window must be < slow_window"

    if kind in ("sma_price", "price_sma", "price_above_sma"):
        w = int(p.get("window", 0))
        if w < 2:
            return False, "SMA window too small"

    # sizing sanity
    if str(spec.portfolio.sizing_mode) == "pct_cash_shares":
        b = float(spec.portfolio.buy_pct_cash)
        s = float(spec.portfolio.sell_pct_shares)
        if not (0.0 < b <= 1.0):
            return False, "buy_pct_cash must be in (0,1]"
        if not (0.0 < s <= 1.0):
            return False, "sell_pct_shares must be in (0,1]"

    return True, "ok"


# ============================================================
# Compute PnL and Profit/Volume from a BacktestBundle
# ============================================================
def compute_pnl_and_efficiency(bundle) -> Tuple[float, float, float]:
    """
    Returns: (total_pnl, total_traded_notional, efficiency=pnl/notional)
    Efficiency = profit / volume (volume approximated by traded notional).
    """
    # PnL: prefer report series 'pnl' if present; else use equity delta
    rep = bundle.report
    if rep is not None and isinstance(rep.series, dict) and "pnl" in rep.series:
        pnl_series = rep.series["pnl"]
        total_pnl = float(pd.to_numeric(pnl_series, errors="coerce").fillna(0.0).sum())
    else:
        eq = bundle.portfolio_result.equity_curve
        total_pnl = float(eq.iloc[-1] - eq.iloc[0]) if len(eq) else 0.0

    # Volume: sum of absolute notional traded
    trades = bundle.portfolio_result.trades
    if trades is None or trades.empty or "notional" not in trades.columns:
        total_notional = 0.0
    else:
        total_notional = float(pd.to_numeric(trades["notional"], errors="coerce").fillna(0.0).sum())

    eff = float(total_pnl / total_notional) if total_notional > 0 else float("-inf" if total_pnl < 0 else 0.0)
    return total_pnl, total_notional, eff


# ============================================================
# Optimizer
# ============================================================
@dataclass
class OptimizeConfig:
    mode: SearchMode = "random"
    n_trials: int = 200                 # random trials
    seed: int = 42
    top_k: int = 30
    objective: ObjectiveMode = "pnl_then_efficiency"
    verbose: bool = False
    # If you want to hard-stop expensive trials:
    max_failures: int = 50


class Optimizer:
    def __init__(
        self,
        base_spec: EngineSpec,
        space: SearchSpace,
        cfg: OptimizeConfig,
        fixed_params: Optional[Params] = None,
    ) -> None:
        self.base_spec = base_spec
        self.space = space
        self.cfg = cfg
        self.fixed_params = dict(fixed_params or {})
        self.rng = random.Random(cfg.seed)

    def _iter_trials(self) -> Iterable[Params]:
        if self.cfg.mode == "grid":
            return self.space.grid()
        return (self.space.sample(self.rng) for _ in range(int(self.cfg.n_trials)))

    def run(self) -> Tuple[TrialResult, pd.DataFrame]:
        failures = 0
        results: List[TrialResult] = []

        for trial_params in self._iter_trials():
            params = dict(self.fixed_params)
            params.update(trial_params)

            # build spec
            spec_i = apply_params_to_spec(self.base_spec, params)

            ok, reason = validate_params(spec_i)
            if not ok:
                if self.cfg.verbose:
                    print(f"[skip] {reason} params={trial_params}")
                continue

            try:
                bundle = BacktestEngine(spec_i).run()
                pnl, notional, eff = compute_pnl_and_efficiency(bundle)

                if self.cfg.objective == "pnl_then_efficiency":
                    score = score_pnl_then_efficiency(pnl, eff)
                else:
                    score = (pnl, eff)

                metrics = dict(bundle.report.metrics or {})
                notes = {
                    "strategy_kind": str(spec_i.strategy.kind),
                    "data_start": spec_i.data.start,
                    "data_end": spec_i.data.end,
                }

                results.append(
                    TrialResult(
                        params=params,
                        pnl=pnl,
                        traded_notional=notional,
                        efficiency=eff,
                        score=score,
                        metrics=metrics,
                        notes=notes,
                    )
                )

            except Exception as e:
                failures += 1
                if self.cfg.verbose:
                    print(f"[fail] {type(e).__name__}: {e} params={trial_params}")
                if failures >= self.cfg.max_failures:
                    break

        if not results:
            raise RuntimeError("No successful trials. Check constraints / data range / parameter space.")

        # Rank: pnl desc, efficiency desc
        results_sorted = sorted(results, key=lambda r: (r.score[0], r.score[1]), reverse=True)
        best = results_sorted[0]

        df = self._results_to_df(results_sorted)
        df = df.head(int(self.cfg.top_k)).reset_index(drop=True)
        return best, df

    @staticmethod
    def _results_to_df(results: List[TrialResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            row = {
                "pnl": r.pnl,
                "traded_notional": r.traded_notional,
                "profit_per_notional": r.efficiency,
                "score_pnl": r.score[0],
                "score_eff": r.score[1],
                "strategy_kind": r.notes.get("strategy_kind"),
                "data_start": r.notes.get("data_start"),
                "data_end": r.notes.get("data_end"),
            }
            # add a few common metrics if present
            for k in ["Sharpe", "CAGR", "Total return", "Max drawdown"]:
                if k in r.metrics:
                    row[k] = r.metrics[k]
            # add params flattened
            for pk, pv in r.params.items():
                row[f"param::{pk}"] = pv
            rows.append(row)
        return pd.DataFrame(rows)


# ============================================================
# One-at-a-time optimization (param importance / sensitivity)
# ============================================================
def optimize_one_at_a_time(
    base_spec: EngineSpec,
    catalog: Dict[str, ParamSpec],
    keys: Sequence[str],
    cfg: OptimizeConfig,
    fixed_params: Optional[Params] = None,
) -> pd.DataFrame:
    """
    For each parameter in `keys`, optimize only that param (others fixed).
    Ranks parameters by best pnl improvement (primary) and efficiency (secondary).
    """
    fixed_params = dict(fixed_params or {})

    # baseline run
    base_bundle = BacktestEngine(base_spec).run()
    base_pnl, base_notional, base_eff = compute_pnl_and_efficiency(base_bundle)

    rows = []
    for k in keys:
        space = build_space(catalog, [k])
        opt = Optimizer(base_spec=base_spec, space=space, cfg=cfg, fixed_params=fixed_params)
        best, _top = opt.run()

        rows.append({
            "param_key": k,
            "baseline_pnl": base_pnl,
            "best_pnl": best.pnl,
            "delta_pnl": best.pnl - base_pnl,
            "baseline_profit_per_notional": base_eff,
            "best_profit_per_notional": best.efficiency,
            "best_value": best.params.get(k),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["delta_pnl", "best_profit_per_notional"], ascending=[False, False]).reset_index(drop=True)
    return df


# ============================================================
# Date-window catalog builder (for uploaded BMCE data)
# ============================================================
def build_date_window_choices_from_uploaded_bmce(
    base_data_cfg: DataConfig,
    symbol: Optional[str] = None,
    min_bars: int = 252,
    step_bars: int = 21,
    max_windows: int = 200,
) -> List[Tuple[str, str]]:
    """
    Loads MarketData ONCE to get index, then creates candidate (start,end) windows.
    Returns list of (start_iso, end_iso).

    Notes:
    - windows are forward in time
    - end is inclusive in your loaders (typically); this is fine for research
    """
    md = load_marketdata(base_data_cfg)
    sym = symbol or base_data_cfg.symbols[0]
    idx = pd.DatetimeIndex(md.bars[sym].index).sort_values()

    windows: List[Tuple[str, str]] = []
    n = len(idx)
    if n < min_bars:
        # fallback: just the full range
        return [(idx[0].date().isoformat(), idx[-1].date().isoformat())]

    # rolling window ends at various points; keep it simple
    count = 0
    for start_i in range(0, n - min_bars, step_bars):
        end_i = min(n - 1, start_i + min_bars - 1)
        # you can also make end_i vary, but keep simple here
        start_iso = idx[start_i].date().isoformat()
        end_iso = idx[-1].date().isoformat()  # full to end by default
        windows.append((start_iso, end_iso))
        count += 1
        if count >= max_windows:
            break

    return windows


def add_date_window_param(
    catalog: Dict[str, ParamSpec],
    windows: Sequence[Tuple[str, str]],
    key: str = "data.window",
) -> None:
    """
    Adds a single param 'data.window' whose choice is a tuple(start_iso, end_iso).
    Runner will map it into data.start/data.end.
    """
    catalog[key] = ChoiceParam(key=key, choices=list(windows))


def expand_window_param(params: Params) -> Params:
    """
    If params contains data.window=(start,end), expand into data.start/data.end.
    """
    if "data.window" in params:
        w = params["data.window"]
        if isinstance(w, (tuple, list)) and len(w) == 2:
            params = dict(params)
            params["data.start"] = str(w[0]) if w[0] is not None else None
            params["data.end"] = str(w[1]) if w[1] is not None else None
    return params


# ============================================================
# High-level convenience API (what your Streamlit page should call)
# ============================================================
def run_optimization(
    base_spec: EngineSpec,
    catalog: Dict[str, ParamSpec],
    active_keys: Sequence[str],
    cfg: OptimizeConfig,
    fixed_params: Optional[Params] = None,
) -> Tuple[TrialResult, pd.DataFrame, EngineSpec]:
    """
    Joint optimization across chosen active keys.

    Returns:
      best_trial, top_trials_df, best_spec
    """
    space = build_space(catalog, active_keys)
    opt = Optimizer(base_spec=base_spec, space=space, cfg=cfg, fixed_params=fixed_params)

    best, df = opt.run()

    # Expand window param if used
    best_params = expand_window_param(best.params)

    best_spec = apply_params_to_spec(base_spec, best_params)
    return best, df, best_spec


# ============================================================
# Default catalog (adapt ranges as you like)
# ============================================================
def default_param_catalog_for_your_app() -> Dict[str, ParamSpec]:
    """
    A sane starting catalog consistent with your Streamlit UI and engine.
    You can freely add/remove keys.
    """
    cat: Dict[str, ParamSpec] = {}

    # Strategy kind (optional to optimize; usually keep fixed)
    cat["strategy.kind"] = ChoiceParam("strategy.kind", ["ma_cross", "sma_price"])

    # MA cross windows
    if strategy_kind == "ma_cross":
        cat["strategy.fast_window"] = IntRangeParam("strategy.fast_window", 5, 80, step=1)
        cat["strategy.slow_window"] = IntRangeParam("strategy.slow_window", 20, 200, step=1)
    else :
        cat["strategy.window"] = IntRangeParam("strategy.window", 10, 200, step=1)

    # Portfolio sizing
    cat["portfolio.buy_pct_cash"] = FloatRangeParam("portfolio.buy_pct_cash", 0.05, 1.0, step=0.05)
    cat["portfolio.sell_pct_shares"] = FloatRangeParam("portfolio.sell_pct_shares", 0.05, 1.0, step=0.05)

    # Portfolio mechanics
    cat["portfolio.rebalance_policy"] = ChoiceParam("portfolio.rebalance_policy", ["on_change", "every_bar"])

    return cat
