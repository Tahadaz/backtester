# optimizer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Param grid
# -----------------------------
@dataclass(frozen=True)
class ParamGrid:
    """
    Deterministic parameter grid.
    Example:
      ParamGrid({"fast_window": [5, 10], "slow_window": [20, 30]})
    """
    grid: Dict[str, List[Any]]

    def iter_params(self) -> Iterable[Dict[str, Any]]:
        keys = sorted(self.grid.keys())
        vals = [self.grid[k] for k in keys]

        def rec(i: int, current: Dict[str, Any]):
            if i == len(keys):
                yield dict(current)
                return
            k = keys[i]
            for v in vals[i]:
                current[k] = v
                yield from rec(i + 1, current)

        yield from rec(0, {})

    def key(self, params: Dict[str, Any]) -> str:
        # stable, human-readable key
        items = ",".join(f"{k}={params[k]}" for k in sorted(params.keys()))
        return items


# -----------------------------
# Objective
# -----------------------------
@dataclass(frozen=True)
class Objective:
    """
    Maps metrics -> scalar score.
    score_fn receives the summary_table (from your Report) or a metrics dict.
    """
    name: str
    score_fn: Callable[[Dict[str, Any]], float]
    higher_is_better: bool = True

    def score(self, metrics: Dict[str, Any]) -> float:
        try:
            s = float(self.score_fn(metrics))
        except Exception:
            s = np.nan
        return s


# -----------------------------
# Run artifact contract
# -----------------------------
@dataclass
class RunArtifact:
    symbol: str
    params: Dict[str, Any]
    params_key: str
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    score: float
    metrics: Dict[str, Any]          # extracted scalar metrics
    equity: pd.Series
    returns: pd.Series
    report: Any                      # your Report object
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Runner (strategy -> execution -> analysis)
# -----------------------------
class BacktestRunner:
    """
    Adapter that knows how to:
      - instantiate strategy from params
      - ensure required features are computed (you can wire your IndicatorEngine here)
      - run execution engine
      - build a report
    """

    def __init__(
        self,
        symbol: str,
        bars: pd.DataFrame,
        feature_provider: Callable[[Dict[str, Any]], pd.DataFrame],
        strategy_factory: Callable[[Dict[str, Any]], Any],
        execution_engine: Any,
        report_builder: Any,
        interval: str = "1D",
        cache: Optional[Dict[str, RunArtifact]] = None,
    ):
        """
        feature_provider(params) -> features_df aligned to bars.index
        strategy_factory(params) -> strategy object with .generate(ctx) returning StrategyOutput (position at t)
        """
        self.symbol = symbol
        self.bars = bars
        self.feature_provider = feature_provider
        self.strategy_factory = strategy_factory
        self.execution_engine = execution_engine
        self.report_builder = report_builder
        self.interval = interval
        self.cache = cache if cache is not None else {}

    def _cache_key(self, start: pd.Timestamp, end: pd.Timestamp, params_key: str) -> str:
        raw = f"{self.symbol}|{start}|{end}|{params_key}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def run(
        self,
        params: Dict[str, Any],
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> RunArtifact:
        idx = self.bars.index
        start = idx.min() if start is None else pd.Timestamp(start)
        end = idx.max() if end is None else pd.Timestamp(end)

        bars_slice = self.bars.loc[start:end].copy()
        params_key = ",".join(f"{k}={params[k]}" for k in sorted(params.keys()))
        ckey = self._cache_key(start, end, params_key)

        if ckey in self.cache:
            return self.cache[ckey]

        # 1) Features
        feats = self.feature_provider(params)  # must be aligned to full bars index
        feats_slice = feats.loc[bars_slice.index]

        # 2) Strategy intent (position at t)
        strat = self.strategy_factory(params)

        # Construct minimal ctx matching your strategy.py
        # ctx.market.bars[symbol] and ctx.features.features[symbol]
        market = type("MD", (), {"bars": {self.symbol: bars_slice}})()
        features = type("FD", (), {"features": {self.symbol: feats_slice}})()
        ctx = type("CTX", (), {"market": market, "features": features, "symbol": self.symbol})()

        out = strat.generate(ctx)  # StrategyOutput with .position (Series)
        desired_pos = out.position.reindex(bars_slice.index)

        # 3) Execution (fills at t+1)
        pf = self.execution_engine.run_single_asset(
            symbol=self.symbol,
            bars=bars_slice,
            desired_position=desired_pos,
        )

        # 4) Analysis/report
        from others.analyzers import portfolio_to_result  # local import to avoid cycles
        result = portfolio_to_result(self.symbol, bars_slice, pf, interval=self.interval)
        report = self.report_builder.build(result)

        # Extract scalar metrics from summary_table
        # summary_table is DataFrame indexed by metric name with column "value"
        summary = report.summary_table
        metrics = {k: float(summary.loc[k, "value"]) if k in summary.index else np.nan
                   for k in ["Total Return", "CAGR", "Ann. Vol", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Trades", "Hit Rate", "Profit Factor", "Expectancy"]}

        artifact = RunArtifact(
            symbol=self.symbol,
            params=params,
            params_key=params_key,
            is_start=start,
            is_end=end,
            score=np.nan,  # filled by optimizer using objective
            metrics=metrics,
            equity=report.series["equity"],
            returns=report.series["returns"],
            report=report,
            meta={"diagnostics": getattr(out, "diagnostics", {})},
        )

        self.cache[ckey] = artifact
        return artifact


# -----------------------------
# Grid Search Optimizer
# -----------------------------
@dataclass
class GridSearchResult:
    table: pd.DataFrame
    best: RunArtifact
    artifacts: List[RunArtifact]


class GridSearchOptimizer:
    """
    Runs a param grid on a given time range (typically in-sample).
    """
    def __init__(self, runner: BacktestRunner, objective: Objective):
        self.runner = runner
        self.objective = objective

    def run(
        self,
        grid: ParamGrid,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        top_k: int = 10,
    ) -> GridSearchResult:
        artifacts: List[RunArtifact] = []

        for params in grid.iter_params():
            art = self.runner.run(params, start=start, end=end)
            art.score = self.objective.score(art.metrics)
            artifacts.append(art)

        # Build results table
        rows = []
        for a in artifacts:
            row = {
                "params": a.params_key,
                "score": a.score,
                **a.metrics,
                "start": a.is_start,
                "end": a.is_end,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("score", ascending=not self.objective.higher_is_better).reset_index(drop=True)

        best = artifacts[df.index[0]] if len(df) > 0 else artifacts[0]
        # safer: pick by argmax/argmin on score
        if self.objective.higher_is_better:
            best = max(artifacts, key=lambda x: (-np.inf if np.isnan(x.score) else x.score))
        else:
            best = min(artifacts, key=lambda x: (np.inf if np.isnan(x.score) else x.score))

        # Keep top_k artifacts
        top_keys = set(df.head(top_k)["params"].tolist())
        top_artifacts = [a for a in artifacts if a.params_key in top_keys]

        return GridSearchResult(table=df, best=best, artifacts=top_artifacts)


# -----------------------------
# Walk-Forward Optimization (WFO)
# -----------------------------
@dataclass
class WFOSegment:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def make_wfo_segments(
    index: pd.DatetimeIndex,
    is_bars: int,
    oos_bars: int,
    step_bars: Optional[int] = None,
) -> List[WFOSegment]:
    """
    Rolling WFO segments:
      optimize on [i : i+is_bars)
      test on     [i+is_bars : i+is_bars+oos_bars)
      step forward by step_bars (default = oos_bars)
    """
    step = oos_bars if step_bars is None else step_bars
    segs: List[WFOSegment] = []
    n = len(index)

    i = 0
    while True:
        is_start_i = i
        is_end_i = i + is_bars - 1
        oos_start_i = is_end_i + 1
        oos_end_i = oos_start_i + oos_bars - 1

        if oos_end_i >= n:
            break

        segs.append(WFOSegment(
            is_start=index[is_start_i],
            is_end=index[is_end_i],
            oos_start=index[oos_start_i],
            oos_end=index[oos_end_i],
        ))
        i += step

    return segs


@dataclass
class WFOResult:
    oos_equity: pd.Series
    oos_returns: pd.Series
    fold_table: pd.DataFrame          # per-fold metrics + chosen params
    param_stability: pd.DataFrame     # distribution of best params
    per_fold_best: List[RunArtifact]  # best IS artifacts
    per_fold_oos: List[RunArtifact]   # OOS artifacts evaluated with best params


class WalkForwardOptimizer:
    """
    Institutional WFO:
      For each fold:
        - grid search on IS window
        - take best params
        - evaluate once on OOS window
      Stitch OOS equity and compute OOS report-ready series.
    """
    def __init__(self, runner: BacktestRunner, objective: Objective):
        self.runner = runner
        self.objective = objective

    def run(
        self,
        grid: ParamGrid,
        is_bars: int,
        oos_bars: int,
        step_bars: Optional[int] = None,
    ) -> WFOResult:
        idx = self.runner.bars.index
        segs = make_wfo_segments(idx, is_bars=is_bars, oos_bars=oos_bars, step_bars=step_bars)

        per_fold_best: List[RunArtifact] = []
        per_fold_oos: List[RunArtifact] = []
        fold_rows: List[Dict[str, Any]] = []

        for fold_id, seg in enumerate(segs, start=1):
            # 1) Optimize on IS
            gso = GridSearchOptimizer(self.runner, self.objective)
            is_res = gso.run(grid, start=seg.is_start, end=seg.is_end, top_k=5)
            best_is = is_res.best
            per_fold_best.append(best_is)

            # 2) Evaluate best on OOS
            oos_art = self.runner.run(best_is.params, start=seg.oos_start, end=seg.oos_end)
            oos_art.score = self.objective.score(oos_art.metrics)
            per_fold_oos.append(oos_art)

            fold_rows.append({
                "fold": fold_id,
                "is_start": seg.is_start, "is_end": seg.is_end,
                "oos_start": seg.oos_start, "oos_end": seg.oos_end,
                "best_params": best_is.params_key,
                "is_score": best_is.score,
                "oos_score": oos_art.score,
                **{f"is_{k}": v for k, v in best_is.metrics.items()},
                **{f"oos_{k}": v for k, v in oos_art.metrics.items()},
            })

        fold_table = pd.DataFrame(fold_rows)

        # Stitch OOS equity (concatenate, avoiding overlapping endpoints if any)
        equities = []
        for art in per_fold_oos:
            eq = art.equity.copy()
            equities.append(eq)

        oos_equity = pd.concat(equities).sort_index()
        oos_returns = oos_equity.pct_change().rename("oos_returns")

        # Param stability table
        stability = self._param_stability(per_fold_best)

        return WFOResult(
            oos_equity=oos_equity,
            oos_returns=oos_returns,
            fold_table=fold_table,
            param_stability=stability,
            per_fold_best=per_fold_best,
            per_fold_oos=per_fold_oos,
        )

    def _param_stability(self, best_arts: List[RunArtifact]) -> pd.DataFrame:
        """
        Breakdown of chosen params across folds. Helps detect parameter fragility.
        """
        keys = [a.params_key for a in best_arts]
        vc = pd.Series(keys).value_counts().rename_axis("params").reset_index(name="count")
        vc["freq"] = vc["count"] / vc["count"].sum()
        return vc


# -----------------------------
# Plot helpers for optimizer outputs
# -----------------------------
def plot_wfo_oos_equity(oos_equity: pd.Series):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    oos_equity.plot(ax=ax)
    ax.set_title("Walk-Forward Out-of-Sample Equity (stitched)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    return fig


def plot_param_timeline(fold_table: pd.DataFrame):
    """
    Simple diagnostic: best_params chosen over folds.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = fold_table["fold"]
    y = fold_table["best_params"]
    ax.plot(x, range(len(y)))  # placeholder line to force a plot area
    ax.set_title("Best Params per Fold (see labels)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Index (labels below)")
    # annotate
    for i, txt in enumerate(y.tolist(), start=1):
        ax.annotate(txt, (i, i - 1))
    return fig
