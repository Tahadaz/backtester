# backtest_engine.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import json
import os
import hashlib
import pandas as pd


# ---------- Types you already have (import in your project) ----------
# from data import BaseDataSource, MarketData
# from indicators import IndicatorEngine, FeatureSpec, FeaturesData
# from analyzers import ReportBuilder, portfolio_to_result, Report
# from execution_portfolio import ExecutionEngine

# Minimal Protocol-like aliases
MarketData = Any
FeaturesData = Any
FeatureSpec = Any
BaseDataSource = Any
IndicatorEngine = Any
ExecutionEngine = Any
ReportBuilder = Any
Report = Any


# -----------------------------
# Experiment spec
# -----------------------------
@dataclass(frozen=True)
class ExperimentSpec:
    """
    Defines one end-to-end run.

    For v1 single-stock:
      - symbol: str
      - strategy_factory: params -> strategy instance
      - feature_specs_factory: params -> list[FeatureSpec]
    """
    name: str
    symbol: str
    params: Dict[str, Any]

    # Factories to construct strategy and requested features (keeps engine generic)
    strategy_factory: Callable[[Dict[str, Any]], Any]
    feature_specs_factory: Callable[[Dict[str, Any]], List[FeatureSpec]]

    # Date slicing
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None

    # Meta
    data_kwargs: Dict[str, Any] = field(default_factory=dict)

    tags: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Run bundle returned by engine
# -----------------------------
@dataclass
class RunBundle:
    run_id: str
    spec: ExperimentSpec
    market_data: MarketData
    features_data: FeaturesData
    portfolio: Any
    report: Report
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Result store (organized outputs)
# -----------------------------
class ResultStore:
    def __init__(self, root_dir: str = "runs"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def make_run_dir(self, run_id: str) -> str:
        run_dir = os.path.join(self.root_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        for sub in ["tables", "series", "figures"]:
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
        return run_dir

    def save_json(self, run_dir: str, name: str, obj: Dict[str, Any]) -> None:
        path = os.path.join(run_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, default=str, indent=2)

    def save_table_csv(self, run_dir: str, name: str, df: pd.DataFrame) -> None:
        path = os.path.join(run_dir, "tables", f"{name}.csv")
        df.to_csv(path, index=True)

    def save_series_parquet(self, run_dir: str, name: str, s: pd.Series) -> None:
        # parquet is ideal; fall back to csv if needed
        path = os.path.join(run_dir, "series", f"{name}.parquet")
        s.to_frame(name).to_parquet(path)

    def save_figure_png(self, run_dir: str, name: str, fig: Any) -> None:
        path = os.path.join(run_dir, "figures", f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")

from results import BacktestResult, ResultsBuilder
import pandas as pd

def _get_first_series_attr(obj, candidates):
    for name in candidates:
        if hasattr(obj, name):
            val = getattr(obj, name)
            # allow DataFrame with column 'total' etc.
            if isinstance(val, pd.DataFrame) and val.shape[1] == 1:
                return val.iloc[:, 0]
            if isinstance(val, pd.Series):
                return val
    return None

# -----------------------------
# Backtest engine (Cerebro-like orchestrator)
# -----------------------------
class BacktestEngine:
    """
    Central orchestrator:
      data -> features -> strategy intent -> execution -> analyzers -> persistence -> return bundle

    Inspired by:
      - Backtrader Cerebro as central coordinator :contentReference[oaicite:4]{index=4}
      - QuantConnect modular pipeline :contentReference[oaicite:5]{index=5}
      - QuantStart component swapping + system outline :contentReference[oaicite:6]{index=6}
    """

    def __init__(self, data_source, indicator_engine, execution_engine, report_builder, result_store=None):
        self.data_source = data_source
        self.indicator_engine = indicator_engine
        self.execution_engine = execution_engine
        self.report_builder = report_builder
        self.store = result_store

    def _run_id(self, spec: ExperimentSpec) -> str:
        payload = {
            "name": spec.name,
            "symbol": spec.symbol,
            "params": {k: spec.params[k] for k in sorted(spec.params)},
            "start": str(spec.start) if spec.start is not None else "",
            "end": str(spec.end) if spec.end is not None else "",
            "data_kwargs": spec.data_kwargs,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def run(self, spec: ExperimentSpec, persist: bool = True):
        run_id = self._run_id(spec)

        # 1) Load MarketData using your real signature
        # spec.data_kwargs should include: paths, interval, etc.
        market_data = self.data_source.load(
            symbols=[spec.symbol],
            start=str(spec.start) if spec.start is not None else None,
            end=str(spec.end) if spec.end is not None else None,
            **spec.data_kwargs,
        )

        bars = market_data.bars[spec.symbol]

        # 2) Compute features
        feature_specs = spec.feature_specs_factory(spec.params)
        

        features_data = self.indicator_engine.compute(
            market_data=market_data,
            specs=feature_specs,
            symbols=[spec.symbol],
        )
        feats_df = features_data.features[spec.symbol].reindex(bars.index)

        # 3) Strategy intent
        strategy = spec.strategy_factory(spec.params)

        # wrappers
        md = type("MD", (), {"bars": {spec.symbol: bars}})()
        fd = type("FD", (), {"features": {spec.symbol: feats_df}})()

        # single source-of-truth mapping: FeatureSpec.name -> produced column name
        feature_map = {fs.name: fs.canonical_name() for fs in feature_specs if getattr(fs, "name", None)}

        ctx = type("CTX", (), {
            "market": md,
            "features": fd,
            "symbol": spec.symbol,
            "feature_map": feature_map,
        })()


        strat_out = strategy.generate(ctx)
        desired_position = strat_out.position.reindex(bars.index)

        # 4) Execution
        portfolio = self.execution_engine.run_single_asset(
            symbol=spec.symbol,
            bars=bars,
            desired_position=desired_position,
        )

        # 1) Core series
        prices = bars["Close"].astype(float)
        equity = _get_first_series_attr(
            portfolio,
            candidates=["equity", "equity_curve", "total", "portfolio_value", "value", "nav"]
        )
        if equity is None:
            # Some implementations store a dataframe of history
            hist = _get_first_series_attr(portfolio, candidates=["history", "curve", "df", "results"])
            if hist is not None:
                equity = hist
        if equity is None:
            raise AttributeError(
                "Could not find an equity curve on Portfolio. "
                "Expected one of: equity/equity_curve/total/portfolio_value/value/nav "
                "or a history/results dataframe."
            )

        equity = equity.astype(float)
        returns = equity.pct_change().fillna(0.0)

        # 2) Trades (if you don't have a ledger yet, pass None and results.py will infer from position)
        trades_df = getattr(portfolio, "trades", None)

        # 3) Position (exec


        # 5) Report
        from others.analyzers import portfolio_to_result
        interval = spec.data_kwargs.get("interval", "1d")
        result = portfolio_to_result(spec.symbol, bars, portfolio, interval=str(interval))
        report = self.report_builder.build(result)

        bundle = RunBundle(
            run_id=run_id,
            spec=spec,
            market_data=market_data,
            features_data=features_data,
            portfolio=portfolio,
            report=report,
            meta={"strategy_diagnostics": getattr(strat_out, "diagnostics", {})},
        )


        # optional persistence (if you already implemented ResultStore)
        if persist and self.store is not None:
            self._persist(bundle)

        return bundle

    def _persist(self, bundle: RunBundle) -> None:
        run_dir = self.store.make_run_dir(bundle.run_id)

        # Meta
        meta = {
            "run_id": bundle.run_id,
            "experiment": {
                "name": bundle.spec.name,
                "symbol": bundle.spec.symbol,
                "params": bundle.spec.params,
                "start": str(bundle.spec.start),
                "end": str(bundle.spec.end),
                "interval": bundle.spec.data_kwargs.get("interval", ""),
                "tags": bundle.spec.tags,
            },
            "bundle_meta": bundle.meta,
        }
        self.store.save_json(run_dir, "meta", meta)

        # Tables
        self.store.save_table_csv(run_dir, "summary", bundle.report.summary_table)
        self.store.save_table_csv(run_dir, "trades", bundle.report.trade_table)
        self.store.save_table_csv(run_dir, "monthly_returns", bundle.report.monthly_table)
        self.store.save_table_csv(run_dir, "drawdowns", bundle.report.drawdown_table)

        # Series
        for name, series in bundle.report.series.items():
            if isinstance(series, pd.Series):
                self.store.save_series_parquet(run_dir, name, series)

        # Figures
        for name, fig in bundle.report.figures.items():
            self.store.save_figure_png(run_dir, name, fig)


# -----------------------------
# Batch runner for multiple strategies / param sets
# -----------------------------
@dataclass
class BatchResult:
    table: pd.DataFrame
    bundles: List[RunBundle]


class BacktestBatch:
    """
    Runs multiple ExperimentSpec and produces a ranking table for quick comparison.
    """
    def __init__(self, engine: BacktestEngine):
        self.engine = engine

    def run(self, specs: Sequence[ExperimentSpec], persist: bool = True) -> BatchResult:
        bundles: List[RunBundle] = []
        rows: List[Dict[str, Any]] = []

        for spec in specs:
            b = self.engine.run(spec, persist=persist)
            bundles.append(b)

            # Extract common summary metrics
            st = b.report.summary_table
            def get(metric: str):
                return float(st.loc[metric, "value"]) if metric in st.index else float("nan")

            rows.append({
                "run_id": b.run_id,
                "name": spec.name,
                "symbol": spec.symbol,
                "params": json.dumps(spec.params, sort_keys=True),
                "Total Return": get("Total Return"),
                "CAGR": get("CAGR"),
                "Sharpe": get("Sharpe"),
                "Max Drawdown": get("Max Drawdown"),
                "Calmar": get("Calmar"),
                "Trades": get("Trades"),
            })

        table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
        return BatchResult(table=table, bundles=bundles)
