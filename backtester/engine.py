# engine.py
from __future__ import annotations

from data import YahooFinanceDataSource   # your class
from data import MarketData
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Callable, Literal, Union

from pathlib import Path

# ---- Project modules (adapt import paths if you have a package folder) ----
from data import MarketData
from data import BMCEDataSource
from indicators import IndicatorEngine, FeatureSpec, FeaturesData
from strategy import (
    BaseStrategy,
    MovingAverageCrossStrategy,
    MovingAverageCrossParams,
    SignalFrame,
)
from portfolio import PortfolioEngine, PortfolioConfig, PortfolioResult
from results import ResultsAnalyzer, BacktestReport


DataSourceKind = Literal["bmce", "yfinance"]
StrategyKind = Literal["ma_cross"]  # extend later


# -----------------------------
# Engine Spec Objects
# -----------------------------
@dataclass(frozen=True)
class DataConfig:
    source: DataSourceKind
    symbols: List[str]

    timezone: str = "GMT"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None

    # BMCE inputs:
    # - for single symbol: str/Path
    # - for multi symbols: dict {symbol: str/Path}
    bmce_paths: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None

    # yfinance inputs
    yf_period: str = "max"
    yf_interval: str = "1d"
    yf_auto_adjust: bool = False


@dataclass(frozen=True)
class IndicatorsConfig:
    """
    You can supply:
      - specs directly, OR
      - a builder that returns specs, OR
      - leave empty and let the engine infer specs from strategy kind (only for known strategies)
    """
    specs: Optional[List[FeatureSpec]] = None
    spec_builder: Optional[Callable[[], List[FeatureSpec]]] = None
    
    cache_dir: Optional[str] = ".cache/features"
    enable_disk_cache: bool = True
    enable_memory_cache: bool = True
    engine_version: str = "v1"


@dataclass(frozen=True)
class StrategyConfig:
    kind: StrategyKind
    params: Dict[str, Any] = field(default_factory=dict)

BenchmarkSource = Literal["yahoo"]

@dataclass(frozen=True)
class BenchmarkConfig:
    enabled: bool = False
    symbol: str = "SPY"
    source: BenchmarkSource = "yahoo"
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"
    auto_adjust: bool = False


@dataclass(frozen=True)
class EngineSpec:
    data: DataConfig
    indicators: IndicatorsConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    plot_indicators: List[str] = field(default_factory=list)
    periods_per_year: int = 252
    rf_annual: float = 0.0



@dataclass(frozen=True)
class BacktestBundle:
    md: MarketData
    feats: FeaturesData
    signals: SignalFrame
    portfolio_result: PortfolioResult
    report: BacktestReport
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Indicator spec inference helpers
# -----------------------------
def sma_specs_for_ma_cross(fast_window: int, slow_window: int) -> List[FeatureSpec]:
    """
    Ensures features are named exactly as your strategy expects:
      sma_{fast_window}, sma_{slow_window}
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


def resolve_specs(ind_cfg: IndicatorsConfig, strat_cfg: StrategyConfig) -> List[FeatureSpec]:
    if ind_cfg.specs is not None:
        if not ind_cfg.specs:
            raise ValueError("IndicatorsConfig.specs is empty.")
        return ind_cfg.specs

    if ind_cfg.spec_builder is not None:
        specs = ind_cfg.spec_builder()
        if not specs:
            raise ValueError("IndicatorsConfig.spec_builder returned empty specs list.")
        return specs

    # Infer from known strategies
    if strat_cfg.kind == "ma_cross":
        fast = int(strat_cfg.params.get("fast_window", 20))
        slow = int(strat_cfg.params.get("slow_window", 50))
        return sma_specs_for_ma_cross(fast, slow)

    raise ValueError(
        "No indicator specs provided and no inference rule exists for this strategy kind. "
        "Provide IndicatorsConfig.specs or spec_builder."
    )


# -----------------------------
# Strategy registry
# -----------------------------
def build_strategy(cfg: StrategyConfig) -> BaseStrategy:
    if cfg.kind == "ma_cross":
        params = MovingAverageCrossParams(
            fast_window=int(cfg.params.get("fast_window", 20)),
            slow_window=int(cfg.params.get("slow_window", 50)),
            allow_short=bool(cfg.params.get("allow_short", True)),
            nan_policy=str(cfg.params.get("nan_policy", "flat")),
        )
        return MovingAverageCrossStrategy(params)

    raise ValueError(f"Unknown strategy kind: {cfg.kind}")


# -----------------------------
# Data loaders
# -----------------------------
def load_marketdata_bmce(cfg: DataConfig) -> MarketData:
    if cfg.bmce_paths is None:
        raise ValueError("BMCE source selected but bmce_paths is None.")

    ds = BMCEDataSource(timezone=cfg.timezone)

    # BaseDataSource.load(...) in your project returns MarketData (not dict)
    md_or_bars = ds.load(
        symbols=cfg.symbols,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        paths=cfg.bmce_paths,
    )

    # ✅ Case 1: BaseDataSource returns MarketData (your current behavior)
    if isinstance(md_or_bars, MarketData):
        return md_or_bars

    # ✅ Case 2: if you ever change load() to return dict[str, DataFrame]
    if isinstance(md_or_bars, dict):
        for s in cfg.symbols:
            if s not in md_or_bars:
                raise KeyError(f"BMCEDataSource returned no bars for '{s}'. Keys: {list(md_or_bars.keys())}")
        return MarketData(
            bars=md_or_bars,
            source="BMCEDataSource",
            timezone=cfg.timezone,
            interval=cfg.interval,
        )

    raise TypeError(f"BMCEDataSource.load returned unsupported type: {type(md_or_bars)}")


def load_marketdata_yfinance(cfg: DataConfig) -> MarketData:
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed. Add it to requirements or use BMCE source.") from e

    if cfg.interval != "1d":
        # keep explicit
        raise ValueError("yfinance loader in this engine currently supports interval='1d' only.")

    out: Dict[str, Any] = {}
    for sym in cfg.symbols:
        df = yf.download(
            tickers=sym,
            period=cfg.yf_period,
            interval=cfg.yf_interval,
            auto_adjust=cfg.yf_auto_adjust,
            progress=False,
        )
        if df is None or len(df) == 0:
            raise ValueError(f"yfinance returned empty data for symbol '{sym}'.")

        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"yfinance data missing columns {missing}. got={list(df.columns)}")
        out[sym] = df.sort_index()

    return MarketData(bars=out, source="yfinance", timezone=cfg.timezone, interval=cfg.yf_interval)


def load_marketdata(cfg: DataConfig) -> MarketData:
    if cfg.source == "bmce":
        return load_marketdata_bmce(cfg)
    if cfg.source == "yfinance":
        return load_marketdata_yfinance(cfg)
    raise ValueError(f"Unknown data source: {cfg.source}")


# -----------------------------
# Engine
# -----------------------------
class BacktestEngine:
    """
    Final orchestration layer:
      MarketData -> FeaturesData -> SignalFrame -> PortfolioResult -> BacktestReport
    """

    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec

    def run(self) -> BacktestBundle:
        # 1) Data
        md = load_marketdata(self.spec.data)
        symbols = self.spec.data.symbols
        
        # 2) Indicators
        specs = resolve_specs(self.spec.indicators, self.spec.strategy)
        ind = IndicatorEngine(
            cache_dir=self.spec.indicators.cache_dir,
            enable_disk_cache=self.spec.indicators.enable_disk_cache,
            enable_memory_cache=self.spec.indicators.enable_memory_cache,
            engine_version=self.spec.indicators.engine_version,
        )
        feats = ind.compute(md, specs=specs, symbols=symbols)

        # 3) Strategy
        strat = build_strategy(self.spec.strategy)
        sf = strat.generate_signals(md, feats, symbols=symbols)

        # 4) Portfolio
        port = PortfolioEngine(self.spec.portfolio)
        pres = port.run(md, sf, symbols=symbols)

        # inside BacktestEngine.run(), before calling ResultsAnalyzer.analyze(...)
        bmd = None
        bsym = None
        if getattr(self.spec, "benchmark", None) and self.spec.benchmark.enabled:
            bcfg = self.spec.benchmark
            yds = YahooFinanceDataSource(timezone=self.spec.data.timezone)
            bmd = yds.load(
                symbols=[bcfg.symbol],
                start=bcfg.start,
                end=bcfg.end,
                interval=bcfg.interval,
                auto_adjust=bcfg.auto_adjust,
                progress=False,
            )
            bsym = bcfg.symbol

        
        # 5) Results
        analyzer = ResultsAnalyzer(periods_per_year=self.spec.periods_per_year, rf_annual=self.spec.rf_annual)
        report = analyzer.analyze(
            pres,
            market_data=md,
            symbols=symbols,
            features_data=feats,
            plot_indicators=getattr(self.spec, "plot_indicators", None),
            benchmark_market_data=bmd,
            benchmark_symbol=bsym,
        )

        
        meta = {
            "symbols": symbols,
            "data_source": self.spec.data.source,
            "strategy_kind": self.spec.strategy.kind,
            "indicator_names": [s.name or s.indicator for s in specs],
            "portfolio": {k: v for k, v in self.spec.portfolio.__dict__.items() if k != "cost_model"},
        }

        return BacktestBundle(
            md=md,
            feats=feats,
            signals=sf,
            portfolio_result=pres,
            report=report,
            meta=meta,
        )





"""
TEXT EXPLANATION (for Cursor review)

What this engine layer does:
- It is the final orchestration layer after data.py, indicators.py, strategy.py, portfolio.py, results.py.
- You provide an EngineSpec that includes:
  - DataConfig: symbol(s), and whether data comes from BMCE (file paths) or yfinance
  - IndicatorsConfig: FeatureSpec list (or builder), otherwise inferred from strategy kind
  - StrategyConfig: which strategy to use (currently MA cross) and its parameters
  - PortfolioConfig: execution/accounting configuration (already implemented in portfolio.py)
- The engine then runs:
  MarketData -> IndicatorEngine.compute -> Strategy.generate_signals -> PortfolioEngine.run -> ResultsAnalyzer.analyze
- It returns a BacktestBundle containing intermediate artifacts (MarketData, FeaturesData, SignalFrame)
  plus final outputs (PortfolioResult and BacktestReport).

BMCE integration:
- Your BMCEDataSource provides _load_impl(...) and _read_one_file(...).
- This engine assumes BaseDataSource exposes a public `load(...)` that calls _load_impl internally.
- If BaseDataSource uses another public method name, you only change the one line:
    bars = ds.load(...)
  to bars = ds.<actual_method>(...).

Indicator inference:
- For strategy 'ma_cross', the engine auto-builds two FeatureSpec objects with indicator='sma'
  and name='sma_{window}' to match MovingAverageCrossStrategy.
"""
