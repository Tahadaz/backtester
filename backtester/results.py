# results.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestReport:
    metrics: Dict[str, float]
    series: Dict[str, pd.Series]
    tables: Dict[str, pd.DataFrame]
    plots: Dict[str, Any]
    style: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


class ResultsAnalyzer:
    def __init__(self, periods_per_year: int = 252, rf_annual: float = 0.0) -> None:
        self.periods_per_year = periods_per_year
        self.rf_annual = rf_annual

    # -----------------------------
    # Core public API
    # -----------------------------
    def analyze(
        self,
        portfolio_result,
        market_data,
        symbols: Sequence[str],
        features_data=None,
        plot_indicators: Optional[List[str]] = None,
        benchmark_market_data=None,
        benchmark_symbol: Optional[str] = None,
    ) -> BacktestReport:
        # --- core strategy series ---
        equity = portfolio_result.equity_curve.astype(float).sort_index()
        rets = portfolio_result.returns.astype(float).reindex(equity.index).fillna(0.0)

        cum = (1.0 + rets).cumprod() - 1.0
        dd = self._drawdown_from_equity(equity)

        # --- price & indicators panel (single-asset first) ---
        sym0 = symbols[0]
        px = market_data.bars[sym0]["Close"].reindex(equity.index).astype(float)

        ind_df = None
        if features_data is not None and plot_indicators:
            feats_sym = features_data.features[sym0]
            cols = [c for c in plot_indicators if c in feats_sym.columns]
            if cols:
                ind_df = feats_sym[cols].reindex(equity.index)

        trades = portfolio_result.trades.copy()
        trades = self._prepare_trades_table(trades)

        # --- benchmark series ---
        bench_cum = None
        bench_rets = None
        rel_cum = None
        if benchmark_market_data is not None:
            # benchmark symbol
            bsym = benchmark_symbol or list(benchmark_market_data.bars.keys())[0]
            bpx = benchmark_market_data.bars[bsym]["Close"].astype(float)

            # align to strategy index
            bpx = bpx.reindex(equity.index).ffill()
            bench_rets = bpx.pct_change().fillna(0.0)
            bench_cum = (1.0 + bench_rets).cumprod() - 1.0
            rel_cum = (1.0 + rets).cumprod() / (1.0 + bench_rets).cumprod() - 1.0

        # --- monthly & yearly returns ---
        monthly = self._monthly_return_matrix(rets)
        yearly = self._yearly_returns(rets)

        # --- headline metrics (strategy + optionally benchmark) ---
        metrics = self._metrics_block(rets, equity, dd, bench_rets)

        # --- tables ---
        timeseries = pd.DataFrame({
            "equity": equity,
            "returns": rets,
            "cum_returns": cum,
            "drawdown": dd,
        })
        if bench_cum is not None:
            timeseries["bench_cum_returns"] = bench_cum
            timeseries["rel_cum_returns"] = rel_cum

        strat_vs_bench = None
        if bench_rets is not None:
            strat_vs_bench = self._strategy_vs_benchmark_table(rets, bench_rets)

        tables = {
            "trades": trades,
            "timeseries": timeseries,
            "monthly_returns": monthly,
            "yearly_returns": yearly.to_frame("year_return"),
        }
        if strat_vs_bench is not None:
            tables["strategy_vs_benchmark"] = strat_vs_bench

        # --- plot payloads (plot-ready objects, no matplotlib here) ---
        plots = {
            "price_panel": {
                "price": px,
                "indicators": ind_df,   # may be None
                "trades": trades,       # contains timestamp/qty/side/price
            },
            "cum_vs_bench": {
                "strategy": cum,
                "benchmark": bench_cum,
            },
            "drawdown": dd,
            "monthly_heatmap": monthly,
            "yearly_bar": yearly,
        }

        style = self._style_spec()

        return BacktestReport(
            metrics=metrics,
            series={
                "equity": equity,
                "returns": rets,
                "cum_returns": cum,
                "drawdown": dd,
                **({} if bench_cum is None else {
                    "bench_cum_returns": bench_cum,
                    "bench_returns": bench_rets,
                    "rel_cum_returns": rel_cum,
                })
            },
            tables=tables,
            plots=plots,
            style=style,
            meta={
                "symbols": list(symbols),
                "benchmark_symbol": benchmark_symbol,
            }
        )

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _prepare_trades_table(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades is None or trades.empty:
            return pd.DataFrame(columns=["timestamp", "symbol", "qty", "side", "price", "notional", "cost"])

        df = trades.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "qty" in df.columns and "side" not in df.columns:
            df["side"] = np.where(df["qty"] > 0, "BUY", "SELL")
        # keep only useful columns (keep extra costs if you want)
        keep = [c for c in ["timestamp", "symbol", "qty", "side", "price", "notional", "cost", "commission_ht", "vat", "slippage"] if c in df.columns]
        df = df[keep].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return df

    def _drawdown_from_equity(self, equity: pd.Series) -> pd.Series:
        peak = equity.cummax()
        dd = equity / peak - 1.0
        return dd.fillna(0.0)

    def _annualized_return(self, rets: pd.Series) -> float:
        # geometric
        n = rets.shape[0]
        if n <= 1:
            return np.nan
        total = (1.0 + rets).prod()
        return total ** (self.periods_per_year / n) - 1.0

    def _annualized_vol(self, rets: pd.Series) -> float:
        return float(rets.std(ddof=1) * np.sqrt(self.periods_per_year))

    def _sharpe(self, rets: pd.Series) -> float:
        rf = self.rf_annual
        mu = self._annualized_return(rets)
        vol = self._annualized_vol(rets)
        if vol == 0 or np.isnan(vol):
            return np.nan
        return (mu - rf) / vol

    def _max_drawdown(self, dd: pd.Series) -> float:
        return float(dd.min()) if dd is not None and len(dd) else np.nan

    def _metrics_block(self, rets: pd.Series, equity: pd.Series, dd: pd.Series, bench_rets: Optional[pd.Series]) -> Dict[str, float]:
        out = {
            "Total return": float((1.0 + rets).prod() - 1.0),
            "CAGR": float(self._annualized_return(rets)),
            "Vol (ann.)": float(self._annualized_vol(rets)),
            "Sharpe": float(self._sharpe(rets)),
            "Max drawdown": float(self._max_drawdown(dd)),
        }
        if bench_rets is not None:
            out["Bench total return"] = float((1.0 + bench_rets).prod() - 1.0)
        return out

    def _monthly_return_matrix(self, rets: pd.Series) -> pd.DataFrame:
        # group by year-month
        r = rets.copy()
        r.index = pd.to_datetime(r.index)
        m = r.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)
        if m.empty:
            return pd.DataFrame()

        df = m.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot(index="year", columns="month", values="ret").sort_index()
        # make month labels 1..12; app can rename to Jan..Dec
        return pivot

    def _yearly_returns(self, rets: pd.Series) -> pd.Series:
        r = rets.copy()
        r.index = pd.to_datetime(r.index)
        y = r.resample("Y").apply(lambda x: (1.0 + x).prod() - 1.0)
        if y.empty:
            return pd.Series(dtype=float)
        y.index = y.index.year
        y.name = "year_return"
        return y

    def _strategy_vs_benchmark_table(self, strat_rets: pd.Series, bench_rets: pd.Series) -> pd.DataFrame:
        # Align
        s = strat_rets.reindex(bench_rets.index).fillna(0.0)
        b = bench_rets.reindex(s.index).fillna(0.0)

        sdd = self._drawdown_from_equity((1.0 + s).cumprod())
        bdd = self._drawdown_from_equity((1.0 + b).cumprod())

        rows = [
            ("Total return", float((1.0 + s).prod() - 1.0), float((1.0 + b).prod() - 1.0)),
            ("CAGR", float(self._annualized_return(s)), float(self._annualized_return(b))),
            ("Vol (ann.)", float(self._annualized_vol(s)), float(self._annualized_vol(b))),
            ("Sharpe", float(self._sharpe(s)), float(self._sharpe(b))),
            ("Max drawdown", float(sdd.min()), float(bdd.min())),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Strategy", "Benchmark"]).set_index("Metric")

    def _style_spec(self) -> Dict[str, Any]:
        # App uses this to color tables/metrics consistently
        return {
            "good_high": {"metrics": ["Total return", "CAGR", "Sharpe"], "tables": ["Strategy", "Benchmark"]},
            "good_low": {"metrics": ["Vol (ann.)", "Max drawdown"], "tables": []},
        }
