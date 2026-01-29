# results.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Annualization = Literal["D"]  # daily bars for now


# -----------------------------
# Report containers
# -----------------------------
@dataclass(frozen=True)
class BacktestReport:
    """
    A structured report object that you can print, serialize, or feed to a UI layer.
    """
    metrics: Dict[str, float]
    series: Dict[str, pd.Series]
    tables: Dict[str, pd.DataFrame]
    meta: Dict[str, Any]


# -----------------------------
# Results / Analytics layer
# -----------------------------
class ResultsAnalyzer:
    """
    Consumes ONLY portfolio outputs (and optional prices) and produces:
      - headline metrics (CAGR, Sharpe, max DD, etc.)
      - series (cum returns, drawdown, exposure)
      - tables (monthly returns, trade summary)

    This mirrors how serious backtesters separate:
      portfolio/accounting  ->  analyzers/reporting
    """

    def __init__(
        self,
        annualization: Annualization = "D",
        periods_per_year: int = 252,
        rf_annual: float = 0.0,
    ):
        self.annualization = annualization
        self.ppy = int(periods_per_year)
        self.rf_annual = float(rf_annual)

    # -----------------------------
    # Public API
    # -----------------------------
    def analyze(
        self,
        portfolio_result: Any,  # expects .equity_curve, .returns, .positions, .trades, .meta
        market_data: Optional[Any] = None,  # optional, to compute exposures/trade markers
        symbols: Optional[Sequence[str]] = None,
        close_col: str = "Close",
    ) -> BacktestReport:
        """
        portfolio_result is expected to look like your PortfolioResult:
          - equity_curve: pd.Series
          - returns: pd.Series
          - positions: pd.DataFrame (shares)
          - trades: pd.DataFrame (timestamp,symbol,qty,price,notional,cost,...)
          - meta: dict

        market_data is optional. If provided, must expose market_data.bars[symbol][close_col].
        """
        eq = self._coerce_series(portfolio_result.equity_curve, name="equity_curve")
        rets = self._coerce_series(portfolio_result.returns, name="returns")

        if symbols is None:
            if hasattr(portfolio_result, "positions") and isinstance(portfolio_result.positions, pd.DataFrame):
                symbols = list(portfolio_result.positions.columns)
            else:
                symbols = []
        symbols = list(symbols)

        positions = self._coerce_positions(portfolio_result.positions, symbols)

        trades = self._coerce_trades(getattr(portfolio_result, "trades", None))

        # Align index
        idx = eq.index.intersection(rets.index)
        eq = eq.loc[idx].astype(float)
        rets = rets.loc[idx].astype(float)
        positions = positions.reindex(idx).fillna(0).astype(int)

        # Core derived series
        cumret = self._cumulative_returns_from_equity(eq)
        dd, dd_max, dd_start, dd_trough, dd_recovery, dd_duration = self._drawdown_stats(eq)

        # Optional exposures (need close prices)
        close_prices = None
        if market_data is not None and symbols:
            close_prices = self._build_close_matrix(market_data, idx, symbols, close_col=close_col)

        exposure = {}
        if close_prices is not None:
            exposure = self._compute_exposures(eq, positions, close_prices)

        # Metrics
        metrics = {}
        metrics.update(self._perf_metrics(eq, rets, dd_max))
        metrics.update(self._drawdown_metrics(dd_max, dd_duration))
        metrics.update(self._trade_metrics(trades, eq))

        # If exposure exists, add key exposure metrics
        if exposure:
            metrics.update(self._exposure_metrics(exposure))

        # Tables
        monthly = self._monthly_returns_table(rets)
        trade_summary = self._trade_summary_table(trades)

        series = {
            "equity": eq,
            "returns": rets,
            "cum_returns": cumret,
            "drawdown": dd,
        }
        series.update(exposure)

        tables = {
            "monthly_returns": monthly,
            "trades": trade_summary,
        }

        meta = {
            "periods_per_year": self.ppy,
            "rf_annual": self.rf_annual,
            "input_meta": getattr(portfolio_result, "meta", {}),
        }

        return BacktestReport(metrics=metrics, series=series, tables=tables, meta=meta)

    # -----------------------------
    # Plot helpers (matplotlib)
    # -----------------------------
    def plot_cumulative_returns(self, report: BacktestReport, title: str = "Cumulative Returns") -> None:
        s = report.series["cum_returns"].dropna()
        plt.figure(figsize=(12, 5))
        plt.plot(s.index, s.values)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative return")
        plt.grid(True)
        plt.show()

    def plot_equity_curve(self, report: BacktestReport, title: str = "Equity Curve") -> None:
        s = report.series["equity"].dropna()
        plt.figure(figsize=(12, 5))
        plt.plot(s.index, s.values)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Equity (NAV)")
        plt.grid(True)
        plt.show()

    def plot_drawdown(self, report: BacktestReport, title: str = "Drawdown") -> None:
        s = report.series["drawdown"].dropna()
        plt.figure(figsize=(12, 4))
        plt.plot(s.index, s.values)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.show()

    def plot_returns_hist(self, report: BacktestReport, title: str = "Returns Histogram", bins: int = 50) -> None:
        r = report.series["returns"].dropna()
        plt.figure(figsize=(10, 4))
        plt.hist(r.values, bins=bins)
        plt.title(title)
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    # -----------------------------
    # Core metric computations
    # -----------------------------
    def _perf_metrics(self, equity: pd.Series, returns: pd.Series, max_dd: float) -> Dict[str, float]:
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

        n = len(equity)
        years = n / float(self.ppy) if self.ppy > 0 else np.nan
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else np.nan

        vol_ann = float(returns.std(ddof=1) * np.sqrt(self.ppy)) if len(returns) > 1 else np.nan

        rf_period = (1.0 + self.rf_annual) ** (1.0 / self.ppy) - 1.0
        excess = returns - rf_period
        sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(self.ppy)) if excess.std(ddof=1) and excess.std(ddof=1) > 0 else np.nan

        downside = returns.copy()
        downside[downside > 0] = 0.0
        downside_std = float(downside.std(ddof=1))
        sortino = float(excess.mean() / downside_std * np.sqrt(self.ppy)) if downside_std and downside_std > 0 else np.nan

        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

        return {
            "total_return": total_return,
            "CAGR": cagr,
            "vol_annual": vol_ann,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
        }

    @staticmethod
    def _drawdown_stats(equity: pd.Series) -> Tuple[pd.Series, float, Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp], float]:
        """
        Returns:
          drawdown series (<=0),
          max_dd (most negative),
          start, trough, recovery timestamps,
          duration in bars (peak->recovery) for max DD episode when recover exists.
        """
        eq = equity.astype(float)
        peak = eq.cummax()
        dd = eq / peak - 1.0

        max_dd = float(dd.min()) if len(dd) else np.nan
        trough = dd.idxmin() if len(dd) else None

        # Identify peak (start) before trough
        start = None
        recovery = None
        duration = np.nan

        if trough is not None and not np.isnan(max_dd):
            peak_before = eq.loc[:trough].idxmax()
            start = peak_before

            # recovery: first time equity exceeds previous peak after trough
            peak_level = float(eq.loc[start])
            post = eq.loc[trough:]
            rec = post[post >= peak_level]
            if len(rec) > 0:
                recovery = rec.index[0]
                duration = float((equity.index.get_loc(recovery) - equity.index.get_loc(start)))
            else:
                recovery = None
                duration = np.nan

        return dd, max_dd, start, trough, recovery, duration

    @staticmethod
    def _drawdown_metrics(max_dd: float, dd_duration: float) -> Dict[str, float]:
        return {
            "max_drawdown": float(max_dd),
            "max_dd_duration_bars": float(dd_duration) if not np.isnan(dd_duration) else np.nan,
        }

    @staticmethod
    def _trade_metrics(trades: pd.DataFrame, equity: pd.Series) -> Dict[str, float]:
        """
        Lightweight, robust trade stats using fill-level data.
        For round-trip PnL you’d need matching logic; we keep this MVP-safe.
        """
        if trades is None or trades.empty:
            return {
                "num_fills": 0.0,
                "turnover": 0.0,
                "total_cost": 0.0,
                "cost_as_pct_initial_equity": 0.0,
            }

        # Turnover: sum(abs(notional)) / avg equity
        total_notional = float(trades["notional"].abs().sum()) if "notional" in trades.columns else 0.0
        avg_equity = float(equity.mean()) if len(equity) else np.nan
        turnover = float(total_notional / avg_equity) if avg_equity and avg_equity > 0 else np.nan

        total_cost = float(trades["cost"].sum()) if "cost" in trades.columns else 0.0
        cost_pct_init = float(total_cost / float(equity.iloc[0])) if len(equity) and float(equity.iloc[0]) > 0 else np.nan

        return {
            "num_fills": float(len(trades)),
            "turnover": turnover,
            "total_cost": total_cost,
            "cost_as_pct_initial_equity": cost_pct_init,
        }

    @staticmethod
    def _trade_summary_table(trades: pd.DataFrame) -> pd.DataFrame:
        if trades is None or trades.empty:
            return pd.DataFrame(columns=["timestamp", "symbol", "qty", "price", "notional", "cost"])

        cols = [c for c in ["timestamp", "symbol", "qty", "price", "notional", "cost"] if c in trades.columns]
        out = trades[cols].copy()
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"])
            out = out.sort_values("timestamp")
        return out.reset_index(drop=True)

    def _monthly_returns_table(self, returns: pd.Series) -> pd.DataFrame:
        """
        Classic monthly returns heatmap table (as a DataFrame).
        """
        r = returns.dropna().copy()
        if r.empty:
            return pd.DataFrame()

        monthly = (1.0 + r).resample("M").prod() - 1.0
        df = monthly.to_frame("return")
        df["year"] = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot(index="year", columns="month", values="return").sort_index()
        pivot.columns = [self._month_name(m) for m in pivot.columns]
        return pivot

    @staticmethod
    def _month_name(m: int) -> str:
        names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        return names.get(int(m), str(m))

    # -----------------------------
    # Exposure computations (optional)
    # -----------------------------
    def _build_close_matrix(
        self,
        market_data: Any,
        idx: pd.Index,
        symbols: Sequence[str],
        close_col: str = "Close",
    ) -> pd.DataFrame:
        """
        Build a Close price matrix aligned to portfolio timestamps.
        Requires: market_data.bars[symbol] DataFrame with close_col and datetime index.
        """
        mats = []
        for s in symbols:
            bars = market_data.bars[s]
            if close_col not in bars.columns:
                raise KeyError(f"Bars for '{s}' missing '{close_col}' column.")
            ser = bars[close_col].reindex(idx).astype(float)
            mats.append(ser.rename(s))
        close_px = pd.concat(mats, axis=1)
        return close_px

    @staticmethod
    def _compute_exposures(
        equity: pd.Series,
        positions: pd.DataFrame,
        close_prices: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        """
        Exposure series:
          - position_value per symbol
          - weights per symbol
          - gross_exposure, net_exposure
          - time_in_market (binary series)
        """
        # position values
        pos_val = positions.astype(float) * close_prices.astype(float)
        # weights: value / equity
        w = pos_val.div(equity, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        gross = w.abs().sum(axis=1)
        net = w.sum(axis=1)
        in_mkt = (positions.abs().sum(axis=1) > 0).astype(float)

        out = {
            "gross_exposure": gross.rename("gross_exposure"),
            "net_exposure": net.rename("net_exposure"),
            "time_in_market": in_mkt.rename("time_in_market"),
        }
        return out

    @staticmethod
    def _exposure_metrics(exposure: Dict[str, pd.Series]) -> Dict[str, float]:
        gross = exposure.get("gross_exposure")
        net = exposure.get("net_exposure")
        tim = exposure.get("time_in_market")

        return {
            "avg_gross_exposure": float(gross.mean()) if gross is not None and len(gross) else np.nan,
            "avg_net_exposure": float(net.mean()) if net is not None and len(net) else np.nan,
            "pct_time_in_market": float(tim.mean()) if tim is not None and len(tim) else np.nan,
        }

    # -----------------------------
    # Utility
    # -----------------------------
    @staticmethod
    def _cumulative_returns_from_equity(equity: pd.Series) -> pd.Series:
        eq = equity.astype(float).dropna()
        if eq.empty:
            return pd.Series(dtype="float64", name="cum_returns")
        cum = eq / float(eq.iloc[0]) - 1.0
        cum.name = "cum_returns"
        return cum

    @staticmethod
    def _coerce_series(x: Any, name: str) -> pd.Series:
        if not isinstance(x, pd.Series):
            raise TypeError(f"{name} must be a pd.Series, got {type(x)}")
        if x.index is None:
            raise ValueError(f"{name} must have an index.")
        return x.copy()

    @staticmethod
    def _coerce_positions(x: Any, symbols: Sequence[str]) -> pd.DataFrame:
        if not isinstance(x, pd.DataFrame):
            raise TypeError(f"positions must be a pd.DataFrame, got {type(x)}")
        missing = [s for s in symbols if s not in x.columns]
        if missing:
            raise KeyError(f"positions missing symbols: {missing}. columns={list(x.columns)}")
        return x.copy()

    @staticmethod
    def _coerce_trades(x: Any) -> pd.DataFrame:
        if x is None:
            return pd.DataFrame()
        if not isinstance(x, pd.DataFrame):
            raise TypeError(f"trades must be a pd.DataFrame, got {type(x)}")
        return x.copy()


"""
TEXT EXPLANATION (for Cursor review)

What this file does:
- Implements the Results/Analytics layer that comes AFTER portfolio.py.
- It takes PortfolioResult (equity_curve, returns, positions, trades) and produces a BacktestReport with:
  1) metrics: total return, CAGR, annual vol, Sharpe, Sortino, Calmar, max drawdown + duration, turnover, total costs
  2) series: equity, returns, cumulative returns, drawdown, and (if MarketData is provided) gross/net exposure + time in market
  3) tables: monthly returns table (year x month) and a trade summary table

Key conventions:
- Annualization uses 252 trading days/year by default.
- Risk-free rate is configurable (rf_annual).
- Drawdown is computed from equity curve as equity/cummax(equity)-1.
- Turnover is computed as sum(abs(trade notional)) divided by average equity.
- Exposure series are computed only if you pass market_data so we can obtain Close prices.

Plotting:
- Provides helper methods using matplotlib:
  plot_cumulative_returns, plot_equity_curve, plot_drawdown, plot_returns_hist

Integration:
- After you run PortfolioEngine and get `res`:
    analyzer = ResultsAnalyzer()
    report = analyzer.analyze(res, market_data=md, symbols=[...])
    analyzer.plot_cumulative_returns(report)
"""
