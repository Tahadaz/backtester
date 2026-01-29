# results.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

        round_trips = self._round_trips_from_fills(portfolio_result.trades)
        curve_vs_bench = self._curve_vs_benchmark_table(
            strat_rets=rets,
            strat_equity=equity,
            strat_dd=dd,
            round_trips=round_trips,
            bench_rets=bench_rets,
            bench_equity=(1.0 + bench_rets).cumprod() if bench_rets is not None else None,
            bench_dd=self._drawdown_from_equity((1.0 + bench_rets).cumprod()) if bench_rets is not None else None,
        )

        trade_tbl = self._trade_table(round_trips)
        time_tbl = self._time_table(rets)

        # --- monthly & yearly returns ---
        monthly = self.plot_monthly_heatmap(rets)
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
        tables["curve_vs_benchmark"] = curve_vs_bench
        tables["trade_summary"] = trade_tbl
        tables["time_summary"] = time_tbl
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
    # results.py (add/merge into your existing ResultsAnalyzer)


    def _sortino(self, rets: pd.Series) -> float:
        r = rets.dropna()
        if r.empty:
            return np.nan
        rf = self.rf_annual
        n = len(r)
        total = (1.0 + r).prod()
        cagr = total ** (self.periods_per_year / n) - 1.0
        downside = r[r < 0]
        if downside.empty:
            return np.inf
        downside_dev = downside.std(ddof=1) * np.sqrt(self.periods_per_year)
        if downside_dev == 0 or np.isnan(downside_dev):
            return np.nan
        return (cagr - rf) / downside_dev

    def _max_dd_duration(self, equity: pd.Series) -> int:
        """
        Longest underwater duration in bars (can be interpreted as days for daily data).
        """
        eq = equity.dropna()
        if eq.empty:
            return 0
        peak = eq.cummax()
        underwater = eq < peak
        # longest consecutive True run
        max_run = 0
        run = 0
        for u in underwater.values:
            if u:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return int(max_run)

    def _r_squared(self, strat_rets: pd.Series, bench_rets: pd.Series) -> float:
        s = strat_rets.align(bench_rets, join="inner")[0].fillna(0.0)
        b = strat_rets.align(bench_rets, join="inner")[1].fillna(0.0)
        if len(s) < 3:
            return np.nan
        # OLS with intercept: R^2 = corr^2 in simple 1-factor regression if intercept included
        corr = np.corrcoef(s.values, b.values)[0, 1]
        return float(corr * corr) if not np.isnan(corr) else np.nan

    def _round_trips_from_fills(self, fills: pd.DataFrame) -> pd.DataFrame:
        """
        Build round-trip trades from fills (single-symbol logic, but works per symbol).
        Returns a DataFrame with:
          entry_ts, exit_ts, side, entry_price, exit_price, qty, pnl, ret, days
        """
        if fills is None or fills.empty:
            return pd.DataFrame(columns=[
                "symbol","entry_ts","exit_ts","side","entry_price","exit_price","qty",
                "pnl","ret","days"
            ])

        df = fills.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        if "side" not in df.columns:
            df["side"] = np.where(df["qty"] > 0, "BUY", "SELL")

        out_rows = []
        for sym, g in df.groupby("symbol", sort=False):
            pos = 0.0
            entry_ts = None
            entry_price = None
            entry_qty = 0.0
            entry_side = None
            entry_notional = None
            entry_cost = 0.0

            for _, r in g.iterrows():
                ts = r["timestamp"]
                qty = float(r["qty"])
                px = float(r["price"])
                cost = float(r["cost"]) if "cost" in r and pd.notna(r["cost"]) else 0.0

                prev_pos = pos
                pos = prev_pos + qty

                # open trade when going from 0 -> nonzero
                if prev_pos == 0 and pos != 0:
                    entry_ts = ts
                    entry_price = px
                    entry_qty = pos
                    entry_side = "LONG" if pos > 0 else "SHORT"
                    entry_notional = abs(entry_qty * entry_price)
                    entry_cost = cost
                    continue

                # close trade when going back to 0 or flipping sign
                closed = (prev_pos != 0 and pos == 0) or (prev_pos != 0 and np.sign(prev_pos) != np.sign(pos) and pos != 0)
                if closed and entry_ts is not None:
                    exit_ts = ts
                    exit_price = px

                    # PnL: long -> (exit-entry)*qty ; short -> (entry-exit)*abs(qty)
                    qty0 = entry_qty
                    if qty0 > 0:
                        pnl = (exit_price - entry_price) * qty0
                    else:
                        pnl = (entry_price - exit_price) * abs(qty0)

                    # subtract costs (entry + exit fill cost; this is conservative)
                    pnl_net = pnl - entry_cost - cost

                    ret = pnl_net / entry_notional if entry_notional and entry_notional > 0 else np.nan
                    days = (exit_ts - entry_ts).days

                    out_rows.append({
                        "symbol": sym,
                        "entry_ts": entry_ts,
                        "exit_ts": exit_ts,
                        "side": entry_side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty0,
                        "pnl": pnl_net,
                        "ret": ret,
                        "days": days,
                    })

                    # if flip (pos != 0), immediately open new trade at same fill
                    if pos != 0:
                        entry_ts = ts
                        entry_price = px
                        entry_qty = pos
                        entry_side = "LONG" if pos > 0 else "SHORT"
                        entry_notional = abs(entry_qty * entry_price)
                        entry_cost = cost
                    else:
                        entry_ts = None
                        entry_price = None
                        entry_qty = 0.0
                        entry_side = None
                        entry_notional = None
                        entry_cost = 0.0

        return pd.DataFrame(out_rows)

    def _trades_per_year(self, equity_index: pd.DatetimeIndex, round_trips: pd.DataFrame) -> float:
        if equity_index is None or len(equity_index) < 2:
            return np.nan
        years = (equity_index[-1] - equity_index[0]).days / 365.25
        if years <= 0:
            return np.nan
        ntr = 0 if round_trips is None else len(round_trips)
        return float(ntr / years)

    def _curve_vs_benchmark_table(
        self,
        strat_rets: pd.Series,
        strat_equity: pd.Series,
        strat_dd: pd.Series,
        round_trips: pd.DataFrame,
        bench_rets: Optional[pd.Series] = None,
        bench_equity: Optional[pd.Series] = None,
        bench_dd: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        def pack(rets, equity, dd):
            total = (1.0 + rets).prod() - 1.0
            cagr = self._annualized_return(rets)
            sharpe = self._sharpe(rets)
            sortino = self._sortino(rets)
            vol = self._annualized_vol(rets)
            maxdd = float(dd.min()) if dd is not None and len(dd) else np.nan
            dd_dur = self._max_dd_duration(equity)
            return total, cagr, sharpe, sortino, vol, maxdd, dd_dur

        s_total, s_cagr, s_sh, s_so, s_vol, s_mdd, s_dddur = pack(strat_rets, strat_equity, strat_dd)
        tpy = self._trades_per_year(strat_equity.index, round_trips)

        if bench_rets is not None:
            b_total, b_cagr, b_sh, b_so, b_vol, b_mdd, b_dddur = pack(bench_rets, bench_equity, bench_dd)
            r2 = self._r_squared(strat_rets, bench_rets)
            df = pd.DataFrame({
                "Strategy": [s_total, s_cagr, s_sh, s_so, s_vol, r2, s_mdd, s_dddur, tpy],
                "Benchmark": [b_total, b_cagr, b_sh, b_so, b_vol, np.nan, b_mdd, b_dddur, np.nan],
            }, index=[
                "Total Return",
                "CAGR",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Annual Volatility",
                "R-Squared",
                "Max Daily Drawdown",
                "Max Drawdown Duration",
                "Trades per Year",
            ])
        else:
            df = pd.DataFrame({
                "Strategy": [s_total, s_cagr, s_sh, s_so, s_vol, np.nan, s_mdd, s_dddur, tpy],
            }, index=[
                "Total Return",
                "CAGR",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Annual Volatility",
                "R-Squared",
                "Max Daily Drawdown",
                "Max Drawdown Duration",
                "Trades per Year",
            ])

        return df

    def _trade_table(self, round_trips: pd.DataFrame) -> pd.DataFrame:
        if round_trips is None or round_trips.empty:
            idx = [
                "Trade Winning %",
                "Average Trade %",
                "Average Win %",
                "Average Loss %",
                "Best Trade %",
                "Worst Trade %",
                "Worst Trade Date",
                "Avg Days in Trade",
                "Trades",
            ]
            return pd.DataFrame({"Value": [np.nan]*6 + ["TBD", np.nan, 0]}, index=idx)

        r = round_trips["ret"].astype(float)
        wins = r[r > 0]
        losses = r[r < 0]

        win_pct = (len(wins) / len(r)) * 100.0 if len(r) else np.nan
        avg_trade = r.mean() * 100.0
        avg_win = wins.mean() * 100.0 if len(wins) else np.nan
        avg_loss = losses.mean() * 100.0 if len(losses) else np.nan
        best = r.max() * 100.0
        worst = r.min() * 100.0

        worst_row = round_trips.loc[r.idxmin()] if len(r) else None
        worst_date = worst_row["exit_ts"].date().isoformat() if worst_row is not None else "TBD"
        avg_days = float(round_trips["days"].mean()) if "days" in round_trips else np.nan
        ntr = int(len(round_trips))

        df = pd.DataFrame({"Value": [
            win_pct,
            avg_trade,
            avg_win,
            avg_loss,
            best,
            worst,
            worst_date,
            avg_days,
            ntr,
        ]}, index=[
            "Trade Winning %",
            "Average Trade %",
            "Average Win %",
            "Average Loss %",
            "Best Trade %",
            "Worst Trade %",
            "Worst Trade Date",
            "Avg Days in Trade",
            "Trades",
        ])
        return df

    def _time_table(self, rets: pd.Series) -> pd.DataFrame:
        # monthly
        m = rets.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)
        y = rets.resample("Y").apply(lambda x: (1.0 + x).prod() - 1.0)

        def stats(x: pd.Series):
            x = x.dropna()
            if x.empty:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            wins = x[x > 0]
            losses = x[x < 0]
            win_pct = (len(wins) / len(x)) * 100.0 if len(x) else np.nan
            avg_win = wins.mean() * 100.0 if len(wins) else np.nan
            avg_loss = losses.mean() * 100.0 if len(losses) else np.nan
            best = x.max() * 100.0
            worst = x.min() * 100.0
            return (win_pct, avg_win, avg_loss, best, worst)

        m_win, m_avgw, m_avgl, m_best, m_worst = stats(m)
        y_win, _, _, y_best, y_worst = stats(y)

        df = pd.DataFrame({"Value": [
            m_win,
            m_avgw,
            m_avgl,
            m_best,
            m_worst,
            y_win,
            y_best,
            y_worst,
        ]}, index=[
            "Winning Months %",
            "Average Winning Month %",
            "Average Losing Month %",
            "Best Month %",
            "Worst Month %",
            "Winning Years %",
            "Best Year %",
            "Worst Year %",
        ])
        return df


    def plot_monthly_heatmap(monthly: pd.DataFrame):
        fig = plt.figure(figsize=(12, 5))
        ax = plt.gca()

        if monthly is None or monthly.empty:
            ax.set_title("Monthly returns heatmap (no data)")
            return fig

        data = monthly.values.astype(float)
        if np.isnan(data).all():
            ax.set_title("Monthly returns heatmap (no data)")
            return fig

        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        if vmin < 0 < vmax:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(data, aspect="auto", norm=norm, cmap="RdYlGn")

        ax.set_title("Monthly Returns Heatmap")
        ax.set_yticks(range(len(monthly.index)))
        ax.set_yticklabels(monthly.index.astype(str).tolist())
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

        # ✅ write values in each cell
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        return fig


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
