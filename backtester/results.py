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
    """
    Compute-only reporting layer.

    Responsibilities:
      - Compute strategy performance series/metrics
      - Build "summary tables" (Curve vs Benchmark / Trade / Time)
      - Build month heatmap matrix and yearly returns
      - Prepare plot-ready payloads (Series/DataFrames) for the UI

    Non-responsibilities:
      - No matplotlib/plotly/streamlit rendering
      - No colormaps, fills, marker styles (UI layer handles that)
    """

    def __init__(self, periods_per_year: int = 252, rf_annual: float = 0.0) -> None:
        self.periods_per_year = int(periods_per_year)
        self.rf_annual = float(rf_annual)

    # =============================
    # Public API
    # =============================
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
        symbols = list(symbols)
        if not symbols:
            raise ValueError("symbols must be non-empty")

        # --- core strategy series ---
        equity = portfolio_result.equity_curve.astype(float).sort_index()
        rets = portfolio_result.returns.astype(float).reindex(equity.index).fillna(0.0)
        

        cum = (1.0 + rets).cumprod() - 1.0
        dd = self._drawdown_from_equity(equity)
        pnl = equity.diff().fillna(0.0)              # currency PnL per bar
        cum_pnl = pnl.cumsum()


        # --- price + indicators payload (single asset for now) ---
        sym0 = symbols[0]
        px = market_data.bars[sym0]["Close"].reindex(equity.index).astype(float)
        bars0 = market_data.bars[sym0].reindex(equity.index).copy()

        # ensure float for plot libs
        for c in ["Open", "High", "Low", "Close"]:
            if c in bars0.columns:
                bars0[c] = bars0[c].astype(float)

        ind_df = None
        if features_data is not None and plot_indicators:
            feats_sym = features_data.features[sym0]
            cols = [c for c in plot_indicators if c in feats_sym.columns]
            if cols:
                ind_df = feats_sym[cols].reindex(bars0.index)
        
        # trades payload (fills)
        trades = self._prepare_trades_table(portfolio_result.trades)
        trade_ledger = self._trade_ledger_from_fills(trades)
        trade_perf = self._trade_performance_summary(trade_ledger)
        # --- benchmark series (optional) ---
        bench_rets = None
        bench_cum = None
        bench_equity = None
        bench_dd = None
        rel_cum = None

        if benchmark_market_data is not None:
            bsym = benchmark_symbol or list(benchmark_market_data.bars.keys())[0]
            bpx = benchmark_market_data.bars[bsym]["Close"].astype(float)
            bpx = bpx.reindex(equity.index).ffill()

            bench_rets = bpx.pct_change().fillna(0.0)
            bench_equity = (1.0 + bench_rets).cumprod()
            bench_cum = bench_equity - 1.0
            bench_dd = self._drawdown_from_equity(bench_equity)
            rel_cum = (1.0 + rets).cumprod() / (1.0 + bench_rets).cumprod() - 1.0

        # --- monthly/yearly returns ---
        monthly_mat = self._monthly_return_matrix(rets)   # year x month (1..12)
        yearly = self._yearly_returns(rets)               # index=year, values=return

        # --- round trips from fills (needed for Trade table) ---
        round_trips = self._round_trips_from_fills(trades)

        # --- 3 summary tables like your screenshot ---
        curve_vs_bench = self._curve_vs_benchmark_table(
            strat_rets=rets,
            strat_equity=(1.0 + rets).cumprod(),
            strat_dd=dd,
            round_trips=round_trips,
            bench_rets=bench_rets,
            bench_equity=bench_equity,
            bench_dd=bench_dd,
        )
        trade_tbl = self._trade_table(round_trips)
        time_tbl = self._time_table(rets)

        # --- headline metrics (for quick display) ---
        metrics = self._headline_metrics(rets, dd, bench_rets)

        # --- time series table ---
        ts = pd.DataFrame(
            {
                "equity": equity,
                "returns": rets,
                "cum_returns": cum,
                "drawdown": dd,
            },
            index=equity.index,
        )
        if bench_cum is not None:
            ts["bench_cum_returns"] = bench_cum
            ts["rel_cum_returns"] = rel_cum

        # --- outputs ---
        tables: Dict[str, pd.DataFrame] = {
            "trades": trades,
            "timeseries": ts,
            "curve_vs_benchmark": curve_vs_bench,
            "trade_summary": trade_tbl,
            "time_summary": time_tbl,
            "monthly_returns": monthly_mat,
            "yearly_returns": yearly.to_frame("year_return"),
            "trade_ledger":trade_ledger,
            "trade_performance":trade_perf,
        }

        plots: Dict[str, Any] = {
            "price_panel": {
                "symbol": sym0,
                "bars": bars0,            # <-- NEW: full OHLC
                "price": bars0["Close"],  # optional convenience
                "indicators": ind_df,
                "trades": trades,
                "indicator_cols": plot_indicators,
            },
            "cum_vs_bench": {"strategy": cum, "benchmark": bench_cum},
            "drawdown": dd,
            "monthly_heatmap": monthly_mat,  # app will render + annotate
            "yearly_bar": yearly,
        }

        series: Dict[str, pd.Series] = {
            "equity": equity,
            "returns": rets,
            "cum_returns": cum,
            "drawdown": dd,
            "pnl":pnl,
            "cum_pnl":cum_pnl,
        }
        if bench_cum is not None:
            series.update(
                {
                    "bench_returns": bench_rets,
                    "bench_cum_returns": bench_cum,
                    "rel_cum_returns": rel_cum,
                }
            )

        style = self._style_spec()

        return BacktestReport(
            metrics=metrics,
            series=series,
            tables=tables,
            plots=plots,
            style=style,
            meta={
                "symbols": symbols,
                "benchmark_symbol": benchmark_symbol,
            },
        )

    # =============================
    # Styling hints for the UI
    # =============================
    def _style_spec(self) -> Dict[str, Any]:
        return {
            # metrics where higher is better
            "good_high": [
                "Total Return",
                "CAGR",
                "Sharpe Ratio",
                "Sortino Ratio",
                "R-Squared",
                "Trade Winning %",
                "Average Trade %",
                "Average Win %",
                "Best Trade %",
                "Winning Months %",
                "Average Winning Month %",
                "Best Month %",
                "Winning Years %",
                "Best Year %",
            ],
            # metrics where lower is better (or "less negative" drawdown)
            "good_low": [
                "Annual Volatility",
                "Max Daily Drawdown",
                "Max Drawdown Duration",
                "Average Losing Month %",
                "Worst Month %",
                "Worst Year %",
                "Average Loss %",
                "Worst Trade %",
            ],
        }

    # =============================
    # Core series metrics
    # =============================
    def _drawdown_from_equity(self, equity: pd.Series) -> pd.Series:
        e = equity.astype(float).copy()
        peak = e.cummax()
        dd = e / peak - 1.0
        return dd.fillna(0.0)

    def _annualized_return(self, rets: pd.Series) -> float:
        r = rets.dropna()
        n = len(r)
        if n <= 1:
            return np.nan
        total = (1.0 + r).prod()
        return float(total ** (self.periods_per_year / n) - 1.0)

    def _annualized_vol(self, rets: pd.Series) -> float:
        r = rets.dropna()
        if len(r) <= 1:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(self.periods_per_year))

    def _sharpe(self, rets: pd.Series) -> float:
        vol = self._annualized_vol(rets)
        if vol == 0 or np.isnan(vol):
            return np.nan
        return float((self._annualized_return(rets) - self.rf_annual) / vol)

    def _sortino(self, rets: pd.Series) -> float:
        r = rets.dropna()
        if r.empty:
            return np.nan
        downside = r[r < 0]
        if downside.empty:
            return np.inf
        downside_dev = downside.std(ddof=1) * np.sqrt(self.periods_per_year)
        if downside_dev == 0 or np.isnan(downside_dev):
            return np.nan
        return float((self._annualized_return(r) - self.rf_annual) / downside_dev)

    def _headline_metrics(self, rets: pd.Series, dd: pd.Series, bench_rets: Optional[pd.Series]) -> Dict[str, float]:
        out = {
            "Total return": float((1.0 + rets).prod() - 1.0),
            "CAGR": float(self._annualized_return(rets)),
            "Vol (ann.)": float(self._annualized_vol(rets)),
            "Sharpe": float(self._sharpe(rets)),
            "Sortino": float(self._sortino(rets)),
            "Max drawdown": float(dd.min()) if dd is not None and len(dd) else np.nan,
        }
        if bench_rets is not None:
            out["Bench total return"] = float((1.0 + bench_rets).prod() - 1.0)
        return out

    # =============================
    # Trades / fills
    # =============================
    def _prepare_trades_table(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades is None or trades.empty:
            return pd.DataFrame(columns=["timestamp", "symbol", "qty", "side", "price", "notional", "cost"])

        df = trades.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "side" not in df.columns:
            df["side"] = np.where(df["qty"].astype(float) > 0, "BUY", "SELL")

        keep = [c for c in ["timestamp", "symbol", "qty", "side", "price", "notional", "cost", "commission_ht", "vat", "slippage"] if c in df.columns]
        df = df[keep].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return df

    def _round_trips_from_fills(self, fills: pd.DataFrame) -> pd.DataFrame:
        """
        Build round-trip trades from fills.
        Assumes fills change position over time; closes when position goes to 0 or flips sign.
        """
        if fills is None or fills.empty:
            return pd.DataFrame(columns=[
                "symbol","entry_ts","exit_ts","side","entry_price","exit_price","qty",
                "pnl","ret","days"
            ])

        df = fills.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

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

                # open when 0 -> nonzero
                if prev_pos == 0 and pos != 0:
                    entry_ts = ts
                    entry_price = px
                    entry_qty = pos
                    entry_side = "LONG" if pos > 0 else "SHORT"
                    entry_notional = abs(entry_qty * entry_price)
                    entry_cost = cost
                    continue

                # close when back to 0 OR flip sign
                closing = (prev_pos != 0 and pos == 0) or (prev_pos != 0 and pos != 0 and np.sign(prev_pos) != np.sign(pos))
                if closing and entry_ts is not None:
                    exit_ts = ts
                    exit_price = px
                    qty0 = entry_qty

                    if qty0 > 0:
                        pnl = (exit_price - entry_price) * qty0
                    else:
                        pnl = (entry_price - exit_price) * abs(qty0)

                    pnl_net = pnl - entry_cost - cost
                    ret = pnl_net / entry_notional if entry_notional and entry_notional > 0 else np.nan
                    days = int((exit_ts - entry_ts).days)

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

                    # if flip sign, start new trade immediately at same fill
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

    def _trades_per_year(self, idx: pd.DatetimeIndex, round_trips: pd.DataFrame) -> float:
        if idx is None or len(idx) < 2:
            return np.nan
        years = (idx[-1] - idx[0]).days / 365.25
        if years <= 0:
            return np.nan
        return float((0 if round_trips is None else len(round_trips)) / years)

    # =============================
    # Monthly / yearly returns
    # =============================
    def _monthly_return_matrix(self, rets: pd.Series) -> pd.DataFrame:
        r = rets.copy()
        r.index = pd.to_datetime(r.index)
        m = r.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)
        if m.empty:
            return pd.DataFrame()
        df = m.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        return df.pivot(index="year", columns="month", values="ret").sort_index()

    def _yearly_returns(self, rets: pd.Series) -> pd.Series:
        r = rets.copy()
        r.index = pd.to_datetime(r.index)
        y = r.resample("Y").apply(lambda x: (1.0 + x).prod() - 1.0)
        if y.empty:
            return pd.Series(dtype=float)
        y.index = y.index.year
        y.name = "year_return"
        return y

    # =============================
    # Summary tables (3 panels)
    # =============================
    def _max_dd_duration(self, equity: pd.Series) -> int:
        eq = equity.dropna()
        if eq.empty:
            return 0
        peak = eq.cummax()
        underwater = eq < peak
        max_run, run = 0, 0
        for u in underwater.values:
            if bool(u):
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return int(max_run)

    def _r_squared(self, strat_rets: pd.Series, bench_rets: pd.Series) -> float:
        s, b = strat_rets.align(bench_rets, join="inner")
        s = s.fillna(0.0)
        b = b.fillna(0.0)
        if len(s) < 3:
            return np.nan
        corr = np.corrcoef(s.values, b.values)[0, 1]
        return float(corr * corr) if not np.isnan(corr) else np.nan

    def _curve_vs_benchmark_table(
        self,
        strat_rets: pd.Series,
        strat_equity: pd.Series,
        strat_dd: pd.Series,
        round_trips: pd.DataFrame,
        bench_rets: Optional[pd.Series],
        bench_equity: Optional[pd.Series],
        bench_dd: Optional[pd.Series],
    ) -> pd.DataFrame:
        def pack(rets, equity, dd):
            total = (1.0 + rets).prod() - 1.0
            cagr = self._annualized_return(rets)
            sharpe = self._sharpe(rets)
            sortino = self._sortino(rets)
            vol = self._annualized_vol(rets)
            maxdd = float(dd.min()) if dd is not None and len(dd) else np.nan
            dd_dur = self._max_dd_duration(equity)
            return float(total), float(cagr), float(sharpe), float(sortino), float(vol), float(maxdd), float(dd_dur)

        s_total, s_cagr, s_sh, s_so, s_vol, s_mdd, s_dddur = pack(strat_rets, strat_equity, strat_dd)
        tpy = self._trades_per_year(strat_equity.index, round_trips)

        index = [
            "Total Return",
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Annual Volatility",
            "R-Squared",
            "Max Daily Drawdown",
            "Max Drawdown Duration",
            "Trades per Year",
        ]

        if bench_rets is not None and bench_equity is not None and bench_dd is not None:
            b_total, b_cagr, b_sh, b_so, b_vol, b_mdd, b_dddur = pack(bench_rets, bench_equity, bench_dd)
            r2 = self._r_squared(strat_rets, bench_rets)

            df = pd.DataFrame(
                {
                    "Strategy": [s_total, s_cagr, s_sh, s_so, s_vol, r2, s_mdd, s_dddur, tpy],
                    "Benchmark": [b_total, b_cagr, b_sh, b_so, b_vol, np.nan, b_mdd, b_dddur, np.nan],
                },
                index=index,
            )
            return df

        df = pd.DataFrame({"Strategy": [s_total, s_cagr, s_sh, s_so, s_vol, np.nan, s_mdd, s_dddur, tpy]}, index=index)
        return df

    def _trade_table(self, round_trips: pd.DataFrame) -> pd.DataFrame:
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
        if round_trips is None or round_trips.empty:
            return pd.DataFrame({"Value": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "TBD", np.nan, 0]}, index=idx)

        r = round_trips["ret"].astype(float)
        wins = r[r > 0]
        losses = r[r < 0]

        win_pct = (len(wins) / len(r)) * 100.0 if len(r) else np.nan
        avg_trade = r.mean() * 100.0
        avg_win = wins.mean() * 100.0 if len(wins) else np.nan
        avg_loss = losses.mean() * 100.0 if len(losses) else np.nan
        best = r.max() * 100.0
        worst = r.min() * 100.0

        worst_date = "TBD"
        if len(r):
            worst_row = round_trips.loc[r.idxmin()]
            worst_date = pd.to_datetime(worst_row["exit_ts"]).date().isoformat()

        avg_days = float(round_trips["days"].mean()) if "days" in round_trips.columns else np.nan
        ntr = int(len(round_trips))

        return pd.DataFrame({"Value": [win_pct, avg_trade, avg_win, avg_loss, best, worst, worst_date, avg_days, ntr]}, index=idx)

    def _time_table(self, rets: pd.Series) -> pd.DataFrame:
        idx = [
            "Winning Months %",
            "Average Winning Month %",
            "Average Losing Month %",
            "Best Month %",
            "Worst Month %",
            "Winning Years %",
            "Best Year %",
            "Worst Year %",
        ]

        r = rets.copy()
        r.index = pd.to_datetime(r.index)

        m = r.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)
        y = r.resample("Y").apply(lambda x: (1.0 + x).prod() - 1.0)

        def stats(x: pd.Series) -> Tuple[float, float, float, float, float]:
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

        return pd.DataFrame({"Value": [m_win, m_avgw, m_avgl, m_best, m_worst, y_win, y_best, y_worst]}, index=idx)
    def _trade_ledger_from_fills(self, fills: pd.DataFrame) -> pd.DataFrame:
        """
        Build closed-trade ledger (round trips) from fills using FIFO lot matching.

        Expected fills columns:
        timestamp (datetime), symbol (str), qty (signed), price (float), cost (float)

        Output columns:
        entry_time, exit_time, symbol, side, qty,
        entry_price, exit_price,
        gross_pnl, entry_cost, exit_cost, net_pnl,
        return_pct, hold_days
        """
        if fills is None or fills.empty:
            return pd.DataFrame(
                columns=[
                    "entry_time", "exit_time", "symbol", "side", "qty",
                    "entry_price", "exit_price",
                    "gross_pnl", "entry_cost", "exit_cost", "net_pnl",
                    "return_pct", "hold_days",
                ]
            )

        f = fills.copy()

        # Normalize
        f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce")
        f = f.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        for c in ["qty", "price", "cost"]:
            if c in f.columns:
                f[c] = pd.to_numeric(f[c], errors="coerce")
        f = f.dropna(subset=["qty", "price"])
        if "cost" not in f.columns:
            f["cost"] = 0.0
        f["cost"] = f["cost"].fillna(0.0)

        # FIFO lots per symbol
        # lot: qty_signed, price, time, entry_cost_total
        lots: dict[str, list[dict]] = {}
        out_rows: list[dict] = []

        for _, row in f.iterrows():
            ts = row["timestamp"]
            sym = str(row["symbol"])
            qty = int(row["qty"])
            price = float(row["price"])
            cost_total = float(row["cost"])

            if qty == 0:
                continue

            if sym not in lots:
                lots[sym] = []

            # Allocate exit/entry cost pro-rata if a fill both closes and opens (flip)
            fill_abs = abs(qty)
            fill_cost_total = cost_total

            # Helper: allocate cost proportional to a used quantity from this fill
            def alloc_fill_cost(used_abs_qty: int) -> float:
                if fill_abs == 0:
                    return 0.0
                return fill_cost_total * (float(used_abs_qty) / float(fill_abs))

            remaining_qty = qty  # signed

            # If there are open lots with opposite sign, we are closing them
            while remaining_qty != 0 and lots[sym]:
                lot = lots[sym][0]
                lot_qty = int(lot["qty"])
                if lot_qty == 0:
                    lots[sym].pop(0)
                    continue

                # Same direction -> stop closing; this fill is opening/increasing
                if np.sign(lot_qty) == np.sign(remaining_qty):
                    break

                # Opposite direction -> close
                close_abs = min(abs(remaining_qty), abs(lot_qty))

                side = "LONG" if lot_qty > 0 else "SHORT"
                entry_price = float(lot["price"])
                exit_price = price

                # Gross PnL
                if side == "LONG":
                    gross = close_abs * (exit_price - entry_price)
                else:
                    gross = close_abs * (entry_price - exit_price)

                # Allocate entry cost from lot pro-rata
                lot_abs_before = abs(lot_qty)
                entry_cost_part = float(lot["entry_cost"]) * (float(close_abs) / float(lot_abs_before))

                # Allocate exit cost from this fill pro-rata
                exit_cost_part = alloc_fill_cost(close_abs)

                net = gross - entry_cost_part - exit_cost_part
                notional_entry = close_abs * entry_price
                ret_pct = (net / notional_entry) if notional_entry != 0 else np.nan

                hold_days = (pd.Timestamp(ts) - pd.Timestamp(lot["timestamp"])).days

                out_rows.append(
                    {
                        "entry_time": lot["timestamp"],
                        "exit_time": ts,
                        "symbol": sym,
                        "side": side,
                        "qty": int(close_abs),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": float(gross),
                        "entry_cost": float(entry_cost_part),
                        "exit_cost": float(exit_cost_part),
                        "net_pnl": float(net),
                        "return_pct": float(ret_pct),
                        "hold_days": int(hold_days),
                    }
                )

                # Reduce lot and remaining fill qty
                # Update lot qty toward zero
                if lot_qty > 0:
                    lot["qty"] = lot_qty - close_abs
                else:
                    lot["qty"] = lot_qty + close_abs

                # Reduce lot entry cost proportionally
                lot["entry_cost"] = float(lot["entry_cost"]) - float(entry_cost_part)

                # Reduce remaining fill qty toward zero
                if remaining_qty > 0:
                    remaining_qty -= close_abs
                else:
                    remaining_qty += close_abs

                # Remove depleted lots
                if int(lot["qty"]) == 0:
                    lots[sym].pop(0)

            # Any remaining_qty opens a new lot (or increases same direction)
            if remaining_qty != 0:
                open_abs = abs(remaining_qty)
                entry_cost_for_open = alloc_fill_cost(open_abs)

                lots[sym].append(
                    {
                        "timestamp": ts,
                        "qty": int(remaining_qty),
                        "price": float(price),
                        "entry_cost": float(entry_cost_for_open),
                    }
                )

        ledger = pd.DataFrame(out_rows)
        if ledger.empty:
            return ledger

        ledger = ledger.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
        return ledger
    def _trade_performance_summary(self, ledger: pd.DataFrame) -> pd.DataFrame:
        if ledger is None or ledger.empty:
            return pd.DataFrame(index=[
                "Trades", "Win Rate", "Avg Net PnL", "Total Net PnL",
                "Avg Return %", "Profit Factor", "Avg Hold Days"
            ], data={"Value": [0, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan]})

        net = ledger["net_pnl"].astype(float)
        wins = net[net > 0]
        losses = net[net < 0]

        trades = int(len(net))
        win_rate = float((net > 0).mean())
        avg_net = float(net.mean())
        total_net = float(net.sum())
        avg_ret = float(ledger["return_pct"].astype(float).mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else np.inf
        avg_hold = float(ledger["hold_days"].astype(float).mean())

        return pd.DataFrame(
            {"Value": [trades, win_rate, avg_net, total_net, avg_ret, profit_factor, avg_hold]},
            index=["Trades", "Win Rate", "Avg Net PnL", "Total Net PnL",
                "Avg Return %", "Profit Factor", "Avg Hold Days"],
        )

