# portfolio.py
from __future__ import annotations
from typing import Literal

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal, Any, Tuple

import numpy as np
import pandas as pd

SizingMode = Literal["target_weight", "pct_cash_shares"]

# -----------------------------
# Contracts expected from upstream layers
# -----------------------------
# MarketDataLike:
#   market_data.bars: dict[symbol, pd.DataFrame] with index datetime-like and columns including Open, Close.
#
# SignalFrame:
#   sf.signals: pd.DataFrame index datetime-like, columns symbols, values in {-1,0,1} (or {0,1} if long-only)
#   sf.validity: pd.DataFrame booleans same shape
#
# NOTE: We intentionally avoid importing your project modules to keep this file standalone.
# You can add typing imports later (e.g., from .types import MarketDataLike, SignalFrame).


# -----------------------------
# Configuration dataclasses
# -----------------------------
RebalancePolicy = Literal["on_change", "every_bar"]
FillPriceModel = Literal["next_open"]  # extend later: "next_close", "vwap", "mid", etc.
MarkToMarketModel = Literal["close_t1"]  # your choice: close(t+1)


@dataclass(frozen=True)
class CostModel:
    """
    Transaction cost model for Moroccan equities (configurable).

    Defaults are set to a commonly-cited "max / standard brochure" style:
      - Brokerage (commission de courtage): 0.60% HT
      - Exchange fee (commission de bourse): 0.10% HT
      - Settlement/Livraison: 0.20% HT
      - VAT (TVA): 10% applied on commissions (practice varies; keep configurable)

    You can set any component to 0.0 if not applicable to your context.
    """
    brokerage_bps: float = 60.0     # 0.60% = 60 bps (HT)
    exchange_bps: float = 10.0      # 0.10% = 10 bps (HT)
    settlement_bps: float = 20.0    # 0.20% = 20 bps (HT)
    slippage_bps: float = 0.0       # model impact/spread; keep 0 for now

    vat_rate: float = 0.10          # 10% TVA on commissions; set 0.0 if you don't want this
    # If you want: fixed minimum commission, per-order ticket fees, etc. add later.

    def estimate_cost(self, notional: float) -> Tuple[float, Dict[str, float]]:
        """
        Returns (total_cost, breakdown). notional is absolute traded value (>=0).
        """
        notional = float(abs(notional))
        commission_ht = notional * (self.brokerage_bps + self.exchange_bps + self.settlement_bps) / 10000.0
        slippage = notional * (self.slippage_bps / 10000.0)

        vat = commission_ht * float(self.vat_rate)
        total = commission_ht + vat + slippage

        breakdown = {
            "commission_ht": commission_ht,
            "vat": vat,
            "slippage": slippage,
            "total": total,
        }
        return total, breakdown


@dataclass(frozen=True)
class PortfolioConfig:
    # Portfolio semantics
    allow_short: bool = True
    initial_cash: float = 1_000_000.0

    # Exposure / constraints (optional)
    max_gross: float = 1.0
    max_weight_per_asset: Optional[float] = None
    cash_buffer: float = 0.0

    # Mechanics
    rebalance_policy: RebalancePolicy = "on_change"
    fill_price_model: FillPriceModel = "next_open"
    mtm_model: MarkToMarketModel = "close_t1"

    # --- NEW: per-trade sizing ---
    sizing_mode: SizingMode = "target_weight"
    buy_pct_cash: float = 1.0        # 0..1, used in pct_cash_shares mode
    sell_pct_shares: float = 1.0     # 0..1, used in pct_cash_shares mode

    # Prices
    open_col: str = "Open"
    close_col: str = "Close"

    # Costs
    cost_model: CostModel = field(default_factory=CostModel)

    allow_fractional_shares: bool = False


    def __post_init__(self) -> None:
        if not (0.0 < self.buy_pct_cash <= 1.0):
            raise ValueError("buy_pct_cash must be in (0, 1].")
        if not (0.0 < self.sell_pct_shares <= 1.0):
            raise ValueError("sell_pct_shares must be in (0, 1].")
        if not (0.0 <= self.cash_buffer < 1.0):
            raise ValueError("cash_buffer must be in [0,1).")


# -----------------------------
# Records / outputs
# -----------------------------
@dataclass(frozen=True)
class Fill:
    timestamp: pd.Timestamp
    symbol: str
    qty: int                    # signed
    price: float                # fill price
    notional: float             # abs(qty*price)
    cost: float                 # total cost paid (>=0)
    cost_breakdown: Dict[str, float]


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)

    def position(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def set_position(self, symbol: str, qty: int) -> None:
        self.positions[symbol] = int(qty)

    def apply_fill(self, fill: Fill) -> None:
        """
        Update cash and position for a fill.
        Convention:
          - Buy qty>0: cash decreases by qty*price + cost
          - Sell qty<0: cash increases by |qty|*price - cost
        """
        signed_cash_flow = -fill.qty * float(fill.price)  # buy -> negative cash flow
        self.cash += signed_cash_flow
        self.cash -= float(fill.cost)
        self.positions[fill.symbol] = self.position(fill.symbol) + int(fill.qty)

    def mark_to_market(self, close_prices: Dict[str, float]) -> float:
        """
        Compute equity = cash + sum(qty * close).
        """
        equity = float(self.cash)
        for sym, qty in self.positions.items():
            if sym in close_prices:
                equity += int(qty) * float(close_prices[sym])
        return equity


@dataclass
class PortfolioResult:
    equity_curve: pd.Series                 # indexed by timestamps (t+1)
    returns: pd.Series                      # simple returns on equity_curve
    positions: pd.DataFrame                 # rows timestamps, cols symbols, values shares
    trades: pd.DataFrame                    # one row per fill
    meta: Dict[str, Any]


# -----------------------------
# Portfolio Engine (one-file "portfolio layer")
# -----------------------------
class PortfolioEngine:
    """
    One-module portfolio layer:
      signals(t) -> targets(t) -> fill at open(t+1) -> mark-to-market at close(t+1)
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.cfg = config or PortfolioConfig()

    # ---------- public API ----------
    def run(
        self,
        market_data: Any,   # MarketDataLike
        signal_frame: Any,  # SignalFrame
        symbols: Optional[Sequence[str]] = None,
    ) -> PortfolioResult:
        symbols = list(symbols) if symbols is not None else list(signal_frame.signals.columns)

        # Validate data availability
        for s in symbols:
            if s not in market_data.bars:
                raise KeyError(f"MarketData missing symbol '{s}'. Available: {list(market_data.bars.keys())}")
            bars = market_data.bars[s]
            for col in (self.cfg.open_col, self.cfg.close_col):
                if col not in bars.columns:
                    raise KeyError(f"Bars for '{s}' missing required column '{col}'. Columns: {list(bars.columns)}")

        # Align index: we drive by signals index, but require bars contain t+1 open/close for fills/mtm
        idx = pd.Index(signal_frame.signals.index).sort_values()
        if len(idx) < 2:
            raise ValueError("Need at least 2 timestamps to apply t+1 fill semantics.")

        state = PortfolioState(cash=float(self.cfg.initial_cash), positions={s: 0 for s in symbols})

        fills: List[Fill] = []
        equity_points: List[Tuple[pd.Timestamp, float]] = []
        pos_hist: List[Tuple[pd.Timestamp, Dict[str, int]]] = []

        prev_target_weights: Optional[Dict[str, float]] = None

        # iterate up to second last timestamp because we fill at t+1
        for i in range(len(idx) - 1):
            t = pd.Timestamp(idx[i])
            t1 = pd.Timestamp(idx[i + 1])

            # 1) Extract signals at time t
            sig_row = signal_frame.signals.loc[t, symbols]
            if hasattr(signal_frame, "validity") and signal_frame.validity is not None:
                valid_row = signal_frame.validity.loc[t, symbols]
            else:
                valid_row = pd.Series(True, index=symbols)

            # Replace invalid with 0 intent (flat) to preserve safety
            sig_row = sig_row.where(valid_row.astype(bool), 0.0).astype(float)

            # 2) Compute prices at t+1 open/close (fill at open(t+1), mtm at close(t+1))
            open_t1 = {s: float(market_data.bars[s].loc[t1, self.cfg.open_col]) for s in symbols}
            close_t1 = {s: float(market_data.bars[s].loc[t1, self.cfg.close_col]) for s in symbols}

            # 3) Compute equity BEFORE trading at t+1 (marking prior holdings at close(t))
            # For simplicity in daily bars, we size using equity marked at close(t) approximated by close(t1)?.
            # We avoid lookahead by sizing using equity based on latest known state cash + positions marked at close(t).
            # If you want exact, pass close(t) prices from bars; for now we use close(t) if available else close(t1) fallback.
            close_t = self._get_close_t(market_data, symbols, t, fallback=close_t1)
            equity_t = state.mark_to_market(close_t)

            # 4) Target generation: signals -> target weights
            target_weights = self._signals_to_target_weights(sig_row, symbols)

            # 5) Rebalance policy
            if self.cfg.rebalance_policy == "on_change" and prev_target_weights is not None:
                if self._weights_equal(prev_target_weights, target_weights):
                    # no rebalance; still mark-to-market at t+1
                    equity_t1 = state.mark_to_market(close_t1)
                    equity_points.append((t1, equity_t1))
                    pos_hist.append((t1, dict(state.positions)))
                    continue

            if self.cfg.sizing_mode == "target_weight":
                # --- existing behavior ---
                target_weights = self._apply_constraints(target_weights)
                investable_equity = equity_t * (1.0 - float(self.cfg.cash_buffer))
                target_shares = self._weights_to_shares(target_weights, open_t1, investable_equity)

                orders = []
                for s in symbols:
                    current = state.position(s)
                    desired = int(target_shares.get(s, 0))
                    delta = desired - current
                    if delta != 0:
                        orders.append((s, int(delta)))

            else:
                # --- new per-trade sizing behavior ---
                orders = self._deltas_pct_cash_shares(sig_row, state, open_t1, equity_t, symbols)

            # Execute orders at open(t+1)
            for s, delta in orders:
                if delta == 0:
                    continue

                fill_price = float(open_t1[s])
                notional = abs(delta) * fill_price
                cost, breakdown = self.cfg.cost_model.estimate_cost(notional)

                f = Fill(
                    timestamp=t1,
                    symbol=s,
                    qty=int(delta),
                    price=fill_price,
                    notional=float(notional),
                    cost=float(cost),
                    cost_breakdown=breakdown,
                )
                state.apply_fill(f)
                fills.append(f)


            # 9) Mark-to-market at close(t+1) (your requested convention)
            equity_t1 = state.mark_to_market(close_t1)
            equity_points.append((t1, equity_t1))
            pos_hist.append((t1, dict(state.positions)))

            prev_target_weights = dict(target_weights)

        # Build outputs
        equity_curve = pd.Series(
            [v for _, v in equity_points],
            index=pd.Index([ts for ts, _ in equity_points], name="timestamp"),
            name="equity",
            dtype="float64",
        )

        returns = equity_curve.pct_change().fillna(0.0)
        positions = self._positions_history_to_df(pos_hist, symbols)
        trades = self._fills_to_df(fills)

        meta = {
            "config": self.cfg.__dict__,
            "notes": {
                "causality": "decide at t using SignalFrame, execute at open(t+1), mark-to-market at close(t+1)",
                "shares": "integer only (no fractional shares)",
                "constraints": "max_gross always applied; per-asset cap and cash_buffer optional",
            },
        }
        return PortfolioResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            trades=trades,
            meta=meta,
        )

    # ---------- internals ----------
    def _signals_to_target_weights(self, sig_row: pd.Series, symbols: List[str]) -> Dict[str, float]:
        """
        Default mapping:
          signal +1 -> +1 weight
          signal  0 ->  0 weight
          signal -1 -> -1 weight (if allow_short), else 0
        For multi-asset, this produces raw weights; constraints will clamp and gross-scale.
        """
        out: Dict[str, float] = {}
        for s in symbols:
            x = float(sig_row.get(s, 0.0))
            if not self.cfg.allow_short and x < 0:
                x = 0.0
            # Keep in [-1,1] defensively
            x = float(np.clip(x, -1.0, 1.0))
            out[s] = x
        return out

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Applies:
          - per-asset max weight (optional): clamp each |w_i| <= cap
          - max gross exposure: scale down if sum(|w_i|) > max_gross
        """
        w = dict(weights)

        # Per-asset clamp if provided
        if self.cfg.max_weight_per_asset is not None:
            cap = float(self.cfg.max_weight_per_asset)
            if cap <= 0:
                raise ValueError("max_weight_per_asset must be > 0 if provided.")
            for k in w:
                w[k] = float(np.clip(w[k], -cap, cap))

        # Gross scaling
        gross = sum(abs(x) for x in w.values())
        max_gross = float(self.cfg.max_gross)
        if max_gross <= 0:
            raise ValueError("max_gross must be > 0.")
        if gross > max_gross and gross > 0:
            scale = max_gross / gross
            for k in w:
                w[k] *= scale

        return w

    def _weights_to_shares(
        self,
        weights: Dict[str, float],
        prices: Dict[str, float],
        equity: float,
    ) -> Dict[str, int]:
        """
        Convert weights -> integer shares using reference prices (here: open(t+1)).
        For shorts, shares are negative.
        Rounding: toward zero (int()) to preserve "no fractional shares".
        """
        target: Dict[str, int] = {}
        eq = float(equity)
        for sym, w in weights.items():
            p = float(prices[sym])
            if p <= 0:
                raise ValueError(f"Non-positive price for {sym}: {p}")
            desired_notional = float(w) * eq
            desired_shares = desired_notional / p
            # toward zero
            q = int(desired_shares)
            target[sym] = q
        return target

    def _get_close_t(self, market_data: Any, symbols: List[str], t: pd.Timestamp, fallback: Dict[str, float]) -> Dict[str, float]:
        """
        Close(t) for equity marking at decision time.
        If Close(t) not available for a symbol at t, fallback to provided dict (e.g., close(t+1)).
        This keeps engine robust to missing days in some symbol series.
        """
        close_t: Dict[str, float] = {}
        for s in symbols:
            bars = market_data.bars[s]
            if t in bars.index:
                close_t[s] = float(bars.loc[t, self.cfg.close_col])
            else:
                close_t[s] = float(fallback[s])
        return close_t

    @staticmethod
    def _weights_equal(a: Dict[str, float], b: Dict[str, float], tol: float = 1e-12) -> bool:
        if a.keys() != b.keys():
            return False
        for k in a.keys():
            if abs(float(a[k]) - float(b[k])) > tol:
                return False
        return True

    @staticmethod
    def _positions_history_to_df(
        pos_hist: List[Tuple[pd.Timestamp, Dict[str, int]]],
        symbols: List[str],
    ) -> pd.DataFrame:
        rows = []
        idx = []
        for ts, pos in pos_hist:
            idx.append(ts)
            rows.append([int(pos.get(s, 0)) for s in symbols])
        return pd.DataFrame(rows, index=pd.Index(idx, name="timestamp"), columns=symbols, dtype="int64")

    @staticmethod
    def _fills_to_df(fills: List[Fill]) -> pd.DataFrame:
        if not fills:
            return pd.DataFrame(columns=["timestamp", "symbol", "qty", "price", "notional", "cost", "commission_ht", "vat", "slippage"])
        rows = []
        for f in fills:
            rows.append({
                "timestamp": f.timestamp,
                "symbol": f.symbol,
                "qty": f.qty,
                "price": f.price,
                "notional": f.notional,
                "cost": f.cost,
                "commission_ht": f.cost_breakdown.get("commission_ht", np.nan),
                "vat": f.cost_breakdown.get("vat", np.nan),
                "slippage": f.cost_breakdown.get("slippage", np.nan),
            })
        df = pd.DataFrame(rows)
        df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return df
    
    def _available_cash(self, state: PortfolioState, equity_t: float) -> float:
        """
        Keep cash_buffer * equity_t reserved in cash.
        """
        reserve = float(self.cfg.cash_buffer) * float(equity_t)
        return max(0.0, float(state.cash) - reserve)


    def _max_abs_shares_cap(self, equity_t: float, price: float) -> int:
        """
        Cap gross exposure in shares (single-asset practical cap).
        """
        if price <= 0:
            return 0
        investable_equity = float(equity_t) * (1.0 - float(self.cfg.cash_buffer))
        cap_notional = float(self.cfg.max_gross) * investable_equity

        # if max_weight_per_asset is set, it should also cap the asset
        if self.cfg.max_weight_per_asset is not None:
            cap_notional = min(cap_notional, float(self.cfg.max_weight_per_asset) * investable_equity)

        return int(cap_notional / float(price))  # floor


    def _deltas_pct_cash_shares(
        self,
        sig_row: pd.Series,
        state: PortfolioState,
        open_t1: Dict[str, float],
        equity_t: float,
        symbols: List[str],
    ) -> List[Tuple[str, int]]:
        """
        Build (symbol, delta_shares) orders for pct_cash_shares sizing.

        Rules (single-asset friendly, multi-asset works by splitting budgets):
        - If signal==0: reduce existing position by sell_pct_shares (partial exit)
        - If signal==+1: if short, close full short first; then buy with buy_pct_cash of available cash
        - If signal==-1: if long, close full long first; then short with buy_pct_cash of investable equity (simplified)
        """
        orders: List[Tuple[str, int]] = []

        # Budget splitting across active (non-zero) signals for entries
        active_syms = [s for s in symbols if float(sig_row.get(s, 0.0)) != 0.0]
        n_active = max(1, len(active_syms))

        avail_cash = self._available_cash(state, equity_t)

        for s in symbols:
            sig = float(sig_row.get(s, 0.0))
            if not self.cfg.allow_short and sig < 0:
                sig = 0.0
            sig = float(np.clip(sig, -1.0, 1.0))

            pos = int(state.position(s))
            px = float(open_t1[s])
            cap_abs = self._max_abs_shares_cap(equity_t, px)

            # ---- FLAT: partial exit ----
            if sig == 0.0:
                if pos == 0:
                    continue
                q = int(np.ceil(self.cfg.sell_pct_shares * abs(pos)))
                q = min(q, abs(pos))
                delta = -q if pos > 0 else +q  # sell long / cover short
                orders.append((s, int(delta)))
                continue

            # ---- LONG intent ----
            if sig > 0:
                # If currently short, close it fully first (intent-consistent)
                if pos < 0:
                    orders.append((s, -pos))  # buy to cover

                    # after closing, treat as flat for entry sizing
                    pos = 0

                # Enter/increase long using % of available cash
                cash_budget = (self.cfg.buy_pct_cash * avail_cash) / n_active
                buy_qty = int(cash_budget / px)  # floor, integer shares
                if buy_qty <= 0:
                    continue

                # Cap by gross limits
                desired_pos = min(pos + buy_qty, cap_abs)
                delta = desired_pos - pos
                if delta != 0:
                    orders.append((s, int(delta)))
                continue

            # ---- SHORT intent ----
            if sig < 0 and self.cfg.allow_short:
                # If currently long, close it fully first
                if pos > 0:
                    orders.append((s, -pos))  # sell to flat
                    pos = 0

                investable_equity = float(equity_t) * (1.0 - float(self.cfg.cash_buffer))
                short_notional = (self.cfg.buy_pct_cash * investable_equity) / n_active
                short_qty = int(short_notional / px)  # floor
                if short_qty <= 0:
                    continue

                desired_pos = max(pos - short_qty, -cap_abs)
                delta = desired_pos - pos
                if delta != 0:
                    orders.append((s, int(delta)))
                continue

        return orders



"""
TEXT EXPLANATION (for Cursor review)

What this file does:
- Implements a complete "portfolio layer" as a single module: sizing/targets, constraints, execution simulation, and accounting.
- It consumes MarketData (market_data.bars[symbol]) and SignalFrame (signals + validity) and produces:
  * equity_curve (marked at close(t+1))
  * returns series
  * positions history (shares)
  * trade ledger (fills with cost breakdown)

Key causal conventions:
- Signals are read at time t.
- Orders are executed at Open(t+1) (FillPriceModel = next_open).
- Equity is marked-to-market at Close(t+1) (as requested).

Sizing and constraints:
- Signals map to target weights in [-1,1] (long/short).
- Optional per-asset max weight and cash buffer exist but do not have to be set.
- max_gross is always applied (default 1.0), scaling weights down if needed.
- Weights are converted to integer shares using Open(t+1). No fractional shares.

Costs:
- CostModel is configurable. Defaults use commonly cited Morocco brochure-style levels:
  brokerage 0.60% HT + exchange 0.10% HT + settlement 0.20% HT, VAT 10% on commissions, plus optional slippage.
- estimate_cost() returns a breakdown (commission_ht, vat, slippage) stored in each Fill.

How to integrate:
- Create PortfolioEngine(config) then call run(market_data, signal_frame, symbols=...).
- Ensure bars have 'Open' and 'Close' columns (canonicalized).
"""
