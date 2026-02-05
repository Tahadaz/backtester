# portfolio.py
from __future__ import annotations
from typing import Literal
import math

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
class PortfolioStats:
    final_equity: float
    pnl: float
    traded_notional: float
    n_fills: int

@dataclass(frozen=True)
class CostModel:
    """
    Transaction cost model for Moroccan equities (configurable).

    Defaults are set to a commonly-cited "max / standard brochure" style:
      - Brokerage (commission de courtage): 0.60% HT
      - Exchange fee (commission de bourse): 0.10% HT
      - Settlement/Livraison: 0.20% HT
      - tva (TVA): 10% applied on commissions (practice varies; keep configurable)

    You can set any component to 0.0 if not applicable to your context.
    """
    brokerage_bps: float = 60.0     # 0.60% = 60 bps (HT)
    comm_bourse_bps: float = 10.0      # 0.10% = 10 bps (HT)
    reg_liv_bps: float = 20.0    # 0.20% = 20 bps (HT)
    slippage_bps: float = 0.0       # model impact/spread; keep 0 for now

    tva_rate: float = 0.10          # 10% TVA on commissions; set 0.0 if you don't want this
    # If you want: fixed minimum commission, per-order ticket fees, etc. add later.

    def estimate_cost(self, notional: float) -> Tuple[float, Dict[str, float]]:
        """
        Returns (total_cost, breakdown). notional is absolute traded value (>=0).
        """
        notional = float(abs(notional))
        commission_ht = notional * (self.brokerage_bps + self.comm_bourse_bps + self.reg_liv_bps) / 10000.0
        slippage = notional * (self.slippage_bps / 10000.0)

        tva = commission_ht * float(self.tva_rate)
        total = commission_ht + tva + slippage

        breakdown = {
            "commission_ht": commission_ht,
            "tva": tva,
            "slippage": slippage,
            "total": total,
        }
        return total, breakdown


@dataclass(frozen=True)
class PortfolioConfig:
    # Portfolio semantics
    allow_short: bool = True
    initial_cash: float = 100_000.0

    # Exposure / constraints (optional)
    max_gross: float = 1.0
    max_weight_per_asset: Optional[float] = None
    cash_buffer: float = 0.0

    # Mechanics
    rebalance_policy: RebalancePolicy = "on_change"
    fill_price_model: FillPriceModel = "next_open"
    mtm_model: MarkToMarketModel = "close_t1"

    # -----------------------------
    # Liquidity / Volume constraints
    # -----------------------------
    volume_col: str = "Volume"

    # Layer 3: execution cap (participation)
    use_participation_cap: bool = False
    participation_rate: float = 0.05   # 5% of volume (typical research default)
    participation_basis: str = "adv"    # "bar" or "adv"
    adv_window: int = 20               # used if participation_basis == "adv"

    # What to do with unfilled remainder:
    # - If False: cancel remainder (simple, your current behavior)
    # - If True: carry remainder forward (more realistic, more code)
    carry_unfilled: bool = False

    # Layer 1: optional gate (see section B)
    use_volume_gate: bool = False
    volume_gate_kind: str = "min_ratio_adv"  # "min_abs" or "min_ratio_adv"
    min_volume_abs: float = 0.0        # e.g., 50_000 shares
    min_volume_ratio_adv: float = 0.1  # e.g., Volume >= 0.3 * ADV
    volume_gate_adv_window: int = 20


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

    cooldown_bars: int = 0



    def __post_init__(self) -> None:
        if not (0.0 < self.buy_pct_cash <= 1.0):
            raise ValueError("buy_pct_cash must be in (0, 1].")
        if not (0.0 < self.sell_pct_shares <= 1.0):
            raise ValueError("sell_pct_shares must be in (0, 1].")
        if not (0.0 <= self.cash_buffer < 1.0):
            raise ValueError("cash_buffer must be in [0,1).")
        if not (0.0 < float(self.participation_rate) <= 1.0):
            raise ValueError("participation_rate must be in (0, 1].")

        if str(self.participation_basis) not in ("bar", "adv"):
            raise ValueError("participation_basis must be 'bar' or 'adv'.")

        if int(self.adv_window) < 1:
            raise ValueError("adv_window must be >= 1.")

        if str(self.volume_gate_kind) not in ("min_abs", "min_ratio_adv"):
            raise ValueError("volume_gate_kind must be 'min_abs' or 'min_ratio_adv'.")

        if int(self.volume_gate_adv_window) < 1:
            raise ValueError("volume_gate_adv_window must be >= 1.")

        if float(self.min_volume_ratio_adv) < 0.0:
            raise ValueError("min_volume_ratio_adv must be >= 0.")



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

        last_trade_i: Dict[str, int] = {s: -10**9 for s in symbols}  # bar index of last executed trade per symbol
        cooldown = int(getattr(self.cfg, "cooldown_bars", 0) or 0)

        # -----------------------------
        # Precompute volume and any ADV series we need
        # -----------------------------
        need_volume = bool(self.cfg.use_participation_cap or self.cfg.use_volume_gate)

        vol_series: Dict[str, pd.Series] = {}
        adv_by_window: Dict[int, Dict[str, pd.Series]] = {}  # window -> (symbol -> ADV series)

        if need_volume:
            # Determine which ADV windows are required
            adv_windows: List[int] = []

            if self.cfg.use_participation_cap and str(self.cfg.participation_basis) == "adv":
                adv_windows.append(int(self.cfg.adv_window))

            if self.cfg.use_volume_gate and str(self.cfg.volume_gate_kind) == "min_ratio_adv":
                adv_windows.append(int(self.cfg.volume_gate_adv_window))

            # Unique + sorted windows
            adv_windows = sorted(set(w for w in adv_windows if w >= 1))

            for s in symbols:
                bars = market_data.bars[s]
                if self.cfg.volume_col not in bars.columns:
                    raise KeyError(
                        f"Bars for '{s}' missing required volume column '{self.cfg.volume_col}'. "
                        f"Columns: {list(bars.columns)}"
                    )

                v = pd.to_numeric(bars[self.cfg.volume_col], errors="coerce").astype("float64")
                v = v.reindex(idx).fillna(0.0)
                vol_series[s] = v

            # Precompute all ADV series needed (per window, per symbol)
            for w in adv_windows:
                adv_by_window[w] = {}
                for s in symbols:
                    adv_by_window[w][s] = vol_series[s].rolling(w, min_periods=1).mean()



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
            # --- cooldown filter: block trades if too soon since last trade ---
            if cooldown > 0 and orders:
                filtered = []
                for s, delta in orders:
                    if delta == 0:
                        continue
                    # trade executes at t1 which corresponds to loop index i+1
                    if (i + 1) - last_trade_i.get(s, -10**9) < cooldown:
                        continue  # skip trade for this symbol due to cooldown
                    filtered.append((s, delta))
                orders = filtered
            # -----------------------------
            # Layer 1: volume gate (option)
            # -----------------------------
            if self.cfg.use_volume_gate and orders:
                gated = []
                for s, delta in orders:
                    if delta == 0:
                        continue

                    # ENTRY-ONLY gating:
                    # - If delta increases exposure in the direction of the signal, treat as entry.
                    # - Simpler proxy: gate buys only (delta > 0) for long-only.
                    is_entry = (delta > 0)  # adjust if you allow short entries too

                    if not is_entry:
                        gated.append((s, delta))
                        continue

                    v_t1 = float(vol_series[s].loc[t1])

                    ok = True
                    if self.cfg.volume_gate_kind == "min_abs":
                        ok = v_t1 >= float(self.cfg.min_volume_abs)
                    else:
                        w = int(self.cfg.volume_gate_adv_window)
                        adv_t1 = float(adv_by_window[w][s].loc[t1])
                        ok = v_t1 >= float(self.cfg.min_volume_ratio_adv) * adv_t1

                    if ok:
                        gated.append((s, delta))
                    # else: skip this order entirely

                orders = gated


            # Execute orders at open(t+1)
            for s, delta in orders:
                if delta == 0:
                    continue
                # -----------------------------
                # Layer 3: participation cap (volume-constrained fills)
                # -----------------------------
                if self.cfg.use_participation_cap:
                    # determine liquidity basis at t1
                    if self.cfg.participation_basis == "bar":
                        liq_vol = float(vol_series[s].loc[t1])
                    else:
                        w = int(self.cfg.adv_window)
                        liq_vol = float(adv_by_window[w][s].loc[t1])

                    max_fill = int(max(0.0, float(self.cfg.participation_rate) * liq_vol))

                    # If max_fill is 0, you cannot trade this bar
                    if max_fill <= 0:
                        continue

                    # Cap absolute shares
                    abs_delta = abs(int(delta))
                    if abs_delta > max_fill:
                        # Simple mode: cancel remainder
                        # (carry_unfilled will be handled later if you choose)
                        delta = int(np.sign(delta) * max_fill)
                        if delta == 0:
                            continue

                fill_price = float(open_t1[s])
                notional = abs(delta) * fill_price
                cost, breakdown = self.cfg.cost_model.estimate_cost(notional)
                last_trade_i[s] = i + 1
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
    # portfolio.py  (inside class PortfolioEngine)

    def run_stats_only_arrays(
        self,
        open_px: np.ndarray,
        close_px: np.ndarray,
        sig: np.ndarray,
        vol_px: Optional[np.ndarray] = None,
        adv_cap_px: Optional[np.ndarray] = None,
        adv_gate_px: Optional[np.ndarray] = None,
    ) -> PortfolioStats:
        """
        Ultra-fast stats-only path:
        - open_px, close_px: aligned arrays (same length)
        - sig: aligned signal array values in {-1,0,1} or {0,1}
        """
        pnl, traded, n_fills, final_eq = self._run_stats_fast_single(
            np.asarray(open_px, dtype=np.float64),
            np.asarray(close_px, dtype=np.float64),
            np.asarray(sig, dtype=np.float64),
            np.asarray(vol_px, dtype=np.float64) if vol_px is not None else None,
            np.asarray(adv_cap_px, dtype=np.float64) if adv_cap_px is not None else None,
            np.asarray(adv_gate_px, dtype=np.float64) if adv_gate_px is not None else None,
        )
        return PortfolioStats(
            final_equity=float(final_eq),
            pnl=float(pnl),
            traded_notional=float(traded),
            n_fills=int(n_fills),
        )

    
    def run_stats_only(
        self,
        market_data: Any,   # MarketDataLike
        signal_frame: Any,  # SignalFrame
        symbols: Optional[Sequence[str]] = None,
    ) -> PortfolioStats:
        """
        FAST path for optimization.

        Design goal: be *fast and functional*, not feature-complete.

        Assumptions:
          - Best performance for single-symbol optimization.
          - Uses next-open execution (t -> trade at Open[t+1]) and MTM at Close[t+1] like run().
          - Costs are linear in notional (CostModel), so we compute them cheaply.

        Falls back to the slow full run() if:
          - multiple symbols are provided, or
          - required columns are missing, or
          - not enough bars.
        """
        symbols = list(symbols) if symbols is not None else list(signal_frame.signals.columns)
        if len(symbols) != 1:
            # Keep behavior correct for multi-asset: use the existing full engine
            pres = self.run(market_data, signal_frame, symbols=symbols)
            pnl = float(pres.equity_curve.iloc[-1]) - float(self.cfg.initial_cash) if len(pres.equity_curve) else 0.0
            traded = float(pres.trades["notional"].sum()) if (not pres.trades.empty and "notional" in pres.trades.columns) else 0.0
            n_fills = int(len(pres.trades))
            final_eq = float(pres.equity_curve.iloc[-1]) if len(pres.equity_curve) else float(self.cfg.initial_cash)
            return PortfolioStats(final_equity=final_eq, pnl=pnl, traded_notional=traded, n_fills=n_fills)

        sym = symbols[0]
        if sym not in market_data.bars:
            raise KeyError(f"MarketData missing symbol '{sym}'")

        bars = market_data.bars[sym]
        for col in (self.cfg.open_col, self.cfg.close_col):
            if col not in bars.columns:
                raise KeyError(f"Bars for '{sym}' missing required column '{col}'")

        idx = pd.Index(signal_frame.signals.index).sort_values()
        if len(idx) < 2:
            raise ValueError("Need at least 2 timestamps to apply t+1 fill semantics.")
        # Align bars to signals index (this is the only pandas alignment we do here)
        bars = bars.reindex(idx)
        open_px = bars[self.cfg.open_col].to_numpy(dtype=np.float64, copy=False)
        close_px = bars[self.cfg.close_col].to_numpy(dtype=np.float64, copy=False)

        sig = signal_frame.signals[sym].reindex(idx).to_numpy(dtype=np.float64, copy=False)

        pnl, traded, n_fills, final_eq = self._run_stats_fast_single(open_px, close_px, sig, vol_px=None, adv_cap_px=None, adv_gate_px=None)

        return PortfolioStats(
            final_equity=float(final_eq),
            pnl=float(pnl),
            traded_notional=float(traded),
            n_fills=int(n_fills),
        )

    def _run_stats_fast_single(
        self,
        open_px: np.ndarray,
        close_px: np.ndarray,
        sig: np.ndarray,
        vol_px: Optional[np.ndarray] = None,
        adv_cap_px: Optional[np.ndarray] = None,
        adv_gate_px: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, int, float]:
        """
        Tight numpy loop (no pandas, no dicts, minimal allocations).
        Semantics: decide at i using sig[i], execute at open[i+1], MTM at close[-1].
        Returns: (pnl, traded_notional, n_fills, final_equity)
        """
        n = int(len(open_px))
        if n < 2:
            final_eq = float(self.cfg.initial_cash)
            return 0.0, 0.0, 0, final_eq

        cash = float(self.cfg.initial_cash)
        pos = 0.0

        traded = 0.0
        n_fills = 0

        cm = self.cfg.cost_model
        k = (
            ((cm.brokerage_bps + cm.comm_bourse_bps + cm.reg_liv_bps) / 10000.0) * (1.0 + cm.tva_rate)
            + (cm.slippage_bps / 10000.0)
        )

        cooldown = int(self.cfg.cooldown_bars or 0)
        last_exec_i = -10**9  # store execution index (i+1) when trade happened

        allow_short = bool(self.cfg.allow_short)
        sizing_mode = str(self.cfg.sizing_mode)

        buy_pct_cash = float(self.cfg.buy_pct_cash)
        sell_pct_shares = float(self.cfg.sell_pct_shares)

        cash_buffer = float(self.cfg.cash_buffer)
        max_gross = float(self.cfg.max_gross)
        allow_frac = bool(self.cfg.allow_fractional_shares)

        on_change = (str(self.cfg.rebalance_policy) == "on_change")
        prev_desired = 0.0

        use_gate = bool(self.cfg.use_volume_gate)
        gate_kind = str(self.cfg.volume_gate_kind)

        use_cap = bool(self.cfg.use_participation_cap)
        cap_basis = str(self.cfg.participation_basis)
        prate = float(self.cfg.participation_rate)

        min_abs = float(self.cfg.min_volume_abs)
        min_ratio = float(self.cfg.min_volume_ratio_adv)

        for i in range(n - 1):
            # --- desired position direction from signal ---
            s = sig[i]
            if not np.isfinite(s):
                desired = 0.0
            else:
                if allow_short:
                    desired = -1.0 if s < 0 else (1.0 if s > 0 else 0.0)
                else:
                    desired = 1.0 if s > 0 else 0.0

            exec_i = i + 1  # execution happens at open[i+1]

            # cooldown gate (based on execution index)
            if cooldown > 0 and (exec_i - last_exec_i) < cooldown:
                prev_desired = desired
                continue

            px = float(open_px[exec_i])
            if not np.isfinite(px) or px <= 0.0:
                prev_desired = desired
                continue

            # --- compute order qty ---
            qty = 0.0

            if sizing_mode == "target_weight":
                equity = cash + pos * px
                investable = max(0.0, equity * (1.0 - cash_buffer))

                target_w = desired
                if not allow_short:
                    target_w = max(0.0, target_w)
                # clamp gross for single-asset
                target_w = max(-max_gross, min(max_gross, target_w))

                target_pos = (target_w * investable) / px
                if not allow_frac:
                    target_pos = math.floor(target_pos) if target_pos >= 0 else -math.floor(abs(target_pos))

                qty = float(target_pos - pos)

                if on_change and desired == prev_desired and abs(qty) < 1e-12:
                    prev_desired = desired
                    continue

            else:
                # pct_cash_shares ACTION MODE:
                # +1 => buy using buy_pct_cash of cash
                #  0 => hold (do nothing)
                # -1 => sell sell_pct_shares of current position
                if desired > 0.0:
                    spend = cash * buy_pct_cash
                    buy_shares = spend / px
                    if not allow_frac:
                        buy_shares = math.floor(buy_shares)
                    qty = float(buy_shares)

                elif desired < 0.0:
                    # sell fraction of existing (long or short)
                    sell_shares = abs(pos) * sell_pct_shares
                    if not allow_frac:
                        sell_shares = math.floor(sell_shares)
                    if pos > 0:
                        qty = float(-sell_shares)
                    elif pos < 0:
                        qty = float(sell_shares)  # buy-to-cover
                    else:
                        qty = 0.0

                else:
                    # HOLD
                    qty = 0.0

                if on_change and desired == prev_desired and abs(qty) < 1e-12:
                    prev_desired = desired
                    continue

            if qty == 0.0 or not np.isfinite(qty):
                prev_desired = desired
                continue

            # ENTRY-ONLY proxy for gate: only gate when increasing absolute exposure
            is_entry = (abs(pos + qty) > abs(pos) + 1e-12)

            # --- Layer 1: volume / adv gate ---
            if use_gate and is_entry:
                if gate_kind == "min_abs":
                    if vol_px is None:
                        prev_desired = desired
                        continue
                    v = float(vol_px[exec_i])
                    if (not np.isfinite(v)) or (v < min_abs):
                        prev_desired = desired
                        continue

                elif gate_kind == "min_ratio_adv":
                    if (vol_px is None) or (adv_gate_px is None):
                        prev_desired = desired
                        continue
                    v = float(vol_px[exec_i])
                    adv = float(adv_gate_px[exec_i])
                    if (not np.isfinite(v)) or (not np.isfinite(adv)) or adv <= 0.0:
                        prev_desired = desired
                        continue
                    if v < (min_ratio * adv):
                        prev_desired = desired
                        continue

                else:
                    # unknown gate kind -> safest is no-trade
                    prev_desired = desired
                    continue

            # --- Layer 3: participation cap (bar or adv) ---
            if use_cap:
                liq = None
                if cap_basis == "bar":
                    if vol_px is None:
                        prev_desired = desired
                        continue
                    liq = float(vol_px[exec_i])
                else:
                    if adv_cap_px is None:
                        prev_desired = desired
                        continue
                    liq = float(adv_cap_px[exec_i])

                if (liq is None) or (not np.isfinite(liq)) or liq <= 0.0:
                    prev_desired = desired
                    continue

                max_fill = prate * liq
                if max_fill <= 0.0:
                    prev_desired = desired
                    continue

                abs_qty = abs(qty)
                if abs_qty > max_fill:
                    qty = float(math.copysign(math.floor(max_fill), qty))
                    if qty == 0.0:
                        prev_desired = desired
                        continue

            # --- execute ---
            notional = abs(qty) * px
            cost = notional * k

            if qty > 0:
                cash -= (notional + cost)
            else:
                cash += (notional - cost)

            pos += qty
            traded += notional
            n_fills += 1
            last_exec_i = exec_i
            prev_desired = desired

        final_eq = cash + float(pos) * float(close_px[-1])
        pnl = final_eq - float(self.cfg.initial_cash)
        return float(pnl), float(traded), int(n_fills), float(final_eq)

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
            return pd.DataFrame(columns=["timestamp", "symbol", "qty", "price", "notional", "cost", "commission_ht", "tva", "slippage"])
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
                "tva": f.cost_breakdown.get("tva", np.nan),
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
        Action-mode sizing (long-only, percentages):
        +1 -> buy using buy_pct_cash of available cash (split across +1 symbols)
        0 -> do nothing (hold)
        -1 -> sell sell_pct_shares of current long position (partial/total exit)
        """
        orders: List[Tuple[str, int]] = []

        # Treat signals as actions: only +1 counts as buy
        buy_syms = [s for s in symbols if float(sig_row.get(s, 0.0)) > 0.0]
        n_buy = max(1, len(buy_syms))

        avail_cash = self._available_cash(state, equity_t)

        for s in symbols:
            sig = float(sig_row.get(s, 0.0))
            sig = float(np.clip(sig, -1.0, 1.0))

            pos = int(state.position(s))
            px = float(open_t1[s])
            cap_abs = self._max_abs_shares_cap(equity_t, px)

            # ---- SELL ONLY on -1 ----
            if sig == -1.0:
                if pos <= 0:
                    continue
                q = int(np.ceil(self.cfg.sell_pct_shares * abs(pos)))
                q = min(q, abs(pos))
                delta = -q
                if delta != 0:
                    orders.append((s, int(delta)))
                continue

            # ---- HOLD on 0 ----
            if sig == 0.0:
                continue

            # ---- BUY ONLY on +1 ----
            if sig == 1.0:
                cash_budget = (self.cfg.buy_pct_cash * avail_cash) / n_buy
                buy_qty = int(cash_budget / px)
                if buy_qty <= 0:
                    continue

                desired_pos = min(pos + buy_qty, cap_abs)
                delta = desired_pos - pos
                if delta != 0:
                    orders.append((s, int(delta)))
                continue

        return orders




