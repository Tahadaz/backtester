# app.py
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

st.set_page_config(page_title="SMA Backtest + Optimize (Excel)", layout="wide")

# -----------------------------
# Data loader (simple)
# -----------------------------
@st.cache_data
def load_bmce_excel(uploaded_file, sheet_name=0, start=None, end=None) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    df = df.rename(columns={
        "Date": "Date",
        "Ouvt": "Open",
        "+Haut": "High",
        "+Bas": "Low",
        "Clôture": "Close",
        "Cloture": "Close",
        "Volume": "Volume",
    })

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    if start is not None:
        df = df.loc[pd.to_datetime(start):]
    if end is not None:
        df = df.loc[:pd.to_datetime(end)]

    df = df.dropna(subset=["Open", "Close"])
    return df


# -----------------------------
# Fast SMA via cumsum
# -----------------------------
def sma_cumsum(x: np.ndarray, window: int) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or window > n:
        return out
    cs = np.cumsum(x, dtype=np.float64)
    out[window - 1:] = (cs[window - 1:] - np.concatenate(([0.0], cs[:-window]))) / window
    return out


def make_positions_long_flat(close: np.ndarray, sma: np.ndarray) -> np.ndarray:
    signal = (close > sma).astype(np.int8)
    pos = np.zeros(close.shape[0], dtype=np.int8)
    pos[1:] = signal[:-1]
    return pos


# -----------------------------
# Backtest (simple, trade only on change) + ledger
# performance = realized_pnl / peak_invested
# -----------------------------
def backtest_simple(open_px, close_px, window,
                    buy_pct_cash=1.0, sell_pct_shares=1.0,
                    initial_cash=100_000.0, round_shares=True,
                    index=None):
    open_px = np.asarray(open_px, float)
    close_px = np.asarray(close_px, float)
    n = len(open_px)
    if n == 0:
        return {"pnl": 0.0, "ledger": pd.DataFrame()}

    sma = sma_cumsum(close_px, int(window))
    pos = make_positions_long_flat(close_px, sma)

    cash, shares, avg_cost = float(initial_cash), 0.0, 0.0
    equity = np.empty(n, float)
    ledger = []

    net_inv, peak_inv, realized = 0.0, 0.0, 0.0

    for i in range(n):
        px_open = open_px[i]
        px_close = close_px[i]
        if not (np.isfinite(px_open) and px_open > 0 and np.isfinite(px_close)):
            equity[i] = cash + shares * (px_close if np.isfinite(px_close) else 0.0)
            continue

        if i > 0:
            dpos = int(pos[i]) - int(pos[i - 1])

            if dpos == 1:  # ENTER long
                spend = cash * max(0.0, min(1.0, float(buy_pct_cash)))
                qty = spend / px_open
                if round_shares:
                    qty = np.floor(qty)
                if qty > 0:
                    notional = qty * px_open
                    cash -= notional

                    new_sh = shares + qty
                    avg_cost = ((avg_cost * shares) + notional) / new_sh if new_sh > 0 else 0.0
                    shares = new_sh

                    net_inv += notional
                    peak_inv = max(peak_inv, net_inv)

                    ledger.append({
                        "time": (index[i] if index is not None else i),
                        "side": "BUY",
                        "qty": float(qty),
                        "price": float(px_open),
                        "notional": float(notional),
                        "net_invested": float(net_inv),
                        "peak_invested": float(peak_inv),
                        "realized_pnl": float(realized),
                        "performance": float(realized / (peak_inv + 1e-12)),
                    })

            elif dpos == -1:  # EXIT long (set sell_pct_shares=1.0 if you want full flat)
                qty = shares * max(0.0, min(1.0, float(sell_pct_shares)))
                if round_shares:
                    qty = np.floor(qty)
                if qty > 0:
                    notional = qty * px_open
                    cash += notional
                    shares -= qty

                    realized += notional - (avg_cost * qty)
                    net_inv -= notional

                    if shares <= 1e-12:
                        shares, avg_cost = 0.0, 0.0

                    ledger.append({
                        "time": (index[i] if index is not None else i),
                        "side": "SELL",
                        "qty": float(qty),
                        "price": float(px_open),
                        "notional": float(notional),
                        "net_invested": float(net_inv),
                        "peak_invested": float(peak_inv),
                        "realized_pnl": float(realized),
                        "performance": float(realized / (peak_inv + 1e-12)),
                    })

        equity[i] = cash + shares * px_close

    final_equity = float(equity[-1])
    pnl = final_equity - float(initial_cash)
    perf = float(realized / (peak_inv + 1e-12))  # simple efficiency metric

    return {
        "final_equity": final_equity,
        "pnl": float(pnl),
        "equity_curve": equity,
        "ledger": pd.DataFrame(ledger),
        "realized_pnl": float(realized),
        "peak_invested": float(peak_inv),
        "performance": perf,
    }


# -----------------------------
# Optimizer (grid search)
# objective: "pnl" or "performance"
# -----------------------------
def optimize_grid(open_px, close_px, index,
                  windows, buy_pcts, sell_pcts,
                  initial_cash, objective):
    rows = []
    best = None

    for w in windows:
        for b in buy_pcts:
            for s in sell_pcts:
                out = backtest_simple(
                    open_px=open_px,
                    close_px=close_px,
                    window=int(w),
                    buy_pct_cash=float(b),
                    sell_pct_shares=float(s),
                    initial_cash=float(initial_cash),
                    round_shares=True,
                    index=None,   # keep ledger indices numeric for speed
                )

                score = out[objective]
                row = {
                    "window": int(w),
                    "buy_pct_cash": float(b),
                    "sell_pct_shares": float(s),
                    "score": float(score),
                    "pnl": float(out["pnl"]),
                    "performance": float(out["performance"]),
                    "realized_pnl": float(out["realized_pnl"]),
                    "peak_invested": float(out["peak_invested"]),
                }
                rows.append(row)

                if best is None or row["score"] > best["score"]:
                    best = dict(row)
    # ... after the loops
    if not rows:
        # nothing was evaluated (empty parameter grid or all trials skipped)
        return pd.DataFrame(), {}

    res = pd.DataFrame(rows)

    # If you accidentally used another name, fix it here
    if "score" not in res.columns:
        raise ValueError(
            f"optimize_grid: missing 'score' column. Columns are: {list(res.columns)}. "
            f"Example row keys: {list(rows[0].keys())}"
        )



    res = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return res, (best if best is not None else {})

def pick_best_median_worst(res: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # assume res already has a "score" column
    res_sorted = res.sort_values("score", ascending=False).reset_index(drop=True)

    best = res_sorted.head(k)

    worst = res_sorted.tail(k).sort_values("score", ascending=False).reset_index(drop=True)

    mid_idx = len(res_sorted) // 2
    start = max(0, mid_idx - k // 2)
    mid = res_sorted.iloc[start:start + k].reset_index(drop=True)

    return best.reset_index(drop=True), mid, worst


# -----------------------------
# UI
# -----------------------------
st.title("SMA Backtest + Optimization (Excel .xlsx)")

uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload an .xlsx with columns like: Date, Ouvt, +Haut, +Bas, Clôture, Volume")
    st.stop()

with st.sidebar:
    st.header("Data")
    sheet = st.text_input("Sheet name / index", value="0")
    use_date_filter = st.checkbox("Filter by date", value=False)
    start = st.date_input("Start date (optional)", value=None)
    end = st.date_input("End date (optional)", value=None)

    st.header("Backtest")
    window = st.number_input("SMA window", 2, 500, 20, 1)
    buy_pct_cash = st.slider("Buy % cash (on entry)", 0.0, 1.0, 1.0, 0.05)
    sell_pct_shares = st.slider("Sell % shares (on exit)", 0.0, 1.0, 1.0, 0.05)
    initial_cash = st.number_input("Initial cash", min_value=0.0, value=100000.0, step=1000.0)

    run_bt = st.button("Run backtest", type="primary")

    st.header("Optimize")

    w_min, w_max = st.slider("Window range", 2, 500, (5, 100), 1)
    w_step = st.number_input("Window step", 1, 50, 1, 1)

    st.subheader("Buy % of cash grid")
    buy_min  = st.number_input("buy min", 0.0, 1.0, 0.25, 0.01)
    buy_max  = st.number_input("buy max", 0.0, 1.0, 1.00, 0.01)
    buy_step = st.number_input("buy step", 0.01, 1.0, 0.25, 0.01)

    st.subheader("Sell % of shares grid")
    sell_min  = st.number_input("sell min", 0.0, 1.0, 1.00, 0.01)
    sell_max  = st.number_input("sell max", 0.0, 1.0, 1.00, 0.01)
    sell_step = st.number_input("sell step", 0.01, 1.0, 0.25, 0.01)

    objective = st.selectbox("Objective", ["pnl", "performance"], index=0)
    run_opt = st.button("Optimize", type="secondary")


sheet_name = int(sheet) if sheet.strip().isdigit() else sheet.strip()
try:
    start = pd.to_datetime(start) if use_date_filter else None
    end   = pd.to_datetime(end)   if use_date_filter else None

    df = load_bmce_excel(
        uploaded_file=uploaded,
        sheet_name=sheet_name,
        start=start,
        end=end,
    )
except Exception as e:
    st.error(f"Failed to load Excel: {e}")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)

open_px = df["Open"].to_numpy(dtype=float)
close_px = df["Close"].to_numpy(dtype=float)

if run_bt:
    out = backtest_simple(
        open_px=open_px,
        close_px=close_px,
        window=int(window),
        buy_pct_cash=float(buy_pct_cash),
        sell_pct_shares=float(sell_pct_shares),
        initial_cash=float(initial_cash),
        round_shares=True,
        index=df.index,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Equity", f"{out['final_equity']:,.2f}")
    c2.metric("PnL", f"{out['pnl']:,.2f}")
    c3.metric("Realized PnL", f"{out['realized_pnl']:,.2f}")
    c4.metric("Peak Invested", f"{out['peak_invested']:,.2f}")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Equity Curve")
        st.line_chart(pd.Series(out["equity_curve"], index=df.index, name="Equity"))
    with right:
        st.subheader("Ledger")
        st.dataframe(out["ledger"], use_container_width=True, height=450)


if run_opt:
    windows = list(range(int(w_min), int(w_max) + 1, int(w_step)))

    def parse_list(s):
        return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

    buy_pcts = np.arange(buy_min, buy_max + 1e-12, buy_step)
    sell_pcts = np.arange(sell_min, sell_max + 1e-12, sell_step)

    # optional: round to avoid floating display noise (0.30000000004)
    buy_pcts = np.round(buy_pcts, 6).tolist()
    sell_pcts = np.round(sell_pcts, 6).tolist()

    res, best = optimize_grid(
        open_px=open_px,
        close_px=close_px,
        index=df.index,
        windows=windows,
        buy_pcts=buy_pcts,
        sell_pcts=sell_pcts,
        initial_cash=float(initial_cash),
        objective=objective,
    )

    st.subheader("Optimization Results")
    st.dataframe(res, use_container_width=True, height=350)

    if best:
        st.subheader("Best Params")
        st.write(best)

        # Re-run best with full ledger dates for display
        best_out = backtest_simple(
            open_px=open_px,
            close_px=close_px,
            window=int(best["window"]),
            buy_pct_cash=float(best["buy_pct_cash"]),
            sell_pct_shares=float(best["sell_pct_shares"]),
            initial_cash=float(initial_cash),
            round_shares=True,
            index=df.index,
        )

        left, right = st.columns([2, 3])
        with left:
            st.subheader("Best Equity Curve")
            st.line_chart(pd.Series(best_out["equity_curve"], index=df.index, name="Equity"))
        with right:
            st.subheader("Best Ledger")
            st.dataframe(best_out["ledger"], use_container_width=True, height=450)
    best5, mid5, worst5 = pick_best_median_worst(res, k=5)

    st.subheader("Top 5 strategies")
    st.dataframe(best5, use_container_width=True)

    st.subheader("Middle 5 strategies (around median score)")
    st.dataframe(mid5, use_container_width=True)

    st.subheader("Worst 5 strategies")
    st.dataframe(worst5, use_container_width=True)

