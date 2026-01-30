# test_optimize_real_file.py
from __future__ import annotations

from pathlib import Path

from optimize import OptimizeConfig, ParamDef, run_optimization
from engine import EngineSpec, DataConfig, StrategyConfig
from portfolio import PortfolioConfig, CostModel


def main():
    # -------------------------
    # 1) Your local file path
    # -------------------------
    # Put your real path here:
    FILE_PATH = Path(r"C:\Users\taha\Downloads\backtester\backtester\DATA IAM.xlsx")  # <-- CHANGE THIS
    SYMBOL = "IAM"  # just a label for your dict key / reporting

    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    # -------------------------
    # 2) Base spec (single symbol)
    # -------------------------
    data_cfg = DataConfig(
        source="bmce",
        symbols=[SYMBOL],
        timezone="GMT",
        interval="1d",
        start=None,
        end=None,
        bmce_paths=str(FILE_PATH),  # can be str or Path
    )

    # Choose one strategy to test
    STRATEGY_KIND = "sma_price"   # "sma_price" or "ma_cross"

    if STRATEGY_KIND == "sma_price":
        strat_cfg = StrategyConfig(
            kind="sma_price",
            params={
                "window": 50,
                "allow_short": False,
                "nan_policy": "flat",
            },
        )
    else:
        strat_cfg = StrategyConfig(
            kind="ma_cross",
            params={
                "fast_window": 20,
                "slow_window": 50,
                "allow_short": False,
                "nan_policy": "flat",
            },
        )

    port_cfg = PortfolioConfig(
        allow_short=False,
        initial_cash=100_000.0,
        sizing_mode="pct_cash_shares",   # <-- your percentages mode
        buy_pct_cash=0.25,
        sell_pct_shares=1.00,
        cooldown_bars=0,
        cost_model=CostModel(
            brokerage_bps=0.0,
            exchange_bps=0.0,
            settlement_bps=0.0,
            slippage_bps=0.0,
            vat_rate=0.0,
        ),
    )

    base_spec = EngineSpec(
        data=data_cfg,
        indicators=None,      # optimizer ignores engine.indicators
        strategy=strat_cfg,
        portfolio=port_cfg,
        benchmark=None,
        plot_indicators=[],
        periods_per_year=252,
        rf_annual=0.0,
    )

    # -------------------------
    # 3) Active params (domains)
    #    IMPORTANT: create ParamDef with desired ranges here
    #    (don't mutate .domain later in Streamlit)
    # -------------------------
    active_params = []

    if STRATEGY_KIND == "sma_price":
        active_params += [
            ParamDef("strategy.window", "int", (10, 20, 5), int),
        ]
    else:
        active_params += [
            ParamDef("strategy.fast_window", "int", (5, 20, 1), int),
            ParamDef("strategy.slow_window", "int", (20, 40, 1), int),
        ]

    active_params += [
        ParamDef("portfolio.buy_pct_cash", "float", (0.10, 1.00, 0.10), float),
        ParamDef("portfolio.sell_pct_shares", "float", (0.25, 1.00, 0.25), float),
        ParamDef("portfolio.cooldown_bars", "int", (0, 20, 2), int),
        # ParamDef("data.window", "date_window", [("2021-01-01","2023-12-31"), ...], lambda x: x),
    ]

    # -------------------------
    # 4) Run optimization
    # -------------------------
    cfg = OptimizeConfig(
        method="grid",   # "grid" or "random"
        seed=42,
        n_trials=200,    # used only for random
        top_k=20,
        enable_disk_cache=False,
        enable_memory_cache=True,
        feature_cache_dir=".cache/features",
    )

    best, top_df, best_params, best_spec = run_optimization(
        base_spec=base_spec,
        active_params=active_params,
        cfg=cfg,
    )

    # -------------------------
    # 5) Print results (PNL then efficiency)
    # -------------------------
    print("\n========================")
    print("BEST RESULT (ranked by PNL then efficiency)")
    print("========================")
    print("PNL:", best.pnl)
    print("Traded notional:", best.traded_notional)
    print("Efficiency (PNL / notional):", best.efficiency)
    print("Nb fills:", best.n_fills)
    print("Params:", best.params)
    print("Error:", best.error)

    print("\n========================")
    print("TOP RESULTS")
    print("========================")

    show_cols = [*best.params.keys(), "pnl", "efficiency", "traded_notional", "n_fills", "error"]
    show_cols = [c for c in show_cols if c in top_df.columns]
    print(top_df[show_cols].head(20).to_string(index=False))

    print("\n========================")
    print("BEST SPEC (sanity)")
    print("========================")
    print("Strategy:", best_spec.strategy)
    print("Portfolio:", best_spec.portfolio)


if __name__ == "__main__":
    main()


"""
TEXT EXPLANATION (for Cursor review)

- This script runs optimize.py using a REAL BMCE CSV/XLSX file (no Streamlit).
- It builds EngineSpec with DataConfig(source='bmce', bmce_paths=<file>).
- It defines ParamDef domains directly (so no frozen-dataclass mutation).
- It runs run_optimization() and prints PNL + efficiency clearly.
"""
