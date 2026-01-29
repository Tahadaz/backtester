# test_engine.py
from engine import (
    BacktestEngine, EngineSpec,
    DataConfig, IndicatorsConfig, StrategyConfig
)
from portfolio import PortfolioConfig, CostModel
import matplotlib.pyplot as plt


def main():
    # ---- EDIT THIS ----
    PATH = r"C:\Users\taha\Downloads\backtester\Data IAM.xlsx"  # or .csv
    SYMBOL = "IAM"

    spec = EngineSpec(
        data=DataConfig(
            source="bmce",
            symbols=[SYMBOL],
            bmce_paths=PATH,      # single symbol => single path
            timezone="GMT",
            interval="1d",
        ),
        indicators=IndicatorsConfig(
            specs=None,  # inferred from strategy
        ),
        strategy=StrategyConfig(
            kind="ma_cross",
            params={
                "fast_window": 20,
                "slow_window": 50,
                "allow_short": True,
                "nan_policy": "flat",
            },
        ),
        portfolio=PortfolioConfig(
            allow_short=True,
            initial_cash=1_000_000.0,
            rebalance_policy="on_change",
            max_gross=1.0,
            cash_buffer=0.0,
            # start with 0 costs for sanity
            cost_model=CostModel(
                brokerage_bps=0.0,
                exchange_bps=0.0,
                settlement_bps=0.0,
                slippage_bps=0.0,
                vat_rate=0.0,
            ),
        ),
        periods_per_year=252,
        rf_annual=0.0,
    )

    bundle = BacktestEngine(spec).run()

    # Print headline metrics
    print("=== METRICS ===")
    for k, v in bundle.report.metrics.items():
        print(f"{k:28s} {v}")

    # Plot cumulative returns
    cum = bundle.report.series["cum_returns"].dropna()
    plt.figure(figsize=(12, 5))
    plt.plot(cum.index, cum.values)
    plt.title("Cumulative Returns (MA Cross)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
