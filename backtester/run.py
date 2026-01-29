# run_signals.py
from data import YahooFinanceDataSource
from indicators import IndicatorEngine
from strategy import make_ma_cross_strategy  # from your strategy.py
from portfolio import PortfolioEngine, PortfolioConfig
from results import ResultsAnalyzer
# test_portfolio_plot.py
import matplotlib.pyplot as plt

from data import MarketData
from indicators import IndicatorEngine  # or whatever class computes FeaturesData
from strategy import MovingAverageCrossStrategy  # adapt to your real strategy class
from portfolio import PortfolioEngine, PortfolioConfig, CostModel

def main() -> None:
    # 1) Load AAPL data
    srcY = YahooFinanceDataSource(timezone="UTC")
    mdY = srcY.load(
        symbols=["IAM"],
        paths={"IAM": "DATA IAM.xlsx"},
        start="2000-01-01",
        end="2014-01-01",
        interval="1d",
    )

    # 2) Build strategy (SMA 15/50 crossover by default)
    strat = make_ma_cross_strategy(fast_window=100, slow_window=400, allow_short=False, nan_policy="flat")

    # 3) Compute required features for this strategy
    specs = strat.required_features()
    ind = IndicatorEngine(cache_dir=".cache/features", enable_disk_cache=True)
    featsY = ind.compute(mdY, specs, symbols=["IAM"])

    # 4) Generate signals (intent at time t)
    sf = strat.generate_signals(mdY, featsY, symbols=["IAM"])
    sig = sf.signals["IAM"]

    # 5) Show when signals are generated / change
    # "Generated" here means the timestamps where the signal takes a value (after warmup).
    valid = sf.validity["IAM"] if sf.validity is not None else sig.notna()

    # 4) Run portfolio
    cfg = PortfolioConfig(
        allow_short=True,
        initial_cash=1_000_000.0,
        rebalance_policy="on_change",
        max_gross=1.0,
        cash_buffer=0.0,
        # Costs: set to 0 for your first sanity test, then re-enable
        cost_model=CostModel(
            brokerage_bps=0.0,
            exchange_bps=0.0,
            settlement_bps=0.0,
            slippage_bps=0.0,
            vat_rate=0.0,
        ),
    )
    engine = PortfolioEngine(cfg)
    res = engine.run(mdY, sf, symbols=["IAM"])



    an = ResultsAnalyzer(periods_per_year=252, rf_annual=0.0)
    rep = an.analyze(res, market_data=mdY, symbols=["IAM"])

    an.plot_cumulative_returns(rep)
    an.plot_drawdown(rep)
    print(rep.metrics)
    print(rep.tables["monthly_returns"])



if __name__ == "__main__":
    main()





