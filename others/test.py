from others.analyzers import portfolio_to_result, ReportBuilder

# Suppose you have:
# pf = engine.run_single_asset(symbol, bars, desired_position)
# bars = market_data.bars[symbol]

result = portfolio_to_result(symbol, bars, pf, interval="1D")
report = ReportBuilder(rolling_window=63).build(result)

# Tables
print(report.summary_table)
print(report.monthly_table)       # monthly returns pivot (years x months)
print(report.drawdown_table.head(10))
print(report.trade_table.tail(10))

# Plots
# report.figures["equity_curve"].show()  # in notebooks
# Or save:
for name, fig in report.figures.items():
    fig.savefig(f"{name}.png", dpi=150, bbox_inches="tight")
