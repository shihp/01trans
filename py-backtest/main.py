from pathlib import Path

from core.data_loader import load_ohlc_csv
from core.strategy import MovingAverageCrossConfig, moving_average_cross_signal
from core.backtester import backtest
from core.metrics import (
    annualized_return,
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "sample.csv"

    df = load_ohlc_csv(str(data_path))
    prices = df["close"]

    config = MovingAverageCrossConfig(short_window=3, long_window=5)
    signal = moving_average_cross_signal(prices, config=config)

    result = backtest(prices, signal)

    strat_ret = result["strategy_returns"]
    equity = result["equity_curve"]

    print("===== Backtest Summary (Sample Data) =====")
    print(f"Annualized Return : {annualized_return(strat_ret):.2%}")
    print(f"Annualized Vol    : {annualized_vol(strat_ret):.2%}")
    print(f"Sharpe Ratio      : {sharpe_ratio(strat_ret):.2f}")
    print(f"Max Drawdown      : {max_drawdown(equity):.2%}")

    # Plot equity curve
    try:
        import matplotlib.pyplot as plt

        equity.plot(title="Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(Plotting skipped due to error: {exc})")


if __name__ == "__main__":
    main()

