import pandas as pd


def backtest(
    prices: pd.Series,
    signal: pd.Series,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Very simple long-only backtester.

    Assumptions:
    - Trades executed at close price
    - No fees, no slippage
    - `signal` represents target position (0 or 1) in units of capital fraction
    """
    if prices.empty:
        raise ValueError("prices series is empty")

    # Align indexes
    prices = prices.sort_index()
    signal = signal.reindex(prices.index).fillna(0)

    # Underlying daily returns
    returns = prices.pct_change().fillna(0.0)

    # Use previous day's signal as position for today's return
    positioned = signal.shift(1).fillna(0)
    strategy_returns = positioned * returns

    equity_curve = (1.0 + strategy_returns).cumprod() * float(initial_capital)

    df = pd.DataFrame(
        {
            "price": prices,
            "signal": signal,
            "position": positioned,
            "returns": returns,
            "strategy_returns": strategy_returns,
            "equity_curve": equity_curve,
        }
    )

    return df


__all__ = ["backtest"]

