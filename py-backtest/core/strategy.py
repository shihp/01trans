from dataclasses import dataclass

import pandas as pd


@dataclass
class MovingAverageCrossConfig:
    short_window: int = 10
    long_window: int = 30

    def __post_init__(self) -> None:
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("Windows must be positive integers")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window")


def moving_average_cross_signal(
    prices: pd.Series,
    config: MovingAverageCrossConfig | None = None,
) -> pd.Series:
    """
    Generate position signal based on a moving average cross strategy.

    Signal semantics:
    - 1: long (fully invested)
    - 0: flat (no position)
    """
    if config is None:
        config = MovingAverageCrossConfig()

    short_ma = prices.rolling(config.short_window).mean()
    long_ma = prices.rolling(config.long_window).mean()

    signal = (short_ma > long_ma).astype(int)
    signal.name = "signal"
    return signal


__all__ = ["MovingAverageCrossConfig", "moving_average_cross_signal"]

