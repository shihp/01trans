import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _to_returns(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(float)
    return s


def annualized_return(returns: pd.Series) -> float:
    r = _to_returns(returns)
    if r.empty:
        return 0.0
    cumulative = (1.0 + r).prod()
    n = r.shape[0]
    return float(cumulative ** (TRADING_DAYS / n) - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    r = _to_returns(returns)
    if r.empty:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    r = _to_returns(returns)
    if r.empty:
        return 0.0

    daily_rf = rf / TRADING_DAYS
    excess = r - daily_rf
    vol = annualized_vol(excess)
    if vol == 0.0:
        return 0.0
    return annualized_return(excess) / vol


def max_drawdown(equity_curve: pd.Series) -> float:
    eq = equity_curve.dropna().astype(float)
    if eq.empty:
        return 0.0
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    return float(drawdown.min())


__all__ = ["annualized_return", "annualized_vol", "sharpe_ratio", "max_drawdown"]

