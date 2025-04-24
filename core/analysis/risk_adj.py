import numpy as np


def sharpe_ratio(V_total):

    returns = np.diff(V_total) / V_total[:-1]

    mu = np.mean(returns)
    sigma = np.std(returns)

    sharpe_ratio = (mu / sigma) * np.sqrt(len(V_total))  # annualise
    return sharpe_ratio


def calmar_ratio(V_total):
    returns = np.diff(V_total) / V_total[:-1]

    mdd = max_drawdown(V_total)
    mu = np.mean(returns)

    annual_return = mu * len(V_total)  # annualise
    calmar_ratio = annual_return / mdd if mdd != 0 else np.nan

    return calmar_ratio


def max_drawdown(V_total):

    peak = np.maximum.accumulate(V_total)
    drawdown = (V_total - peak) / peak
    mdd = -np.min(drawdown)

    return mdd
