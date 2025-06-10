# utils/metrics.py
import numpy as np
import pandas as pd

def compute_sharpe(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0

def compute_volatility(returns):
    return np.std(returns) * np.sqrt(252)

def compute_max_drawdown(cumulative):
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def compute_cagr(cumulative, timestamps, periods=252):
    total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1
    num_years = (timestamps.iloc[-1] - timestamps.iloc[0]).days / 365.25
    return (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
