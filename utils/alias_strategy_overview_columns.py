# utils/alias_strategy_overview_columns.py

def alias_strategy_overview_columns(df):
    return df.rename(columns={
        "Sharpe Ratio": "sharpe_ratio",
        "Total Return": "total_return",
        "Max Drawdown": "max_drawdown",
        "Win Rate": "win_rate",
        "Expectancy": "expectancy",
        "Trades": "trades",
        "STD": "std",
        "TP": "tp",
        "SL": "sl",
        "asset": "asset",
        "strategy": "strategy"
    })
