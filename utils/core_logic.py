# core_logic.py

from datetime import datetime

def simulate_trade(binance_price, bybit_price, trade_size_eth=1.0, fee_pct=0.002):
    if binance_price < bybit_price:
        buy_ex = 'binance'
        sell_ex = 'bybit'
        buy_price = binance_price
        sell_price = bybit_price
    else:
        buy_ex = 'bybit'
        sell_ex = 'binance'
        buy_price = bybit_price
        sell_price = binance_price

    spread = abs(sell_price - buy_price) / min(buy_price, sell_price)
    gross_profit = (sell_price - buy_price) * trade_size_eth
    fee_cost = (buy_price + sell_price) * trade_size_eth * (fee_pct / 2)
    net_profit = gross_profit - fee_cost

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "buy_exchange": buy_ex,
        "sell_exchange": sell_ex,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "spread": spread,
        "gross_profit": gross_profit,
        "net_profit": net_profit
    }
