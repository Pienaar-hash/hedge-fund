# === strategies/ict.py ===
import os
import pandas as pd
import numpy as np
from core.strategy_base import Strategy as StrategyBase

SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
DATA_PATH = f"data/processed/{SYMBOL}_{TIMEFRAME}.csv"
LOG_PATH = f"logs/ict_trades_{SYMBOL.lower()}_{TIMEFRAME}.csv"

Z_THRESHOLD = 2
ATR_WINDOW = 14

class StrategyImpl(StrategyBase):
    def configure(self, params):
        global SYMBOL, TIMEFRAME, DATA_PATH, LOG_PATH
        SYMBOL = params.get("symbol", SYMBOL)
        TIMEFRAME = params.get("timeframe", TIMEFRAME)
        DATA_PATH = f"data/processed/{SYMBOL}_{TIMEFRAME}.csv"
        LOG_PATH = f"logs/ict_trades_{SYMBOL.lower()}_{TIMEFRAME}.csv"
        global Z_THRESHOLD, ATR_WINDOW
        Z_THRESHOLD = params.get("z_threshold", Z_THRESHOLD)
        ATR_WINDOW = params.get("atr_window", ATR_WINDOW)

    def run(self):
        if not os.path.exists(DATA_PATH):
            print(f"❌ Missing file: {DATA_PATH}")
            self.trades_df = pd.DataFrame()
            return

        df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        df['hl2'] = (df['high'] + df['low']) / 2
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['atr'] = df['range'].rolling(ATR_WINDOW).mean()

        df['fvg_up'] = (df['low'].shift(1) > df['high'].shift(2))
        df['fvg_down'] = (df['high'].shift(1) < df['low'].shift(2))

        df['z_displacement'] = (df['body'] - df['body'].rolling(20).mean()) / df['body'].rolling(20).std()
        df.dropna(inplace=True)

        in_position = False
        trades = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            ts = df.index[i]

            if not in_position:
                if row['fvg_up'] and row['z_displacement'] > Z_THRESHOLD:
                    entry_price = row['close']
                    entry_time = ts
                    direction = 1
                    in_position = True

                elif row['fvg_down'] and row['z_displacement'] < -Z_THRESHOLD:
                    entry_price = row['close']
                    entry_time = ts
                    direction = -1
                    in_position = True

            elif in_position:
                ret = (row['close'] - entry_price) / entry_price * direction
                if abs(ret) >= 0.02:
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': ts,
                        'entry_price': entry_price,
                        'exit_price': row['close'],
                        'return': ret
                    })
                    in_position = False

        self.trades_df = pd.DataFrame(trades)

    def log_results(self):
        if not hasattr(self, 'trades_df') or self.trades_df.empty:
            print("⚠️ No ICT trades logged.")
        else:
            self.trades_df.to_csv(LOG_PATH, index=False)
            print(f"✅ Saved {len(self.trades_df)} ICT trades to {LOG_PATH}")
