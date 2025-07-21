# === strategies/volatility_target.py with advanced exit control ===
import os
import pandas as pd
import numpy as np
from core.strategy_base import Strategy

LOG_FOLDER = "logs"

class StrategyImpl(Strategy):
    def run(self):
        all_trades = []

        for asset in self.assets:
            symbol = asset['symbol']
            tf = asset['tf']
            df = self.load_data(symbol, tf)
            if df is None or len(df) < self.lookback:
                continue

            capital = self.starting_equity / len(self.assets)
            equity = capital
            position_open = False
            entry_price, entry_time, entry_idx, leverage, is_long = None, None, 0, 1.0, None

            for i in range(self.lookback, len(df), self.rebalance_period):
                row = df.iloc[i]

                ema = row.get("ema", row["close"])
                ema_slope = df["ema"].pct_change().iloc[i] if "ema" in df.columns else 0

                if self.use_trend_filter:
                    if row["close"] > ema:
                        direction = 1  # long
                    elif self.allow_shorts and (ema - row["close"]) / ema >= self.short_threshold and ema_slope < 0:
                        direction = -1  # short
                    else:
                        continue  # skip trade
                else:
                    direction = 1

                vol = df.iloc[i - self.lookback:i]["log_return"].std()
                if vol == 0 or pd.isna(vol):
                    continue

                dynamic_target_vol = self.target_vol * (1.2 if direction == 1 else 0.8)
                leverage = min(dynamic_target_vol / vol, self.max_leverage)

                if not position_open:
                    entry_price = row["close"]
                    entry_time = df.index[i]
                    entry_idx = i
                    is_long = direction == 1
                    max_price = entry_price if is_long else None
                    min_price = entry_price if not is_long else None
                    position_open = True
                    continue

                exit_flag = False
                exit_reason = None
                atr = row["atr"]
                price = row["close"]
                bars_held = i - entry_idx

                stop_loss = entry_price - 1.5 * atr if is_long else entry_price + 1.5 * atr

                # Early exit if trade moves quickly against us
                if bars_held <= 2 and ((entry_price - price > 0.75 * atr) if is_long else (price - entry_price > 0.75 * atr)):
                    exit_flag = True
                    exit_reason = "early_fail"

                # Hard stop-loss
                elif (is_long and price < stop_loss) or (not is_long and price > stop_loss):
                    exit_flag = True
                    exit_reason = "stop_loss"

                # Trailing stop activates after 2R gain
                elif (is_long and price > entry_price + 2 * atr):
                    max_price = max(max_price, price)
                    trail_stop = max_price - self.atr_trail * atr
                    if price < trail_stop:
                        exit_flag = True
                        exit_reason = "trailing_stop"
                elif (not is_long and price < entry_price - 2 * atr):
                    min_price = min(min_price, price)
                    trail_stop = min_price + self.atr_trail * atr
                    if price > trail_stop:
                        exit_flag = True
                        exit_reason = "trailing_stop"

                # Max hold enforcement
                elif bars_held >= self.max_hold:
                    exit_flag = True
                    exit_reason = "max_hold"

                if exit_flag:
                    exit_price = price
                    exit_time = df.index[i]
                    log_return = np.log(exit_price / entry_price) if is_long else np.log(entry_price / exit_price)
                    fee_cost = self.fee * abs(leverage)
                    net_ret = (log_return * leverage) - fee_cost
                    pnl = capital * net_ret
                    equity += pnl

                    holding_duration = (exit_time - entry_time).total_seconds() / 3600.0
                    trade_outcome = "win" if net_ret > 0 else "loss"
                    expected_move = leverage * self.target_vol
                    rr_ratio = expected_move / fee_cost if fee_cost > 0 else np.nan

                    all_trades.append({
                        "symbol": symbol,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "log_return": log_return,
                        "leverage": leverage,
                        "net_return": net_ret,
                        "pnl_log_return": net_ret,
                        "capital_used": capital,
                        "vol": vol,
                        "exit_reason": exit_reason,
                        "holding_duration": holding_duration,
                        "trade_outcome": trade_outcome,
                        "reward_risk": rr_ratio,
                        "position": "long" if is_long else "short"
                    })

                    position_open = False

        self.trades_df = pd.DataFrame(all_trades)
        self.log_results()

    def load_data(self, symbol, tf):
        path = f"data/processed/{symbol.lower()}_{tf.lower()}.csv"
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            return None
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["vol"] = df["log_return"].rolling(self.lookback).std()
        df["atr"] = df["close"].rolling(14).apply(lambda x: np.mean(np.abs(np.diff(x))))
        if self.use_trend_filter:
            df["ema"] = df["close"].ewm(span=20).mean()
        df.dropna(inplace=True)
        return df

StrategyImpl = StrategyImpl
