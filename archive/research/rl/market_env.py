import numpy as np

class MarketEnv:
    """
    Minimal offline env: feed OHLCV+features, step through, emit reward by PnL delta.
    Train policies offline; evaluate live only after strong backtests.
    """
    def __init__(self, bars, fee=0.0004):
        self.bars = bars.reset_index(drop=True)
        self.fee = fee
        self.t = 0
        self.position = 0  # -1, 0, +1
        self.entry = None
        self.equity = 1.0

    def reset(self):
        self.t = 0
        self.position = 0
        self.entry = None
        self.equity = 1.0
        return self._obs()

    def step(self, action):
        # action: -1 short, 0 flat, +1 long
        price = float(self.bars.loc[self.t, "close"])
        reward = 0.0
        if action != self.position:
            # pay fee for flip/enter
            self.equity *= (1 - self.fee)
            self.entry = price
        if self.position != 0 and action == 0:
            # exit position: realize pnl
            pnl = (price / self.entry - 1.0) * (1 if self.position == 1 else -1)
            self.equity *= (1 + pnl)
            reward = pnl
        self.position = action
        self.t += 1
        done = self.t >= len(self.bars) - 1
        return self._obs(), reward, done, {"equity": self.equity}

    def _obs(self):
        row = self.bars.loc[self.t]
        return np.array([row["close"], row.get("rsi", 50), row.get("z", 0)], dtype=float)
