# ut_bot.py
import numpy as np
import pandas as pd

class UTBotStrategy:
    def __init__(self, sensitivity=1.0, atr_period=10):
        self.sensitivity = sensitivity
        self.atr_period = atr_period
        self.trailing_stop = None
        self.position = 0  # 1 for long, -1 for short, 0 for none

    def compute_atr(self, df):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        return atr

    def evaluate(self, df: pd.DataFrame):
        """
        Input: df with columns ['open', 'high', 'low', 'close']
        Output: signals per row: 'BUY', 'SELL', or 'HOLD'
        """
        signals = []
        atr = self.compute_atr(df)
        src = df["close"]

        trailing_stop = np.zeros(len(df))
        position = np.zeros(len(df))

        for i in range(len(df)):
            prev_stop = trailing_stop[i-1] if i > 0 else src.iloc[0]
            n_loss = self.sensitivity * atr.iloc[i]

            if i == 0 or pd.isna(atr.iloc[i]):
                trailing_stop[i] = prev_stop
                position[i] = 0
                signals.append("HOLD")
                continue

            if src.iloc[i] > prev_stop and src.iloc[i-1] > prev_stop:
                trailing_stop[i] = max(prev_stop, src.iloc[i] - n_loss)
            elif src.iloc[i] < prev_stop and src.iloc[i-1] < prev_stop:
                trailing_stop[i] = min(prev_stop, src.iloc[i] + n_loss)
            elif src.iloc[i] > prev_stop:
                trailing_stop[i] = src.iloc[i] - n_loss
            else:
                trailing_stop[i] = src.iloc[i] + n_loss

            # Position logic
            if src.iloc[i-1] < prev_stop and src.iloc[i] > prev_stop:
                position[i] = 1
                signals.append("BUY")
            elif src.iloc[i-1] > prev_stop and src.iloc[i] < prev_stop:
                position[i] = -1
                signals.append("SELL")
            else:
                position[i] = position[i-1]
                signals.append("HOLD")

        self.trailing_stop = trailing_stop
        self.position = position
        return signals
