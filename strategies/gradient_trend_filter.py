# gradient_trend_filter.py
import numpy as np
import pandas as pd

class GradientTrendFilter:
    def __init__(self, length=25):
        self.length = length
        self.alpha = 2 / (length + 1)
        self.base = None
        self.diff = None

    def noise_filter(self, src):
        nf_1 = np.zeros_like(src)
        nf_2 = np.zeros_like(src)
        nf_3 = np.zeros_like(src)

        for i in range(1, len(src)):
            nf_1[i] = self.alpha * src[i] + (1 - self.alpha) * nf_1[i - 1]
            nf_2[i] = self.alpha * nf_1[i] + (1 - self.alpha) * nf_2[i - 1]
            nf_3[i] = self.alpha * nf_2[i] + (1 - self.alpha) * nf_3[i - 1]

        return nf_3

    def bands(self, base, range_data):
        val = self.noise_filter(range_data)
        upper3 = base + val * 0.618 * 2.5
        upper2 = base + val * 0.382 * 2
        upper1 = base + val * 0.236 * 1.5
        lower1 = base - val * 0.236 * 1.5
        lower2 = base - val * 0.382 * 2
        lower3 = base - val * 0.618 * 2.5
        return upper3, upper2, upper1, lower1, lower2, lower3

    def evaluate(self, df: pd.DataFrame):
        """
        Input: DataFrame with columns ['high', 'low', 'close']
        Output: signal list with values: 'UP', 'DOWN', or 'NONE'
        """
        src = df["close"].values
        range_data = (df["high"] - df["low"]).values

        base = self.noise_filter(src)
        diff = np.zeros_like(base)
        signals = []

        for i in range(len(base)):
            if i < 2:
                signals.append("NONE")
                continue
            diff[i] = base[i] - base[i - 2]

            if diff[i - 1] < 0 and diff[i] > 0:
                signals.append("UP")
            elif diff[i - 1] > 0 and diff[i] < 0:
                signals.append("DOWN")
            else:
                signals.append("NONE")

        self.base = base
        self.diff = diff
        return signals
