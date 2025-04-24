from core.models.base_strat import BaseStrategy
import pandas as pd
import numpy as np


class EWMAMeanReversion(BaseStrategy):
    windows = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    gamma = 0

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test, name="Mean Reversion")

        self.window = None

    def _make_signals(self, data: pd.DataFrame, train: bool = False):

        moving_averages = data['Close'].ewm(span=self.window, adjust=False).mean()
        std = data['Close'].ewm(span=self.window, adjust=False).std()

        upper_band = moving_averages + (self.gamma * std)
        lower_band = moving_averages - (self.gamma * std)

        signals = [0]
        for t in data.index[1:]:

            price = data.loc[t, "Close"]

            if price < lower_band[t]:
                signal = 1
            elif price > upper_band[t]:
                signal = -1
            else:
                signal = 0

            signals.append(signal)

        data["signal"] = signals
        return data

    def fit(self):
        """ Fit Moving Average Model to training data."""

        best_window = None
        best_pnl = -np.inf

        for window in self.windows:

            self.window = window
            V, V_cap, V_total, theta = self.run(self.train.copy(), train=True)  # run mean reversion on training data

            pnl = (V_total[-1] - V_total[0]) - 1

            if pnl > best_pnl:
                best_window = window
                best_pnl = pnl

        self.window = best_window
