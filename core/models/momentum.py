from core.models.base_strat import BaseStrategy
import pandas as pd
import numpy as np

class Momentum(BaseStrategy):
    short_window: int
    long_window: int

    short_window_list: list[int] = [3, 5, 10, 15, 20, 25, 30]
    long_window_list: list[int] = [10, 15, 20, 25, 30, 40, 50, 60, 70]

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test, name="Moving Average Crossover")

    def _make_signals(self, dataframe: pd.DataFrame, train: bool = False):

        dataframe["short_ma"] = dataframe["Close"].rolling(window=self.short_window, min_periods=1,
                                                           closed="neither").mean()
        dataframe["long_ma"] = dataframe["Close"].rolling(window=self.long_window, min_periods=1,
                                                          closed="neither").mean()
        dataframe["signal"] = np.where(dataframe["short_ma"] > dataframe["long_ma"], 1, -1)

        return dataframe

    def fit(self):

        x, y = len(self.short_window_list), len(self.long_window_list)
        mat = np.zeros((x, y))

        best_short_window = None
        best_long_window = None
        best_pnl = -np.inf

        for i, short_window in enumerate(self.short_window_list):
            for j, long_window in enumerate(self.long_window_list):

                if long_window <= short_window:
                    mat[i, j] = -np.inf
                    continue

                self.short_window = short_window
                self.long_window = long_window
                V, V_cap, V_total, theta = self.run(self.train.copy(), train=True)

                pnl = (V_total[-1] - V_total[0]) - 1
                mat[i, j] = pnl * 100

                if pnl > best_pnl:
                    best_short_window = short_window
                    best_long_window = long_window
                    best_pnl = pnl

        self.short_window = best_short_window
        self.long_window = best_long_window
        return mat

