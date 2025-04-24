from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseStrategy(ABC):
    train: pd.DataFrame
    test: pd.DataFrame
    initial_capital: int = 100000
    leverage: int = 10

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, name: str):
        self.train = train.sort_values(by="Date", ascending=True).reset_index(drop=True)
        self.test = test.sort_values(by="Date", ascending=True).reset_index(drop=True)
        self.name = name

    def run(self, data: pd.DataFrame, train: bool = False):
        """
        Cannot exceed leverage limit
        """
        data = data.reset_index(drop=True)
        data = self._make_signals(data, train=train)

        n = data.shape[0]
        V = np.zeros(n)
        V_cap = np.zeros(n)
        V_total = np.zeros(n)
        theta = np.zeros(n)

        V[0] = self.initial_capital * self.leverage
        V_total[0] = self.initial_capital * self.leverage

        for t in range(1, n):
            theta_unbounded = V_total[t - 1] * data.loc[t - 1, "signal"]
            theta[t - 1] = np.clip(theta_unbounded, a_min=-(V_total[t - 1]),
                                   a_max=(V_total[t - 1]))

            delta_V = data.loc[t, "Excess Return"] * theta[t - 1]  # excess returns is shifted 1
            V[t] = V[t - 1] + delta_V

            M = abs(theta[t - 1]) / self.leverage
            delta_V_cap = (V_total[t - 1] - M) * (data.loc[t - 1, "Daily Rate"] / 100)
            V_cap[t] = V_cap[t - 1] + delta_V_cap

            V_total[t] = V_total[t - 1] + delta_V + delta_V_cap

        return V, V_cap, V_total, theta

    def plot_position(self):
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))

        # run on train data
        V, V_cap, V_total, theta = self.run(self.train.copy(), train=True)

        # train position
        ax[0].plot(self.train.Date, theta, label=r"$\theta$", linewidth=2)
        ax[0].plot(self.train.Date, V_total, label=r"$V_t^{total} \cdot L$", linewidth=1, color="red", linestyle="--")
        ax[0].plot(self.train.Date, - (V_total), label=r"$- V_t^{total} \cdot L$", linewidth=1, color="red",
                   linestyle="--")
        ax[0].set_title("Position (train)")
        ax[0].grid(True, linestyle="-", linewidth=0.7, alpha=1)
        ax[0].set_ylabel("$")
        ax[0].set_xlabel("Date")
        ax[0].legend()

        # run on test data
        V, V_cap, V_total, theta = self.run(self.test.copy(), train=False)

        # train position
        ax[1].plot(self.test.Date, theta, label=r"$\theta$", linewidth=2)
        ax[1].plot(self.test.Date, V_total, label=r"$V_t^{total} \cdot L$", linewidth=1, color="red", linestyle="--")
        ax[1].plot(self.test.Date, - (V_total), label=r"$- V_t^{total} \cdot L$", linewidth=1, color="red",
                   linestyle="--")
        ax[1].set_title("Position (test)")
        ax[1].grid(True, linestyle="-", linewidth=0.7, alpha=1)
        ax[1].set_ylabel("$")
        ax[1].set_xlabel("Date")
        ax[1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def plot_pnl(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # run on train data
        V, V_cap, V_total, theta = self.run(self.train.copy(), train=True)
        train_return = ((V_total[-1] / V_total[0]) - 1) * 100
        print("Train Return: ", train_return)
        print("Train V_cap dollar return: ", V_cap[-1] - V_cap[0])
        print("Train Trading dollar return: ", V[-1] - V[0])
        print("Train Trading + money market: ", (V[-1] - V[0]) + V_cap[-1] - V_cap[0])
        print("Train Total dollar return: ", V_total[-1] - V_total[0])
        print("\n")

        # pnl, money market and total - TRAIN
        ax[0, 0].plot(self.train.Date[1:], V_total[1:] - V_total[:-1], color="black", label=r"$\Delta V^{total}$",
                      linewidth=2)
        ax[0, 0].plot(self.train.Date[1:], V[1:] - V[:-1], label=r"$\Delta V$", linewidth=1)
        # ax[0, 0].plot(self.train.Date[1:], V_cap[1:] - V_cap[:-1], label=r"$\Delta V^{cap}$", linewidth=1)
        ax[0, 0].grid(True, linestyle="-", linewidth=0.7, alpha=1)
        ax[0, 0].set_title(f"{self.name} - Deltas (train)")
        ax[0, 0].set_ylabel("$")
        ax[0, 0].set_xlabel("Date")
        # ax[0, 0].legend()
        delta_v_cap_ax_train = ax[0, 0].twinx()
        delta_v_cap_ax_train.plot(self.train.Date[1:], V_cap[1:] - V_cap[:-1], label=r"$\Delta V^{cap}$", linewidth=1,
                                  color="orange")
        delta_v_cap_ax_train.set_ylabel(r"$\Delta V_{cap}$")

        # merge the legends from both y-axes
        handles_0_0, labels_0_0 = ax[0, 0].get_legend_handles_labels()
        handles_2_0_0, labels_2_0_0 = delta_v_cap_ax_train.get_legend_handles_labels()
        ax[0, 0].legend(handles_0_0 + handles_2_0_0, labels_0_0 + labels_2_0_0)

        ax[1, 0].plot(self.train.Date, V_total, color="black", label=r"$V^{total}$", linewidth=2)
        ax[1, 0].plot(self.train.Date, V, label=r"$V$", linewidth=1)
        ax[1, 0].grid(True, linestyle="-", linewidth=0.7, alpha=1)
        ax[1, 0].set_title(f"{self.name} - Cumulative Values (train)")
        ax[1, 0].set_ylabel("$")
        ax[1, 0].set_xlabel("Date")
        v_cap_ax_train = ax[1, 0].twinx()
        v_cap_ax_train.plot(self.train.Date, V_cap, label=r"$V^{cap}$", linewidth=1, color="orange")
        v_cap_ax_train.set_ylabel(r"$V_{cap}$")

        # merge the legends from both y-axes
        handles_1_0, labels_1_0 = ax[1, 0].get_legend_handles_labels()
        handles_2_1_0, labels_2_1_0 = v_cap_ax_train.get_legend_handles_labels()
        ax[1, 0].legend(handles_1_0 + handles_2_1_0, labels_1_0 + labels_2_1_0)

        # run on test data
        V, V_cap, V_total, theta = self.run(self.test.copy(), train=False)
        test_return = ((V_total[-1] / V_total[0]) - 1) * 100
        print("Test Return:", test_return)
        print("Test V_cap dollar return: ", V_cap[-1] - V_cap[0])
        print("Test Trading dollar return: ", V[-1] - V[0])
        print("Test Trading + money market: ", (V[-1] - V[0]) + V_cap[-1] - V_cap[0])
        print("Test Total dollar return: ", V_total[-1] - V_total[0])
        print("\n")

        # pnl, money market and total - TEST
        ax[0, 1].plot(self.test.Date[1:], V_total[1:] - V_total[:-1], color="black", label=r"$\Delta V^{total}$",
                      linewidth=2)
        ax[0, 1].plot(self.test.Date[1:], V[1:] - V[:-1], label=r"$\Delta V$", linewidth=1)
        # ax[0, 1].plot(self.test.Date[1:], V_cap[1:] - V_cap[:-1], label=r"$\Delta V^{cap}$", linewidth=1)
        ax[0, 1].grid(True, linestyle="-", linewidth=0.7, alpha=1)
        ax[0, 1].set_title(f"{self.name} - Deltas (test)")
        ax[0, 1].set_ylabel("$")
        ax[0, 1].set_xlabel("Date")
        # ax[0, 1].legend()
        delta_v_cap_ax_test = ax[0, 1].twinx()
        delta_v_cap_ax_test.plot(self.test.Date[1:], V_cap[1:] - V_cap[:-1], label=r"$\Delta V^{cap}$", linewidth=1,
                                 color="orange")
        delta_v_cap_ax_test.set_ylabel(r"$\Delta V_{cap}$")

        # merge the legends from both y-axes
        handles_0_1, labels_0_1 = ax[0, 1].get_legend_handles_labels()
        handles_2_0_1, labels_2_0_1 = delta_v_cap_ax_test.get_legend_handles_labels()
        ax[0, 1].legend(handles_0_1 + handles_2_0_1, labels_0_1 + labels_2_0_1)

        ax[1, 1].plot(self.test.Date, V_total, color="black", label=r"$V^{total}$", linewidth=2)
        ax[1, 1].plot(self.test.Date, V, label=r"$V$", linewidth=1)
        ax[1, 1].grid(True, linestyle="-", linewidth=0.7, alpha=1)
        ax[1, 1].set_title(f"{self.name} - Cumulative Values (test)")
        ax[1, 1].set_ylabel("$")
        ax[1, 1].set_xlabel("Date")
        v_cap_ax_test = ax[1, 1].twinx()
        v_cap_ax_test.plot(self.test.Date, V_cap, label=r"$V^{cap}$", linewidth=1, color="orange")
        v_cap_ax_test.set_ylabel(r"$V_{cap}$")

        # merge the legends from both y-axes
        handles_1_1, labels_1_1 = ax[1, 1].get_legend_handles_labels()
        handles_2_1_1, labels_2_1_1 = v_cap_ax_test.get_legend_handles_labels()

        ax[1, 1].legend(handles_1_1 + handles_2_1_1, labels_1_1 + labels_2_1_1)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    @abstractmethod
    def _make_signals(self, *args, **kwargs):
        raise NotImplementedError("Override this method")

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Override this method")


