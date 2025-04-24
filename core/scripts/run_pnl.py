import matplotlib.pyplot as plt
from core.analysis.preprocess import preprocess_data, train_test_split
from core.analysis.data_interrogation import plot_autocorrelation
from core.models.momentum import Momentum
from core.models.ewma import EWMAMeanReversion
from core.analysis.risk_adj import sharpe_ratio, calmar_ratio, max_drawdown
import pandas as pd



def plot_data(data):
    fig, ax = plt.subplots(3, 1, figsize=(12, 15))

    # SPTL timeseries
    ax[0].plot(data["Date"], data["Close"], color="black", label="SPTL")
    ax[0].set_title("SPTL Timeseries")
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("$ Value")
    ax[0].legend()

    # excess return
    ax[1].plot(data["Date"], data["Excess Return"], label='SPTL Excess Returns', color="red", linewidth=1)
    ax[1].set_title('SPTL Excess Return')
    ax[1].grid(True, linestyle="-", linewidth=0.7, alpha=1)
    ax[1].set_xlabel("Date")
    ax[1].legend()

    # risk-free rate
    ax[2].plot(data["Date"], data["Daily Rate"], label="EFFR Daily Rate", color="black", linewidth=1)
    ax[2].set_title('EFFR Daily Rate')
    ax[2].grid(True, linestyle="-", linewidth=0.7, alpha=1)
    ax[2].set_xlabel("Date")
    ax[2].legend()


def performance_indicators(strategy, train, test):

    strategy.fit()

    # train metrics
    _, _, V_total, _ = strategy.run(train, train=True)
    _sr_train, _cr_train, _mdd_train = sharpe_ratio(V_total), calmar_ratio(V_total), max_drawdown(V_total)

    # test metrics
    _, _, V_total, _ = strategy.run(test, train=False)
    _sr_test, _cr_test, _mdd_test = sharpe_ratio(V_total), calmar_ratio(V_total), max_drawdown(V_total)

    performance_table = pd.DataFrame({
        "Sharpe Ratio": [_sr_train, _sr_test],
        "Calmar Ratio": [_cr_train, _cr_test],
        "MDD": [_mdd_train, _mdd_test],
    })
    performance_table.index = ["Train", "Test"]
    return performance_table


def momentum():

    # get data
    data = preprocess_data()
    plot_data(data)
    train, test = train_test_split(data)

    # explore auto-correlation
    plot_autocorrelation(train)

    # fit momentum hyperparameters
    _momentum = Momentum(train=train, test=test)
    _ = _momentum.fit()
    print(f"Momentum: short_window={_momentum.short_window}, long_window={_momentum.long_window}")

    # plot position and pnl
    _momentum.plot_position()
    _momentum.plot_pnl()

    # plot performance indicators
    risk_adjusted_metrics = performance_indicators(_momentum, train, test)
    print(risk_adjusted_metrics)


def ewma():

    # get data
    data = preprocess_data()
    plot_data(data)
    train, test = train_test_split(data)

    # explore auto-correlation
    plot_autocorrelation(train)

    # fit momentum hyperparameters
    _ewma_mr = EWMAMeanReversion(train=train, test=test)
    _ = _ewma_mr.fit()
    print(f"Momentum: window_size={_ewma_mr.window}")

    # plot position and pnl
    _ewma_mr.plot_position()
    _ewma_mr.plot_pnl()

    # plot performance indicators
    risk_adjusted_metrics = performance_indicators(_ewma_mr, train, test)
    print(risk_adjusted_metrics)


if __name__ == "__main__":
    momentum()
    ewma()


