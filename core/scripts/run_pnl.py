import matplotlib.pyplot as plt
from core.analysis.preprocess import preprocess_data, train_test_split
from core.analysis.data_interrogation import plot_autocorrelation
from core.models.momentum import Momentum
from core.models.ewma import EWMAMeanReversion


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


def momentum():

    # get data
    data = preprocess_data()
    plot_data(data)
    train, test = train_test_split(data)

    # explore auto-correlation
    plot_autocorrelation(train)

    # fit momentum hyperparameters
    momentum = Momentum(train=train, test=test)
    _ = momentum.fit()
    print(f"Momentum: short_window={momentum.short_window}, long_window={momentum.long_window}")

    # plot position and pnl
    momentum.plot_position()
    momentum.plot_pnl()


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


if __name__ == "__main__":
    momentum()
    ewma()


