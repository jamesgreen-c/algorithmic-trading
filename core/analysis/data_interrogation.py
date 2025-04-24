import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def variance_ratio(price_series):

    lags = [2, 5, 10, 20, 30, 45, 60, 70, 80, 90]
    returns = price_series.pct_change().dropna()
    variance_1 = returns.var()  # 1 period variance

    vr_results = {}
    for k in lags:
        returns_k = price_series.pct_change(k).dropna()
        variance_k = returns_k.var()

        # k-th variace ratio
        VR_k = variance_k / (k * variance_1)
        vr_results[k] = VR_k

    return pd.DataFrame.from_dict(vr_results, orient='index', columns=['Variance Ratio'])


def plot_VR(train: pd.DataFrame):

    vr_df = variance_ratio(train["Close"])

    plt.figure(figsize=(12, 5))
    plt.plot(vr_df.index, vr_df['Variance Ratio'], marker='o', linestyle='-', color='b', label='Variance Ratio')
    plt.axhline(y=1, color='r', linestyle='--', label='Random Walk (VR=1)')

    plt.xlabel("Lag (k)")
    plt.ylabel("Variance Ratio")
    plt.title("Variance Ratio (Train)")
    plt.legend()
    plt.grid(True, linestyle="-", linewidth=0.7, alpha=1)
    plt.show()


def plot_autocorrelation(train: pd.DataFrame):

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plot_acf(train["Close"], ax=ax, title="SPTL Close Autocorrelation (Train)",
             lags=20)  # You can adjust lags as needed
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.show()