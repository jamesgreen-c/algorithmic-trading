import pandas as pd
import os


def find_file_directory(file_name):
    start = os.getcwd().split("algorithmic-trading")[0] + "algorithmic-trading"
    for root, dirs, files in os.walk(start):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


def train_test_split(data: pd.DataFrame, train_ratio: int = 6):

    pivot = data.shape[0] // 10 * train_ratio
    train, test = data[:pivot], data[pivot:]
    return train,test


def preprocess_data():

    # SPTL ETF
    sptl = pd.read_csv(find_file_directory("sptl.csv"))
    sptl = sptl[2:].rename(columns={"Price": "Date"}).reset_index(drop=True)

    # Effective Fed Funds Rate EFFR
    effr = pd.read_excel(find_file_directory("effr.xlsx"))

    # filter for EOD (SPTL) and annual rates (EFFR)
    sptl_eod = sptl[["Date", "Close"]]
    sptl_eod.loc[:, "Close"] = sptl_eod["Close"].astype(float)  # convert close from str to float
    effr_rate = effr[["Effective Date", "Rate (%)"]].rename(
        columns={"Effective Date": "Date", "Rate (%)": "Rate"}).sort_values(by="Date")

    # align dates
    effr_rate["Date"] = effr_rate["Date"].apply(lambda x: f"{x[-4:]}-{x[:2]}-{x[3:5]}")
    sptl_eod = sptl_eod.sort_values(by="Date").reset_index(drop=True)
    effr_rate = effr_rate.sort_values(by="Date").reset_index(drop=True)
    data = pd.merge(left=sptl_eod, right=effr_rate, how="left", on="Date")
    data = data.sort_values(by="Date").reset_index(drop=True)

    # forward fill missing rate value
    data["Rate"] = data["Rate"].ffill()

    # fix Date datatype
    data["Date"] = data["Date"].apply(lambda x: pd.to_datetime(x).date())

    # adjust annual risk free rate
    data["Daily Rate"] = data["Rate"] / 252

    # calculate returns
    data["Return"] = (data["Close"] - data["Close"].shift(1)) / data["Close"].shift(1)
    data["Excess Return"] = data["Return"] - (data["Daily Rate"].shift(1) / 100)

    return data
