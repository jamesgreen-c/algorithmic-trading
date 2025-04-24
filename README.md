# Leveraged Treasury Bond Trading Strategies

This repository contains modular implementations of algorithmic trading strategies on long-term U.S. Treasury bonds (SPTL ETF), leveraging both **EWMA-based mean reversion** and **momentum** signals. The effective federal funds rate (EFFR) is used as a proxy for the risk-free rate to compute excess returns.

The strategies are evaluated on both **train** and **test** splits and are constructed with a **fixed leverage factor**, tracking not only portfolio value but also the impact of a simulated money market account.

---

## 📈 Strategy Overview

- **Assets**: SPTL ETF (long-term U.S. Treasury bonds)
- **Risk-Free Rate**: Daily-adjusted effective federal funds rate
- **Returns**: Excess returns over daily risk-free rate
- **Capital Allocation**:
  - Leveraged position on the ETF
  - Cash balance earning interest (money market)
- **Performance Metrics**:
  - Sharpe Ratio
  - Calmar Ratio
  - Max Drawdown
- **Diagnostics**:
  - Variance Ratio plots
  - Autocorrelation of returns
  - PnL decomposition

---

## 🧠 Strategy Design

- `BaseStrategy`: Abstract base class defining the portfolio mechanics.
- Strategies (e.g., EWMA, momentum) inherit from `BaseStrategy` and override the signal generation logic.

Each strategy:
- Computes a position signal
- Allocates leveraged capital based on signal
- Updates portfolio value using excess returns
- Tracks cumulative value, trading PnL, and capital account

---

## 🗂️ File Structure

```plaintext
core
|
├── data/                      # Raw data files (SPTL prices, EFFR)
│   ├── sptl.csv
│   └── effr.xlsx
│
├── analysis/                  # Core logic and strategy implementations
│   ├── preprocess.py          # Data loading, cleaning, and merging
│   ├── risk_adj.py            # Sharpe, Calmar, drawdown calculations
│   └── data_interrogation.py  # Variance ratio, autocorrelation plots
│
├── models/                    # Interactive analysis and strategy experiments
│   ├── base_strat.py          # Abstract BaseStrategy class
│   ├── ewma.py                # EWMA-based mean reversion strategy
│   └── momentum.py            # Momentum-based strategy
│
├── scripts/                   # Run strategies
│   └── run_pnl.py
│
├── README.md                  
└── requirements.txt           # Python dependencies
```


---

## 📊 Data

- **SPTL**: Long-term U.S. Treasury Bond ETF (SPDR Portfolio Long Term Treasury ETF)
- **EFFR**: Effective Federal Funds Rate (used as the daily risk-free rate)

> ⚠️ You must place `sptl.csv` and `effr.xlsx` in the appropriate directory. The `find_file_directory()` function scans for files starting from the project root.

---

## 🔍 Strategy Mechanics

Each strategy is trained and tested separately on a rolling split of the dataset. Positions are adjusted daily and constrained by:

- **Leverage**: Default 10x
- **Capital Tracking**:
  - `V`: Value from trading
  - `V_cap`: Value from interest on unutilized capital
  - `V_total`: Total leveraged portfolio value

The signal generation logic (momentum, EWMA mean reversion, etc.) must be implemented in derived classes of `BaseStrategy`.

--- 

## 📎 Notes

- Daily returns are excess of the daily risk-free rate, using 252 trading days per year.
- Leverage is capped at a fixed multiple of capital (`leverage=10` by default).
- Missing EFFR values are forward-filled to ensure daily alignment.

---
