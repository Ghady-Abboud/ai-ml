US STOCKS MACHINE LEARNING TRADING SYSTEM

STAGE 1 – DATA ACQUISITION
================================
Goal: get clean historical OHLCV data for all S&P 500 symbols.

Program should:
1. Pull list of current S&P 500 tickers (scrape Wikipedia or use a static CSV).
2. For each symbol, download daily data (5–10 years) from Yahoo Finance via yfinance.
3. Store raw CSVs locally (data/raw/).
Output: pandas.DataFrame with [Open, High, Low, Close, Volume].

STAGE 2 – FEATURE GENERATION
================================
Goal: numeric predictors for each date.

Program should:
1. Compute technical indicators (returns, rolling mean/std, RSI, MACD, etc.).
2. Add lag features (1 d, 5 d, 20 d returns).
3. Normalize or z-score features per symbol.

STAGE 3 – LABEL CREATION
================================
Goal: create supervised-learning target.

Program should:
1. Define future return r_{t+1} = (P_{t+1}/P_t - 1).
2. Label as:
   - classification target → 1 if r_{t+1} > 0 else 0
3. Align features so labels are forward-shifted.

STAGE 4 – MODEL TRAINING
================================
Goal: per-symbol or pooled model that predicts next-day direction.

Program should:
1. Split data chronologically (train / validate / test).
2. Train baseline models:
   - Logistic Regression
   - Gradient Boosting
3. Evaluate metrics 

STAGE 5 – BACKTESTING
================================
Goal: simulate trading using predictions.

Program should:
1. Generate daily signals (+1 = buy, 0 = flat).
2. Plot equity curve.
3. Log metrics

PROJECT GOAL
================================
Can technical indicators derived from daily SP500 constituent data predict next day price direction better than random or naive baselines?.

RECORDINGS
================================
   Random backtesting from 2019-01-01 to 2025-01-01 with initial capital of $1000 yields -0.50% ($995.03)
   Naive backtesting from 2019-01-01 to 2025-01-01 with initial capital of $1000 yields +46.3% ($1463.08)
   Logistic Regression yields a 52.4 % accuracy
   XGBoost yields a 53% accuracy

   Logistic Regression Backtest:
   - Initial Capital: $1000
   - Training occurred from 2019-01-01 to 2023-01-01
      Final Portfolio Value: $2911.10
      Total Return: 191.11%
   
   XGBoost Backtest:
   - Initial Capital: $1000
   - Training occurred from 2019-01-01 to 2023-01-01
      Final Portfolio Value: $5678.89
      Total Return: 467.89%


   Train period: 2019-03-29 to 2022-12-30
   Test period: 2023-01-03 to 2024-12-30
   Train samples: 462421, Test samples: 248820

   Logistic Regression Backtest:
   - Initial Capital: $1000
      Final Portfolio Value: $1262.51
      Total Return: 26.25%
   
   XGBoost Backtest:
   - Initial Capital: $1000
      Final Portfolio Value: $1639.21
      Total Return: 63.92%
   
   **BACKTESTING DOES NOT ACCOUNT FOR SLIIPPAGE OR TRANSACTION COST, % SHOULD BE LOWER THAN WHAT IS CURRENTLY IS**
