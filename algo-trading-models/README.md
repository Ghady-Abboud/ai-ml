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
   - regression target → r_{t+1}
   - classification target → 1 if r_{t+1} > 0 else 0
3. Align features so labels are forward-shifted.

STAGE 4 – MODEL TRAINING
================================
Goal: per-symbol or pooled model that predicts next-day direction.

Program should:
1. Split data chronologically (train / validate / test).
2. Train baseline models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
3. Evaluate metrics (accuracy, precision, Sharpe).
4. Save best model artifact (models/{symbol}.pkl).

STAGE 5 – BACKTESTING
================================
Goal: simulate trading using predictions.

Program should:
1. Generate daily signals (+1 = buy, 0 = flat).
2. Apply transaction costs (e.g. 0.1%).
3. Compute cumulative return, drawdown, Sharpe ratio.
4. Plot equity curve.
5. Log metrics to results/.

STAGE 6 – LIVE INFERENCE
================================
Goal: deploy daily prediction loop.

Program should:
1. At market open, pull latest price for each symbol (Alpaca API).
2. Compute features from most recent window.
3. Load trained model, predict next-day move.
4. Record to a signals.csv.

STAGE 7 – ALERTING / MONITORING
================================
Goal: deliver actionable info, not manual trades.

Program should:
1. Summarize top N buy/sell signals.
2. Send formatted message via Telegram bot.
3. Log timestamp + predictions.
4. Run automatically via cron job or cloud function once per day.

STAGE 8 – RESEARCH LOOP
================================
Goal: continuous improvement.

Every month:
- Re-train on new data.
- Add features or models.
- Compare performance vs baseline.
- Archive experiment configs.

RESEARCH GOAL
================================
Can technical indicators derived from daily SP500 constituent data predict short-term (1-5 day) price direction better than random or naive baselines?.

Recordings
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

Additional Indicators to consider: hma, zlhma
