import matplotlib.pyplot as plt

def backtest_strategy(df, signal_col = "SIGNAL"):
  df["RETURN"] = df.groupby("Ticker")["Close"].pct_change()
  df["STRATEGY_RETURN"] = df["RETURN"] * df.groupby("Ticker")[signal_col].shift(1)

  daily_returns = df.groupby("Date")["STRATEGY_RETURN"].mean().fillna(0)

  initial_capital = 1000
  portfolio = initial_capital * (1 + daily_returns).cumprod()
  plt.figure(figsize=(10,5))
  plt.plot(portfolio.index, portfolio.values, label="Portfolio Value")
  plt.title("Portfolio Value Over Time")
  plt.xlabel("Date")
  plt.ylabel("Portfolio Value ($)")
  plt.legend()
  plt.grid(True)
  plt.show()

  print(f"Final Portfolio Value: ${portfolio.iloc[-1]:.2f}")
  print(f"Total Return: {(portfolio.iloc[-1] - initial_capital) / initial_capital * 100:.2f}%")