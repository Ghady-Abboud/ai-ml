import yaml
import features as ft
import data as dt
import baseline as baseline 
import backtest as bt
import model as ml

if __name__ == "__main__":
  config = yaml.safe_load(open("config.yaml"))
  tickers = dt.grab_tickers()
  if tickers is None:
    print("Failed to grab tickers.")
    exit(1)
  print(f"Successfully grabbed {len(tickers)} tickers.\n")
  if not dt.has_downloaded_data():
    print("Downloading ticker data...")
    dt.download_ticker_data(tickers, config['Date']['start'], config['Date']['end'])
  fe = ft.FeatureEngineer()
  fe.load_data()
  df = fe.extract_features(fe.data)

  # naive_df = baseline.implementNaive(df)
  # random_df = baseline.implementRandom(df)

  # bt.backtest_strategy(naive_df)
  # bt.backtest_strategy(random_df, signal_col="RANDOM_SIGNAL")
  # print("Backtesting completed.")

  model = ml.Model(df)
  model.run()