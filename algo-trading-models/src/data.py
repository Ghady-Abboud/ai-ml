import requests
from io import StringIO
import pandas as pd
import yfinance as yf
import os

def grab_tickers():
  """
  Retrieve the list of S&P 500 ticker symbols from Wikipedia.

  Makes an HTTP request to the S&P 500 Wikipedia page and parses the first
  table to extract the 'Symbol' column.

  Returns:
      list[str] | None: A list of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
          on success; returns None if an error occurs while fetching or
          parsing the page.
  """
  url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:144.0) Gecko/20100101 Firefox/144.0'}
  try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    symbols = tables[0]['Symbol'].tolist()
    return symbols
  except requests.exceptions.RequestException as e:
    print(f"Error occurred during HTTP request: {e}")
    return None
  except Exception as e:
    print(f"Unexpected error occurred: {e}")
    return None

def has_downloaded_data():
  """
  Check if the raw data directory exists and contains files.

  Returns:
      bool: True if data has been downloaded, False otherwise.
  """
  dir_path = "data/raw"
  if not os.path.exists(dir_path):
    return False
  files = os.listdir(dir_path)
  return len(files) > 0

def download_ticker_data(tickers : list, start_date : str, end_date : str):
  """
  Download and save historical OHLCV data for a list of tickers.

  For each ticker in `tickers`, downloads historical price data from Yahoo
  Finance for the specified date range and writes a CSV file named
  "<TICKER>_historical_data.csv" into the "data/raw" directory (created if
  it does not exist).

  Args:
      tickers (list[str]): Iterable of ticker symbols to download.
      start_date (str): Start date in 'YYYY-MM-DD' format (inclusive).
      end_date (str): End date in 'YYYY-MM-DD' format (exclusive).

  Returns:
      None
  """
  try:
    dir_path = "data/raw"
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    for ticker in tickers:
      df = yf.download(ticker, start=start_date, end=end_date)
      if df is None or df.empty:
        continue
      df['Ticker'] = ticker
      df.to_csv(f"{dir_path}/{ticker}_historical_data.csv")

  except Exception as e:
    print(f"Error occurred while downloading ticker data: {e}")