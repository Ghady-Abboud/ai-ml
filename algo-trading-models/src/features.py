import os
import pandas as pd
import talib as ta

class FeatureEngineer:

  def __init__(self):
    self.data = None
    self.features = []

  def load_data(self):
    """
    Load data from CSV files into a single DF

    Returns:
        pd.DataFrame: Combined DataFrame containing data from all CSV files.
    """
    dir_path = "data/raw"
    if not os.path.exists(dir_path):
      raise FileExistsError(f"Directory {dir_path} does not exist.")
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    dfs = []
    for file in files:
      print(f"Loading file: {file}")
      df = pd.read_csv(os.path.join(dir_path , file), skiprows=3, header=None)
      df.columns = ["Date", "Close", "High", "Low", "Open", "Volume", "Ticker"]
      df["Date"] = pd.to_datetime(df["Date"])
      dfs.append(df)
    self.data = pd.concat(dfs, ignore_index=True)
    return self.data
  
  def extract_features(self, df):
    self.compute_momentum(df)
    self.compute_natr(df)
    self.compute_ema(df)
    self.compute_rsi(df)
    self.compute_bbands(df)
    self.compute_macd(df)
    self.compute_obv(df)
    self.compute_returns(df)
    self.normalize_columns(df, self.features)
    return df

  def normalize_columns(self, df, cols):
    for column_name in cols:
      df[column_name] = df.groupby("Ticker")[column_name].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df

  def compute_momentum(self, df):
    df["Momentum_1Y"] = ta.MOM(df["Close"], timeperiod=60)
    self.features.append("Momentum_1Y")
    return df
  
  def compute_rsi(self, df):
    d = 14
    df["RSI_14"] = ta.RSI(df["Close"], timeperiod=d)
    df["RSI_CHANGE_1D"] = df.groupby("Ticker")["RSI_14"].pct_change(1)
    df["RSI_CHANGE_5D"] = df.groupby("Ticker")["RSI_14"].pct_change(5)
    self.features.extend(["RSI_14", "RSI_CHANGE_1D", "RSI_CHANGE_5D"])
    return df

  def compute_natr(self, df):
    d = 14
    df["NATR_14"] = ta.NATR(df["High"], df["Low"], df["Close"], timeperiod=d)
    self.features.append("NATR_14")
    return df
  
  def compute_ema(self, df):
    st = 15
    mt = 50
    lt = 200
    df["EMA_10"] = ta.EMA(df["Close"], timeperiod=st)
    df["EMA_50"] = ta.EMA(df["Close"], timeperiod=mt)
    df["EMA_200"] = ta.EMA(df["Close"], timeperiod=lt)
    self.features.extend(["EMA_10", "EMA_50", "EMA_200"])
    return df

  def compute_bbands(self, df):
    df['BB_LOW'] = ta.BBANDS(df['Close'], timeperiod=20)[0]
    df['BB_MID'] = ta.BBANDS(df['Close'], timeperiod=20)[1]
    df['BB_UP'] = ta.BBANDS(df['Close'], timeperiod=20)[2]
    self.features.extend(["BB_LOW", "BB_MID", "BB_UP"])
    return df

  def compute_macd(self, df):
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.MACD(df['Close'])
    self.features.extend(["MACD", "MACD_Signal", "MACD_Hist"])
    return df

  def compute_obv(self, df):
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])
    self.features.append("OBV")
    return df
  
  def compute_returns(self, df):
    df["RETURN_1D"] = df.groupby("Ticker")["Close"].pct_change(1)
    df["RETURN_5D"] = df.groupby("Ticker")["Close"].pct_change(5)
    df["RETURN_60D"] = df.groupby("Ticker")["Close"].pct_change(60)
    self.features.extend(["RETURN_1D", "RETURN_5D", "RETURN_60D"])
    return df