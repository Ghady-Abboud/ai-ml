import numpy as np
import pandas as pd

def implementNaive(df : pd.DataFrame):
    """
      A simple naive strategy that uses a fixed set of rules to generate buy/sell signals based on technical indicators. Results will be benchmarked against a more sophisticated ML-based strategy to evaluate performance changes.
    
      Args:
          df (pd.DataFrame): DataFrame containing stock data with technical indicators.
      Returns:
          pd.DataFrame: DataFrame containing only stocks with buy/sell signals
    """
    df["RSI_CLASS"] = df["RSI_14"].apply(lambda x: -1 if x >= 0.7 else (1 if x <= 0.3 else 0))
    df["RSI_MOM_CLASS"] = df.apply(lambda row : \
                                   1 if (row["RSI_CHANGE_1D"] > 0 and row["RSI_CHANGE_5D"] > 0) \
                                   else (-1 if (row["RSI_CHANGE_1D"] < 0 and row["RSI_CHANGE_5D"] < 0) \
                                   else 0), \
                                   axis = 1)
    df["MACD_CLASS"] = df["MACD"].apply(lambda x: -1 if x <= 0.5 else 1)
    df["EMA_CLASS"] = df.apply(lambda row: \
                        1 if (row["EMA_10"] > row["EMA_50"] and row["EMA_50"] > row["EMA_200"]) \
                        else (-1 if (row["EMA_10"] < row["EMA_50"] and row["EMA_50"] < row["EMA_200"]) \
                        else 0), \
                        axis = 1)
    df["RETURNS_CLASS"] = df.apply(lambda row: \
                                   1 if (row["RETURN_1D"] > 0 and row["RETURN_5D"] > 0 and row["RETURN_60D"] > 0) \
                                   else (-1 if (row["RETURN_1D"] < 0 and row["RETURN_5D"] < 0 and row["RETURN_60D"] < 0) \
                                   else 0), \
                                   axis = 1)
    df["BUY_SCORE"] = df["RSI_CLASS"] + df["RSI_MOM_CLASS"] + df["MACD_CLASS"] + df["EMA_CLASS"] + df["RETURNS_CLASS"]
    df["SIGNAL"] = df["BUY_SCORE"].apply(lambda x: 1 if x >= 1 else (-1 if x <= -1 else 0))
    return df

def implementRandom(df : pd.DataFrame):
    np.random.seed(42)
    df["RANDOM_SIGNAL"] = np.random.choice([-1, 0, 1], size=len(df))
    return df