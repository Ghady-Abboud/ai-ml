import pandas as pd

def load_data(file_path):
  from src.data.preprocess import preprocess_acled_data
  
  data = preprocess_acled_data(pd.read_csv(file_path))
  return data