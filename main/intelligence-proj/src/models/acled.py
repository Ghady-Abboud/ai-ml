import pandas as pd

def load_data(file_path):
  from src.data.preprocess import preprocess_acled_data
  
  data = preprocess_acled_data(pd.read_csv(file_path))
  data['event_date'] = pd.to_datetime(data['event_date'])
  data = data.sort_values('event_date').reset_index(drop=True)
  return data

def did_conflict_occur(data):
  data_cp = data.copy()

  data_cp['event_severity'] = pd.to_numeric(data_cp['event_severity'], errors='coerce')

  data_cp['conflict_occurred_7d'] = data_cp['event_severity'].rolling(window=7).max().shift(-6) >= 3
  data_cp['conflict_occurred_14d'] = data_cp['event_severity'].rolling(window=14).max().shift(-13) >= 3
  data_cp['conflict_occurred_30d'] = data_cp['event_severity'].rolling(window=30).max().shift(-29) >= 3

  return data_cp

def compute_severity_metrics(data):
  data_cp = data.copy()

  data_cp['average_severity_7d'] = data_cp['event_severity'].rolling(window=7).mean().shift(-6)
  data_cp['max_severity_7d'] = data_cp['event_severity'].rolling(window=7).max().shift(-6)
  data_cp['average_severity_14d'] = data_cp['event_severity'].rolling(window=14).mean().shift(-13)
  data_cp['max_severity_14d'] = data_cp['event_severity'].rolling(window=14).max().shift(-13)
  data_cp['average_severity_30d'] = data_cp['event_severity'].rolling(window=30).mean().shift(-29)
  data_cp['max_severity_30d'] = data_cp['event_severity'].rolling(window=30).max().shift(-29)

  return data_cp