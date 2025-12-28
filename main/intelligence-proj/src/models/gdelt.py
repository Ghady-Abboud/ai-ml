def load_data(file_path):
  from src.data.preprocess import preprocess_gdelt_data
  """
  Load data from a given file path.

  Args:
      file_path (str): The path to the data file.

  Returns:
      pd.DataFrame: Loaded data as a pandas DataFrame.
  """
  import pandas as pd

  data = preprocess_gdelt_data(pd.read_csv(file_path))
  return data

def create_event_text(row):
  """Convert GDELT event row to descriptive text for BERT."""
  text = (f"{row['Actor1Name']} (event code {row['EventCode']}) "
          f"towards {row['Actor2Name']}. "
          f"Goldstein scale: {row['GoldsteinScale']}, "
          f"Tone: {row['AvgTone']}")
  return text

def classify_text_intensity(df, text_column="event_text"):
  """
      Predict event intensity (0-4) using BERT zero-shot classification.

      0: Stable/Cooperative
      1: Verbal Tension
      2: Diplomatic Crisis
      3: Military Posturing
      4: Armed Conflict
  """
  import torch
  from transformers import pipeline

  classifier = pipeline(
      "zero-shot-classification",
      model="facebook/bart-large-mnli",
      device=0 if torch.cuda.is_available() else -1
  )

  labels = [
      "Stable/Cooperative",
      "Verbal Tension",
      "Diplomatic Crisis",
      "Military Posturing",
      "Armed Conflict"
  ]

  intensity_dict = {
      "Stable/Cooperative": 0,
      "Verbal Tension": 1,
      "Diplomatic Crisis": 2,
      "Military Posturing": 3,
      "Armed Conflict": 4
  }

  predictions = []
  for event in df[text_column]:
    result = classifier(event, labels, multi_label=False)
    top_label = result["labels"][0]
    predictions.append(intensity_dict[top_label])

  df = df.copy()
  df["predicted_label"] = [p for p in predictions]
  return df