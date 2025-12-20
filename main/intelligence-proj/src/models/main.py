def load_data(file_path):
    """
    Load data from a given file path.

    Args:
        file_path (str): The path to the data file.
    
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    import pandas as pd

    data = pd.read_csv(file_path)
    return data

def create_event_text(row):
    """Convert GDELT event row to descriptive text for BERT."""
    text = (f"{row['Actor1Name']} (event code {row['EventCode']}) "
            f"towards {row['Actor2Name']}. "
            f"Goldstein scale: {row['GoldsteinScale']}, "
            f"Tone: {row['AvgTone']}")
    return text

def predict_intensity(df, text_column='event_text'):
    """
        Predict event intensity (0-4) using BERT zero-shot classification.

        0: Stable/Cooperative
        1: Verbal Tension
        2: Diplomatic Crisis
        3: Military Posturing
        4: Armed Conflict
    """
    import pandas as pd
    import torch
    from transformers import pipeline

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device = 0 if torch.cuda.is_available() else -1
    )

    labels = [
        "Stable/Cooperative",
        "Verbal Tension",
        "Diplomatic Crisis",
        "Military Posturing",
        "Armed Conflict"
    ]

    intensities = []
    for text in df[text_column]:
        result = classifier(text, labels)
        pred_label = result['labels'][0]
