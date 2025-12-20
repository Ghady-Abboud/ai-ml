import os
import pandas as pd

def preprocess_data(data):
    """
    Clean up and preprocess data files from input_dir and save them to output_dir.

    Args:
        input_dir (str): Path to the input directory
        output_dir (str): Path to the output directory
    """

    data_cleaned = data.drop(columns=['SOURCEURL', 'Actor1Code', 'Actor1CountryCode', 'Actor2Code', 'Actor2CountryCode', 'EventBaseCode', 'EventRootCode', 'QuadClass'])
    return data_cleaned