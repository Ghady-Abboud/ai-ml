import os
import pandas as pd

def preprocess_data(input_dir, output_dir):
    """
    Clean up and preprocess data files from input_dir and save them to output_dir.

    Args:
        input_dir (str): Path to the input directory
        output_dir (str): Path to the output directory
    """

    data = pd.read_csv(os.path.join(input_dir, 'irn_isr_gdelt.csv'))
    data_cleaned = data.drop(columns=['SOURCEURL'])

    data_cleaned.to_csv(os.path.join(output_dir, 'irn_isr_gdelt_cleaned.csv'), index=False)