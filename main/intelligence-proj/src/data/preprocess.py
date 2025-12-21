import os
import pandas as pd

def preprocess_data(data):
    """
    Clean up and preprocess data files from input_dir and save them to output_dir.

    Args:
        input_dir (str): Path to the input directory
        output_dir (str): Path to the output directory
    """
    data = standardize_actor_names(data)
    return data


def standardize_actor_names(df):
    """
    Standardize actor names in Actor1Name and Actor2Name columns.
    Convert variants of Iran/Israel to their standard forms.

    Args:
        df (pd.DataFrame): DataFrame with Actor1Name and Actor2Name columns

    Returns:
        pd.DataFrame: DataFrame with standardized actor names
    """
    df = df.copy()

    iran_variants = [
        'IRAN', 'IRANIAN', 'TEHRAN', 'TEHERAN', 'PERSIA', 'ISFAHAN', 'ESFAHAN', 'SHIRAZ', 'MASHHAD', 'MESHED', 'TABRIZ', 'AHVAZ', 'RASHT', 'RESHT', 'URMIA', 'FARS NEWS'
    ]

    israel_variants = [
        'ISRAEL', 'ISRAELI', 'ISRAIL', 'ISREAL', 'ISREALI', 'YISRAEL', 'YISROEL', 'HEBREW STATE', 'JERUSALEM', 'TEL AVIV', 'HAIFA', 'JAFFA', 'AKKA', 'AKKO', 'CAESAREA', 'GALILEE', 'NEGEV', 'NEGEV DESERT', 'SODOM', 'GOMORRAH', 'JERUSALEM POST'
    ]

    df['Actor1Name'] = df['Actor1Name'].apply(
        lambda x: 'IRAN' if x in iran_variants else ('ISRAEL' if x in israel_variants else x)
    )

    df['Actor2Name'] = df['Actor2Name'].apply(
        lambda x: 'IRAN' if x in iran_variants else ('ISRAEL' if x in israel_variants else x)
    )

    return df