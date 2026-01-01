import pandas as pd

def preprocess_gdelt_data(data):
    """
    Clean up and preprocess GDELT csv files 

    Args:
        data (pd.DataFrame): Pandas DataFrame
    """

    data_cleaned = data.drop(columns=['SOURCEURL', 'Actor1Code', 'Actor1CountryCode', 'Actor2Code', 'Actor2CountryCode', 'EventBaseCode', 'EventRootCode', 'QuadClass'])
    return data_cleaned

def preprocess_acled_data(data):
    """
    Clean up and preprocess ACLED csv files
    Filter for Iran-Israel confrontations including proxies

    Args:
        data (pd.DataFrame): Pandas DataFrame

    Returns:
        pd.DataFrame: Filtered dataframe with Iran-Israel related events
    """
    import re

    df = data.copy()

    iran_proxies = [
        'hezbollah', 'hizballah', 'hizbollah',  # Hezbollah (Lebanon)
        'hamas',  # Hamas (Gaza)
        'houthi', 'ansar allah',  # Houthis (Yemen)
        'islamic jihad', 'pij',  # Palestinian Islamic Jihad
        'popular mobilization', 'pmf', 'hashd',  # Iraqi militias
        'kataib', 'kata\'ib',  # Kataib Hezbollah (Iraq)
        'asaib ahl al-haq',  # Iraqi militia
        'badr'  # Badr Organization (Iraq)
    ]

    iran_pattern = re.compile(r'iran', re.IGNORECASE)
    israel_pattern = re.compile(r'israel', re.IGNORECASE)
    proxy_pattern = re.compile('|'.join(iran_proxies), re.IGNORECASE)

    def is_iran_or_proxy(actor):
        if pd.isna(actor):
            return False
        actor_str = str(actor)
        return bool(iran_pattern.search(actor_str)) or bool(proxy_pattern.search(actor_str))

    def is_israel(actor):
        if pd.isna(actor):
            return False
        return bool(israel_pattern.search(str(actor)))

    # Filter for Iran-Israel confrontations
    iran_israel_mask = (
        (df['actor1'].apply(is_iran_or_proxy) & df['actor2'].apply(is_israel)) |
        (df['actor1'].apply(is_israel) & df['actor2'].apply(is_iran_or_proxy))
    )

    df_filtered = df[iran_israel_mask].copy()

    def normalize_actor(actor):
        if pd.isna(actor):
            return actor
        actor_str = str(actor)
        if is_iran_or_proxy(actor_str):
            return 'IRAN'
        elif is_israel(actor_str):
            return 'ISRAEL'
        return actor

    df_filtered['actor1'] = df_filtered['actor1'].apply(normalize_actor)
    df_filtered['actor2'] = df_filtered['actor2'].apply(normalize_actor)

    relevant_columns = [
        'event_date', 'event_type',
        'sub_event_type', 'actor1', 'actor2', 'fatalities',
    ]

    available_columns = [col for col in relevant_columns if col in df_filtered.columns]
    df_filtered = df_filtered[available_columns]

    # Add severity column based on recent events
    event_severity_mapping = {
        'Explosions/Remote violence': '4',
        'Battles': '4',
        'Violence against civilians': '3',
        'Strategic developments': '2',
        'Riots': '1',
        'Protests': '0',
    }

    df_filtered['event_severity'] = df_filtered['event_type'].map(event_severity_mapping)
    return df_filtered

