import pandas as pd
from datetime import timedelta
from src.models import acled, gdelt


def verify_prediction_capability(gdelt_path, acled_path, n_samples=1000):
    """
    Verify if GDELT data from 30 days prior can predict ACLED events

    Since both datasets start at the same date, we shift ACLED events by 30 days
    to ensure we have GDELT data from the 30 days before each ACLED event.

    Args:
        gdelt_path: Path to GDELT CSV file
        acled_path: Path to ACLED CSV file
        n_samples: Number of samples to use

    Returns:
        pd.DataFrame: Results with predictions and actual events
    """
    print(f"Loading ACLED events...")
    acled_df = acled.load_data(acled_path)
    acled_df_shifted = acled_df[acled_df['event_date'] >= acled_df['event_date'].min() + timedelta(days=30)]
    acled_df_shifted = acled_df_shifted.head(n_samples)

    print(f"Loading GDELT data...")
    gdelt_df = gdelt.load_data(gdelt_path)

    results = []

    print(f"\nProcessing {len(acled_df_shifted)} ACLED events...")
    for idx, acled_row in acled_df_shifted.iterrows():
        acled_date = acled_row['event_date']
        acled_severity = acled_row['event_severity']

        start_date = acled_date - timedelta(days=30)
        end_date = acled_date - timedelta(days=1)

        gdelt_window = gdelt_df[
            (gdelt_df['date'] >= start_date) &
            (gdelt_df['date'] <= end_date)
        ]

        if len(gdelt_window) > 0:
            gdelt_window_copy = gdelt_window.copy()
            gdelt_window_copy['event_text'] = gdelt_window_copy.apply(gdelt.create_event_text, axis=1)

            scored_window = gdelt.classify_text_intensity(gdelt_window_copy, text_column='event_text')

            max_predicted_severity = scored_window['predicted_label'].max()
            avg_predicted_severity = scored_window['predicted_label'].mean()
            event_count = len(gdelt_window)
            avg_goldstein = gdelt_window['GoldsteinScale'].mean()
            min_goldstein = gdelt_window['GoldsteinScale'].min()
            avg_tone = gdelt_window['AvgTone'].mean()
        else:
            max_predicted_severity = 0
            avg_predicted_severity = 0
            event_count = 0
            avg_goldstein = 0
            min_goldstein = 0
            avg_tone = 0

        result = {
            'acled_date': acled_date,
            'acled_event_type': acled_row['event_type'],
            'acled_severity': int(acled_severity),
            'gdelt_event_count': event_count,
            'gdelt_max_predicted_severity': max_predicted_severity,
            'gdelt_avg_predicted_severity': avg_predicted_severity,
            'gdelt_avg_goldstein': avg_goldstein,
            'gdelt_min_goldstein': min_goldstein,
            'gdelt_avg_tone': avg_tone,
            'correct_prediction': max_predicted_severity >= int(acled_severity) if event_count > 0 else False
        }
        results.append(result)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(acled_df_shifted)} events...")

    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("PREDICTION VERIFICATION SUMMARY")
    print("="*80)

    events_with_data = results_df[results_df['gdelt_event_count'] > 0]

    print(f"\nTotal ACLED events analyzed: {len(results_df)}")
    print(f"Events with GDELT data (30 days prior): {len(events_with_data)}")
    print(f"Coverage: {len(events_with_data) / len(results_df) * 100:.2f}%")

    if len(events_with_data) > 0:
        print(f"\nPrediction accuracy (max severity >= actual): {events_with_data['correct_prediction'].sum() / len(events_with_data) * 100:.2f}%")

        print("\n" + "-"*80)
        print("ACLED Severity Distribution:")
        print(results_df['acled_severity'].value_counts().sort_index())

        print("\n" + "-"*80)
        print("GDELT Predicted Severity (max) Distribution:")
        print(events_with_data['gdelt_max_predicted_severity'].value_counts().sort_index())

    return results_df


if __name__ == "__main__":
    gdelt_path = "data/raw/irn_isr_gdelt.csv"
    acled_path = "data/raw/irn_isr_acled.csv"

    results = verify_prediction_capability(gdelt_path, acled_path, n_samples=1000)

    output_path = "data/verification_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")