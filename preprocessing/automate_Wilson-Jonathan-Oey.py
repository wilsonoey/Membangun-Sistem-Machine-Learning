import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def automate_preprocessing(
    input_path: str,
    output_path: str,
    train_path: str = None,
    test_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Perform full preprocessing pipeline on raw vaccination dataset.

    Steps:
    1. Sort by country and date
    2. Forward-fill numeric features per country
    3. Fill remaining NaN with 0
    4. Drop source columns
    5. Scale numeric features using StandardScaler
    6. Save full preprocessed dataset
    7. (Optional) Split and save train/test datasets

    Args:
        input_path:    Path to raw input CSV file.
        output_path:   Path to save full preprocessed CSV.
        train_path:    Path to save training split CSV (if None, skip split).
        test_path:     Path to save test split CSV (if None, skip split).
        test_size:     Proportion of data for test split (default 0.2 = 20%).
        random_state:  Random seed for reproducibility (default 42).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)
    df = df.copy()
    df = df.sort_values(['country', 'date'])

    numeric_cols = [
        'total_vaccinations',
        'people_vaccinated',
        'people_fully_vaccinated',
        'daily_vaccinations_raw',
        'daily_vaccinations',
        'total_vaccinations_per_hundred',
        'people_vaccinated_per_hundred',
        'people_fully_vaccinated_per_hundred',
        'daily_vaccinations_per_million'
    ]

    print("Performing forward-fill for numeric features grouped by country...")
    df[numeric_cols] = df.groupby('country')[numeric_cols].ffill()

    print("Filling remaining missing values with 0...")
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print("Dropping source_website and source_name columns...")
    df = df.drop(columns=['source_website', 'source_name'], errors='ignore')

    print("Scaling numeric features using StandardScaler...")
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save full preprocessed dataset
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing completed! Saved full preprocessed data to: {output_path}")

    # Train/Test split (optional)
    if train_path is not None and test_path is not None:
        print(f"\nSplitting data into train ({int((1 - test_size) * 100)}%) "
              f"and test ({int(test_size * 100)}%) sets...")

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        for path in [train_path, test_path]:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train set: {len(train_df)} rows  → saved to: {train_path}")
        print(f"Test  set: {len(test_df)} rows  → saved to: {test_path}")

    return df


if __name__ == "__main__":
    default_input = "country_vaccinations.csv"
    default_output = "country_vaccinations_preprocessed.csv"
    default_train = "country_vaccinations_train.csv"
    default_test = "country_vaccinations_test.csv"

    if not os.path.exists(default_input) and os.path.exists("country_vaccinations_raw.csv"):
        default_input = "country_vaccinations_raw.csv"

    try:
        automate_preprocessing(
            input_path=default_input,
            output_path=default_output,
            train_path=default_train,
            test_path=default_test,
            test_size=0.2,
            random_state=42,
        )
    except Exception as e:
        print(f"Error during preprocessing: {e}")
