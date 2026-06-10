import argparse
import os
import pandas as pd
from sklearn.linear_model import Ridge
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split


def _make_dummy_split(train_csv: str, test_csv: str):
    """Create small dummy train/test CSV files for demonstration."""

    dummy_data = pd.DataFrame({
        "total_vaccinations": np.linspace(0.1, 1.0, 20),
        "people_vaccinated": np.linspace(0.1, 1.0, 20),
        "people_fully_vaccinated": np.linspace(0.05, 0.5, 20),
        "daily_vaccinations_raw": np.full(20, 0.1),
        "total_vaccinations_per_hundred": np.linspace(0.1, 1.0, 20),
        "people_vaccinated_per_hundred": np.linspace(0.1, 1.0, 20),
        "people_fully_vaccinated_per_hundred": np.linspace(0.05, 0.5, 20),
        "daily_vaccinations_per_million": np.full(20, 0.1),
        "daily_vaccinations": np.linspace(500, 5000, 20),
    })
    train_df, test_df = train_test_split(dummy_data, test_size=0.2, random_state=42)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print("Dummy train/test datasets created for demonstration.")


def _resolve_path(primary: str, fallbacks: list) -> str:
    """Return the first existing path from primary then fallbacks."""
    if os.path.exists(primary):
        return primary
    for fb in fallbacks:
        if os.path.exists(fb):
            return fb
    return primary  # let caller handle missing file error


def main():
    parser = argparse.ArgumentParser(
        description="Train basic Ridge regression model with MLflow autolog"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="country_vaccinations_train.csv",
        help="Path to training split CSV dataset",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="country_vaccinations_test.csv",
        help="Path to test split CSV dataset",
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI",
    )
    args = parser.parse_args()

    # Configure MLflow - gunakan SQLite agar kompatibel dengan MLflow 3.x
    tracking_uri = args.tracking_uri
    if tracking_uri == "file:./mlruns":
        tracking_uri = "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("vaccination-basic-experiment")

    # Resolve train/test paths with fallbacks
    train_fallbacks = ["../preprocessing/country_vaccinations_train.csv"]
    test_fallbacks = ["../preprocessing/country_vaccinations_test.csv"]
    args.train_csv = _resolve_path(args.train_csv, train_fallbacks)
    args.test_csv = _resolve_path(args.test_csv, test_fallbacks)

    # If train/test files are not found, create dummy data
    if not os.path.exists(args.train_csv) or not os.path.exists(args.test_csv):
        print("Train/test files not found. Creating dummy data for demonstration...")
        _make_dummy_split(args.train_csv, args.test_csv)

    print(f"Loading train dataset from {args.train_csv}...")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loading test dataset from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)

    target = "daily_vaccinations"
    for name, df in [("train", train_df), ("test", test_df)]:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in {name} dataset.")

    train_df = train_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    # Feature columns
    non_feature_cols = ["date", "country", "vaccines", "iso_code", target]
    features = [c for c in train_df.columns if c not in non_feature_cols]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"Train size: {len(X_train)} rows | Test size: {len(X_test)} rows")

    # Enable MLflow Autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="basic_ridge_training") as run:
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Test prediction to trigger autolog evaluation metrics
        predictions = model.predict(X_test)  # noqa: F841

        # Simpan model ke folder models/latest untuk Docker build
        os.makedirs("models/latest", exist_ok=True)
        mlflow.sklearn.save_model(model, "models/latest")
        print(f"Model saved to models/latest (run_id: {run.info.run_id})")
        print("Basic Ridge regression training completed.")


if __name__ == "__main__":
    main()
