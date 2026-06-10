import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Attempt to import dagshub
try:
    import dagshub
    HAS_DAGSHUB = True
except ImportError:
    HAS_DAGSHUB = False


def _make_dummy_split(train_csv: str, test_csv: str):
    """Create small dummy train/test CSV files for demonstration."""
    from sklearn.model_selection import train_test_split

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
    return primary


def main():
    parser = argparse.ArgumentParser(
        description="Train tuned Ridge regression model with manual logging and DagsHub integration"
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
        default="",
        help="MLflow tracking URI (opsional, otomatis diatur oleh DagsHub jika tersedia)",
    )
    parser.add_argument(
        "--dagshub_repo_owner",
        type=str,
        default=os.environ.get("DAGSHUB_REPO_OWNER", ""),
        help="DagsHub repo owner/username (atau set env DAGSHUB_REPO_OWNER)",
    )
    parser.add_argument(
        "--dagshub_repo_name",
        type=str,
        default=os.environ.get("DAGSHUB_REPO_NAME", "Membangun-Sistem-Machine-Learning"),
        help="DagsHub repo name (atau set env DAGSHUB_REPO_NAME)",
    )
    args = parser.parse_args()

    # ── Inisialisasi tracking ke DagsHub (tanpa OAuth browser) ──────────────
    # Pendekatan: set MLFLOW_TRACKING_URI langsung ke server DagsHub
    # Kredensial dibaca dari env var MLFLOW_TRACKING_USERNAME & MLFLOW_TRACKING_PASSWORD
    if args.dagshub_repo_owner:
        dagshub_uri = (
            f"https://dagshub.com/{args.dagshub_repo_owner}/{args.dagshub_repo_name}.mlflow"
        )
        mlflow.set_tracking_uri(dagshub_uri)
        print(f"MLflow tracking URI set to DagsHub: {dagshub_uri}")

        # Validasi bahwa credentials tersedia
        username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        if not username or not password:
            print("⚠️  WARNING: MLFLOW_TRACKING_USERNAME atau MLFLOW_TRACKING_PASSWORD belum di-set!")
            print("   Jalankan perintah berikut sebelum menjalankan script ini:")
            print(f"   export MLFLOW_TRACKING_USERNAME=\"{args.dagshub_repo_owner}\"")
            print("   export MLFLOW_TRACKING_PASSWORD=\"<token_dagshub_anda>\"")
            raise EnvironmentError(
                "Kredensial DagsHub tidak ditemukan. "
                "Set MLFLOW_TRACKING_USERNAME dan MLFLOW_TRACKING_PASSWORD terlebih dahulu."
            )
        print(f"Authenticated as: {username}")
    elif args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        print(f"Using custom tracking URI: {args.tracking_uri}")
    else:
        # Fallback ke SQLite agar kompatibel dengan MLflow 3.x
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        print("Using local SQLite tracking: mlruns.db")

    mlflow.set_experiment("vaccination-tuned-experiment")


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

    # Tuning Grid
    alphas = [0.1, 1.0, 10.0]
    best_r2 = -1
    best_alpha = None
    best_model = None

    print(f"Starting hyperparameter tuning over alphas: {alphas}...")
    for alpha in alphas:
        run_name = f"ridge_alpha_{alpha}"
        with mlflow.start_run(run_name=run_name, nested=True):
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            # Manual logging
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("model_type", "Ridge")
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            print(f"Alpha={alpha}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha
                best_model = model

    # Log best model with additional artifacts
    with mlflow.start_run(run_name="best_tuned_model"):
        preds = best_model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(best_r2)

        mlflow.log_param("best_alpha", best_alpha)
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Save and log the model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Save local artifacts
        os.makedirs("models/latest", exist_ok=True)
        mlflow.sklearn.save_model(best_model, "models/latest")

        # Additional artifact 1: model_info.json (Metadata)
        model_info = {
            "best_alpha": best_alpha,
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
            "features": list(X_train.columns),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        with open("models/model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=4)
        mlflow.log_artifact("models/model_info.json", artifact_path="metadata")

        # Additional artifact 2: Feature coefficient plot (Plots)
        plt.figure(figsize=(10, 5))
        coefs = pd.Series(best_model.coef_, index=X_train.columns)
        coefs.sort_values().plot(kind="barh", color="teal")
        plt.title("Feature Coefficients (Ridge Regression)")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("models/feature_coefficients.png")
        plt.close()
        mlflow.log_artifact("models/feature_coefficients.png", artifact_path="plots")

        # Additional artifact 3: Dataset sample preview (Data)
        sample_df = pd.concat([train_df.head(8), test_df.head(2)])
        sample_df.to_csv("models/dataset_sample_preview.csv", index=False)
        mlflow.log_artifact("models/dataset_sample_preview.csv", artifact_path="data")

        print(f"\nTuned model training completed successfully!")
        print(f"Best Alpha: {best_alpha}")
        print(f"Best R2 Score: {best_r2:.4f}")
        print("Logged artifacts: model_info.json, feature_coefficients.png, dataset_sample_preview.csv")


if __name__ == "__main__":
    main()
