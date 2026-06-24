"""
Machine Learning Training Pipeline with MLflow Tracking
Melatih model ML/DL dan tracking metrics, parameters ke DagsHub melalui MLflow
"""

import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


def setup_mlflow_dagshub():
    """Setup MLflow untuk terhubung dengan DagsHub"""
    # Dapatkan credentials dari environment variables
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if not all([mlflow_tracking_uri, mlflow_tracking_username, mlflow_tracking_password]):
        raise ValueError(
            "Missing DagsHub credentials. Set MLFLOW_TRACKING_URI, "
            "MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD"
        )
    
    # Set MLflow tracking server
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set eksperimen
    experiment_name = "COVID_Vaccination_ML_Pipeline"
    mlflow.set_experiment(experiment_name)
    
    print(f"✓ MLflow configured to track at: {mlflow_tracking_uri}")
    print(f"✓ Experiment: {experiment_name}")


def load_and_prepare_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load dan prepare data untuk training"""
    print(f"\n📂 Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Drop non-numeric columns for ML training
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Target variable - gunakan total_vaccinations (atau salah satu dari numeric cols)
    # Anda bisa modify ini sesuai kebutuhan
    if 'total_vaccinations' in numeric_cols:
        target_col = 'total_vaccinations'
        numeric_cols.remove(target_col)
    else:
        target_col = numeric_cols[-1]
        numeric_cols = numeric_cols[:-1]
    
    X = df[numeric_cols].fillna(0)
    y = df[target_col].fillna(0)
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Features: {list(numeric_cols[:5])}... ({len(numeric_cols)} total)")
    print(f"   Target: {target_col}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': numeric_cols,
        'scaler': scaler,
    }


def train_and_log_model(model, model_name: str, X_train, X_test, y_train, y_test, params: dict):
    """Train model dan log ke MLflow"""
    print(f"\n🤖 Training {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Log metrics
        metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
        }
        mlflow.log_metrics(metrics)
        
        # Log model
        if 'sklearn' in str(type(model)):
            mlflow.sklearn.log_model(model, "model")
        
        print(f"   ✓ Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        print(f"   ✓ Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"   ✓ Model logged to MLflow")
        
        return metrics


def train_multiple_models(data, feature_names):
    """Train multiple models dan track semuanya dengan MLflow"""
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    models_config = [
        {
            'name': 'Linear Regression',
            'model': LinearRegression(),
            'params': {}
        },
        {
            'name': 'Random Forest',
            'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': 100,
                'max_depth': 'None',
                'random_state': 42,
            }
        },
        {
            'name': 'Gradient Boosting',
            'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42,
            }
        },
    ]
    
    results = {}
    
    for config in models_config:
        model_name = config['name']
        model = config['model']
        params = config['params']
        
        metrics = train_and_log_model(
            model, model_name, X_train, X_test, y_train, y_test, params
        )
        results[model_name] = metrics
    
    return results


def save_results_summary(results: dict, output_dir: str = 'training/results'):
    """Save hasil training summary ke file"""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    
    # Convert results to JSON-serializable format
    results_json = {}
    for model_name, metrics in results.items():
        results_json[model_name] = {k: float(v) for k, v in metrics.items()}
    
    with open(summary_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n📊 Training summary saved to {summary_path}")
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("🚀 COVID Vaccination ML Training Pipeline with MLflow")
    print("="*60)
    
    # Setup MLflow connection ke DagsHub
    setup_mlflow_dagshub()
    
    # Load preprocessed data
    data_path = '../preprocessing/country_vaccinations_preprocessed.csv'
    data = load_and_prepare_data(data_path)
    
    # Train multiple models dengan MLflow tracking
    results = train_multiple_models(data, data['feature_names'])
    
    # Save summary
    save_results_summary(results)
    
    print("\n✅ Training pipeline completed!")
    print("📈 Check DagsHub MLflow Tracking UI for detailed results")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
