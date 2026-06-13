import json
import os
import time
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, generate_latest

MODEL_PATH = os.environ.get("MODEL_PATH", "models/latest")
MODEL_INFO_PATH = "models/model_info.json"

# Prometheus metrics definition
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total jumlah request inferensi yang diterima"
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total jumlah error saat inferensi"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Durasi endpoint prediksi dalam detik"
)

# New metrics for Advance (observability)
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total jumlah baris data/individu yang diprediksi"
)
MODEL_LOAD_STATUS = Gauge(
    "model_load_status",
    "Status load model (1 = sukses, 0 = gagal)"
)
PREDICTION_VALUE_MEAN = Gauge(
    "prediction_value_mean",
    "Nilai rata-rata prediksi vaksinasi harian dalam batch terakhir"
)
PREDICTION_VALUE_MAX = Gauge(
    "prediction_value_max",
    "Nilai maksimum prediksi vaksinasi harian dalam batch terakhir"
)
PREDICTION_VALUE_MIN = Gauge(
    "prediction_value_min",
    "Nilai minimum prediksi vaksinasi harian dalam batch terakhir"
)
PREDICTION_FEATURES_NAN_TOTAL = Counter(
    "prediction_features_nan_total",
    "Jumlah nilai kosong (NaN) pada fitur input yang diterima"
)
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP request yang diproses",
    ["endpoint", "method", "status"]
)

app = Flask(__name__)
model = None
model_info = {}

def load_model_and_metadata():
    global model, model_info
    
    # Fallback path if models/latest does not exist locally
    actual_model_path = MODEL_PATH
    if not Path(actual_model_path).exists() and Path("../models/latest").exists():
        actual_model_path = "../models/latest"
    elif not Path(actual_model_path).exists() and Path("../Membangun_model/models/latest").exists():
        actual_model_path = "../Membangun_model/models/latest"
    elif not Path(actual_model_path).exists() and Path("models/latest").exists():
        actual_model_path = "models/latest"
        
    actual_info_path = MODEL_INFO_PATH
    if not Path(actual_info_path).exists() and Path("../models/model_info.json").exists():
        actual_info_path = "../models/model_info.json"
    elif not Path(actual_info_path).exists() and Path("../Membangun_model/models/model_info.json").exists():
        actual_info_path = "../Membangun_model/models/model_info.json"
    elif not Path(actual_info_path).exists() and Path("models/model_info.json").exists():
        actual_info_path = "models/model_info.json"

    print(f"Attempting to load model from: {actual_model_path}")
    if Path(actual_model_path).exists():
        model = mlflow.pyfunc.load_model(actual_model_path)
        MODEL_LOAD_STATUS.set(1)
        print("Model loaded successfully.")
    else:
        MODEL_LOAD_STATUS.set(0)
        print("Warning: Model not found. Server running in uninitialized state.")
        
    if Path(actual_info_path).exists():
        with open(actual_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        print("Model metadata loaded successfully.")

@app.route("/", methods=["GET"])
def index():
    HTTP_REQUESTS_TOTAL.labels(endpoint="/", method="GET", status="200").inc()
    return jsonify({
        "message": "Welcome to the Vaccination Prediction Model API",
        "endpoints": {
            "health": "/health [GET]",
            "metrics": "/metrics [GET]",
            "predict": "/predict [POST]"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    HTTP_REQUESTS_TOTAL.labels(endpoint="/health", method="GET", status="200").inc()
    status = "ready" if model is not None else "not_ready"
    return jsonify({
        "status": status,
        "model_path": MODEL_PATH,
        "model_info": model_info
    })

@app.route("/predict", methods=["POST"])
def predict():
    PREDICTION_REQUESTS.inc()
    start = time.time()
    try:
        if model is None:
            HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status="503").inc()
            return jsonify({"error": "Model not loaded on server."}), 503
            
        payload = request.get_json(force=True)
        rows = payload.get("data", [])
        if not rows:
            HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status="400").inc()
            return jsonify({"error": "Payload harus memiliki field 'data' berisi list record."}), 400

        frame = pd.DataFrame(rows)
        
        # Count NaNs in features
        nan_count = frame.isna().sum().sum()
        if nan_count > 0:
            PREDICTION_FEATURES_NAN_TOTAL.inc(int(nan_count))

        preds = model.predict(frame)
        duration = time.time() - start
        PREDICTION_LATENCY.observe(duration)
        
        # Track predictions count and value stats
        PREDICTIONS_TOTAL.inc(len(rows))
        preds_list = list(map(float, preds))
        if preds_list:
            PREDICTION_VALUE_MEAN.set(sum(preds_list) / len(preds_list))
            PREDICTION_VALUE_MAX.set(max(preds_list))
            PREDICTION_VALUE_MIN.set(min(preds_list))

        HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status="200").inc()
        return jsonify({
            "predictions": preds_list,
            "count": len(rows),
            "latency_seconds": duration,
        })
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        HTTP_REQUESTS_TOTAL.labels(endpoint="/predict", method="POST", status="500").inc()
        duration = time.time() - start
        PREDICTION_LATENCY.observe(duration)
        return jsonify({"error": str(exc), "latency_seconds": duration}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    HTTP_REQUESTS_TOTAL.labels(endpoint="/metrics", method="GET", status="200").inc()
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    load_model_and_metadata()
    # Listen on port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
