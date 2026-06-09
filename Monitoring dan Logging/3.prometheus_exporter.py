import json
import os
import time
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

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

app = Flask(__name__)
model = None
model_info = {}

def load_model_and_metadata():
    global model, model_info
    
    # Fallback path if models/latest does not exist locally
    actual_model_path = MODEL_PATH
    if not Path(actual_model_path).exists() and Path("../Membangun_model/models/latest").exists():
        actual_model_path = "../Membangun_model/models/latest"
    elif not Path(actual_model_path).exists() and Path("models/latest").exists():
        actual_model_path = "models/latest"
        
    actual_info_path = MODEL_INFO_PATH
    if not Path(actual_info_path).exists() and Path("../Membangun_model/models/model_info.json").exists():
        actual_info_path = "../Membangun_model/models/model_info.json"
    elif not Path(actual_info_path).exists() and Path("models/model_info.json").exists():
        actual_info_path = "models/model_info.json"

    print(f"Attempting to load model from: {actual_model_path}")
    if Path(actual_model_path).exists():
        model = mlflow.pyfunc.load_model(actual_model_path)
        print("Model loaded successfully.")
    else:
        print("Warning: Model not found. Server running in uninitialized state.")
        
    if Path(actual_info_path).exists():
        with open(actual_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        print("Model metadata loaded successfully.")

@app.route("/health", methods=["GET"])
def health():
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
            return jsonify({"error": "Model not loaded on server."}), 503
            
        payload = request.get_json(force=True)
        rows = payload.get("data", [])
        if not rows:
            return jsonify({"error": "Payload harus memiliki field 'data' berisi list record."}), 400

        frame = pd.DataFrame(rows)
        preds = model.predict(frame)
        duration = time.time() - start
        PREDICTION_LATENCY.observe(duration)

        return jsonify({
            "predictions": list(map(float, preds)),
            "count": len(rows),
            "latency_seconds": duration,
        })
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        duration = time.time() - start
        PREDICTION_LATENCY.observe(duration)
        return jsonify({"error": str(exc), "latency_seconds": duration}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    load_model_and_metadata()
    # Listen on port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
