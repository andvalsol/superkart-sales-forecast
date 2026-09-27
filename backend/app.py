import io
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]
NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]
ALLOWED_CATEGORIES = {
    "Product_Sugar_Content": {"Low Sugar", "Regular", "No Sugar"},
    "Store_Size": {"Small", "Medium", "High"},
    "Store_Location_City_Type": {"Tier 1", "Tier 2", "Tier 3"},
    "Store_Type": {
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
    },
    "Product_Id_char": {"FD", "DR", "NC"},
    "Product_Type_Category": {"Perishables", "Non Perishables"},
}
TRAINING_RANGES = {
    "Product_Weight": (4.0, 22.0),
    "Product_Allocated_Area": (0.004, 0.298),
    "Product_MRP": (31.0, 266.0),
    "Store_Age_Years": (16.0, 38.0),
}

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/superkart_model.joblib"))
if not MODEL_PATH.exists():
    local_path = Path(__file__).resolve().parent / "models" / "superkart_model.joblib"
    MODEL_PATH = local_path if local_path.exists() else MODEL_PATH

MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "10000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))

superkart_api = Flask(__name__)
superkart_api.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

model = None
model_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:  # Readiness reports failure without leaking details.
    model_error = str(exc)


def error_response(message, status=422, details=None):
    body = {"error": message}
    if details:
        body["details"] = details
    return jsonify(body), status


def validate_records(frame):
    missing = [column for column in FEATURES if column not in frame.columns]
    extra = [column for column in frame.columns if column not in FEATURES]
    if missing or extra:
        return None, {
            "missing_columns": missing,
            "unexpected_columns": extra,
        }

    clean = frame[FEATURES].copy()
    clean["Product_Sugar_Content"] = clean["Product_Sugar_Content"].replace({"reg": "Regular"})

    validation_errors = []
    drift_warnings = []
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(clean[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted)
        if invalid.any():
            validation_errors.append(f"{column} must contain finite numeric values")
            continue
        clean[column] = converted
        lower, upper = TRAINING_RANGES[column]
        if ((converted < lower) | (converted > upper)).any():
            drift_warnings.append(f"{column} contains values outside the training range [{lower}, {upper}]")

    if not validation_errors:
        if (clean["Product_Weight"] <= 0).any():
            validation_errors.append("Product_Weight must be positive")
        if (clean["Product_Allocated_Area"] <= 0).any():
            validation_errors.append("Product_Allocated_Area must be positive")
        if (clean["Product_MRP"] <= 0).any():
            validation_errors.append("Product_MRP must be positive")
        if (clean["Store_Age_Years"] < 0).any():
            validation_errors.append("Store_Age_Years cannot be negative")

    for column, allowed in ALLOWED_CATEGORIES.items():
        values = set(clean[column].dropna().astype(str))
        invalid_values = sorted(values - allowed)
        if invalid_values:
            validation_errors.append(f"{column} has unsupported values: {invalid_values}")

    if clean.isna().any().any():
        validation_errors.append("Null values are not allowed")

    if validation_errors:
        return None, {"validation_errors": validation_errors}
    return clean, {"warnings": drift_warnings}


@superkart_api.get("/healthz")
def health():
    return jsonify({"status": "ok"})


@superkart_api.get("/readyz")
def readiness():
    if model is None:
        return error_response("Model is not ready", status=503)
    return jsonify({"status": "ready", "model": MODEL_PATH.name})


@superkart_api.post("/v1/predict")
def predict():
    if model is None:
        return error_response("Model is not ready", status=503)
    if not request.is_json:
        return error_response("Content-Type must be application/json", status=415)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error_response("Request body must be a JSON object")

    frame, validation = validate_records(pd.DataFrame([payload]))
    if frame is None:
        return error_response("Invalid prediction input", details=validation)

    prediction = float(model.predict(frame)[0])
    return jsonify({"prediction": round(prediction, 2), **validation})


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    if model is None:
        return error_response("Model is not ready", status=503)
    if "file" not in request.files:
        return error_response("Upload a CSV using multipart field 'file'", status=400)

    uploaded = request.files["file"]
    if not uploaded.filename.lower().endswith(".csv"):
        return error_response("The uploaded file must have a .csv extension", status=415)
    try:
        content = uploaded.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            return error_response("Uploaded file is too large", status=413)
        frame = pd.read_csv(io.BytesIO(content))
    except Exception:
        return error_response("Unable to parse the uploaded CSV", status=400)

    if frame.empty:
        return error_response("The uploaded CSV has no data rows", status=400)
    if len(frame) > MAX_BATCH_ROWS:
        return error_response(f"Batch exceeds the {MAX_BATCH_ROWS}-row limit", status=413)

    clean, validation = validate_records(frame)
    if clean is None:
        return error_response("Invalid batch input", details=validation)

    predictions = model.predict(clean)
    result = {str(index): round(float(value), 2) for index, value in zip(frame.index, predictions)}
    if validation["warnings"]:
        return jsonify({"predictions": result, **validation})
    return jsonify(result)


@superkart_api.errorhandler(413)
def request_too_large(_error):
    return error_response("Request exceeds the upload-size limit", status=413)


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
