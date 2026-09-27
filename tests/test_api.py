import io
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import app as backend_app


VALID_PAYLOAD = {
    "Product_Weight": 12.66,
    "Product_Sugar_Content": "Low Sugar",
    "Product_Allocated_Area": 0.027,
    "Product_MRP": 117.08,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type2",
    "Product_Id_char": "FD",
    "Store_Age_Years": 16,
    "Product_Type_Category": "Perishables",
}


@pytest.fixture()
def client():
    backend_app.superkart_api.config.update(TESTING=True)
    return backend_app.superkart_api.test_client()


def test_health_and_readiness(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200


def test_single_prediction(client):
    response = client.post("/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert isinstance(response.get_json()["prediction"], float)


def test_invalid_schema_is_rejected(client):
    invalid = VALID_PAYLOAD.copy()
    invalid.pop("Product_MRP")
    response = client.post("/v1/predict", json=invalid)
    assert response.status_code == 422


def test_batch_matches_single_prediction(client):
    frame = pd.DataFrame([VALID_PAYLOAD])
    content = frame.to_csv(index=False).encode("utf-8")
    batch = client.post(
        "/v1/predictbatch",
        data={"file": (io.BytesIO(content), "batch.csv")},
        content_type="multipart/form-data",
    )
    single = client.post("/v1/predict", json=VALID_PAYLOAD)
    assert batch.status_code == 200
    assert batch.get_json()["0"] == single.get_json()["prediction"]


def test_unseen_training_category_returns_drift_warning(client):
    payload = VALID_PAYLOAD.copy()
    payload["Store_Type"] = "Supermarket Type3"
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    assert "unseen during training" in response.get_json()["warnings"][0]


@pytest.mark.parametrize(
    ("field", "value"),
    [("Product_Weight", True), ("Store_Age_Years", 16.5)],
)
def test_invalid_numeric_types_are_rejected(client, field, value):
    payload = VALID_PAYLOAD.copy()
    payload[field] = value
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
