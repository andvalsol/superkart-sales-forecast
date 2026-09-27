# SuperKart Sales Forecast

An end-to-end product-store sales forecasting project with a fitted regression pipeline, Flask API, Streamlit interface, and Docker deployment.

## Architecture

- `backend`: Flask/Gunicorn API on port `7860`
- `frontend`: Streamlit UI on port `8501`
- `backend/models/superkart_model.joblib`: serialized preprocessing and regression pipeline
- `compose.yaml`: reproducible two-container deployment

## Run with Docker

```bash
docker compose up --build -d --wait
curl http://localhost:7860/readyz
```

Open `http://localhost:8501` for the UI.

## Live Codespace

The public demonstration is available while the Codespace is running:

- API: `https://superkart-deployment-w9jwq44vpp72g6gp-7860.app.github.dev`
- Streamlit: `https://superkart-deployment-w9jwq44vpp72g6gp-8501.app.github.dev`

GitHub may show a one-time access warning before opening a forwarded port. The Codespace stops after 30 minutes of inactivity, so these URLs are demonstration endpoints rather than permanent hosting.

## API

Single prediction:

```bash
curl -X POST http://localhost:7860/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Product_Weight": 12.66,
    "Product_Sugar_Content": "Low Sugar",
    "Product_Allocated_Area": 0.027,
    "Product_MRP": 117.08,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type2",
    "Product_Id_char": "FD",
    "Store_Age_Years": 16,
    "Product_Type_Category": "Perishables"
  }'
```

Batch prediction:

```bash
curl -X POST -F file=@Batch_Data_SuperKart.csv http://localhost:7860/v1/predictbatch
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

Only trusted model artifacts should be loaded because Python serialization formats can execute code during deserialization.
