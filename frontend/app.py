import io
import os

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="SK", layout="wide")
st.title("SuperKart Sales Forecast")
st.caption("Estimate product-store sales for inventory and regional planning.")


def show_api_error(response):
    try:
        details = response.json()
    except ValueError:
        details = response.text
    st.error(f"Prediction failed ({response.status_code}): {details}")


single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction"])

with single_tab:
    with st.form("single_prediction"):
        left, right = st.columns(2)
        with left:
            product_weight = st.number_input("Product weight", min_value=0.01, value=12.66)
            sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
            allocated_area = st.number_input(
                "Allocated display-area ratio", min_value=0.001, max_value=1.0, value=0.027, format="%.3f"
            )
            mrp = st.number_input("Product MRP", min_value=0.01, value=117.08)
            product_prefix = st.selectbox("Product ID prefix", ["FD", "DR", "NC"])
        with right:
            store_size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
            city_tier = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
            store_type = st.selectbox(
                "Store type",
                ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"],
                index=3,
            )
            store_age = st.number_input("Store age in years", min_value=0, value=16, step=1)
            product_category = st.selectbox("Product category", ["Perishables", "Non Perishables"])

        submitted = st.form_submit_button("Forecast sales", type="primary")

    if submitted:
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_tier,
            "Store_Type": store_type,
            "Product_Id_char": product_prefix,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_category,
        }
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=REQUEST_TIMEOUT)
            if response.ok:
                result = response.json()
                st.metric("Forecast product-store sales", f"{result['prediction']:,.2f}")
                for warning in result.get("warnings", []):
                    st.warning(warning)
            else:
                show_api_error(response)
        except requests.RequestException as exc:
            st.error(f"Backend unavailable: {exc}")

with batch_tab:
    st.write("Upload a CSV containing the ten model input columns.")
    uploaded_file = st.file_uploader("Batch CSV", type=["csv"])
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        try:
            preview = pd.read_csv(io.BytesIO(content))
            st.dataframe(preview.head(20), use_container_width=True)
        except Exception as exc:
            st.error(f"Cannot preview CSV: {exc}")
            preview = None

        if preview is not None and st.button("Run batch forecast", type="primary"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={"file": (uploaded_file.name, content, "text/csv")},
                    timeout=REQUEST_TIMEOUT,
                )
                if response.ok:
                    body = response.json()
                    predictions = body.get("predictions", body)
                    output = preview.copy()
                    output["Predicted_Sales"] = [predictions[str(index)] for index in output.index]
                    st.success(f"Generated {len(output)} forecasts.")
                    st.dataframe(output, use_container_width=True)
                    st.download_button(
                        "Download predictions",
                        output.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_predictions.csv",
                        mime="text/csv",
                    )
                    for warning in body.get("warnings", []):
                        st.warning(warning)
                else:
                    show_api_error(response)
            except requests.RequestException as exc:
                st.error(f"Backend unavailable: {exc}")
