import streamlit as st
import pandas as pd
import json

from ml_engine.inference.predictor import FraudPredictor

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Ad Click Fraud Detection – ML API",
    layout="centered"
)

st.title("🚨 Ad Click Fraud Detection – ML Service")
st.caption("Inference-only CNN–RNN + XGBoost service")

# -------------------------------------------------
# LOAD MODEL ONCE (CRITICAL)
# -------------------------------------------------
@st.cache_resource
def load_predictor():
    return FraudPredictor()

predictor = load_predictor()

# -------------------------------------------------
# API MODE (Django will POST file here)
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV (API or UI)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success(
            f"CSV Loaded | Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        )

        if st.button("Run Fraud Detection"):
            with st.spinner("Running inference..."):
                result = predictor.predict(df)

            st.success("Inference completed")

            # IMPORTANT: JSON OUTPUT (API)
            st.json(result)

    except Exception as e:
        st.error("Inference failed")
        st.exception(e)
