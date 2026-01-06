import streamlit as st
import pandas as pd

# IMPORTANT: inference only (no training, no Django)
from ml_engine.inference.predictor import FraudPredictor

# -------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Ad Click Fraud Detection - ML Service",
    layout="centered"
)

st.title("🚨 Ad Click Fraud Detection – ML Service")
st.caption("Inference-only service using pre-trained CNN–RNN + XGBoost models")

# -------------------------------------------------
# Load model ONCE (critical for free tier)
# -------------------------------------------------
@st.cache_resource
def load_predictor():
    return FraudPredictor()

predictor = load_predictor()

# -------------------------------------------------
# CSV Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload clickstream CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.success(
            f"CSV loaded successfully | Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        )

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # -------------------------------------------------
        # Run inference
        # -------------------------------------------------
        if st.button("Run Fraud Detection"):
            with st.spinner("Running fraud detection..."):
                results = predictor.predict(df)

            st.success("Fraud detection completed")

            # ==============================
            # DISPLAY RESULTS
            # ==============================

            # ---- Summary ----
            st.subheader("📊 Summary")
            st.json(results["summary"])

            # ---- High Risk IPs ----
            st.subheader("🚨 High-Risk IPs")
            if results["ip_risk"]:
                st.dataframe(pd.DataFrame(results["ip_risk"]))
            else:
                st.info("No high-risk IPs detected")

            # ---- Hourly Fraud Trends ----
            st.subheader("⏰ Hourly Fraud Trends")
            st.dataframe(pd.DataFrame(results["time_trends"]))

            # ---- Business Impact ----
            st.subheader("💰 Business Impact")
            st.json(results["business_impact"])

            # ---- SHAP Explainability ----
            st.subheader("🔍 SHAP Explainability (Top Factors)")
            st.dataframe(pd.DataFrame(results["shap_summary"]))

    except Exception as e:
        st.error("❌ Error during fraud detection")
        st.exception(e)
