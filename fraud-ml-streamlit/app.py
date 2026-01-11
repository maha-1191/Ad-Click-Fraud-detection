import streamlit as st
import pandas as pd
from ml_engine.inference.predictor import FraudPredictor


query_params = st.experimental_get_query_params()

@st.cache_resource
def load_predictor():
    return FraudPredictor()

predictor = load_predictor()

if "api" in query_params:
    uploaded_file = st.file_uploader(
        label="",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        result = predictor.predict(df)

        st.json(result)

    st.stop()


st.set_page_config(
    page_title="Ad Click Fraud Detection - ML Service",
    layout="centered"
)

st.title("Ad Click Fraud Detection – ML Service")
st.caption("Inference-only CNN–RNN + XGBoost service")

uploaded_file = st.file_uploader(
    "Upload clickstream CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success(
        f"CSV Loaded | Rows: {df.shape[0]} | Columns: {df.shape[1]}"
    )
    st.dataframe(df.head())

    if st.button("Run Fraud Detection"):
        with st.spinner("Running inference..."):
            results = predictor.predict(df)

        st.success("Inference completed")

        st.subheader("Summary")
        st.json(results["summary"])

        st.subheader("IP Risk")
        st.dataframe(pd.DataFrame(results["ip_risk"]))

        st.subheader("Hourly Trends")
        st.dataframe(pd.DataFrame(results["time_trends"]))

        st.subheader("Business Impact")
        st.json(results["business_impact"])

        st.subheader("SHAP Explainability")
        st.dataframe(pd.DataFrame(results["shap_summary"]))


