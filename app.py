import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("tuned_xgboost_model.joblib")
FEATURES = [
    "Sex", "Weight", "Height", "Age at Reporting",
    "Differentiated Care Model", "AHD Client", "Risk Categorization",
    "Current Regimen Line", "Months Of Prescription",
    "TB screening at last visit", "Marital status", "Employmet",
    "Family Members", "Alcoho and Drug Use",
]
NUMERIC_FEATURES = {
    "Weight": float,
    "Height": float,
    "Age at Reporting": int,
    "Months Of Prescription": int,
}

st.set_page_config(page_title="ML Predictor", page_icon="🤖")
st.title("🤖 Machine-learning model predictor")
st.caption("Enter one record or upload a CSV containing the 14 training columns.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(str(exc))
    st.info("Upload tuned_xgboost_model.joblib to the repository root, next to app.py.")
    st.stop()

# A preprocessing Pipeline must be saved inside the joblib file for text categories
# to work. If only a bare XGBoost model was saved, upload the already-encoded CSV.
def predict(frame: pd.DataFrame):
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    frame = frame[FEATURES].copy()
    return model.predict(frame)

manual, upload = st.tabs(["Manual input", "CSV upload"])

with manual:
    with st.form("prediction_form"):
        values = {}
        left, right = st.columns(2)
        for index, feature in enumerate(FEATURES):
            target = left if index % 2 == 0 else right
            with target:
                if feature in NUMERIC_FEATURES:
                    values[feature] = st.number_input(
                        feature, value=0.0 if NUMERIC_FEATURES[feature] is float else 0,
                        step=0.1 if NUMERIC_FEATURES[feature] is float else 1,
                    )
                else:
                    values[feature] = st.text_input(feature)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            result = predict(pd.DataFrame([values]))
            st.success(f"Prediction: {result[0]}")
        except Exception as exc:
            st.error("Prediction failed")
            st.code(str(exc))
            st.warning(
                "If this mentions strings or dtypes, the joblib file is probably a bare "
                "XGBoost model. Save and deploy the complete preprocessing Pipeline, "
                "not only the final estimator."
            )

with upload:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
        st.dataframe(data.head())
        missing = [column for column in FEATURES if column not in data.columns]
        if missing:
            st.error("CSV is missing: " + ", ".join(missing))
        elif st.button("Predict uploaded rows"):
            try:
                predictions = predict(data)
                output = data.copy()
                output["prediction"] = predictions
                st.dataframe(output)
                st.download_button(
                    "Download predictions",
                    output.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv",
                )
            except Exception as exc:
                st.error("Prediction failed")
                st.code(str(exc))
