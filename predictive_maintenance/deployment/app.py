import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ── Load Model from Hugging Face Model Hub ─────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="nikhileshmehta1989/predictive_maintenance_vehicle",
        filename="best_model.pkl",
        token=os.getenv("HF_TOKEN"),
    )
    return joblib.load(model_path)

model = load_model()

# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.title("Predictive Maintenance - Engine Condition Predictor")
st.markdown("Enter engine sensor readings to predict the engine condition.")

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input("Engine rpm", min_value=0, max_value=3000, value=750)
    lub_oil_pressure = st.number_input("Lub oil pressure", min_value=0.0, max_value=10.0, value=3.16, format="%.4f")
    fuel_pressure = st.number_input("Fuel pressure", min_value=0.0, max_value=25.0, value=6.20, format="%.4f")

with col2:
    coolant_pressure = st.number_input("Coolant pressure", min_value=0.0, max_value=10.0, value=2.17, format="%.4f")
    lub_oil_temp = st.number_input("lub oil temp", min_value=50.0, max_value=120.0, value=76.82, format="%.4f")
    coolant_temp = st.number_input("Coolant temp", min_value=50.0, max_value=220.0, value=78.35, format="%.4f")

# ── Build Input DataFrame ──────────────────────────────────────────────────────

input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp,
}])

st.subheader("Input Preview")
st.dataframe(input_data)

# ── Predict ────────────────────────────────────────────────────────────────────


if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][int(prediction)]
        confidence_text = f" Confidence: {probability:.1%}"
    else:
        confidence_text = ""

    if prediction == 1:
        st.success(f"Engine condition is predicted as GOOD / NORMAL.{confidence_text}")
    else:
        st.error(f"Engine condition is predicted as BAD / MAINTENANCE REQUIRED.{confidence_text}")
