import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)


# -----------------------------
# Load Model and Scaler
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

with open(MODEL_DIR / "crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# -----------------------------
# Title
# -----------------------------
st.title("🌾 Crop Yield Prediction")

st.write(
    "Enter the agricultural and environmental details below "
    "to predict millet yield."
)


# -----------------------------
# User Inputs
# -----------------------------
Moisture = st.number_input(
    "Moisture",
    min_value=0.0,
    value=12.9
)

rainfall = st.number_input(
    "Rainfall",
    min_value=0.0,
    value=0.07
)

Average_Humidity = st.number_input(
    "Average Humidity",
    min_value=0,
    max_value=100,
    value=50
)

Mean_Temp = st.number_input(
    "Mean Temperature",
    value=76
)

max_Temp = st.number_input(
    "Maximum Temperature",
    value=89
)

Min_temp = st.number_input(
    "Minimum Temperature",
    value=64
)

alkaline = st.selectbox(
    "Alkaline Soil",
    [0, 1]
)

sandy = st.selectbox(
    "Sandy Soil",
    [0, 1]
)

chalky = st.selectbox(
    "Chalky Soil",
    [0, 1]
)

clay = st.selectbox(
    "Clay Soil",
    [0, 1]
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Yield"):

    input_data = pd.DataFrame({
        "Moisture": [Moisture],
        "rainfall": [rainfall],
        "Average Humidity": [Average_Humidity],
        "Mean Temp": [Mean_Temp],
        "max Temp": [max_Temp],
        "Min temp": [Min_temp],
        "alkaline": [alkaline],
        "sandy": [sandy],
        "chalky": [chalky],
        "clay": [clay]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    predicted_yield = prediction[0]

    st.success(
        f"🌾 Predicted Millet Yield: {predicted_yield:.2f}"
    )