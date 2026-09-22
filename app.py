import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "crop_yield_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found:\n\n{MODEL_PATH}"
        )
        st.stop()

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# =========================================================
# LOAD SCALER
# =========================================================

@st.cache_resource
def load_scaler():

    if not SCALER_PATH.exists():
        st.error(
            f"Scaler file not found:\n\n{SCALER_PATH}"
        )
        st.stop()

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return scaler


model = load_model()
scaler = load_scaler()


# =========================================================
# TITLE
# =========================================================

st.title("🌾 Crop Yield Prediction")

st.write(
    "Enter the required agricultural information below "
    "to predict crop yield using a machine learning model."
)


# =========================================================
# GET FEATURE NAMES
# =========================================================

feature_names = None

# If scaler was trained using a pandas DataFrame,
# sklearn usually stores the feature names here.
if hasattr(scaler, "feature_names_in_"):
    feature_names = list(scaler.feature_names_in_)

# Otherwise try the model
elif hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("🌱 Enter Input Values")


if feature_names is not None:

    input_data = {}

    for feature in feature_names:

        input_data[feature] = st.number_input(
            f"{feature}",
            value=0.0
        )

else:

    # Fallback if feature names were not saved
    if hasattr(model, "n_features_in_"):
        number_of_features = model.n_features_in_

    elif hasattr(scaler, "n_features_in_"):
        number_of_features = scaler.n_features_in_

    else:
        st.error(
            "Unable to determine the number of input features."
        )
        st.stop()

    st.warning(
        "Feature names were not saved with the model/scaler. "
        "Please make sure the inputs are entered in the same "
        "order used during model training."
    )

    input_data = {}

    for i in range(number_of_features):

        input_data[f"Feature {i + 1}"] = st.number_input(
            f"Feature {i + 1}",
            value=0.0
        )


# =========================================================
# PREDICTION
# =========================================================

st.write("")


if st.button("🌾 Predict Crop Yield"):

    try:

        # Create DataFrame
        input_df = pd.DataFrame(
            [input_data]
        )

        # If feature names are available,
        # preserve the exact training order.
        if feature_names is not None:
            input_df = input_df[feature_names]

        # Convert to numeric
        input_df = input_df.astype(float)

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)

        prediction_value = prediction[0]

        # Display result
        st.success("Prediction completed successfully! 🌾")

        st.metric(
            label="Predicted Crop Yield",
            value=f"{prediction_value:.2f}"
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 About")

    st.write(
        "This application uses a trained machine learning "
        "regression model to predict crop yield."
    )

    st.write("### 🛠️ Technologies")

    st.write(
        """
        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - Machine Learning
        """
    )