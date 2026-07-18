import streamlit as st
import numpy as np
import joblib


# Load model and scaler
model = joblib.load("logr.pkl")
scaler = joblib.load("scaler.pkl")


# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)


st.title("🩺 Breast Cancer Prediction")
st.write(
    "This application predicts whether a tumor is Benign or Malignant "
    "using Logistic Regression."
)


st.divider()


# Feature names (same order as training data)
features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'fractal_dimension_mean',

    'radius_se',
    'texture_se',
    'perimeter_se',
    'area_se',
    'smoothness_se',
    'compactness_se',
    'concavity_se',
    'concave points_se',
    'symmetry_se',
    'fractal_dimension_se',

    'radius_worst',
    'texture_worst',
    'perimeter_worst',
    'area_worst',
    'smoothness_worst',
    'compactness_worst',
    'concavity_worst',
    'concave points_worst',
    'symmetry_worst',
    'fractal_dimension_worst'
]


st.subheader("Enter Tumor Measurements")


input_values = []


col1, col2, col3 = st.columns(3)


for i, feature in enumerate(features):

    if i % 3 == 0:
        with col1:
            value = st.number_input(feature, value=0.0)
    elif i % 3 == 1:
        with col2:
            value = st.number_input(feature, value=0.0)
    else:
        with col3:
            value = st.number_input(feature, value=0.0)

    input_values.append(value)



st.divider()


if st.button("🔍 Predict"):

    # Convert input into array
    input_data = np.array(input_values).reshape(1, -1)


    # Scaling input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)


    probability = model.predict_proba(input_scaled)


    if prediction[0] == 0:
        st.error("⚠️ Malignant Tumor Detected")
        st.write(
            f"Probability of Malignant: {probability[0][0]*100:.2f}%"
        )

    else:
        st.success("✅ Benign Tumor Detected")
        st.write(
            f"Probability of Benign: {probability[0][1]*100:.2f}%"
        )