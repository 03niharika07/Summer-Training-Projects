import streamlit as st
import numpy as np
import os
import joblib


# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), "logr.pkl")
model = joblib.load(model_path)

scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler = joblib.load(scaler_path)


# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# to remove default padding of streamlit
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🩺 Breast Cancer Prediction")
st.write(
    "Predict whether a tumor is Benign or Malignant using Logistic Regression"
)

st.divider()


# Feature ranges from dataset
feature_ranges = {

    # Mean Features
    "radius_mean": "6.98 - 28.11",
    "texture_mean": "9.71 - 39.28",
    "perimeter_mean": "43.79 - 188.5",
    "area_mean": "143.5 - 2501",
    "smoothness_mean": "0.052 - 0.163",
    "compactness_mean": "0.019 - 0.345",
    "concavity_mean": "0.0 - 0.427",
    "concave points_mean": "0.0 - 0.201",
    "symmetry_mean": "0.106 - 0.304",
    "fractal_dimension_mean": "0.05 - 0.097",


    # SE Features
    "radius_se": "0.112 - 2.873",
    "texture_se": "0.36 - 4.885",
    "perimeter_se": "0.757 - 21.98",
    "area_se": "6.802 - 542.2",
    "smoothness_se": "0.001 - 0.031",
    "compactness_se": "0.002 - 0.135",
    "concavity_se": "0.0 - 0.396",
    "concave points_se": "0.0 - 0.053",
    "symmetry_se": "0.008 - 0.079",
    "fractal_dimension_se": "0.001 - 0.03",


    # Worst Features
    "radius_worst": "7.93 - 36.04",
    "texture_worst": "12.02 - 49.54",
    "perimeter_worst": "50.41 - 251.2",
    "area_worst": "185.2 - 4254",
    "smoothness_worst": "0.071 - 0.223",
    "compactness_worst": "0.027 - 1.058",
    "concavity_worst": "0.0 - 1.252",
    "concave points_worst": "0.0 - 0.291",
    "symmetry_worst": "0.156 - 0.664",
    "fractal_dimension_worst": "0.055 - 0.208"

}


# Features in same order as training
features = list(feature_ranges.keys())


input_values = []


# Function for input box

def create_input(feature):

    st.write(f"**{feature}**")
    st.caption(
        f"Allowed range: {feature_ranges[feature]}"
    )

    value = st.text_input(
        label="",
        placeholder="Enter value",
        key=feature
    )

    return value



# Mean Section
st.subheader("📌 Mean Measurements")

col1, col2, col3 = st.columns(3)

mean_features = features[:10]

for i, feature in enumerate(mean_features):

    with [col1, col2, col3][i % 3]:
        input_values.append(create_input(feature))



st.divider()



# SE Section
st.subheader("📊 Standard Error Measurements")

col1, col2, col3 = st.columns(3)

se_features = features[10:20]

for i, feature in enumerate(se_features):

    with [col1, col2, col3][i % 3]:
        input_values.append(create_input(feature))



st.divider()



# Worst Section
st.subheader("⚠️ Worst Measurements")

col1, col2, col3 = st.columns(3)

worst_features = features[20:30]

for i, feature in enumerate(worst_features):

    with [col1, col2, col3][i % 3]:
        input_values.append(create_input(feature))



st.divider()



# Prediction Button

if st.button("🔍 Predict"):

    try:

        data = np.array(
            [float(x) for x in input_values]
        ).reshape(1,-1)


        # Scaling
        data_scaled = scaler.transform(data)


        # Prediction
        prediction = model.predict(data_scaled)

        probability = model.predict_proba(data_scaled)


        if prediction[0] == 0:

            st.error("⚠️ Malignant Tumor Detected")

            st.write(
                f"Malignant Probability: {probability[0][0]*100:.2f}%"
            )


        else:

            st.success("✅ Benign Tumor Detected")

            st.write(
                f"Benign Probability: {probability[0][1]*100:.2f}%"
            )


    except:

        st.warning(
            "Please enter all feature values correctly."
        )