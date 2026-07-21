import streamlit as st
import numpy as np
import joblib
import os


# Load Model and Scaler
path = os.path.join(os.path.dirname(__file__), "svc_model.pkl")
model = joblib.load(path)
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler = joblib.load(path)


# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ---------------- Sidebar ----------------

with st.sidebar:

    st.title("💳 Fraud Detection")

    st.write(
        """
        ### About Project
        
        This application predicts whether a credit card 
        transaction is fraudulent or normal using SVC.
        """
    )

    st.divider()

    st.subheader("🤖 Model Information")

    st.write("""
    **Algorithm:**  
    Support Vector Classifier (SVC)

    **Type:**  
    Binary Classification

    **Classes:**
    - 0 → Normal Transaction
    - 1 → Fraud Transaction
    """)

    st.divider()

    st.subheader("📊 Model Performance")

    st.metric(
        label="Accuracy",
        value="98.9%"
    )

    st.metric(
        label="Precision",
        value="61.54%"
    )

    st.metric(
        label="Recall",
        value="77.42%"
    )

    st.metric(
        label="F1 Score",
        value="68.57%"
    )

    st.metric(
        label="ROC-AUC Score",
        value="99.22%"
    )


    st.divider()


# ---------------- Main Page ----------------

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter transaction details below to check whether the transaction is fraudulent or not."
)


# Input Features

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    step=100.0
)


transaction_hour = st.number_input(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    step=1
)


foreign_transaction = st.selectbox(
    "Foreign Transaction",
    ["False", "True"]
)


location_mismatch = st.selectbox(
    "Location Mismatch",
    ["False", "True"]
)


device_trust_score = st.number_input(
    "Device Trust Score",
    min_value=0.0,
    max_value=100.0
)


velocity_last_24h = st.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    step=1
)


cardholder_age = st.number_input(
    "Cardholder Age",
    min_value=18,
    max_value=100,
    step=1
)


merchant_category = st.selectbox(
    "Merchant Category",
    [
        "Electronics",
        "Food",
        "Grocery",
        "Travel"
    ]
)


# Merchant Encoding

merchant_category_Electronics = 0
merchant_category_Food = 0
merchant_category_Grocery = 0
merchant_category_Travel = 0


if merchant_category == "Electronics":
    merchant_category_Electronics = 1

elif merchant_category == "Food":
    merchant_category_Food = 1

elif merchant_category == "Grocery":
    merchant_category_Grocery = 1

elif merchant_category == "Travel":
    merchant_category_Travel = 1



# Prediction

if st.button("🔍 Predict"):


    # Convert True/False into 0/1

    foreign_transaction = 1 if foreign_transaction == "True" else 0

    location_mismatch = 1 if location_mismatch == "True" else 0



    input_data = np.array([[
        amount,
        transaction_hour,
        foreign_transaction,
        location_mismatch,
        device_trust_score,
        velocity_last_24h,
        cardholder_age,
        merchant_category_Electronics,
        merchant_category_Food,
        merchant_category_Grocery,
        merchant_category_Travel
    ]])


    # Scaling

    input_scaled = scaler.transform(input_data)


    # Prediction

    prediction = model.predict(input_scaled)



    if prediction[0] == 1:

        st.error(
            "⚠️ Fraud Transaction Detected"
        )

    else:

        st.success(
            "✅ Transaction is Normal"
        )