import streamlit as st
import numpy as np
import joblib


# Load model and scaler
model = joblib.load("svc_model.pkl")
scaler = joblib.load("scaler.pkl")


# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# Title
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check whether the transaction is fraudulent or not.")


# Input fields

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0
)

transaction_hour = st.number_input(
    "Transaction Hour",
    min_value=0,
    max_value=23
)

foreign_transaction = st.selectbox(
    "Foreign Transaction",
    ['False', 'True']
)

location_mismatch = st.selectbox(
    "Location Mismatch",
    ['False', 'True']
)

device_trust_score = st.number_input(
    "Device Trust Score",
    min_value=0.0
)

velocity_last_24h = st.number_input(
    "Number of Transactions in Last 24 Hours",
    min_value=0
)

cardholder_age = st.number_input(
    "Cardholder Age",
    min_value=18,
    max_value=100
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


# Create merchant category encoding

merchant_Electronics = 0
merchant_Food = 0
merchant_Grocery = 0
merchant_Travel = 0


if merchant_category == "Electronics":
    merchant_Electronics = 1

elif merchant_category == "Food":
    merchant_Food = 1

elif merchant_category == "Grocery":
    merchant_Grocery = 1

elif merchant_category == "Travel":
    merchant_Travel = 1



# Prediction button

if st.button("Predict"):

    input_data = np.array([[
        amount,
        transaction_hour,
        foreign_transaction,
        location_mismatch,
        device_trust_score,
        velocity_last_24h,
        cardholder_age,
        merchant_Electronics,
        merchant_Food,
        merchant_Grocery,
        merchant_Travel
    ]])


    # Scaling
    input_scaled = scaler.transform(input_data)


    prediction = model.predict(input_scaled)


    if prediction[0] == 1:
        st.error("⚠️ Fraud Transaction Detected")

    else:
        st.success("✅ Transaction is Normal")