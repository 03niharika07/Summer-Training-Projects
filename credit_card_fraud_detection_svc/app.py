import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os


# ---------------- Load Model Files ----------------

model_path = os.path.join(os.path.dirname(__file__), "svc_model.pkl")
model = joblib.load(model_path)

scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler = joblib.load(scaler_path)

feature_path = os.path.join(os.path.dirname(__file__), "features.pkl")
features = joblib.load(feature_path)

# ---------------- Page Configuration ----------------

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
        transaction is Fraud or Normal using 
        Support Vector Classifier.
        """
    )

    st.divider()

    st.subheader("🤖 Model Information")

    st.write(
        """
        **Algorithm:** SVC (Support Vector Classifier)

        **Kernel:** RBF

        **Problem Type:** Binary Classification

        **Classes:**

        0 → Normal Transaction

        1 → Fraud Transaction
        """
    )


    st.divider()

    st.subheader("📊 Model Performance")

    st.metric(
        "Accuracy",
        "98.9%"
    )

    st.metric(
        "Precision",
        "61.54%"
    )

    st.metric(
        "Recall",
        "77.42%"
    )

    st.metric(
        "F1 Score",
        "68.57%"
    )

    st.metric(
        "ROC-AUC",
        "99.22%"
    )


    st.divider()

    st.caption(
        "Built using Python | Scikit-learn | Streamlit"
    )



# ---------------- Main Page ----------------

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter transaction details to check whether the transaction is fraudulent."
)



# ---------------- User Inputs ----------------


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



# ---------------- Prediction ----------------

if st.button("🔍 Predict"):


    # Convert True/False into 0/1

    foreign_transaction = (
        1 if foreign_transaction == "True" else 0
    )


    location_mismatch = (
        1 if location_mismatch == "True" else 0
    )



    # Create input dataframe

    input_data = pd.DataFrame(
        [[
            amount,
            transaction_hour,
            foreign_transaction,
            location_mismatch,
            device_trust_score,
            velocity_last_24h,
            cardholder_age,
            0,
            0,
            0,
            0
        ]],
        columns=features
    )



    # Merchant encoding

    if merchant_category == "Electronics":
        input_data["merchant_category_Electronics"] = 1

    elif merchant_category == "Food":
        input_data["merchant_category_Food"] = 1

    elif merchant_category == "Grocery":
        input_data["merchant_category_Grocery"] = 1

    elif merchant_category == "Travel":
        input_data["merchant_category_Travel"] = 1



    # Scaling

    input_scaled = scaler.transform(
        input_data
    )



    # Prediction

    prediction = model.predict(
        input_scaled
    )



    # Output

    if prediction[0] == 1:

        st.error(
            "⚠️ Fraud Transaction Detected"
        )

    else:

        st.success(
            "✅ Transaction is Normal"
        )