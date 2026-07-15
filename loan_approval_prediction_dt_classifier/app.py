import streamlit as st
import pandas as pd
import joblib
import os


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)


# ---------------- LOAD MODEL ----------------
model_path = os.path.join(os.path.dirname(__file__), "classifier.pkl")
model = joblib.load(model_path)


# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center; color:#1f4e79;'>
    🏦 Loan Approval Prediction
    </h1>
    <p style='text-align:center;'>
    Predict whether your loan will be approved or not
    </p>
    """,
    unsafe_allow_html=True
)


st.divider()


# ---------------- USER INPUT ----------------

city = st.selectbox(
    "Select City",
    ["Delhi", "Mumbai", "Jaipur", "Bangalore", "Chennai"]
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=700
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=100000
)

years_employed = st.number_input(
    "Years Employed",
    min_value=0,
    value=5
)

points = st.number_input(
    "Points",
    min_value=0,
    value=50
)


# ---------------- PREDICTION ----------------

if st.button("Predict Loan Status"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "income": [income],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "years_employed": [years_employed],
        "points": [points],
        "city": [city]
    })


    # Same encoding as training
    input_data = pd.get_dummies(
        input_data,
        columns=["city"],
        drop_first=True
    )


    # Match training columns
    model_columns = model.feature_names_in_

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )


    prediction = model.predict(input_data)


    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")