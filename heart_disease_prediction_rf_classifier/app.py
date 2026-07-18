import streamlit as st
import pandas as pd
import joblib
import os


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>

    /* Medical background */
    [data-testid="stAppViewContainer"] {

        background: linear-gradient(
            135deg,
            #eaf6ff 0%,
            #ffffff 45%,
            #ffeef0 100%
        );

    }


    /* Main content spacing */
    .block-container {

        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;

    }


    /* Heading */
    h1 {

        text-align:center;
        color:#0b5394 !important;
        font-size:45px;
        font-weight:800;

    }


    /* Normal text */
    p {

        color:#333333 !important;

    }


    /* Input labels */
    label {

        color:#0b5394 !important;
        font-weight:600;

    }


    /* Cards */
    .card {

        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0px 5px 18px rgba(0,0,0,0.08);
        border-left:6px solid #0b5394;

    }


    /* Button */

    .stButton > button {

        width:100%;
        height:3.2rem;
        background:#0b5394;
        color:white;
        font-size:20px;
        font-weight:bold;
        border-radius:15px;
        border:none;

    }


    .stButton > button:hover {

        background:#063970;
        color:white;

    }


    /* Success result */

    .success {

        background:#e6ffed;
        color:#176b3a;
        padding:25px;
        border-radius:20px;
        text-align:center;
        font-size:25px;
        font-weight:bold;

    }


    /* Danger result */

    .danger {

        background:#ffe6e6;
        color:#b00020;
        padding:25px;
        border-radius:20px;
        text-align:center;
        font-size:25px;
        font-weight:bold;

    }


    /* Sidebar */

    section[data-testid="stSidebar"] {

        background:#f0f8ff;

    }


    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- LOAD MODEL ----------------

path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
model = joblib.load(path)



# ---------------- TITLE ----------------


st.title("❤️ Heart Disease Prediction")


st.write(
    "Enter patient details below to predict the possibility of heart disease."
)



# ---------------- INPUTS ----------------


Age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=40
)


Gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)


Cholesterol = st.number_input(
    "Cholesterol",
    value=200
)


Blood_Pressure = st.number_input(
    "Blood Pressure",
    value=120
)


Heart_Rate = st.number_input(
    "Heart Rate",
    value=80
)


Smoking = st.selectbox(
    "Smoking",
    ["Yes","No"]
)


Alcohol_Intake = st.selectbox(
    "Alcohol Intake",
    ["Low","Medium","High"]
)


Exercise_Hours = st.number_input(
    "Exercise Hours",
    value=3
)


Family_History = st.selectbox(
    "Family History",
    ["Yes","No"]
)


Diabetes = st.selectbox(
    "Diabetes",
    ["Yes","No"]
)


Obesity = st.selectbox(
    "Obesity",
    ["Yes","No"]
)


Stress_Level = st.number_input(
    "Stress Level",
    value=5
)


Blood_Sugar = st.number_input(
    "Blood Sugar",
    value=120
)


Exercise_Induced_Angina = st.selectbox(
    "Exercise Induced Angina",
    ["Yes","No"]
)


Chest_Pain_Type = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)



# ---------------- DATAFRAME ----------------


input_data = pd.DataFrame({

    "Age":[Age],
    "Gender":[Gender],
    "Cholesterol":[Cholesterol],
    "Blood Pressure":[Blood_Pressure],
    "Heart Rate":[Heart_Rate],
    "Smoking":[Smoking],
    "Alcohol Intake":[Alcohol_Intake],
    "Exercise Hours":[Exercise_Hours],
    "Family History":[Family_History],
    "Diabetes":[Diabetes],
    "Obesity":[Obesity],
    "Stress Level":[Stress_Level],
    "Blood Sugar":[Blood_Sugar],
    "Exercise Induced Angina":[Exercise_Induced_Angina],
    "Chest Pain Type":[Chest_Pain_Type]

})



# ---------------- PREDICTION ----------------


if st.button("Predict Heart Disease"):


    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    confidence = max(probability[0])*100



    if prediction[0] == 1:

        st.markdown(
            f"""
            <div class="prediction-box"
            style="background:#ffcccc;color:#990000">

            ⚠️ High Risk of Heart Disease

            <br><br>

            Confidence: {confidence:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            f"""
            <div class="prediction-box"
            style="background:#d8f3dc;color:#1b4332">

            ✅ Low Risk of Heart Disease

            <br><br>

            Confidence: {confidence:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )



st.markdown("---")




