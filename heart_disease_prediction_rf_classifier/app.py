import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# REMOVE DEFAULT STREAMLIT MARGIN 

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        h1 {
            text-align: center;
            color: #d6336c;
        }

        .stButton>button {
            width: 100%;
            height: 3em;
            border-radius: 10px;
            font-size: 18px;
            background-color: #d6336c;
            color: white;
        }

        .prediction-box {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# Load Model

path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
model = joblib.load(path)


# TITLE

st.markdown(
    """
    # ❤️ Heart Disease Prediction
    """
)


st.write(
    "Enter patient details below to predict the possibility of heart disease."
)


# INPUTS


Age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=40
)


Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


Cholesterol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=400,
    value=200
)


Blood_Pressure = st.number_input(
    "Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)


Heart_Rate = st.number_input(
    "Heart Rate",
    min_value=40,
    max_value=200,
    value=80
)


Smoking = st.selectbox(
    "Smoking",
    ["Yes", "No"]
)


Alcohol_Intake = st.selectbox(
    "Alcohol Intake",
    ["Low", "Medium", "High"]
)


Exercise_Hours = st.number_input(
    "Exercise Hours",
    min_value=0.0,
    value=3.0
)


Family_History = st.selectbox(
    "Family History",
    ["Yes", "No"]
)


Diabetes = st.selectbox(
    "Diabetes",
    ["Yes", "No"]
)


Obesity = st.selectbox(
    "Obesity",
    ["Yes", "No"]
)


Stress_Level = st.number_input(
    "Stress Level",
    min_value=0,
    max_value=10,
    value=5
)


Blood_Sugar = st.number_input(
    "Blood Sugar",
    min_value=50,
    max_value=400,
    value=120
)


Exercise_Induced_Angina = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)


Chest_Pain_Type = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina",
     "Atypical Angina",
     "Non-anginal Pain",
     "Asymptomatic"]
)



# CREATE DATAFRAME 


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


# PREDICTION


if st.button("Predict Heart Disease"):

    prediction = model.predict(input_data)


    if prediction[0] == 1:

        st.markdown(
            """
            <div class="prediction-box">
            ⚠️ High Risk of Heart Disease
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="prediction-box">
            ✅ Low Risk of Heart Disease
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("---")

