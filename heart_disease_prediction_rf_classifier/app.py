import streamlit as st
import pandas as pd
import joblib
import os


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# ================= CSS =================

st.markdown(
    """
    <style>

    /* Page background */
    .stApp {
        background: #f7f9fc;
    }


    /* Remove excess top gap */
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* Input cards */
    .card {

        background:white;
        padding:25px;
        border-radius:18px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.08);
        margin-bottom:20px;

    }


    /* Button */

    div.stButton > button {

        width:100%;
        height:3.2rem;
        background:#d90429;
        color:white;
        font-size:20px;
        font-weight:bold;
        border-radius:12px;

    }


    div.stButton > button:hover {

        background:#9d0208;
        color:white;

    }


    /* Prediction */

    .danger {

        background:#ffe5e5;
        color:#b00020;
        padding:25px;
        border-radius:18px;
        text-align:center;
        font-size:25px;
        font-weight:bold;

    }


    .success {

        background:#e0f7e9;
        color:#1b4332;
        padding:25px;
        border-radius:18px;
        text-align:center;
        font-size:25px;
        font-weight:bold;

    }


    </style>

    """,
    unsafe_allow_html=True
)



# ================= LOAD MODEL =================

path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
model = joblib.load(path)


# ================= SIDEBAR =================

with st.sidebar:

    st.markdown(
        """
        <h2 style="color:#d90429;">
        ❤️ About Project
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        **Heart Disease Prediction**

        Algorithm:
        Random Forest Classifier

        Technologies:

        🐍 Python  
        📊 Pandas  
        🤖 Scikit-learn  
        🎨 Streamlit
        """
    )



# ================= HEADER =================


st.markdown(
    """
    <h1 style="
    text-align:center;
    color:#d90429 !important;
    font-size:45px;
    font-weight:800;
    margin-bottom:5px;">
    ❤️ Heart Disease Prediction
    </h1>


    <p style="
    text-align:center;
    color:#333333 !important;
    font-size:20px;">
    AI based healthcare risk prediction using Random Forest Classifier
    </p>

    <br>

    """,

    unsafe_allow_html=True
)



# ================= PATIENT DETAILS =================


st.markdown(
    """
    <div class="card">

    <h2 style="color:#d90429;">
    👤 Patient Information
    </h2>

    </div>
    """,
    unsafe_allow_html=True
)



col1, col2 = st.columns(2)


with col1:

    Age = st.number_input(
        "Age",
        1,
        100,
        40
    )


    Gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )


    Cholesterol = st.number_input(
        "Cholesterol",
        50,
        400,
        200
    )


    Blood_Pressure = st.number_input(
        "Blood Pressure",
        50,
        250,
        120
    )



with col2:

    Heart_Rate = st.number_input(
        "Heart Rate",
        40,
        200,
        80
    )


    Blood_Sugar = st.number_input(
        "Blood Sugar",
        50,
        400,
        120
    )


    Exercise_Hours = st.number_input(
        "Exercise Hours",
        0,
        20,
        3
    )


    Stress_Level = st.number_input(
        "Stress Level",
        0,
        10,
        5
    )



# ================= LIFESTYLE =================


st.markdown(
    """
    <div class="card">

    <h2 style="color:#d90429;">
    🩺 Lifestyle & Medical History
    </h2>

    </div>
    """,
    unsafe_allow_html=True
)



col3, col4 = st.columns(2)



with col3:


    Smoking = st.selectbox(
        "Smoking",
        ["Yes","No"]
    )


    Alcohol_Intake = st.selectbox(
        "Alcohol Intake",
        ["Low","Medium","High"]
    )


    Family_History = st.selectbox(
        "Family History",
        ["Yes","No"]
    )


    Diabetes = st.selectbox(
        "Diabetes",
        ["Yes","No"]
    )



with col4:


    Obesity = st.selectbox(
        "Obesity",
        ["Yes","No"]
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



# ================= INPUT DATA =================


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



# ================= PREDICT =================


st.write("")


if st.button("🔍 Predict Heart Disease Risk"):


    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    confidence = max(probability[0])*100



    if prediction[0] == 1:

        st.markdown(
            f"""
            <div class="danger">

            ⚠️ High Risk of Heart Disease

            <br><br>

            Confidence : {confidence:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            f"""
            <div class="success">

            ✅ Low Risk of Heart Disease

            <br><br>

            Confidence : {confidence:.2f}%

            </div>
            """,
            unsafe_allow_html=True
        )



st.markdown(
    """
    <br>
    <center>
    Built with ❤️ using Machine Learning & Streamlit
    </center>
    """,
    unsafe_allow_html=True
)



