
import streamlit as st
import pandas as pd
import joblib
import os


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    .stApp {
        background: linear-gradient(
            135deg,
            #fff5f5,
            #f0f7ff
        );
    }


    .title {
        text-align:center;
        font-size:45px;
        font-weight:800;
        color:#c1121f;
    }


    .subtitle {
        text-align:center;
        font-size:20px;
        color:#555;
        margin-bottom:30px;
    }


    .card {

        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0px 5px 20px rgba(0,0,0,0.08);
        margin-bottom:20px;

    }


    .section-title {

        font-size:25px;
        font-weight:700;
        color:#c1121f;

    }


    div.stButton > button {

        width:100%;
        height:3em;
        background:#c1121f;
        color:white;
        font-size:20px;
        font-weight:bold;
        border-radius:15px;
        border:none;

    }


    div.stButton > button:hover {

        background:#780000;
        color:white;

    }


    .danger {

        background:#ffd6d6;
        padding:25px;
        border-radius:20px;
        text-align:center;
        font-size:25px;
        font-weight:bold;
        color:#b00020;

    }


    .success {

        background:#d8f3dc;
        padding:25px;
        border-radius:20px;
        text-align:center;
        font-size:25px;
        font-weight:bold;
        color:#1b4332;

    }


    </style>

    """,
    unsafe_allow_html=True
)



# ---------------- LOAD MODEL ----------------

path = os.path.join(os.path.dirname(__file__), "heart_disease_model.pkl")
model = joblib.load(path)



# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3004/3004458.png",
        width=120
    )

    st.title("❤️ About Project")

    st.write(
        """
        **Heart Disease Prediction System**

        Machine Learning model:
        Random Forest Classifier

        Features:
        - Patient Health Parameters
        - Lifestyle Factors
        - Medical History

        Built using:
        - Python
        - Scikit-learn
        - Streamlit
        """
    )



# ---------------- HEADER ----------------

st.markdown(
    """
    <div class="title">
    ❤️ Heart Disease Prediction
    </div>

    <div class="subtitle">
    AI-powered healthcare risk assessment using Random Forest Classifier
    </div>

    """,
    unsafe_allow_html=True
)



# ---------------- INPUT SECTION ----------------


st.markdown(
    """
    <div class="card">

    <div class="section-title">
    👤 Patient Information
    </div>

    </div>
    """,
    unsafe_allow_html=True
)



col1, col2 = st.columns(2)


with col1:

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


with col2:


    Heart_Rate = st.number_input(
        "Heart Rate",
        value=80
    )


    Blood_Sugar = st.number_input(
        "Blood Sugar",
        value=120
    )


    Exercise_Hours = st.number_input(
        "Exercise Hours",
        value=3
    )



    Stress_Level = st.number_input(
        "Stress Level",
        value=5
    )



# ---------------- LIFESTYLE ----------------


st.markdown(
    """
    <div class="card">

    <div class="section-title">
    🏃 Lifestyle & Medical History
    </div>

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


st.write("")


if st.button("🔍 Predict Heart Disease Risk"):


    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)


    confidence = max(probability[0])*100



    st.write("")


    if prediction[0] == 1:


        st.markdown(
            f"""
            <div class="danger">

            ⚠️ High Risk of Heart Disease

            <br><br>

            Model Confidence: {confidence:.2f}%

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

            Model Confidence: {confidence:.2f}%

            </div>

            """,
            unsafe_allow_html=True
        )



st.markdown(
    """
    ---
    <center>
    Built with ❤️ using Machine Learning & Streamlit
    </center>
    """,
    unsafe_allow_html=True
)



