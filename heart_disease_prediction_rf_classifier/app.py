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


# ================= CUSTOM CSS =================

st.markdown(
    """
    <style>

    /* Remove unnecessary Streamlit spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* Background */
    .stApp {
        background: linear-gradient(
            135deg,
            #fff5f5,
            #f1f8ff
        );
    }


    /* Main heading */
    .title {

        text-align:center;
        font-size:45px;
        font-weight:800;
        color:#b30000 !important;
        margin-bottom:10px;

    }


    /* Subtitle */
    .subtitle {

        text-align:center;
        font-size:20px;
        color:#444 !important;
        margin-bottom:35px;

    }



    /* Cards */

    .card {

        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0px 5px 20px rgba(0,0,0,0.10);
        margin-bottom:20px;

    }



    .section {

        font-size:25px;
        font-weight:700;
        color:#b30000 !important;

    }



    /* Button */

    div.stButton > button {

        width:100%;
        height:3.2em;
        background:#b30000;
        color:white;
        font-size:20px;
        font-weight:bold;
        border-radius:15px;
        border:none;

    }


    div.stButton > button:hover {

        background:#800000;
        color:white;

    }



    /* Prediction boxes */

    .danger {

        background:#ffd6d6;
        color:#8b0000;
        padding:25px;
        border-radius:20px;
        text-align:center;
        font-size:26px;
        font-weight:bold;

    }



    .success {

        background:#d8f3dc;
        color:#1b4332;
        padding:25px;
        border-radius:20px;
        text-align:center;
        font-size:26px;
        font-weight:bold;

    }



    /* Sidebar */

    section[data-testid="stSidebar"] {

        background:#fff5f5;

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

    st.title("❤️ About Project")

    st.write(
        """
        ### Heart Disease Prediction

        Machine Learning Model:
        **Random Forest Classifier**

        Features:
        - Health Parameters
        - Lifestyle Factors
        - Medical History

        Built Using:

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
        color:#b30000 !important;
        font-size:45px;
        font-weight:800;
        margin-top:20px;
        margin-bottom:10px;">
        ❤️ Heart Disease Prediction
    </h1>


    <p style="
        text-align:center;
        color:#333333 !important;
        font-size:20px;
        margin-bottom:40px;">
        AI-based healthcare risk prediction using Random Forest Classifier
    </p>
    """,
    unsafe_allow_html=True
)


# ================= PERSONAL DETAILS =================

st.markdown(
    """
    <div class="card">
    <div class="section">
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



# ================= MEDICAL DETAILS =================


st.markdown(
    """
    <div class="card">
    <div class="section">
    🩺 Lifestyle & Medical History
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



# ================= PREDICTION =================


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

            Confidence: {confidence:.2f}%

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

            Confidence: {confidence:.2f}%

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







