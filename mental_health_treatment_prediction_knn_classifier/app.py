import streamlit as st
import pandas as pd
import joblib


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1 {
    text-align: center;
    color: #00c2cb;
    font-size: 42px;
}

p {
    color: #cccccc;
    text-align: center;
    font-size: 18px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #161b22;
}

section[data-testid="stSidebar"] h2 {
    color: #00c2cb;
}


/* Labels */

label {
    color: white !important;
    font-weight: 600;
}


/* Input boxes */

div[data-baseweb="select"] > div {
    background-color: #262730;
    color: white;
    border-radius: 10px;
}


input {
    background-color: #262730 !important;
    color: white !important;
}


/* Button */

.stButton button {

    width:100%;
    height:50px;

    background: linear-gradient(
        90deg,
        #00c2cb,
        #0077ff
    );

    color:white;
    font-size:20px;
    font-weight:bold;

    border-radius:12px;
    border:none;
}


.stButton button:hover {

    background: linear-gradient(
        90deg,
        #0077ff,
        #00c2cb
    );

}


/* Result Cards */

.success {

background-color:#064e3b;
padding:25px;
border-radius:15px;
text-align:center;
font-size:22px;
color:white;

}


.warning {

background-color:#7f1d1d;
padding:25px;
border-radius:15px;
text-align:center;
font-size:22px;
color:white;

}


</style>
""", unsafe_allow_html=True)



# ---------------- LOAD MODEL ----------------

model = joblib.load("mental_health_knn.pkl")



# ---------------- TITLE ----------------


st.title("🧠 Mental Health Treatment Prediction")

st.markdown(
"""
AI-powered prediction system using **K-Nearest Neighbors (KNN) Algorithm**  
to analyze workplace and personal factors.
"""
)



st.divider()



# ---------------- SIDEBAR ----------------


with st.sidebar:

    st.header("👤 Personal Details")


    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )


    Gender = st.selectbox(
        "Gender",
        ["Male","Female","Other"]
    )


    Country = st.text_input(
        "Country",
        "United States"
    )



    self_employed = st.selectbox(
        "Self Employed?",
        ["Yes","No"]
    )


    family_history = st.selectbox(
        "Family History of Mental Health Issues?",
        ["Yes","No"]
    )



    st.header("🏢 Workplace Details")


    work_interfere = st.selectbox(
        "Work Interference",
        ["Never","Rarely","Sometimes","Often"]
    )


    no_employees = st.selectbox(
        "Company Size",
        [
            "1-5",
            "6-25",
            "26-100",
            "100-500",
            "500-1000",
            "More than 1000"
        ]
    )


    remote_work = st.selectbox(
        "Remote Work?",
        ["Yes","No"]
    )


    tech_company = st.selectbox(
        "Tech Company?",
        ["Yes","No"]
    )



    st.header("🧠 Mental Health Environment")


    benefits = st.selectbox(
        "Mental Health Benefits?",
        ["Yes","No","Don't know"]
    )


    care_options = st.selectbox(
        "Care Options Available?",
        ["Yes","No","Not sure"]
    )


    wellness_program = st.selectbox(
        "Wellness Program?",
        ["Yes","No","Don't know"]
    )


    seek_help = st.selectbox(
        "Seek Help Resources?",
        ["Yes","No","Don't know"]
    )


    anonymity = st.selectbox(
        "Anonymity Protected?",
        ["Yes","No","Don't know"]
    )


    leave = st.selectbox(
        "Mental Health Leave",
        [
            "Very easy",
            "Somewhat easy",
            "Somewhat difficult",
            "Very difficult",
            "Don't know"
        ]
    )


    mental_health_consequence = st.selectbox(
        "Mental Health Consequences?",
        ["Yes","No","Maybe"]
    )


    phys_health_consequence = st.selectbox(
        "Physical Health Consequences?",
        ["Yes","No","Maybe"]
    )


    coworkers = st.selectbox(
        "Discuss with Coworkers?",
        [
            "Yes",
            "No",
            "Some of them"
        ]
    )


    supervisor = st.selectbox(
        "Discuss with Supervisor?",
        [
            "Yes",
            "No",
            "Some of them"
        ]
    )


    mental_health_interview = st.selectbox(
        "Mental Health Interview?",
        ["Yes","No","Maybe"]
    )


    phys_health_interview = st.selectbox(
        "Physical Health Interview?",
        ["Yes","No","Maybe"]
    )


    mental_vs_physical = st.selectbox(
        "Mental vs Physical Health?",
        ["Yes","No","Don't know"]
    )


    obs_consequence = st.selectbox(
        "Observed Consequence?",
        ["Yes","No"]
    )



# ---------------- PREDICTION ----------------


st.write("")

if st.button("🔍 Predict"):


    input_data = pd.DataFrame({

        "Age":[Age],
        "Gender":[Gender],
        "Country":[Country],
        "self_employed":[self_employed],
        "family_history":[family_history],
        "work_interfere":[work_interfere],
        "no_employees":[no_employees],
        "remote_work":[remote_work],
        "tech_company":[tech_company],
        "benefits":[benefits],
        "care_options":[care_options],
        "wellness_program":[wellness_program],
        "seek_help":[seek_help],
        "anonymity":[anonymity],
        "leave":[leave],
        "mental_health_consequence":[mental_health_consequence],
        "phys_health_consequence":[phys_health_consequence],
        "coworkers":[coworkers],
        "supervisor":[supervisor],
        "mental_health_interview":[mental_health_interview],
        "phys_health_interview":[phys_health_interview],
        "mental_vs_physical":[mental_vs_physical],
        "obs_consequence":[obs_consequence]

    })



    prediction = model.predict(input_data)[0]



    st.divider()



    if prediction == "Yes":

        st.markdown(
        """
        <div class="warning">
        ⚠️ Higher possibility of seeking mental health treatment
        </div>
        """,
        unsafe_allow_html=True
        )


    else:

        st.markdown(
        """
        <div class="success">
        ✅ Lower possibility of seeking mental health treatment
        </div>
        """,
        unsafe_allow_html=True
        )

