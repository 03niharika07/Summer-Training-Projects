import streamlit as st
import pandas as pd
import joblib


# Load Model
model = joblib.load("mental_health_knn.pkl")


# Page Configuration
st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="centered"
)


# Title
st.title("🧠 Mental Health Treatment Prediction")
st.write(
    "This application predicts whether a person is likely to seek mental health treatment based on workplace and personal factors."
)


st.divider()


# User Inputs

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

Country = st.text_input(
    "Country",
    "United States"
)

self_employed = st.selectbox(
    "Are you self employed?",
    ["Yes", "No"]
)

family_history = st.selectbox(
    "Family history of mental health issues?",
    ["Yes", "No"]
)

work_interfere = st.selectbox(
    "Does mental health interfere with work?",
    ["Never", "Rarely", "Sometimes", "Often"]
)

no_employees = st.selectbox(
    "Number of employees in company",
    ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
)

remote_work = st.selectbox(
    "Do you work remotely?",
    ["Yes", "No"]
)

tech_company = st.selectbox(
    "Is it a tech company?",
    ["Yes", "No"]
)

benefits = st.selectbox(
    "Does company provide mental health benefits?",
    ["Yes", "No", "Don't know"]
)

care_options = st.selectbox(
    "Are mental health care options available?",
    ["Yes", "No", "Not sure"]
)

wellness_program = st.selectbox(
    "Is there a wellness program?",
    ["Yes", "No", "Don't know"]
)

seek_help = st.selectbox(
    "Does company provide help resources?",
    ["Yes", "No", "Don't know"]
)

anonymity = st.selectbox(
    "Is anonymity protected?",
    ["Yes", "No", "Don't know"]
)

leave = st.selectbox(
    "Is mental health leave available?",
    ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"]
)

mental_health_consequence = st.selectbox(
    "Mental health consequence at workplace?",
    ["Yes", "No", "Maybe"]
)

phys_health_consequence = st.selectbox(
    "Physical health consequence at workplace?",
    ["Yes", "No", "Maybe"]
)

coworkers = st.selectbox(
    "Discuss mental health with coworkers?",
    ["Yes", "No", "Some of them"]
)

supervisor = st.selectbox(
    "Discuss mental health with supervisor?",
    ["Yes", "No", "Some of them"]
)

mental_health_interview = st.selectbox(
    "Asked about mental health in interview?",
    ["Yes", "No", "Maybe"]
)

phys_health_interview = st.selectbox(
    "Asked about physical health in interview?",
    ["Yes", "No", "Maybe"]
)

mental_vs_physical = st.selectbox(
    "Mental health treated as seriously as physical health?",
    ["Yes", "No", "Don't know"]
)

obs_consequence = st.selectbox(
    "Observed negative consequences?",
    ["Yes", "No"]
)


# Prediction Button

if st.button("Predict"):

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
        st.error(
            "⚠️ The person may be likely to seek mental health treatment."
        )
    else:
        st.success(
            "✅ The person may not be seeking mental health treatment."
        )