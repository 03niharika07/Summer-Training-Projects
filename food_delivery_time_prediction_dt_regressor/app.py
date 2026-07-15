import streamlit as st
import pandas as pd
import joblib


# Load Model
model = joblib.load("model.pkl")


# Page Configuration
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🚚"
)



st.title("🚚 Food Delivery Time Prediction")
st.write("Predict Delivery Time : ")


# User Inputs

distance = st.number_input(
    "Distance (km)",
    min_value=0.0
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Cloudy", "Rainy"]
)

traffic = st.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

time = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

vehicle = st.selectbox(
    "Vehicle Type",
    ["Bike", "Car", "Scooter"]
)

prep_time = st.number_input(
    "Preparation Time (min)",
    min_value=0
)

experience = st.number_input(
    "Courier Experience (years)",
    min_value=0
)


# Encoding same as training

weather_map = {
    "Clear": 0,
    "Cloudy": 1,
    "Rainy": 2
}

traffic_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

time_map = {
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Night": 3
}

vehicle_map = {
    "Bike": 0,
    "Car": 1,
    "Scooter": 2
}



if st.button("Predict Delivery Time"):

    input_data = pd.DataFrame({

        "Distance_km": [distance],

        "Weather": [
            weather_map[weather]
        ],

        "Traffic_Level": [
            traffic_map[traffic]
        ],

        "Time_of_Day": [
            time_map[time]
        ],

        "Vehicle_Type": [
            vehicle_map[vehicle]
        ],

        "Preparation_Time_min": [
            prep_time
        ],

        "Courier_Experience_yrs": [
            experience
        ]
    })


    prediction = model.predict(input_data)


    st.success(
        f"Estimated Delivery Time: {prediction[0]:.2f} minutes"
    )


