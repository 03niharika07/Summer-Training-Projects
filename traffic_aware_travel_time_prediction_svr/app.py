import streamlit as st
import pandas as pd
import joblib


# Load Model

model = joblib.load("traffic_svr_model.pkl")


# Load Dataset for Dropdown Values

data = pd.read_csv("delhi_traffic_features.csv")

start_locations = sorted(data['start_area'].unique())

end_locations = sorted(data['end_area'].unique())


# Page Configuration

st.set_page_config(
    page_title="Traffic-Aware Travel Time Prediction",
    page_icon="🚦",
    layout="wide"
)


# Sidebar

st.sidebar.title("🚦 Model Performance")

st.sidebar.write("### Support Vector Regression Metrics")


st.sidebar.metric(
    "MAE",
    "6.37 minutes"
)

st.sidebar.metric(
    "RMSE",
    "9.19 minutes"
)

st.sidebar.metric(
    "R² Score",
    "90.5%"
)


st.sidebar.divider()


st.sidebar.info(
    """
    **Algorithm Used:**
    Support Vector Regression (SVR)

    **Target:**
    Travel Time (minutes)

    **Features Used:**

    • Start Area
    • End Area
    • Distance
    • Time of Day
    • Day of Week
    • Weather Condition
    • Traffic Density Level
    • Road Type
    """
)


# Main Heading

st.title(
    "🚦 Traffic-Aware Travel Time Prediction Using Support Vector Regression (SVR)"
)


st.write(
    "Estimate journey duration based on traffic conditions and route information."
)


st.divider()


# Input Section

col1, col2 = st.columns(2)


with col1:

    start_area = st.selectbox(
        "📍 Starting Area",
        start_locations
    )


    end_area = st.selectbox(
        "🏁 Destination Area",
        end_locations
    )


    route_data = data[
    (data['start_area'] == start_area) &
    (data['end_area'] == end_area)
]


if not route_data.empty:
    distance = route_data['distance_km'].iloc[0]
    
    st.write(
        f"📏 Distance: {distance} km"
    )

else:
    distance = 0
    st.warning("Distance information not available")


    time_of_day = st.selectbox(
        "🕒 Time of Day",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night"
        ]
    )


with col2:

    day_of_week = st.selectbox(
        "📅 Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


    weather_condition = st.selectbox(
        "🌦 Weather Condition",
        [
            "Clear",
            "Cloudy",
            "Rainy"
        ]
    )


    traffic_density_level = st.selectbox(
        "🚦 Traffic Density Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )


    road_type = st.selectbox(
        "🛣 Road Type",
        [
            "Highway",
            "Main Road",
            "Street"
        ]
    )


st.divider()


# Prediction

if st.button("🚗 Predict Travel Time"):


    input_data = pd.DataFrame(
        {
            "start_area": [start_area],
            "end_area": [end_area],
            "distance_km": [distance],
            "time_of_day": [time_of_day],
            "day_of_week": [day_of_week],
            "weather_condition": [weather_condition],
            "traffic_density_level": [traffic_density_level],
            "road_type": [road_type]
        }
    )


    prediction = model.predict(input_data)


    st.success(
        f"⏱ Estimated Travel Time: {prediction[0]:.2f} minutes"
    )