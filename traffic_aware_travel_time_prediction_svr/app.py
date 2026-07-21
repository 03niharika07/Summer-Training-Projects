import streamlit as st
import pandas as pd
import joblib
import os


# ---------------- LOAD MODEL ----------------

model_path = os.path.join(os.path.dirname(__file__), "traffic_svr_model.pkl")
model = joblib.load(model_path)


# Load dataset for dropdowns and distance

data = pd.read_csv("delhi_traffic_features.csv")


start_locations = sorted(data["start_area"].unique())

end_locations = sorted(data["end_area"].unique())


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Traffic-Aware Travel Time Prediction",
    page_icon="🚦",
    layout="wide"
)


# ---------------- SIDEBAR ----------------

st.sidebar.title("🚦 Traffic-Aware Prediction")

st.sidebar.write(
    """
    ### Model Overview

    This application predicts the estimated 
    travel time using traffic and route 
    related information.
    """
)


st.sidebar.divider()


st.sidebar.subheader("📊 Model Performance")


st.sidebar.metric(
    label="Mean Absolute Error (MAE)",
    value="6.37 min"
)

st.sidebar.caption(
    "Average prediction error of the model"
)


st.sidebar.metric(
    label="Root Mean Squared Error (RMSE)",
    value="9.19 min"
)

st.sidebar.caption(
    "Measures larger prediction errors"
)


st.sidebar.metric(
    label="R² Score",
    value="90.5%"
)

st.sidebar.caption(
    "Model explains 90.5% variation in travel time"
)


st.sidebar.divider()


st.sidebar.subheader("🤖 Algorithm")

st.sidebar.write(
    """
    **Support Vector Regression (SVR)**

    Used for predicting continuous values
    like travel time in minutes.
    """
)


st.sidebar.subheader("📌 Input Features")

st.sidebar.write(
    """
    • Starting Area  
    • Destination Area  
    • Distance  
    • Time of Day  
    • Day of Week  
    • Weather Condition  
    • Traffic Density  
    • Road Type
    """
)



# ---------------- MAIN PAGE ----------------

st.title(
    "🚦 Traffic-Aware Travel Time Prediction Using Support Vector Regression (SVR)"
)


st.write(
    "Estimate the time required to reach your destination based on traffic and route conditions."
)


st.divider()



# ---------------- INPUT SECTION ----------------


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


    # Finding distance automatically

    route = data[
        (data["start_area"] == start_area) &
        (data["end_area"] == end_area)
    ]


    if not route.empty:

        distance = route["distance_km"].iloc[0]

        st.info(
            f"📏 Route Distance: {distance} km"
        )

    else:

        distance = 0

        st.warning(
            "Distance not available for this route"
        )



with col2:

    time_of_day = st.selectbox(
        "🕒 Time of Day",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night"
        ]
    )


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



# ---------------- PREDICTION ----------------


if st.button(
    "🚗 Predict Travel Time",
    use_container_width=True
):


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