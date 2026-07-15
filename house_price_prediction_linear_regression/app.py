import streamlit as st
import joblib
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
model = pickle.load(open("model.pkl","wb"))

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;
color:white;
padding:20px;
border-radius:15px;
background:linear-gradient(90deg,#4F46E5,#7C3AED);'>
🏠 House Price Prediction
</h1>
""", unsafe_allow_html=True)

st.write("### Enter the property details : ")

# ---------------- BUTTON CSS ----------------
st.markdown("""
<style>
div.stButton > button {
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color:white;
    border:none;
    border-radius:10px;
    padding:12px 30px;
    font-size:18px;
    font-weight:bold;
}
div.stButton > button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏠 Property Details")

st.sidebar.info(
    "Enter property details and click Predict."
)

# ---------------- KPI CARDS ----------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Average Area", "1500 sqft")

with col2:
    st.metric("Average Price", "₹4.2M")

st.divider()

# ---------------- INPUTS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input(
        "Area",
        min_value=500.0,
        max_value=5000.0,
        value=1000.0,
        key="area"
    )

with col2:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=2,
        key="bedrooms"
    )

with col3:
    floors = st.number_input(
        "Floors",
        min_value=1,
        max_value=5,
        value=2,
        key="floors"
    )

st.write("")

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict Price"):

    data = pd.DataFrame(
        [[area, bedrooms, floors]],
        columns=["Area", "Bedrooms", "Floors"]
    )

    prediction = model.predict(data)

    predicted_price = float(prediction[0])

    st.markdown(f"""
    <div style="
    background: linear-gradient(135deg,#1E3A8A,#7C3AED);
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-top:20px;">
        <h2>Predicted House Price</h2>
        <h1>₹ {predicted_price:,.0f}</h1>
    </div>
    """, unsafe_allow_html=True)