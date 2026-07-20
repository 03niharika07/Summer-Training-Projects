import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# PAGE CONFIG

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


# LOAD MODEL 

kmeans = joblib.load("kmeans.pkl")
scaler = joblib.load("scaler.pkl")


# LOAD DATA

df = pd.read_csv("Mall_Customers.csv")

df.drop("CustomerID", axis=1, inplace=True)

df['Gender'] = df['Gender'].map({
    "Male":0,
    "Female":1
})


# Create clusters for visualization
X = df[['Gender',
        'Age',
        'Annual Income (k$)',
        'Spending Score (1-100)']]

X_scaled = scaler.transform(X)

df['Cluster'] = kmeans.predict(X_scaled)


# CLUSTER NAMES 

cluster_names = {
    0: "Mature Average Customers",
    1: "High Income Low Spenders",
    2: "Young Potential Customers",
    3: "Premium Customers",
    4: "Young Moderate Spenders",
    5: "Middle Age Regular Customers"
}


recommendations = {
    "Mature Average Customers":
        "Offer traditional deals and value-based offers.",

    "High Income Low Spenders":
        "Use personalized offers to increase engagement.",

    "Young Potential Customers":
        "Promote trendy products and loyalty programs.",

    "Premium Customers":
        "Provide exclusive rewards and premium services.",

    "Young Moderate Spenders":
        "Use discounts and membership benefits.",

    "Middle Age Regular Customers":
        "Focus on retention and regular promotions."
}


# TITLE 

st.title("🛍️ Mall Customer Segmentation using K-Means")
st.write(
    "Segment customers based on their demographic and spending behaviour."
)


# SIDEBAR INPUT 

st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

age = st.sidebar.slider(
    "Age",
    18,
    70,
    30
)

income = st.sidebar.slider(
    "Annual Income (k$)",
    10,
    140,
    50
)

spending = st.sidebar.slider(
    "Spending Score",
    1,
    100,
    50
)


gender_value = 0 if gender=="Male" else 1


# PREDICTION

if st.sidebar.button("Predict Segment"):

    customer = np.array(
        [[
            gender_value,
            age,
            income,
            spending
        ]]
    )

    customer_scaled = scaler.transform(customer)

    cluster = kmeans.predict(customer_scaled)[0]


    segment = cluster_names[cluster]


    st.success(
        f"Customer Segment: {segment}"
    )

    st.info(
        recommendations[segment]
    )


# VISUALIZATION 

st.divider()

st.subheader("📊 Customer Segmentation Analysis")


col1, col2 = st.columns(2)


# Cluster Distribution

with col1:

    st.write("### Cluster Distribution")

    cluster_count = df['Cluster'].value_counts()

    fig, ax = plt.subplots()

    sns.barplot(
        x=cluster_count.index,
        y=cluster_count.values,
        ax=ax
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)



# Income vs Spending

with col2:

    st.write("### Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(6,5))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="viridis",
        s=100,
        ax=ax
    )

    ax.set_title(
        "Customer Segmentation"
    )

    st.pyplot(fig)



# CLUSTER SUMMARY 

st.subheader("📌 Cluster Summary")

summary = df.groupby("Cluster").mean(numeric_only=True)

summary = summary.round(0)

# Convert encoded gender into readable format
summary["Gender"] = summary["Gender"].apply(
    lambda x: "Female" if x >= 0.5 else "Male"
)

summary["Customer Segment"] = summary.index.map(cluster_names)

st.dataframe(summary)