import streamlit as st
import joblib
import numpy as np

# Load model and scaler
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    return scaler, kmeans

scaler, kmeans = load_models()


# Cluster meaning dictionary
cluster_meanings = {
    0: "💰 High Income – High Spending (Premium Customers)",
    1: "💸 Low Income – Low Spending (Careful Customers)",
    2: "🧠 High Income – Low Spending (Smart Savers)",
    3: "🎯 Low Income – High Spending (Impulse Buyers)",
    4: "⚖ Average Income – Average Spending (Balanced Customers)"
}

# Page title
st.title("🧑‍💼 Customer Segmentation using K-Means")
st.write("Predict customer type based on income and spending behavior")

st.divider()

# Inputs (whole numbers only)
annual_income = st.number_input(
    "Annual Income (in k$)",
    min_value=0,
    max_value=200,
    step=1
)

spending_score = st.number_input(
    "Spending Score (1–100)",
    min_value=1,
    max_value=100,
    step=1
)

# Prediction
if st.button("Predict Customer Type"):
    input_data = np.array([[annual_income, spending_score]])
    scaled_data = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_data)[0]

    st.success(f"🧾 Customer Category:\n\n**{cluster_meanings[cluster]}**")
