import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load('models/energy_model.pkl')

# -----------------------------
# Title
# -----------------------------
st.title("⚡ Energy Forecast Dashboard")

# -----------------------------
# Columns Layout
# -----------------------------
col1, col2 = st.columns([1, 2])

# -----------------------------
# LEFT → INPUTS + BUTTON
# -----------------------------
with col1:
    st.subheader("🔧 Inputs")

    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 0, 6, 3)

    lag_1 = st.number_input("1h ago", value=100.0)
    lag_2 = st.number_input("2h ago", value=100.0)
    lag_3 = st.number_input("3h ago", value=100.0)

    # Prediction button here (fixed position)
    predict_btn = st.button("Predict")

# -----------------------------
# RIGHT → GRAPH + OUTPUT
# -----------------------------
with col2:
    st.subheader("📊 Energy Trend")

    data = pd.read_csv('data/energy.csv')

    # Smaller graph (IMPORTANT FIX)
    st.line_chart(data['Energy'].head(150), height=300)

    # Show prediction near graph (so it's visible)
    if predict_btn:
        features = np.array([[hour, day, lag_1, lag_2, lag_3]])
        prediction = model.predict(features)

        st.success(f"⚡ Predicted Energy: {prediction[0]:.2f}")