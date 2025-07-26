# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and selected features
with open("model_PSO.pkl", "rb") as f:
    model = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# App title and instructions
st.title("🎗️ Breast Cancer Prediction App (Using PSO)")
st.markdown("""
Use the sidebar to enter diagnostic values from a breast tissue sample.  
The model will predict whether the tumor is likely **benign** or **malignant**.
""")

# Sidebar input form
st.sidebar.header("Patient Feature Input")

# Collect input from user for each selected feature
input_data = {}
for feature in selected_features:
    input_data[feature] = st.sidebar.number_input(f"{feature}", format="%.4f")

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Show input summary
st.subheader("🔬 Diagnostic Feature Summary")
st.write(input_df)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    result = "🔴 **Malignant Tumor Detected**" if prediction == 1 else "🟢 **Benign Tumor Detected**"
    st.success(f"**Prediction Result:** {result}")
    st.info(f"**Prediction Probabilities:**\n- Benign: {proba[0]:.4f}\n- Malignant: {proba[1]:.4f}")
