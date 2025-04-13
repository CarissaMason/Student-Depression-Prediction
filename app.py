import streamlit as st
import joblib
import numpy as np

# Load trained model and feature names
model = joblib.load("student_depression_model.pkl")
features = joblib.load("model_features.pkl")

st.title("Student Depression Risk Predictor")
st.markdown("This tool uses a machine learning model to estimate the likelihood that a student is experiencing depression based on their academic and lifestyle factors.")

# Input form for each feature
user_input = []
for feature in features:
    if "Gender" in feature:
        val = st.selectbox("Gender (0 = Male, 1 = Female)", [0, 1])
    elif "Have you ever had suicidal thoughts" in feature:
        val = st.selectbox("Suicidal Thoughts? (0 = No, 1 = Yes)", [0, 1])
    elif "Academic Pressure" in feature or "Work Pressure" in feature:
        val = st.number_input(f"{feature} (1–5)", min_value=1, max_value=5, step=1)
    elif "Financial Stress" in feature:
        val = st.number_input(f"{feature} (1–5)", min_value=1, max_value=5, step=1)
    elif "Work/Study Hours" in feature:
        val = st.number_input(f"{feature} (0–24)", min_value=0, max_value=24, step=1)
    elif "Age" in feature:
        val = st.number_input("Age", min_value=18, max_value=65, step=1)
    else:
        val = st.number_input(f"{feature}", step=1.0)
    user_input.append(val)

# Predict button
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"Likely Depressed (Confidence: {prob:.2f})")
    else:
        st.success(f"Not Depressed (Confidence: {1 - prob:.2f})")

st.caption("This prediction is for educational use only and should not replace professional diagnosis.")

