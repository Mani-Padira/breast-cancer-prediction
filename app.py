import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("🔬 Breast Cancer Prediction App")

# Collect input
def user_input():
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    
    data = {}
    for feature in feature_names:
        data[feature] = st.number_input(f"{feature}", value=0.0, format="%.5f")
    
    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "🔴 Malignant" if prediction == 1 else "🟢 Benign"
    st.subheader(f"Prediction: {result}")
