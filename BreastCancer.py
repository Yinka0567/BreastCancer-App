import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and top features
model = joblib.load("rf_top10_model.pkl")
scaler = joblib.load("scaler_top10.pkl")
top_features = joblib.load("top10_features.pkl")

st.title("🩺 Breast Cancer Predictor (Top 10 Features)")

user_input = {}
for feature in top_features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.01)

if st.button("Predict"):
    df = pd.DataFrame([user_input])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0]

    st.subheader("Prediction:")
    st.success("🟢 Benign (0)" if prediction == 0 else "🔴 Malignant (1)")

    st.subheader("Prediction Probabilities:")
    st.write(f"Benign: {prob[0]:.2f} | Malignant: {prob[1]:.2f}")
