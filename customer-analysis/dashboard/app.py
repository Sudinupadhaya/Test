import streamlit as st
import pandas as pd
from src.preprocess import load_and_preprocess
from xgboost import XGBClassifier

st.title("📊 Customer Churn Prediction Dashboard")

X_train, X_test, y_train, y_test = load_and_preprocess()
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

st.subheader("Enter Customer Details:")
monthly_charges = st.slider("Monthly Charges", 0, 200, 70)
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-Month", "One year", "Two year"])

sample = {
    "MonthlyCharges": monthly_charges,
    "tenure": tenure,
    "Contract": 0 if contract == "Month-to-Month" else (1 if contract == "One year" else 2),
    "TotalCharges": monthly_charges * tenure,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "gender": 1
}
df = pd.DataFrame([sample])

if st.button("Predict"):
    pred = model.predict(df)[0]
    st.write("Prediction:", "⚠️ Likely to Churn" if pred == 1 else "✅ Will Stay")
