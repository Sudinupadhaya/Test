import pandas as pd
from src.preprocess import load_and_preprocess
from xgboost import XGBClassifier

# Example prediction for new data
def predict_new(data_dict):
    X_train, X_test, y_train, y_test = load_and_preprocess()
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    new_data = pd.DataFrame([data_dict])
    prediction = model.predict(new_data)
    return "Churn" if prediction[0] == 1 else "No Churn"

if __name__ == "__main__":
    sample = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 5,
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 1,
        "OnlineSecurity": 0,
        "OnlineBackup": 1,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 1,
        "StreamingMovies": 1,
        "Contract": 0,
        "PaperlessBilling": 1,
        "PaymentMethod": 2,
        "MonthlyCharges": 70,
        "TotalCharges": 350
    }
    print(predict_new(sample))
