# Customer-Churn-Prediction

Customer Churn Prediction - README
Project Overview
This project predicts telecom customer churn (whether a customer will leave the company). It demonstrates an end-to-end machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), model training and evaluation, and an interactive dashboard built with Streamlit.
Dataset
- Source: Telco Customer Churn Dataset (Kaggle)
- Rows: 7043
- Features: Customer demographics, contract type, monthly charges, services used, etc.
- Target: Churn (Yes/No)

Place the dataset file in the path: data/telco_churn.csv
Project Structure
customer-churn-prediction/
│-- data/                     # Dataset (CSV file)
│-- src/                      # Source code
│   ├── preprocess.py         # Data preprocessing
│   ├── train_model.py        # Train ML models
│   ├── predict.py            # Predict churn for new data
│-- models/                   # Saved models
│-- results/                  # Plots, metrics
│-- dashboard/                # Streamlit dashboard
│   └── app.py
│-- requirements.txt          # Dependencies
│-- README.md                 # Documentation
Installation & Setup
1. Clone the repository:
   git clone https://github.com/<your-username>/customer-churn-prediction.git
   cd customer-churn-prediction


2. Install dependencies:
   pip install -r requirements.txt
Usage
Train Model:
   python -m src.train_model

Run Prediction:
   python -m src.predict

Launch Dashboard:
   streamlit run dashboard/app.py
Results
- Random Forest ROC-AUC: ~0.82
- XGBoost ROC-AUC: ~0.84
- Key predictors: Contract type, Tenure, MonthlyCharges, OnlineSecurity

Saved outputs:
- results/confusion_matrix.png
- results/feature_importance.png
