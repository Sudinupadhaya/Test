import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path="data/telco_churn.csv"):
    df = pd.read_csv(path)

    # Drop customerID (not useful)
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode categorical variables
    for col in df.select_dtypes(include="object").columns:
        if col != "Churn":
            df[col] = LabelEncoder().fit_transform(df[col])

    # Target encode
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Split
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
