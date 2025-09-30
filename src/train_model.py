from src.preprocess import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        print(f"\n{name}:")
        print("ROC-AUC:", auc)
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate()
