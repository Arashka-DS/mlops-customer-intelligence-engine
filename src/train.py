import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import shap
from lifelines import KaplanMeierFitter
import joblib

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Customer_Retention_XGBoost")

def generate_synthetic_data(n=2000):
    np.random.seed(42)
    df = pd.DataFrame({
        "recency_days": np.random.randint(1, 180, n),
        "frequency_tx": np.random.randint(1, 50, n),
        "monetary_value": np.random.uniform(50, 5000, n),
        "app_sessions_last_30d": np.random.randint(0, 30, n),
        "support_tickets": np.random.randint(0, 5, n),
        "tenure_months": np.random.randint(1, 60, n),
    })
    
    # Target Generation (Churn)
    churn_prob = (df["recency_days"] > 90).astype(int) * 0.4 + \
                 (df["support_tickets"] > 2).astype(int) * 0.3 + \
                 (df["app_sessions_last_30d"] < 2).astype(int) * 0.2
    df["churn"] = (np.random.rand(n) < churn_prob).astype(int)
    return df

def train_and_log():
    df = generate_synthetic_data()
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    with mlflow.start_run():
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(X, y)
        
        # Explainability (SHAP)
        explainer = shap.TreeExplainer(model)
        
        # Survival Analysis (Kaplan-Meier)
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df["tenure_months"], event_observed=df["churn"])
        half_life = kmf.median_survival_time_
        
        # Log to MLflow
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("cohort_half_life_months", half_life)
        mlflow.xgboost.log_model(model, "xgboost_churn")
        
        # Save local artifacts for fast API boot
        os.makedirs("mlruns/local_models", exist_ok=True)
        joblib.dump(model, "mlruns/local_models/xgb_model.pkl")
        joblib.dump(explainer, "mlruns/local_models/shap_explainer.pkl")
        
        print(f"Training complete. Cohort Half-Life: {half_life} months.")

if __name__ == "__main__":
    train_and_log()
