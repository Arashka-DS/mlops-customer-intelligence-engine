import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import mlflow
import mlflow.xgboost
import joblib
import os

# 1. Generate Synthetic Customer Data
np.random.seed(42)
n_samples = 5000
data = pd.DataFrame({
    'tenure_months': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'support_tickets_6m': np.random.poisson(1, n_samples),
    'is_senior': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

# Heuristic Churn Logic
risk = (data['monthly_charges'] * 0.05) - (data['tenure_months'] * 0.1) + (data['support_tickets_6m'] * 2)
data['churn'] = (risk > np.median(risk)).astype(int)
# Heuristic CLV (Historical Value + Expected Future Value)
data['historical_clv'] = data['tenure_months'] * data['monthly_charges']

X = data[['tenure_months', 'monthly_charges', 'support_tickets_6m', 'is_senior']]
y_churn = data['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=0.2, random_state=42)

# 2. MLflow Experiment Tracking
mlflow.set_experiment("Customer_Retention_Engine")
with mlflow.start_run():
    print("Training XGBoost Churn Model...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    loss = log_loss(y_test, preds)
    
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("log_loss", loss)
    mlflow.xgboost.log_model(model, "xgboost_churn")

    # 3. Kaplan-Meier Survival Analysis (Cohort Half-Life)
    print("Fitting Kaplan-Meier Survival Curve...")
    kmf = KaplanMeierFitter()
    # T = duration, E = observed event (churn)
    kmf.fit(durations=data['tenure_months'], event_observed=data['churn'])
    median_survival_time = kmf.median_survival_time_
    mlflow.log_metric("global_cohort_half_life", median_survival_time)

    # Save artifacts for API
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_churn.pkl")
    joblib.dump(kmf, "models/kmf_survival.pkl")
    print(f"Run complete. AUC: {auc:.4f} | Median Half-Life: {median_survival_time} months.")
