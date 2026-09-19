import os
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Intelligence API")

# Load Models
try:
    model = joblib.load("mlruns/local_models/xgb_model.pkl")
    explainer = joblib.load("mlruns/local_models/shap_explainer.pkl")
except Exception as e:
    print("Models not found. Awaiting training.")

class CustomerProfile(BaseModel):
    customer_id: str
    recency_days: int
    frequency_tx: int
    monetary_value: float
    app_sessions_last_30d: int
    support_tickets: int
    tenure_months: int

@app.post("/predict")
def predict_retention(profile: CustomerProfile):
    data = pd.DataFrame([profile.dict(exclude={"customer_id"})])
    
    # Scoring
    churn_prob = float(model.predict_proba(data)[0][1])
    
    # SHAP Explainability
    shap_values = explainer.shap_values(data)
    feature_impacts = list(zip(data.columns, shap_values[0]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_drivers = [{"feature": f, "impact": float(v)} for f, v in feature_impacts[:3]]
    
    # Prescriptive Action Logic
    action = "No Action Needed"
    if churn_prob > 0.70:
        action = "High Risk: Trigger Account Manager Call"
    elif churn_prob > 0.40 and profile.monetary_value > 1000:
        action = "Medium Risk (High LTV): Send 20% Win-Back Discount"
        
    return {
        "customer_id": profile.customer_id,
        "churn_probability": round(churn_prob, 4),
        "marketing_action": action,
        "top_drivers": top_drivers
    }
