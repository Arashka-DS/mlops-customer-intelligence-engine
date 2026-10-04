import os
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Intelligence & Prescriptive Engine API")

# Load Models
try:
    model = joblib.load("mlruns/local_models/xgb_model.pkl")
    explainer = joblib.load("mlruns/local_models/shap_explainer.pkl")
except Exception as e:
    print("Models not found. Awaiting training pipeline completion.")

class CustomerProfile(BaseModel):
    customer_id: str
    recency_days: int
    frequency_tx: int
    monetary_value: float
    app_sessions_last_30d: int
    support_tickets: int
    tenure_months: int

def calculate_rfm_segment(recency, frequency, monetary):
    """Business Heuristics: RFM (Recency, Frequency, Monetary) Segmentation"""
    if recency <= 30 and frequency >= 15 and monetary >= 2500:
        return "Champion VIP"
    elif recency <= 60 and frequency >= 10:
        return "Loyal Customer"
    elif recency > 90 and frequency > 15 and monetary >= 1500:
        return "At-Risk High-Value"
    elif recency > 90 and monetary < 500:
        return "Low-Value Churner"
    else:
        return "Standard Active"

@app.post("/predict")
def predict_retention(profile: CustomerProfile):
    data = pd.DataFrame([profile.dict(exclude={"customer_id"})])
    
    # 1. ML Scoring (Predictive)
    churn_prob = float(model.predict_proba(data)[0][1])
    
    # 2. Business Logic (Heuristic)
    segment = calculate_rfm_segment(profile.recency_days, profile.frequency_tx, profile.monetary_value)
    
    # 3. SHAP Explainability (Diagnostic)
    shap_values = explainer.shap_values(data)
    feature_impacts = list(zip(data.columns, shap_values[0]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_drivers = [{"feature": f, "impact": round(float(v), 3)} for f, v in feature_impacts[:3]]
    
    # 4. Prescriptive Action Matrix
    action = "Monitor - No immediate action required."
    
    if churn_prob > 0.75:
        if "VIP" in segment or "High-Value" in segment:
            action = "🚨 CRITICAL: Alert Account Exec. Deploy 30% Annual Retention Discount."
        else:
            action = "Automated Win-Back Email Sequence (10% Discount)."
    elif churn_prob > 0.45:
        if "VIP" in segment:
            action = "Proactive Concierge Check-in Call (No Sales Pitch)."
        elif profile.support_tickets > 2:
            action = "Priority Support Routing & Apology Campaign."
        else:
            action = "Targeted Feature-Adoption Drip Campaign."
    else:
        if "VIP" in segment:
            action = "Send Exclusive Beta Access Invite to maintain loyalty."
            
    return {
        "customer_id": profile.customer_id,
        "rfm_segment": segment,
        "churn_probability_score": round(churn_prob, 4),
        "prescriptive_action": action,
        "primary_churn_drivers": top_drivers
    }
