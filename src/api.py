import os
import pandas as pd
import joblib
import psycopg2
import threading
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Intelligence & Prescriptive Engine API")

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
    # Scaled to Toman (e.g., 25,000,000 = 25M Toman)
    if recency <= 30 and frequency >= 15 and monetary >= 25000000:
        return "Champion VIP"
    elif recency <= 60 and frequency >= 10:
        return "Loyal Customer"
    elif recency > 90 and frequency > 15 and monetary >= 15000000:
        return "At-Risk High-Value"
    elif recency > 90 and monetary < 5000000:
        return "Low-Value Churner"
    else:
        return "Standard Active"

def log_inference(customer_id, tenure, monetary, churn_prob, segment, action, top_driver):
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "customer_intelligence_db"),
            user=os.getenv("DB_USER", "crm_admin"), 
            password=os.getenv("DB_PASSWORD", "crm_password")
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO customer_inferences 
            (customer_id, tenure_months, monetary_value_toman, churn_probability, rfm_segment, prescriptive_action, top_churn_driver)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (customer_id, tenure, monetary, churn_prob, segment, action, top_driver))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}", flush=True)

@app.post("/predict")
def predict_retention(profile: CustomerProfile):
    data = pd.DataFrame([profile.dict(exclude={"customer_id"})])
    
    churn_prob = float(model.predict_proba(data)[0][1])
    segment = calculate_rfm_segment(profile.recency_days, profile.frequency_tx, profile.monetary_value)
    
    shap_values = explainer.shap_values(data)
    feature_impacts = list(zip(data.columns, shap_values[0]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_driver_name = feature_impacts[0][0]
    top_drivers = [{"feature": f, "impact": round(float(v), 3)} for f, v in feature_impacts[:3]]
    
    action = "Monitor - No immediate action required."
    
    if churn_prob > 0.75:
        if "VIP" in segment or "High-Value" in segment:
            action = "🚨 CRITICAL: Deploy 30% Annual Retention Discount (up to 2,000,000 Toman)."
        else:
            action = "Automated Win-Back Email Sequence (10% Discount)."
    elif churn_prob > 0.45:
        if "VIP" in segment:
            action = "Proactive Concierge Check-in Call (No Sales Pitch)."
        elif profile.support_tickets > 2:
            action = "Priority Support Routing & Apology Campaign."
        else:
            action = "Targeted Feature-Adoption Drip Campaign."
            
    # Save to Postgres asynchronously
    threading.Thread(
        target=log_inference, 
        args=(profile.customer_id, profile.tenure_months, profile.monetary_value, churn_prob, segment, action, top_driver_name)
    ).start()
            
    return {
        "customer_id": profile.customer_id,
        "rfm_segment": segment,
        "churn_probability_score": round(churn_prob, 4),
        "prescriptive_action": action,
        "primary_churn_drivers": top_drivers
    }
