from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import shap
import psycopg2
import os

app = FastAPI(title="Customer Intelligence & Retention API")

model = joblib.load("models/xgb_churn.pkl")
kmf = joblib.load("models/kmf_survival.pkl")
explainer = shap.TreeExplainer(model)
cohort_half_life = float(kmf.median_survival_time_)

class CustomerData(BaseModel):
    customer_id: str
    tenure_months: int = Field(..., ge=0)
    monthly_charges: float = Field(..., gt=0)
    support_tickets_6m: int = Field(..., ge=0)
    is_senior: int = Field(..., ge=0, le=1)
    days_since_last_purchase: int = Field(..., ge=0) # New RFM variable

def log_inference_to_db(payload: dict, customer: CustomerData):
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database="retention_warehouse", user="bi_admin", password="bi_password"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO customer_inferences 
            (customer_id, tenure_months, monthly_charges, churn_probability, predicted_clv, rfm_segment, recommended_marketing_action, top_churn_driver)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            customer.customer_id, customer.tenure_months, customer.monthly_charges, 
            payload['churn_probability'], payload['predicted_clv'], payload['rfm_segment'],
            payload['recommended_action'], payload['explainability']['top_risk_drivers'][0]
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Log Error: {e}")

def calculate_rfm_and_action(churn_prob, clv, recency):
    """Business logic for RFM segmentation and prescriptive marketing."""
    segment = "Standard"
    if clv > 2000 and recency < 30:
        segment = "VIP Champion"
    elif clv > 1000 and recency >= 60:
        segment = "At-Risk Loyal"
    elif recency > 90:
        segment = "Hibernating"

    # Prescriptive Action Logic
    action = "Monitor"
    if churn_prob > 0.7 and clv > 1500:
        action = "Assign Account Manager"
    elif churn_prob > 0.5 and segment == "At-Risk Loyal":
        action = "Send 20% Win-Back Discount"
    elif churn_prob > 0.8 and clv < 200:
        action = "Do Not Contact (Low ROI)"
        
    return segment, action

@app.post("/predict")
def predict_retention(customer: CustomerData, bg_tasks: BackgroundTasks):
    features = pd.DataFrame([{
        'tenure_months': customer.tenure_months,
        'monthly_charges': customer.monthly_charges,
        'support_tickets_6m': customer.support_tickets_6m,
        'is_senior': customer.is_senior
    }])

    churn_prob = float(model.predict_proba(features)[0][1])
    expected_remaining_months = max(1.0, cohort_half_life - customer.tenure_months)
    predicted_clv = round(customer.monthly_charges * expected_remaining_months, 2)
    
    segment, action = calculate_rfm_and_action(churn_prob, predicted_clv, customer.days_since_last_purchase)

    shap_values = explainer.shap_values(features)[0]
    impact_map = sorted(zip(features.columns.tolist(), shap_values), key=lambda x: x[1], reverse=True)
    
    risk_drivers = [f"{feat} ({val:.2f})" for feat, val in impact_map if val > 0]
    ret_drivers = [f"{feat} ({val:.2f})" for feat, val in reversed(impact_map) if val < 0]

    response = {
        "churn_probability": round(churn_prob, 4),
        "predicted_clv": predicted_clv,
        "rfm_segment": segment,
        "recommended_action": action,
        "explainability": {
            "top_risk_drivers": risk_drivers[:3] if risk_drivers else ["None"],
            "top_retention_drivers": ret_drivers[:3] if ret_drivers else ["None"]
        }
    }

    bg_tasks.add_task(log_inference_to_db, response, customer)
    return response
