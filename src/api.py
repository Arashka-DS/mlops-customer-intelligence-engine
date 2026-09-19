from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import shap
import psycopg2
import os

app = FastAPI(title="Customer Intelligence & Retention API")

# Load Models
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

def log_inference_to_db(payload: dict):
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database="retention_warehouse", user="bi_admin", password="bi_password"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO customer_inferences 
            (customer_id, tenure_months, monthly_charges, churn_probability, predicted_clv, cohort_half_life, top_positive_driver, top_negative_driver)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            payload['customer_id'], payload['tenure_months'], payload['monthly_charges'], 
            payload['churn_probability'], payload['predicted_clv'], payload['cohort_half_life'],
            payload['explainability']['top_risk_drivers'][0], payload['explainability']['top_retention_drivers'][0]
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Log Error: {e}")

@app.post("/predict")
def predict_retention(customer: CustomerData, bg_tasks: BackgroundTasks):
    # 1. Prepare Feature DataFrame
    features = pd.DataFrame([{
        'tenure_months': customer.tenure_months,
        'monthly_charges': customer.monthly_charges,
        'support_tickets_6m': customer.support_tickets_6m,
        'is_senior': customer.is_senior
    }])

    # 2. Predict Churn & CLV Heuristic
    churn_prob = float(model.predict_proba(features)[0][1])
    # Expected Life = Median Cohort Life - Current Tenure (floor at 1 month)
    expected_remaining_months = max(1.0, cohort_half_life - customer.tenure_months)
    predicted_clv = round(customer.monthly_charges * expected_remaining_months, 2)

    # 3. SHAP Explainability
    shap_values = explainer.shap_values(features)[0]
    feature_names = features.columns.tolist()
    
    # Map feature names to their SHAP impact values
    impact_map = sorted(zip(feature_names, shap_values), key=lambda x: x[1], reverse=True)
    
    # Positive SHAP values push churn UP (Risk), Negative push churn DOWN (Retention)
    risk_drivers = [f"{feat} ({val:.2f})" for feat, val in impact_map if val > 0]
    ret_drivers = [f"{feat} ({val:.2f})" for feat, val in reversed(impact_map) if val < 0]

    response = {
        "customer_id": customer.customer_id,
        "churn_probability": round(churn_prob, 4),
        "predicted_clv": predicted_clv,
        "cohort_half_life": cohort_half_life,
        "explainability": {
            "top_risk_drivers": risk_drivers[:3] if risk_drivers else ["None"],
            "top_retention_drivers": ret_drivers[:3] if ret_drivers else ["None"]
        }
    }

    bg_tasks.add_task(log_inference_to_db, {**response, **customer.dict()})
    return response
