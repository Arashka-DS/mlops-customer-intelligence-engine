# MLOps Customer Intelligence & Retention Engine

A production-grade predictive pipeline that transitions business analytics from descriptive (what happened) to prescriptive (what should we do). This engine predicts customer churn and outputs dynamic marketing actions based on real-time ML scoring and SHAP explainability.

## 🏛️ System Architecture
- **Experiment Tracking:** `MLflow` logs all XGBoost metrics, hyperparameters, and cohort half-lives.
- **Inference Service:** `FastAPI` serves predictions with strict Pydantic validation.
- **Explainability:** `SHAP` (SHapley Additive exPlanations) provides the exact top 3 risk and retention drivers per customer.
- **Survival Analysis:** `lifelines` computes Kaplan-Meier cohort half-lives to estimate baseline churn horizons.
- **Frontend Dashboard:** `Streamlit` powers a CRM interface for marketing executives to simulate user profiles.
- **Business Intelligence:** PostgreSQL acts as the backbone for operational `Metabase` reporting and executive `Power BI` analytics.

## 🚀 Quick Start
1. **Clone the repository.**
2. **Spin up the complete infrastructure:**
   ```bash
   docker-compose up -d --build
   ```
   (Note: The API container is configured to automatically generate synthetic data and train the initial XGBoost model on boot).
3. **Access the Interfaces:**
- **Streamlit CRM Dashboard:** `http://localhost:8501` (Interactive scoring and SHAP charts)
- **MLflow Tracking Server:** `http://localhost:5000` (View model experiments and Kaplan-Meier metrics)
- **FastAPI Docs:** `http://localhost:8000/docs`
- **Metabase BI:** `http://localhost:3000`
