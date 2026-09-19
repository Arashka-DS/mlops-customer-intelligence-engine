# MLOps Customer Intelligence & Retention Engine

A production-grade predictive pipeline that transitions business analytics from descriptive (what happened) to prescriptive (what should we do). This engine predicts customer churn, estimates Customer Lifetime Value (CLV), and outputs dynamic marketing actions based on real-time RFM segmentation and SHAP explainability.

## System Architecture
- **Experiment Tracking:** `MLflow` logs all XGBoost metrics, hyperparameters, and artifacts.
- **Inference Service:** `FastAPI` serves predictions with strict Pydantic validation.
- **Explainability:** `SHAP` (SHapley Additive exPlanations) provides the exact top 3 risk and retention drivers per customer.
- **Survival Analysis:** `lifelines` computes Kaplan-Meier cohort half-lives to feed dynamic CLV equations.
- **Business Intelligence:** Asynchronous PostgreSQL logging powers real-time Metabase and Power BI executive dashboards.

## Quick Start
1. Clone the repository.
2. Train the model and log to MLflow: `python src/train.py`
3. Spin up the BI infrastructure: `docker-compose up -d --build`
4. Post a customer payload to `http://localhost:8000/predict` to receive a scored response including a prescribed marketing action (e.g., "Send 20% Win-Back Discount").
5. Connect Power BI to `localhost:5432` and apply the included JSON theme for an executive schematic layout.
