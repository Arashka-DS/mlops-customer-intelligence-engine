CREATE DATABASE mlflow_tracking_db;

CREATE TABLE IF NOT EXISTS customer_inferences (
    inference_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    tenure_months INTEGER,
    monetary_value_toman NUMERIC,
    churn_probability NUMERIC,
    rfm_segment VARCHAR(50),
    prescriptive_action VARCHAR(255),
    top_churn_driver VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
