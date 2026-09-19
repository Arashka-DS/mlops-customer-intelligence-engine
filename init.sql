CREATE TABLE IF NOT EXISTS customer_inferences (
    inference_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    tenure_months INTEGER,
    monthly_charges NUMERIC,
    churn_probability NUMERIC,
    predicted_clv NUMERIC,
    cohort_half_life NUMERIC,
    top_positive_driver VARCHAR(100),
    top_negative_driver VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
