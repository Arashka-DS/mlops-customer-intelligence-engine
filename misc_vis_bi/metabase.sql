-- Prescriptive Action Queue (Table)

SELECT
  customer_id,
  predicted_clv,
  recommended_marketing_action
FROM customer_inferences
WHERE recommended_marketing_action != 'Monitor'
ORDER BY predicted_clv DESC;

-- CLV by RFM Segment (Bar Chart)

SELECT
  rfm_segment,
  SUM(predicted_clv)
FROM customer_inferences
GROUP BY 1;

-- Primary Churn Drivers (Pie Chart)

SELECT
  top_churn_driver,
  COUNT(*)
FROM customer_inferences
WHERE churn_probability > 0.6
GROUP BY 1;
