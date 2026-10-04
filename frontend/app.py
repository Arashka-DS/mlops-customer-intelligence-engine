import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customer Intelligence CRM", layout="wide")

st.title("🎯 Customer Intelligence & Retention CRM")
st.caption("Predictive Churn Scoring, SHAP Feature Analysis, and Prescriptive Marketing Actions.")

API_URL = "http://crm_api:8000"

st.sidebar.header("User Profile")
recency = st.sidebar.slider("Recency (Days since last tx)", 1, 180, 45)
freq = st.sidebar.slider("Frequency (Total tx)", 1, 100, 15)
# Updated to Toman scale
monetary = st.sidebar.number_input("Monetary Value (Toman)", 1_000_000, 100_000_000, 15_000_000, step=1_000_000)
sessions = st.sidebar.slider("App Sessions (Last 30d)", 0, 50, 5)
tickets = st.sidebar.slider("Support Tickets", 0, 10, 1)
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)

if st.button("Score Customer Health", type="primary"):
    payload = {
        "customer_id": "CUST-9991",
        "recency_days": recency,
        "frequency_tx": freq,
        "monetary_value": monetary,
        "app_sessions_last_30d": sessions,
        "support_tickets": tickets,
        "tenure_months": tenure
    }
    
    try:
        res = requests.post(f"{API_URL}/predict", json=payload).json()
        
        c1, c2 = st.columns(2)
        c1.metric("Churn Risk Probability", f"{res['churn_probability_score'] * 100:.1f}%")
        c2.metric("RFM Customer Segment", res['rfm_segment'])
        
        st.subheader("💡 Prescriptive Action")
        if res['churn_probability_score'] >= 0.45:
            st.error(res['prescriptive_action'])
        else:
            st.success(res['prescriptive_action'])
        
        st.subheader("🔍 SHAP Value Explainability (Why?)")
        # Updated key to map to the new API schema
        df_shap = pd.DataFrame(res["primary_churn_drivers"])
        fig = px.bar(df_shap, x="impact", y="feature", orientation='h', 
                     title="Top Factors Influencing Churn (Positive = Higher Risk)",
                     color="impact", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"API Error: {e}")
