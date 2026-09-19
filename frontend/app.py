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
monetary = st.sidebar.number_input("Monetary Value ($)", 50, 10000, 1500)
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
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Risk Probability", f"{res['churn_probability'] * 100:.1f}%")
        
        action_color = "normal" if res['churn_probability'] < 0.4 else "inverse"
        c2.metric("Prescribed Action", res['marketing_action'], delta="Automated Rule", delta_color=action_color)
        
        st.subheader("🔍 SHAP Value Explainability (Why?)")
        df_shap = pd.DataFrame(res["top_drivers"])
        fig = px.bar(df_shap, x="impact", y="feature", orientation='h', 
                     title="Top Factors Influencing Churn (Positive = Higher Risk)",
                     color="impact", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"API Error: {e}")
