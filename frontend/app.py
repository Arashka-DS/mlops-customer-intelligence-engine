import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="CRM Intelligence Portal", layout="wide", initial_sidebar_state="expanded")

# Styling to match the Dark PMO Schematic vibe
st.markdown("""
    <style>
    .big-font { font-size:24px !important; color: #00F2FE; font-weight: bold;}
    .metric-card { background-color: #1E293B; padding: 20px; border-radius: 10px; border: 1px solid #334155; }
    .action-high { color: #FF0844; font-weight: bold; }
    .action-safe { color: #92FE9D; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Customer Success & Retention Portal")
st.markdown("Enter customer details below to generate real-time churn risk, CLV, and prescriptive marketing actions.")

# Input Layout
with st.sidebar:
    st.header("Customer Search / Input")
    customer_id = st.text_input("Customer ID", value="CUST-9921")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    charges = st.number_input("Monthly Charges ($)", value=49.99)
    tickets = st.number_input("Support Tickets (Last 6M)", value=1)
    recency = st.number_input("Days Since Last Purchase", value=15)
    is_senior = st.selectbox("Senior Citizen?", [0, 1])
    
    predict_btn = st.button("Analyze Customer Risk", use_container_width=True, type="primary")

# API Interaction & Display
if predict_btn:
    payload = {
        "customer_id": customer_id,
        "tenure_months": tenure,
        "monthly_charges": charges,
        "support_tickets_6m": tickets,
        "is_senior": is_senior,
        "days_since_last_purchase": recency
    }
    
    with st.spinner("Querying ML Engine..."):
        try:
            # Hit the FastAPI container backend
            res = requests.post("http://retention_api:8000/predict", json=payload)
            res.raise_for_status()
            data = res.json()
            
            # --- ROW 1: Top Line Metrics ---
            st.markdown('<p class="big-font">Executive Summary</p>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            risk_color = "inverse" if data['churn_probability'] > 0.5 else "normal"
            col1.metric("Churn Risk", f"{data['churn_probability']*100:.1f}%", delta_color=risk_color)
            col2.metric("Predicted CLV", f"${data['predicted_clv']:,.2f}")
            col3.metric("RFM Segment", data['rfm_segment'])
            
            action = data['recommended_action']
            if action in ["Assign Account Manager", "Send 20% Win-Back Discount"]:
                col4.markdown(f"**Recommended Action:**<br><span class='action-high'>🚨 {action}</span>", unsafe_allow_html=True)
            else:
                col4.markdown(f"**Recommended Action:**<br><span class='action-safe'>✅ {action}</span>", unsafe_allow_html=True)

            st.divider()

            # --- ROW 2: SHAP Explainability ---
            st.markdown('<p class="big-font">SHAP Explainability (Why?)</p>', unsafe_allow_html=True)
            col_risk, col_ret = st.columns(2)
            
            with col_risk:
                st.error("🔴 Top Risk Drivers (Pushing Churn Up)")
                for driver in data['explainability']['top_risk_drivers']:
                    st.write(f"- {driver}")
                    
            with col_ret:
                st.success("🟢 Top Retention Drivers (Pushing Churn Down)")
                for driver in data['explainability']['top_retention_drivers']:
                    st.write(f"- {driver}")

        except Exception as e:
            st.error(f"Failed to connect to ML API. Ensure Docker containers are running. Error: {e}")
