import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Loan Intelligence System",
    layout="wide",
    page_icon="🏦"
)

model = joblib.load("best_xgboost_loan_model.pkl")

# ===================== CSS =====================
st.markdown("""
<style>

body {
    background-color: #f4f6fb;
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #1f3c88;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    text-align: center;
}

/* Button */
.stButton>button {
    width: 100%;
    padding: 12px;
    background: linear-gradient(90deg,#1f3c88,#2563eb);
    color: white;
    font-size: 16px;
    border-radius: 10px;
}

section[data-testid="stSidebar"] {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("<div class='title'>🏦 Loan Intelligence System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered credit decision engine</div>", unsafe_allow_html=True)

st.divider()

# ===================== SIDEBAR =====================
st.sidebar.header("Applicant Information")

age = st.sidebar.number_input("Age", 18, 100, 30)
salary = st.sidebar.number_input("Salary", 1000, 1000000, 50000)
experience = st.sidebar.number_input("Experience", 0, 50, 5)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
existing_loans = st.sidebar.number_input("Existing Loans", 0, 10, 1)
emi_burden = st.sidebar.number_input("EMI Burden", 0, 100, 20)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 2000000, 100000)

job_type = st.sidebar.selectbox("Job Type", ["Private", "Government", "Self-Employed"])
city = st.sidebar.selectbox("City", ["Metro", "Urban", "Semi-Urban", "Rural"])
education = st.sidebar.selectbox("Education", ["High School", "Graduate", "Post Graduate"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Car Loan", "Home Loan", "Personal Loan", "Education Loan"])

# ===================== FEATURES =====================
risk_level = "High" if credit_score < 500 else "Low"

input_data = pd.DataFrame([[
    age, salary, experience, credit_score,
    existing_loans, emi_burden, loan_amount,
    job_type, city, education, marital_status,
    loan_purpose, risk_level
]])

# ===================== TOP CARDS =====================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<div class='card'><h4>Credit Score</h4><h2>{credit_score}</h2></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='card'><h4>Loan Amount</h4><h2>{loan_amount}</h2></div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='card'><h4>Salary</h4><h2>{salary}</h2></div>", unsafe_allow_html=True)

st.divider()

# ===================== BUTTON =====================
if st.button("🚀 Run Credit Analysis"):

    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    colA, colB = st.columns(2)

    with colA:
        if pred[0] == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

    with colB:
        st.progress(float(prob))
        st.metric("Approval Probability", f"{prob:.2f}")

    st.divider()

    st.subheader("Risk Summary")

    st.info(f"Risk Level: {risk_level}")
    st.info(f"Debt Ratio: {loan_amount/(salary+1):.2f}")