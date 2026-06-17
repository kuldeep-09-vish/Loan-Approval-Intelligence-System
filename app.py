import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Loan Intelligence System",
    page_icon="💳",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_xgboost_loan_model.pkl")

# =========================
# HEADER
# =========================
st.markdown("""
    <div style="text-align:center;">
        <h1>Loan Approval Intelligence System</h1>
        <p>AI-powered credit risk analysis for instant loan decisioning</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Applicant Details")

age = st.sidebar.number_input("Age", 18, 100, 30)
salary = st.sidebar.number_input("Monthly Salary", 1000, 1000000, 50000)
experience = st.sidebar.number_input("Experience (Years)", 0, 50, 5)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
existing_loans = st.sidebar.number_input("Existing Loans", 0, 10, 1)
emi_burden = st.sidebar.number_input("EMI Burden (%)", 0, 100, 20)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 2000000, 100000)

job_type = st.sidebar.selectbox("Job Type", ["Private", "Government", "Self-Employed"])
city = st.sidebar.selectbox("City Type", ["Metro", "Urban", "Semi-Urban", "Rural"])
education = st.sidebar.selectbox("Education", ["High School", "Graduate", "Post Graduate"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Car Loan", "Home Loan", "Personal Loan", "Education Loan"])

# =========================
# FEATURE ENGINEERING
# =========================
risk_level = "High" if credit_score < 500 else "Low"

age_group = (
    "Young" if age < 28 else
    "Adult" if age < 50 else
    "Senior"
)

income_per_loan = salary / (loan_amount + 1)
loan_to_salary = loan_amount / (salary + 1)
total_burden = existing_loans + emi_burden

input_data = pd.DataFrame([[
    age, salary, experience, credit_score,
    existing_loans, emi_burden, loan_amount,
    job_type, city, education, marital_status,
    loan_purpose, risk_level, age_group,
    income_per_loan, loan_to_salary, total_burden
]], columns=[
    "Age","Salary","Experience","Credit_Score",
    "Existing_Loans","EMI_Burden","Loan_Amount",
    "Job_Type","City","Education","Marital_Status",
    "Loan_Purpose","Risk_Level","Age_Group",
    "Income_per_Loan","Loan_to_Salary","Total_Burden"
])

# =========================
# MAIN SECTION
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Credit Score", credit_score)

with col2:
    st.metric("Loan Amount", f"{loan_amount}")

with col3:
    st.metric("Salary", f"{salary}")

st.divider()

# =========================
# PREDICTION
# =========================
if st.button("Run AI Analysis", use_container_width=True):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Decision Output")

    left, right = st.columns(2)

    with left:
        if prediction[0] == 1:
            st.success("Loan Approved")
            status = "Approved"
        else:
            st.error("Loan Rejected")
            status = "Rejected"

    with right:
        st.write("Approval Confidence")
        st.progress(float(probability))

        st.metric("Probability Score", f"{probability:.2f}")

    st.divider()

    # =========================
    # RISK INSIGHTS PANEL
    # =========================
    st.subheader("Risk Analysis Summary")

    st.write(f"- Risk Level: **{risk_level}**")
    st.write(f"- Age Group: **{age_group}**")
    st.write(f"- Loan-to-Salary Ratio: **{loan_to_salary:.2f}**")
    st.write(f"- Income per Loan: **{income_per_loan:.2f}**")
    st.write(f"- Total Burden Score: **{total_burden}**")