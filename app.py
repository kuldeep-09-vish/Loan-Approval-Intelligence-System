import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Loan Intelligence System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LIGHT THEME CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Background */
.main {
    background-color: #f5f7fb;
}

/* Header */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #1f3c88;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #1f3c88, #2e6cff);
    color: white;
    font-size: 18px;
    padding: 10px;
    border-radius: 10px;
    border: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_xgboost_loan_model.pkl")

# =========================
# HEADER
# =========================
st.markdown('<div class="title">🏦 Loan Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered credit risk engine for instant loan decisions</div>', unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Applicant Details")

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
# TOP METRICS CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <h3>Credit Score</h3>
        <h2>{credit_score}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <h3>Loan Amount</h3>
        <h2>{loan_amount}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <h3>Salary</h3>
        <h2>{salary}</h2>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =========================
# PREDICTION
# =========================
if st.button("Run AI Credit Analysis"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Decision Result")

    colA, colB = st.columns(2)

    with colA:
        if prediction[0] == 1:
            st.success("✅ Loan Approved")
            status = "Approved"
        else:
            st.error("❌ Loan Rejected")
            status = "Rejected"

    with colB:
        st.write("Approval Confidence")
        st.progress(float(probability))
        st.metric("Probability Score", f"{probability:.2f}")

    st.divider()

    # =========================
    # INSIGHTS SECTION
    # =========================
    st.subheader("Risk Insights")

    st.info(f"Risk Level: {risk_level}")
    st.info(f"Age Group: {age_group}")
    st.info(f"Loan-to-Salary Ratio: {loan_to_salary:.2f}")
    st.info(f"Income per Loan: {income_per_loan:.2f}")
    st.info(f"Total Burden Score: {total_burden}")