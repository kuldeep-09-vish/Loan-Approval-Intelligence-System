import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Loan AI System", layout="wide")

model = joblib.load("best_xgboost_loan_model.pkl")

# =========================
# CUSTOM CSS (UI UPGRADE)
# =========================
st.markdown("""
<style>

html, body, [class*="css"] {
    height: 100%;
    background: #0f172a;
    color: white;
}

/* Full height app */
.main {
    background: #0f172a;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.06);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    backdrop-filter: blur(10px);
    transition: 0.3s ease-in-out;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.6);
}

/* Title */
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
}

/* Metric box */
.metric-box {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

/* Button */
.stButton button {
    width: 100%;
    background: #3b82f6;
    color: white;
    padding: 12px;
    border-radius: 10px;
    border: none;
    font-size: 16px;
    transition: 0.3s;
}

.stButton button:hover {
    background: #2563eb;
    transform: scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="title">💳 Loan Approval Intelligence System</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered credit risk engine</p>", unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR INPUT (CARD STYLE)
# =========================
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Applicant Profile")

    age = st.number_input("Age", 18, 100, 30)
    salary = st.number_input("Monthly Salary", 1000, 1000000, 50000)
    experience = st.number_input("Experience (Years)", 0, 50, 5)
    credit_score = st.number_input("Credit Score", 300, 900, 600)
    existing_loans = st.number_input("Existing Loans", 0, 10, 1)
    emi_burden = st.number_input("EMI Burden (%)", 0, 100, 20)
    loan_amount = st.number_input("Loan Amount", 1000, 2000000, 100000)

    job_type = st.selectbox("Job Type", ["Private", "Government", "Self-Employed"])
    city = st.selectbox("City Type", ["Metro", "Urban", "Semi-Urban", "Rural"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    loan_purpose = st.selectbox("Loan Purpose", ["Car Loan", "Home Loan", "Personal Loan", "Education Loan"])

    st.markdown('</div>', unsafe_allow_html=True)

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
# TOP METRICS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-box">Credit Score<br><h2>{}</h2></div>'.format(credit_score), unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">Loan Amount<br><h2>{}</h2></div>'.format(loan_amount), unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">Salary<br><h2>{}</h2></div>'.format(salary), unsafe_allow_html=True)

st.divider()

# =========================
# BUTTON
# =========================
if st.button("🚀 Run AI Analysis"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Decision Output")

    colA, colB = st.columns(2)

    with colA:
        if prediction[0] == 1:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Rejected ❌")

    with colB:
        st.write("Approval Confidence")
        st.progress(float(probability))
        st.metric("Probability", f"{probability:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # =========================
    # RISK PANEL
    # =========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Risk Analysis")

    st.write(f"Risk Level: **{risk_level}**")
    st.write(f"Age Group: **{age_group}**")
    st.write(f"Loan-to-Salary: **{loan_to_salary:.2f}**")
    st.write(f"Income per Loan: **{income_per_loan:.2f}**")
    st.write(f"Total Burden: **{total_burden}**")

    st.markdown('</div>', unsafe_allow_html=True)