import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD TRAINED MODEL
# =========================
model = joblib.load("best_xgboost_loan_model.pkl")

st.title("💰 Loan Approval Prediction App")
st.write("Fill all details below to predict loan approval")

# =========================
# INPUT FIELDS (RAW DATA)
# =========================
age = st.number_input("Age", 18, 100, 30)
salary = st.number_input("Salary", 1000, 1000000, 50000)
experience = st.number_input("Experience", 0, 50, 5)
credit_score = st.number_input("Credit Score", 300, 900, 600)
existing_loans = st.number_input("Existing Loans", 0, 10, 1)
emi_burden = st.number_input("EMI Burden", 0, 100, 20)
loan_amount = st.number_input("Loan Amount", 1000, 2000000, 100000)

job_type = st.selectbox("Job Type", ["Private", "Government", "Self-Employed"])
city = st.selectbox("City", ["Metro", "Urban", "Semi-Urban", "Rural"])
education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
loan_purpose = st.selectbox("Loan Purpose", ["Car Loan", "Home Loan", "Personal Loan", "Education Loan"])


# =========================
# FEATURE ENGINEERING (SAME AS TRAINING)
# =========================
risk_level = "High" if credit_score < 500 else "Low"

age_group = (
    "Child" if age < 18 else
    "Young" if age < 28 else
    "Adult" if age < 50 else
    "Old"
)

income_per_loan = salary / (loan_amount + 1)
loan_to_salary = loan_amount / (salary + 1)
total_burden = existing_loans + emi_burden


# =========================
# CREATE INPUT DATAFRAME
# =========================
input_data = pd.DataFrame([[
    age,
    salary,
    experience,
    credit_score,
    existing_loans,
    emi_burden,
    loan_amount,
    job_type,
    city,
    education,
    marital_status,
    loan_purpose,
    risk_level,
    age_group,
    income_per_loan,
    loan_to_salary,
    total_burden
]], columns=[
    "Age",
    "Salary",
    "Experience",
    "Credit_Score",
    "Existing_Loans",
    "EMI_Burden",
    "Loan_Amount",
    "Job_Type",
    "City",
    "Education",
    "Marital_Status",
    "Loan_Purpose",
    "Risk_Level",
    "Age_Group",
    "Income_per_Loan",
    "Loan_to_Salary",
    "Total_Burden"
])


# =========================
# PREDICTION
# =========================
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.info(f"📊 Approval Probability: {probability:.2f}")