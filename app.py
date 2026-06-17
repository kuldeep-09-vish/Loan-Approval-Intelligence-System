import streamlit as st
import pandas as pd
import joblib

# ==========================================
# CONFIG & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="CreditPulse | AI Underwriting Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock Model fallback for testing (Remove if your file exists)
try:
    model = joblib.load("best_xgboost_loan_model.pkl")
except:
    class MockModel:
        def predict(self, df): return [1]
        def predict_proba(self, df): return [[0.15, 0.85]]
    model = MockModel()

# ==========================================
# MODERN FINTECH CSS THEME
# ==========================================
st.markdown("""
<style>
    /* Global Background & Font Tuning */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* Input Fields Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Primary Action Button Customization */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 14px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    /* Custom Badge & Micro-copy */
    .header-badge {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR: APPLICANT PROFILE
# ==========================================
with st.sidebar:
    st.markdown('<span class="header-badge">INPUT PARAMETERS</span>', unsafe_allow_html=True)
    st.title("User Profile")
    st.caption("Enter the details of the primary applicant to evaluate default risk metrics.")
    st.write("---")
    
    # Structural Input groupings
    age = st.number_input("Age (Years)", 18, 100, 30)
    salary = st.number_input("Monthly Income ($)", 1000, 1000000, 50000)
    experience = st.number_input("Work Experience (Years)", 0, 50, 5)
    credit_score = st.number_input("Credit Score (FICO)", 300, 900, 680)
    existing_loans = st.number_input("Active Active Loans", 0, 10, 1)
    emi_burden = st.number_input("Current Debt Burden / EMI (%)", 0, 100, 20)
    loan_amount = st.number_input("Requested Loan Amount ($)", 1000, 2000000, 120000)
    
    st.write("---")
    job_type = st.selectbox("Employment Sector", ["Private", "Government", "Self-Employed"])
    city = st.selectbox("Demographic Zone", ["Metro", "Urban", "Semi-Urban", "Rural"])
    education = st.selectbox("Highest Education Level", ["High School", "Graduate", "Post Graduate"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    loan_purpose = st.selectbox("Loan Utility Intent", ["Car Loan", "Home Loan", "Personal Loan", "Education Loan"])

# ==========================================
# FEATURE ENGINEERING PIPELINE
# ==========================================
risk_level = "High" if credit_score < 600 else ("Medium" if credit_score < 720 else "Low")
age_group = "Young" if age < 28 else ("Adult" if age < 50 else "Senior")

income_per_loan = salary / (loan_amount + 1)
loan_to_salary = loan_amount / (salary + 1)
total_burden = existing_loans + (emi_burden / 100)

input_data = pd.DataFrame([[ 
    age, salary, experience, credit_score, existing_loans, emi_burden, loan_amount,
    job_type, city, education, marital_status, loan_purpose, risk_level, age_group,
    income_per_loan, loan_to_salary, total_burden
]], columns=[
    "Age","Salary","Experience","Credit_Score","Existing_Loans","EMI_Burden","Loan_Amount",
    "Job_Type","City","Education","Marital_Status","Loan_Purpose","Risk_Level","Age_Group",
    "Income_per_Loan","Loan_to_Salary","Total_Burden"
])

# ==========================================
# MAIN DASHBOARD INTERFACE
# ==========================================
st.markdown('<span class="header-badge">CREDIT UNDERWRITING SYSTEMS</span>', unsafe_allow_html=True)
st.title("💳 CreditPulse Engine Dashboard")
st.markdown("<p style='color: #94a3b8; font-size:16px;'>Machine learning powered real-time credit decision-making matrix.</p>", unsafe_allow_html=True)

# Top KPI Summary Cards using Streamlit Native Containers
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    with st.container(border=True):
        st.caption("Credit Score Baseline")
        st.subheader(f"📊 {credit_score}")
        
with m_col2:
    with st.container(border=True):
        st.caption("Requested Principal")
        st.subheader(f"💰 ${loan_amount:,}")

with m_col3:
    with st.container(border=True):
        st.caption("Monthly Liquid Income")
        st.subheader(f"💵 ${salary:,}")

with m_col4:
    with st.container(border=True):
        st.caption("Calculated Risk Group")
        color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.subheader(f"{color_map.get(risk_level, '⚪')} {risk_level}")

st.write("##")

# ==========================================
# RUN AI ANALYSIS & DECISION BLOCKS
# ==========================================
if st.button("🚀 EXECUTE QUANTITATIVE ANALYSIS"):
    
    # Model inference execution
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.write("### 🛠️ Execution Pipeline Outputs")
    
    # Container grouping for primary output
    with st.container(border=True):
        res_col1, res_col2 = st.columns([1.2, 2])
        
        with res_col1:
            st.write("#### Underwriter Decision")
            if prediction[0] == 1:
                st.markdown("""
                    <div style="background-color: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; padding: 20px; border-radius: 8px; text-align: center;">
                        <h2 style="color: #22c55e; margin: 0;">APPROVED ✅</h2>
                        <p style="color: #86efac; margin: 5px 0 0 0; font-size:14px;">Applicant meets internal risk limits.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; padding: 20px; border-radius: 8px; text-align: center;">
                        <h2 style="color: #ef4444; margin: 0;">DECLINED ❌</h2>
                        <p style="color: #fca5a5; margin: 5px 0 0 0; font-size:14px;">High risk probability detected.</p>
                    </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            st.write("#### Confidence Probability Matrix")
            st.write("")
            st.progress(float(probability))
            
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Approval Pass Rate", f"{probability * 100:.1f}%")
            p_col2.metric("Default Variance Threshold", f"{(1 - probability) * 100:.1f}%")

    st.write("##")
    
    # Segmented breakdown details 
    st.write("### 📈 Comprehensive Risk Vectors")
    
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        with st.container(border=True):
            st.markdown("**Demographic Context**")
            st.markdown(f"Age Segmentation: `{age_group}`")
            st.markdown(f"Employment Profile: `{job_type}`")
            st.markdown(f"Education Level: `{education}`")

    with v_col2:
        with st.container(border=True):
            st.markdown("**Financial Leverage Factors**")
            st.markdown(f"Loan-to-Salary Ratio: `{loan_to_salary:.2f}x`")
            st.markdown(f"Liquidity Score: `{income_per_loan:.2f}`")

    with v_col3:
        with st.container(border=True):
            st.markdown("**Debt Profiles**")
            st.markdown(f"Active Debt Vectors: `{existing_loans}`")
            st.markdown(f"Total Combined Burden Factor: `{total_burden:.2f}`")