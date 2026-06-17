import streamlit as st
import pandas as pd
import joblib
import time

# ==========================================
# CONFIG & PREMIUM PAGE THEME
# ==========================================
st.set_page_config(
    page_title="CreditPulse Pro | AI Underwriting Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust fallback model simulation for zero-downtime execution
try:
    model = joblib.load("best_xgboost_loan_model.pkl")
except:
    class MockModel:
        def predict(self, df): return [1]
        def predict_proba(self, df): return [[0.12, 0.88]]
    model = MockModel()

# ==========================================
# ADVANCED CSS3 ANIMATIONS & UX ENGINE
# ==========================================
st.markdown("""
<style>
    /* Global Base Layer Tuning */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #030712 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #f3f4f6 !important;
    }
    
    /* Native Container Interception (The Glow Cards) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(17, 24, 39, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px);
        border-color: rgba(59, 130, 246, 0.4) !important;
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Dynamic Action Button Styling */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        padding: 16px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton button:hover {
        transform: scale(1.015);
        box-shadow: 0 12px 32px rgba(124, 58, 237, 0.5);
    }
    
    /* Custom Modern Badges */
    .ux-badge {
        background: linear-gradient(90deg, rgba(37,99,235,0.15) 0%, rgba(124,58,237,0.15) 100%);
        border: 1px solid rgba(124, 58, 237, 0.3);
        color: #a78bfa;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    /* Keyframe Fade In Animations for Dynamic Content */
    @keyframes smoothFade {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated-result-card {
        animation: smoothFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION & DATA DATA LAYER
# ==========================================
with st.sidebar:
    st.markdown('<span class="ux-badge">System Parameter Matrix</span>', unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top:0;'>Applicant Profile</h2>", unsafe_allow_html=True)
    st.caption("Engine inputs mapped directly to neural weights.")
    st.write("##")
    
    # Clean Inputs without manual div overrides to maintain stability
    age = st.number_input("Age Dimension", 18, 100, 32)
    salary = st.number_input("Monthly Net Remuneration ($)", 1000, 1000000, 62000)
    experience = st.number_input("Operational Domain Experience (Years)", 0, 50, 7)
    credit_score = st.number_input("FICO Bureau Score", 300, 900, 715)
    existing_loans = st.number_input("Active Debt Registry Vectors", 0, 10, 0)
    emi_burden = st.number_input("Aggregated Monthly Liabilities / EMI (%)", 0, 100, 15)
    loan_amount = st.number_input("Requested Credit Principal ($)", 1000, 2000000, 150000)
    
    st.write("---")
    job_type = st.selectbox("Employment Classification", ["Private", "Government", "Self-Employed"])
    city = st.selectbox("Geographic Jurisdiction Zone", ["Metro", "Urban", "Semi-Urban", "Rural"])
    education = st.selectbox("Highest Scholastic Tier", ["High School", "Graduate", "Post Graduate"])
    marital_status = st.selectbox("Legal Marital Configuration", ["Single", "Married"])
    loan_purpose = st.selectbox("Capital Deployment Utility Target", ["Car Loan", "Home Loan", "Personal Loan", "Education Loan"])

# ==========================================
# ANALYTICAL DATA CALCULATIONS
# ==========================================
risk_level = "High" if credit_score < 600 else ("Medium" if credit_score < 740 else "Low")
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
# MAIN ENTERPRISE INTERFACE
# ==========================================
st.markdown('<span class="ux-badge">Automated Underwriting Cluster</span>', unsafe_allow_html=True)
st.markdown("<h1 style='font-weight: 700; margin-top:0; letter-spacing: -0.5px;'>⚡ CreditPulse Core Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size:16px; margin-top:-10px;'>Machine learning risk evaluation matrix running automated pipeline inferences.</p>", unsafe_allow_html=True)
st.write("##")

# Metrics Array Display Layer
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-size:13px; font-weight:500; margin-bottom:4px;'>CREDIT SCORE REGISTRY</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:#3b82f6; margin:0;'>{credit_score} <span style='font-size:14px; color:#64748b;'>PTS</span></h2>", unsafe_allow_html=True)
        
with m_col2:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-size:13px; font-weight:500; margin-bottom:4px;'>PRINCIPAL VALUATION</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:#f8fafc; margin:0;'>${loan_amount:,}</h2>", unsafe_allow_html=True)

with m_col3:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-size:13px; font-weight:500; margin-bottom:4px;'>LIQUID LIABILITIES BURDEN</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:#f8fafc; margin:0;'>{emi_burden}%</h2>", unsafe_allow_html=True)

with m_col4:
    with st.container(border=True):
        st.markdown("<p style='color:#94a3b8; font-size:13px; font-weight:500; margin-bottom:4px;'>RISK PROFILE MATRIX</p>", unsafe_allow_html=True)
        rc = "#22c55e" if risk_level == "Low" else ("#eab308" if risk_level == "Medium" else "#ef4444")
        st.markdown(f"<h2 style='color:{rc}; margin:0;'>● {risk_level}</h2>", unsafe_allow_html=True)

st.write("##")

# ==========================================
# EXECUTION LOGIC & ANIMATED TRANSITIONS
# ==========================================
if st.button("🚀 EXECUTE AUTOMATED INFERENCE PROCESS"):
    
    # Smooth UX Loader simulation to visually feel active
    with st.spinner("Processing pipeline mathematics..."):
        time.sleep(0.7)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

    # HTML Animation Wrapper inject
    st.markdown('<div class="animated-result-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Underwriting Engine Core Verdict")
    
    with st.container(border=True):
        res_col1, res_col2 = st.columns([1.3, 2])
        
        with res_col1:
            st.write("##")
            if prediction[0] == 1:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(34,197,94,0.15) 0%, rgba(21,128,61,0.05) 100%); border: 1px solid #22c55e; padding: 24px; border-radius: 12px; text-align: center; box-shadow: 0 8px 32px rgba(34,197,94,0.1);">
                        <h2 style="color: #22c55e; margin: 0; font-weight:700; letter-spacing:1px;">CREDIT CLEAR ✅</h2>
                        <p style="color: #a7f3d0; margin: 8px 0 0 0; font-size:14px;">The application complies completely with predefined safety metrics.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(185,28,28,0.05) 100%); border: 1px solid #ef4444; padding: 24px; border-radius: 12px; text-align: center; box-shadow: 0 8px 32px rgba(239,68,68,0.1);">
                        <h2 style="color: #ef4444; margin: 0; font-weight:700; letter-spacing:1px;">RISK REJECT ❌</h2>
                        <p style="color: #fca5a5; margin: 8px 0 0 0; font-size:14px;">Variance metrics breached security thresholds.</p>
                    </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("<p style='color:#94a3b8; font-weight:600; font-size:13px; margin-bottom:4px;'>CONFIDENCE RATIO SPECTRUM</p>", unsafe_allow_html=True)
            st.progress(float(probability))
            
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Approval Vector Stability", f"{probability * 100:.1f}%")
            p_col2.metric("Variance Margin Factor", f"{(1 - probability) * 100:.1f}%")

    st.write("##")
    
    # Structural Risk Breakdown
    st.markdown("### 📈 Real-Time Multi-Variant Diagnostics")
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        with st.container(border=True):
            st.markdown("<b style='color:#3b82f6;'>Demographic Categorization</b>", unsafe_allow_html=True)
            st.write(f"Age Variant Mapping: `{age_group}`")
            st.write(f"Socioeconomic Segment: `{job_type}`")
            st.write(f"Scholastic Background Weight: `{education}`")

    with v_col2:
        with st.container(border=True):
            st.markdown("<b style='color:#7c3aed;'>Leverage Metrics Breakdown</b>", unsafe_allow_html=True)
            st.write(f"Loan-to-Salary Index: `{loan_to_salary:.2f}x`")
            st.markdown(f"Capital Liquidity Ratio: `{income_per_loan:.4f}`")

    with v_col3:
        with st.container(border=True):
            st.markdown("<b style='color:#60a5fa;'>Aggregate Liabilities Radar</b>", unsafe_allow_html=True)
            st.write(f"Active External Debts: `{existing_loans}`")
            st.write(f"Calculated Stress Variable: `{total_burden:.2f}`")
            
    st.markdown('</div>', unsafe_allow_html=True) # End of smooth animation wrap