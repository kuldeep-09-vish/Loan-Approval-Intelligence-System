import pandas as pd
import numpy as np

np.random.seed(42)

n = 50000

# -----------------------
# NUMERIC FEATURES (REALISTIC DISTRIBUTION)
# -----------------------
age = np.random.randint(21, 65, n)

salary = np.random.normal(60000, 30000, n).astype(int)
salary = np.clip(salary, 10000, 300000)

experience = np.clip(age - np.random.randint(18, 25, n), 0, 40)

credit_score = np.random.randint(300, 900, n)

existing_loans = np.random.choice([0,1,2,3,4], n, p=[0.4,0.25,0.2,0.1,0.05])

emi_burden = np.random.randint(10, 90, n)

loan_amount = np.random.randint(50000, 1500000, n)

# -----------------------
# CATEGORICAL FEATURES
# -----------------------
job_type = np.random.choice(
    ["Government Job", "Private Job", "Self Employed"],
    n, p=[0.3, 0.5, 0.2]
)

city = np.random.choice(
    ["Indore", "Bhopal", "Delhi", "Mumbai", "Jaipur"],
    n
)

education = np.random.choice(
    ["High School", "Graduate", "Post Graduate"],
    n, p=[0.3, 0.5, 0.2]
)

marital_status = np.random.choice(
    ["Single", "Married", "Divorced"],
    n, p=[0.4, 0.5, 0.1]
)

loan_purpose = np.random.choice(
    ["Home Loan", "Car Loan", "Education Loan", "Personal Loan"],
    n
)

# -----------------------
# REALISTIC LOAN APPROVAL LOGIC
# -----------------------
score = (
    (salary > 50000).astype(int) * 2 +
    (credit_score > 650).astype(int) * 3 +
    (experience > 2).astype(int) * 1 +
    (existing_loans <= 2).astype(int) * 2 +
    (emi_burden < 50).astype(int) * 2 +
    (job_type == "Government Job").astype(int) * 1 +
    (education != "High School").astype(int) * 1
)

# risk penalties
score -= (credit_score < 500).astype(int) * 2
score -= (emi_burden > 70).astype(int) * 2
score -= (existing_loans >= 3).astype(int) * 2

# noise (real life randomness)
noise = np.random.randint(0, 3, n)

final_score = score + noise

loan_approved = (final_score > 6).astype(int)

# -----------------------
# DATAFRAME
# -----------------------
df = pd.DataFrame({
    "Age": age,
    "Salary": salary,
    "Experience": experience,
    "Credit_Score": credit_score,
    "Existing_Loans": existing_loans,
    "EMI_Burden": emi_burden,
    "Loan_Amount": loan_amount,
    "Job_Type": job_type,
    "City": city,
    "Education": education,
    "Marital_Status": marital_status,
    "Loan_Purpose": loan_purpose,
    "Loan_Approved": loan_approved
})

# -----------------------
# SAVE
# -----------------------
df.to_csv("loan_real_dataset.csv", index=False)

print("Realistic dataset created!")
print("Shape:", df.shape)
print(df["Loan_Approved"].value_counts())