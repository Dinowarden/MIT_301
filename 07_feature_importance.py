# ============================================================
# PART 7: Feature Importance & Interpretation
# File: 07_feature_importance.py
# MIT 8301 – Machine Learning and AI Fundamentals
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------
# 1. Load and Prepare Data
# ------------------------------------------------------------
df = pd.read_csv("german_credit_data-1.csv")

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')
df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})

X = df.drop('Risk', axis=1)
y = df['Risk']

categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*65)
print("PART 7: FEATURE IMPORTANCE & INTERPRETATION")
print("="*65)

# ------------------------------------------------------------
# 2. Train the Best Model (Logistic Regression)
# ------------------------------------------------------------
best_model = LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
best_model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 3. Extract Feature Importance (Coefficients)
# ------------------------------------------------------------
feature_names = X_encoded.columns
coefficients = best_model.coef_[0]

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

# Sort by absolute coefficient value
importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False)

print("\n--- Ranked Feature Importance (Logistic Regression Coefficients) ---")
print(importance_df[['Feature', 'Coefficient']].to_string(index=False))

# ------------------------------------------------------------
# 4. Visualize Feature Importance
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.barplot(
    x='Coefficient',
    y='Feature',
    data=importance_df,
    palette='coolwarm'
)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.title('Feature Importance - Logistic Regression Coefficients', fontsize=14)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Top 10 features
print("\n--- Top 10 Most Influential Features ---")
print(importance_df.head(10)[['Feature', 'Coefficient']].to_string(index=False))

# ------------------------------------------------------------
# 5. Business Interpretation for Credit Risk Decisions
# ------------------------------------------------------------
print("\n" + "="*65)
print("BUSINESS INTERPRETATION & DECISION-MAKING INSIGHTS")
print("="*65)

print("""
How these insights can inform credit risk decision-making:

1. Checking Account Status
   - Customers with little or no checking account history tend to have 
     higher risk of default.
   - Recommendation: Apply stricter scrutiny or request additional 
     guarantees for applicants with weak checking account status.

2. Credit Amount
   - Higher loan amounts are generally associated with higher risk.
   - Recommendation: Implement tiered interest rates or lower approval 
     limits for very high credit amounts, especially for higher-risk profiles.

3. Loan Duration
   - Longer loan durations increase the probability of default.
   - Recommendation: Prefer shorter-term loans for riskier customers 
     or adjust pricing (higher interest) for long-duration loans.

4. Savings Account Status
   - Limited or no savings is a strong indicator of higher credit risk.
   - Recommendation: Encourage customers to build savings or use 
     savings level as a positive scoring factor.

5. Housing Status
   - People who own houses generally show lower risk compared to 
     those who rent or live for free.
   - Recommendation: Home ownership can be treated as a positive 
     factor in the credit scoring model.

6. Age
   - Younger applicants tend to show slightly higher risk in many cases.
   - Recommendation: Combine age with other stability indicators 
     (job, housing, savings) rather than using age alone.

Practical Applications:
- Develop a transparent credit scoring system based on these coefficients.
- Create risk-based pricing (higher interest rates for higher-risk profiles).
- Design targeted credit products for different customer segments.
- Improve manual review processes by highlighting the most influential factors.
- Support regulatory compliance by providing explainable decisions.
""")

print("\n" + "="*65)
print("PART 7 COMPLETED SUCCESSFULLY")
print("="*65)
print("""
Summary:
- Extracted coefficients from the best Logistic Regression model
- Ranked features by importance
- Visualized feature importance
- Provided clear business interpretation for credit risk decisions
""")