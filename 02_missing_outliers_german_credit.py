# ============================================================
# PART 2: Handling Missing Values & Outliers
# File: 02_missing_outliers_german_credit.py
# MIT 8301 – Machine Learning and AI Fundamentals
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
df = pd.read_csv("german_credit_data-1.csv")

# Drop unnecessary index column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

print("="*65)
print("PART 2: HANDLING MISSING VALUES & OUTLIERS")
print("="*65)

# ------------------------------------------------------------
# 2. Missing Values Analysis
# ------------------------------------------------------------
print("\n--- Missing Values Before Treatment ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage (%)': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

print("""
Justification for handling method:
- 'Saving accounts' → 183 missing values (18.3%)
- 'Checking account' → 394 missing values (39.4%)

These are categorical variables. In credit risk data, missing account 
information often has meaning (customer may not have that account).
Therefore, we create a new category called 'unknown' instead of 
deleting rows or using mode imputation. This preserves information.
""")

# ------------------------------------------------------------
# 3. Handle Missing Values
# ------------------------------------------------------------
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

print("--- Missing Values After Treatment ---")
print(df.isnull().sum())
print("\nMissing values successfully handled.")

# ------------------------------------------------------------
# 4. Outlier Detection using IQR
# ------------------------------------------------------------
print("\n" + "="*65)
print("OUTLIER DETECTION (IQR METHOD)")
print("="*65)

numerical_cols = ['Age', 'Credit amount', 'Duration']

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

for col in numerical_cols:
    count, lower, upper = detect_outliers_iqr(df, col)
    print(f"\n{col}:")
    print(f"  Lower bound : {lower:.2f}")
    print(f"  Upper bound : {upper:.2f}")
    print(f"  Outliers    : {count} ({count/len(df)*100:.1f}%)")

# ------------------------------------------------------------
# 5. Visualize Outliers
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Boxplot of {col}', fontsize=12)

plt.suptitle('Outlier Detection - Numerical Features', fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Outlier Treatment
# ------------------------------------------------------------
print("\n" + "="*65)
print("OUTLIER TREATMENT DECISION")
print("="*65)

print("""
Treatment Strategy:
1. Age            → Keep outliers (realistic elderly customers)
2. Credit amount  → Cap extreme values (Winsorization)
3. Duration       → Cap extreme values (Winsorization)

Reason: Dataset is small (1000 rows). Deleting outliers would 
cause unnecessary loss of information. Capping reduces the 
influence of extreme values while preserving all records.
""")

def cap_outliers(data, column):
    """Cap outliers using IQR method (Winsorization)"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    return data

# Apply capping
df = cap_outliers(df, 'Credit amount')
df = cap_outliers(df, 'Duration')

print("Outliers in 'Credit amount' and 'Duration' have been capped.")
print("Age outliers retained.")

# ------------------------------------------------------------
# 7. Final Verification
# ------------------------------------------------------------
print("\n--- Summary Statistics After Treatment ---")
print(df[['Age', 'Credit amount', 'Duration']].describe().round(2))

print("\n--- Final Missing Values Check ---")
print(df.isnull().sum())

print("\n" + "="*65)
print("PART 2 COMPLETED SUCCESSFULLY")
print("="*65)