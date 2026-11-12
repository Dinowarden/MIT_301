# ============================================================
# PART 3: Categorical Encoding & Feature Scaling
# File: 03_encoding_scaling_german_credit.py
# MIT 8301 – Machine Learning and AI Fundamentals
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ------------------------------------------------------------
# 1. Load and prepare the data (from previous steps)
# ------------------------------------------------------------
df = pd.read_csv("german_credit_data-1.csv")

# Drop unnecessary index column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Handle missing values (from Part 2)
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

print("="*65)
print("PART 3: CATEGORICAL ENCODING & SCALING")
print("="*65)

# ------------------------------------------------------------
# 2. Identify feature types
# ------------------------------------------------------------
print("\n--- Feature Types ---")
print("Numerical features :", ['Age', 'Job', 'Credit amount', 'Duration'])
print("Categorical features:", ['Sex', 'Housing', 'Saving accounts', 
                                'Checking account', 'Purpose'])
print("Target             : Risk")

# ------------------------------------------------------------
# 3. Encode the Target Variable
# ------------------------------------------------------------
# Risk: good → 0, bad → 1  (we want to predict the minority class 'bad')
df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})

print("\n--- Target Encoding ---")
print(df['Risk'].value_counts())
print("good → 0 | bad → 1")

# ------------------------------------------------------------
# 4. Split features and target
# ------------------------------------------------------------
X = df.drop('Risk', axis=1)
y = df['Risk']

# ------------------------------------------------------------
# 5. Train-Test Split (IMPORTANT: Do this BEFORE scaling)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y          # preserves class balance
)

print("\n--- Train-Test Split ---")
print(f"Training set : {X_train.shape[0]} samples")
print(f"Testing set  : {X_test.shape[0]} samples")
print(f"\nTarget distribution in training set:")
print(y_train.value_counts(normalize=True).round(3))

# ------------------------------------------------------------
# 6. Encode Categorical Variables
# ------------------------------------------------------------
print("\n" + "="*65)
print("CATEGORICAL ENCODING")
print("="*65)

# Binary / low-cardinality → Label Encoding
# Higher-cardinality / nominal → One-Hot Encoding

categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

# We will use One-Hot Encoding for all categorical features
# (safer and avoids implying ordinal relationships)

X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded  = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns (in case some categories are missing in test set)
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join='left', axis=1, fill_value=0
)

print("\nShape after One-Hot Encoding:")
print(f"X_train_encoded : {X_train_encoded.shape}")
print(f"X_test_encoded  : {X_test_encoded.shape}")

print("\nEncoded feature names:")
print(X_train_encoded.columns.tolist())

# ------------------------------------------------------------
# 7. Feature Scaling (StandardScaler)
# ------------------------------------------------------------
print("\n" + "="*65)
print("FEATURE SCALING")
print("="*65)

print("""
Justification:
- We use StandardScaler (Z-score normalization)
- Scaling is applied ONLY on the training set to fit the scaler
- Then the same scaler is used to transform the test set
- This prevents data leakage
""")

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train_encoded)

# Transform both train and test
X_test_scaled = scaler.transform(X_test_encoded)

# Convert back to DataFrame for easier inspection (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns, index=X_train_encoded.index)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns, index=X_test_encoded.index)

print("\n--- Scaling completed successfully ---")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape : {X_test_scaled.shape}")

print("\nFirst 5 rows of scaled training data:")
print(X_train_scaled.head().round(3))

# ------------------------------------------------------------
# 8. Final Summary
# ------------------------------------------------------------
print("\n" + "="*65)
print("PART 3 COMPLETED SUCCESSFULLY")
print("="*65)
print("""
Summary of actions taken:
1. Target variable encoded (good=0, bad=1)
2. Train-test split performed BEFORE any scaling (no data leakage)
3. Categorical variables encoded using One-Hot Encoding
4. Numerical features scaled using StandardScaler
5. Scaler fitted only on training data
""")