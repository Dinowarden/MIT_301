# ============================================================
# PART 1: Data Loading & Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ------------------------------------------------------------
# 1.1 Load the dataset
# ------------------------------------------------------------
df = pd.read_csv("german_credit_data-1.csv")   # use the correct filename

print("="*60)
print("1. DATASET LOADED SUCCESSFULLY")
print("="*60)

# ------------------------------------------------------------
# 1.2 Basic Information
# ------------------------------------------------------------
print("\n--- Shape of the dataset ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Last 5 rows ---")
print(df.tail())

# ------------------------------------------------------------
# 1.3 Descriptive Statistics
# ------------------------------------------------------------
print("\n" + "="*60)
print("2. DESCRIPTIVE STATISTICS")
print("="*60)

print("\n--- Numerical features ---")
print(df.describe())

print("\n--- Categorical features ---")
print(df.describe(include=['object']))

# ------------------------------------------------------------
# 1.4 Target Variable Balance
# ------------------------------------------------------------
print("\n" + "="*60)
print("3. TARGET VARIABLE (Risk) DISTRIBUTION")
print("="*60)

print(df['Risk'].value_counts())
print("\nProportions:")
print(df['Risk'].value_counts(normalize=True).round(3))

# ------------------------------------------------------------
# 1.5 Identify Categorical vs Numerical Features
# ------------------------------------------------------------
print("\n" + "="*60)
print("4. FEATURE TYPES")
print("="*60)

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove the index-like column if present
if 'Unnamed: 0' in numerical_features:
    numerical_features.remove('Unnamed: 0')

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# ------------------------------------------------------------
# 1.6 Missing Values Overview
# ------------------------------------------------------------
print("\n" + "="*60)
print("5. MISSING VALUES")
print("="*60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# ------------------------------------------------------------
# 1.7 Visualizations
# ------------------------------------------------------------
print("\n" + "="*60)
print("6. VISUALIZATIONS")
print("="*60)

# --- Target distribution ---
plt.figure(figsize=(7, 5))
sns.countplot(x='Risk', data=df, palette='Set2')
plt.title('Target Variable Distribution (Risk)', fontsize=14)
plt.ylabel('Count')
plt.show()

# --- Age distribution by Risk ---
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Age', hue='Risk', bins=20, kde=True, palette='Set1')
plt.title('Age Distribution by Credit Risk')
plt.show()

# --- Credit amount distribution by Risk ---
plt.figure(figsize=(10, 5))
sns.boxplot(x='Risk', y='Credit amount', data=df, palette='Set2')
plt.title('Credit Amount by Risk Status')
plt.show()

# --- Duration by Risk ---
plt.figure(figsize=(10, 5))
sns.boxplot(x='Risk', y='Duration', data=df, palette='Set2')
plt.title('Loan Duration by Risk Status')
plt.show()

# --- Categorical features vs Risk ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(x='Sex', hue='Risk', data=df, ax=axes[0,0], palette='Set2')
axes[0,0].set_title('Sex vs Risk')

sns.countplot(x='Housing', hue='Risk', data=df, ax=axes[0,1], palette='Set2')
axes[0,1].set_title('Housing vs Risk')

sns.countplot(x='Saving accounts', hue='Risk', data=df, ax=axes[1,0], palette='Set2')
axes[1,0].set_title('Saving accounts vs Risk')
axes[1,0].tick_params(axis='x', rotation=45)

sns.countplot(x='Checking account', hue='Risk', data=df, ax=axes[1,1], palette='Set2')
axes[1,1].set_title('Checking account vs Risk')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# --- Purpose distribution ---
plt.figure(figsize=(12, 5))
sns.countplot(y='Purpose', hue='Risk', data=df, palette='Set2', order=df['Purpose'].value_counts().index)
plt.title('Loan Purpose vs Risk')
plt.show()

# --- Correlation heatmap (numerical features only) ---
plt.figure(figsize=(8, 6))
corr = df[['Age', 'Job', 'Credit amount', 'Duration']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap – Numerical Features')
plt.show()

print("\nEDA completed successfully.")