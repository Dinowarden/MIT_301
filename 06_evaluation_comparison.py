# ============================================================
# PART 6: Model Evaluation & Comparison
# File: 06_evaluation_comparison.py
# MIT 8301 – Machine Learning and AI Fundamentals
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
import warnings
warnings.filterwarnings('ignore')

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
print("PART 6: MODEL EVALUATION & COMPARISON")
print("="*65)

# ------------------------------------------------------------
# 2. Train the Best Models (using previously found parameters)
# ------------------------------------------------------------
# Logistic Regression
lr_model = LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# SVM
svm_model = SVC(C=1, kernel='rbf', gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Naïve Bayes
nb_model = GaussianNB(var_smoothing=1e-9)
nb_model.fit(X_train_scaled, y_train)

models = {
    'Logistic Regression': lr_model,
    'SVM': svm_model,
    'Naïve Bayes': nb_model
}

# ------------------------------------------------------------
# 3. Evaluate All Models
# ------------------------------------------------------------
results = []

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results)
results_df = results_df.round(4)

print("\n--- Performance Comparison Table ---")
print(results_df.to_string(index=False))

# ------------------------------------------------------------
# 4. Detailed Classification Reports
# ------------------------------------------------------------
for name, model in models.items():
    print("\n" + "-"*65)
    print(f"Classification Report: {name}")
    print("-"*65)
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=['Good (0)', 'Bad (1)']))

# ------------------------------------------------------------
# 5. Confusion Matrices
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. ROC Curves
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=plt.gca())

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title('ROC Curves - Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Strengths and Weaknesses Discussion
# ------------------------------------------------------------
print("\n" + "="*65)
print("STRENGTHS AND WEAKNESSES ANALYSIS")
print("="*65)

print("""
1. Logistic Regression
   Strengths:
   - Highly interpretable (coefficients show feature importance)
   - Fast training and prediction
   - Performs well when relationship is roughly linear
   - Less prone to overfitting on small datasets
   Weaknesses:
   - Assumes linear decision boundary
   - May underperform if complex non-linear patterns exist

2. Support Vector Machine (SVM)
   Strengths:
   - Effective in high-dimensional spaces
   - Kernel trick allows modelling non-linear relationships
   - Robust to overfitting (especially with proper C)
   Weaknesses:
   - Less interpretable (black-box)
   - Computationally more expensive
   - Sensitive to feature scaling and hyperparameter choices

3. Naïve Bayes
   Strengths:
   - Extremely fast training and prediction
   - Works well with small datasets
   - Performs surprisingly well even when independence assumption is violated
   Weaknesses:
   - Strong independence assumption rarely holds in real data
   - Usually lower predictive performance than LR and SVM on this dataset
   - Zero-frequency problem (handled by var_smoothing)

Overall Recommendation:
Logistic Regression is preferred for this credit risk problem because:
- It offers the best balance of performance and interpretability
- Feature coefficients can be directly used for business decision-making
- Regulatory environments in credit scoring often require explainable models
""")

print("\n" + "="*65)
print("PART 6 COMPLETED SUCCESSFULLY")
print("="*65)