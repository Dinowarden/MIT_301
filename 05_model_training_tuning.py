# ============================================================
# PART 5: Model Training & Hyperparameter Tuning
# File: 05_model_training_tuning.py
# MIT 8301 – Machine Learning and AI Fundamentals
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. Load and Prepare Data
# ------------------------------------------------------------
df = pd.read_csv("german_credit_data-1.csv")

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Handle missing values
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

# Encode target
df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})

X = df.drop('Risk', axis=1)
y = df['Risk']

# One-hot encoding
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*65)
print("PART 5: MODEL TRAINING & HYPERPARAMETER TUNING")
print("="*65)

# ============================================================
# 2. Importance of Hyperparameter Tuning
# ============================================================
print("""
Why Hyperparameter Tuning is Important:
---------------------------------------
Hyperparameters control the learning process of a model.
Unlike model parameters (weights), hyperparameters are set before training.

Good hyperparameter choices can significantly improve:
- Accuracy and generalization
- Ability to handle class imbalance
- Control of overfitting / underfitting

Techniques used:
1. GridSearchCV  → Exhaustive search over specified parameter grid
2. RandomizedSearchCV → Faster search over random combinations
Both use Cross-Validation to ensure robust performance estimates.
""")

# ============================================================
# 3. Logistic Regression - Hyperparameter Tuning
# ============================================================
print("\n" + "-"*65)
print("3.1 Logistic Regression - GridSearchCV")
print("-"*65)

log_reg = LogisticRegression(max_iter=1000, random_state=42)

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_lr = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_lr.fit(X_train_scaled, y_train)

print(f"Best Parameters : {grid_lr.best_params_}")
print(f"Best CV F1-Score: {grid_lr.best_score_:.4f}")

best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
print(f"Test Accuracy   : {accuracy_score(y_test, y_pred_lr):.4f}")

# ============================================================
# 4. Support Vector Machine (SVM) - Hyperparameter Tuning
# ============================================================
print("\n" + "-"*65)
print("3.2 Support Vector Machine - GridSearchCV")
print("-"*65)

svm = SVC(probability=True, random_state=42)

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(
    estimator=svm,
    param_grid=param_grid_svm,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_svm.fit(X_train_scaled, y_train)

print(f"Best Parameters : {grid_svm.best_params_}")
print(f"Best CV F1-Score: {grid_svm.best_score_:.4f}")

best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)
print(f"Test Accuracy   : {accuracy_score(y_test, y_pred_svm):.4f}")

# ============================================================
# 5. Naïve Bayes - Hyperparameter Tuning
# ============================================================
print("\n" + "-"*65)
print("3.3 Gaussian Naïve Bayes - GridSearchCV")
print("-"*65)

nb = GaussianNB()

param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=20)
}

grid_nb = GridSearchCV(
    estimator=nb,
    param_grid=param_grid_nb,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_nb.fit(X_train_scaled, y_train)

print(f"Best Parameters : {grid_nb.best_params_}")
print(f"Best CV F1-Score: {grid_nb.best_score_:.4f}")

best_nb = grid_nb.best_estimator_
y_pred_nb = best_nb.predict(X_test_scaled)
print(f"Test Accuracy   : {accuracy_score(y_test, y_pred_nb):.4f}")

# ============================================================
# 6. Summary of Best Models
# ============================================================
print("\n" + "="*65)
print("SUMMARY OF TUNED MODELS")
print("="*65)

results = {
    'Model': ['Logistic Regression', 'SVM', 'Naïve Bayes'],
    'Best Parameters': [grid_lr.best_params_, grid_svm.best_params_, grid_nb.best_params_],
    'Best CV F1': [grid_lr.best_score_, grid_svm.best_score_, grid_nb.best_score_],
    'Test Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_nb)
    ]
}

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "="*65)
print("PART 5 COMPLETED SUCCESSFULLY")
print("="*65)
print("""
Summary:
- Trained three models: Logistic Regression, SVM, and Naïve Bayes
- Applied GridSearchCV with 5-fold cross-validation
- Optimized relevant hyperparameters for each model
- Used F1-score as scoring metric due to class imbalance
""")