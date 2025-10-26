# 🧠 Loan Approval Prediction – Logistic Regression & Decision Tree Pipeline

A complete end-to-end machine learning project for **loan approval prediction**, focusing on robust preprocessing, model evaluation, and interpretability.  
This repository contains implementations of **Logistic Regression** (L1, L2, ElasticNet) **and Decision Tree Classifier**, with hyperparameter tuning (GridSearchCV), evaluation, and optional MLflow experiment tracking.

---

## 📘 Project Overview

This project builds a full machine learning pipeline that predicts whether a loan application should be approved based on applicant and financial features. It follows best practices in:

- Data preprocessing (handling missing values, encoding, and scaling)  
- Model training and comparison (Logistic Regression and Decision Tree)  
- Hyperparameter tuning with `GridSearchCV`  
- Performance evaluation using key classification metrics  
- Visualization (confusion matrix and metric comparison)  
- Optional experiment tracking with MLflow

The goal is to establish reliable baseline models that are interpretable and can be compared easily.

---

## 🧩 Features & Workflow

### 1️⃣ Data Preparation
- Drop identifier columns (e.g., `Loan_ID`)  
- Create derived features (e.g., `TotalIncome = ApplicantIncome + CoapplicantIncome`)  
- Handle missing values using `SimpleImputer` (median for numeric, most frequent for categorical)  
- Encode categorical features with `OneHotEncoder`  
- Scale numeric features using `StandardScaler` (important for Logistic Regression)  
- Combine transformations with `ColumnTransformer` and ship them in `Pipeline`

### 2️⃣ Models Implemented
- **Logistic Regression**
  - Variants: `L2` (Ridge), `L1` (Lasso), and `ElasticNet` (mix of L1 & L2)
  - Solver: `saga` (supports L1 & ElasticNet)
  - Regularization parameter `C` tuned via GridSearchCV

- **Decision Tree Classifier**  *(NEW — added)*
  - Non-linear, rule-based model that splits data by feature thresholds
  - Key hyperparameters tuned via GridSearchCV:
    - `criterion`: `gini` or `entropy`
    - `max_depth`: depth limit to control tree complexity
    - `min_samples_split`: min samples required to split an internal node
    - `min_samples_leaf`: min samples required to be at a leaf node
  - Produces **feature importances** (useful for interpretation)
  - Does **not** require feature scaling

Both algorithms are trained as part of a unified pipeline that includes the same preprocessing (imputation + encoding). For Logistic Regression we add `StandardScaler` in the pipeline; Decision Tree pipelines omit scaling.

### 3️⃣ Hyperparameter Tuning
- `GridSearchCV` with `StratifiedKFold` is used to:
  - Find the best `C`, `penalty`, and `l1_ratio` for LogisticRegression
  - Find best `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion` for DecisionTree
- Scoring metric for grid search: **F1-score** (balanced consideration of precision & recall)

### 4️⃣ Evaluation
For each model (baseline and tuned):
- Train & Test metrics reported:
  - **Accuracy**, **Precision**, **Recall**, **F1-score**
- Confusion matrix plotted for the test set
- Summary table with train/test results across all experiments
- For Decision Tree: plot or print top **feature importances**

### 5️⃣ Comparison & Practical Notes
- **Logistic Regression**
  - Best when relationships are roughly linear in log-odds
  - Interpretable coefficients, simple regularization (L1 -> sparsity, L2 -> shrinkage)
  - Requires scaling and careful feature engineering

- **Decision Tree**
  - Handles non-linear relationships and interactions automatically
  - Produces human-readable decision rules
  - Prone to overfitting if unrestricted — control via `max_depth`, `min_samples_leaf`, etc.
  - No scaling needed; robust to feature monotonic transformations

**Which to choose?**
- Prefer **Logistic Regression** when interpretability and simplicity are required and the problem is approximately linear.  
- Prefer **Decision Trees** when you expect non-linear interactions and want rule-based explanations. Use **Random Forest** or **XGBoost** when you need better accuracy (at the cost of interpretability).

---

## 📁 Repository Structure

