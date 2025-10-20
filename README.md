# 🧠 Loan Approval Prediction – Logistic Regression Pipeline

A complete end-to-end machine learning project for **loan approval prediction**, focusing on robust preprocessing, model evaluation, and interpretability using various forms of **Logistic Regression** (L1, L2, and ElasticNet regularization).

---

## 📘 Project Overview

This project builds a full **machine learning pipeline** that predicts whether a loan application should be approved based on applicant and financial features.  

It follows best practices in:
- Data preprocessing (handling missing values, encoding, and scaling)
- Model training and comparison
- Performance evaluation using key classification metrics
- Visualization (confusion matrix and score summary)

The goal is to establish a **reliable and interpretable baseline** before exploring more complex algorithms (e.g., Decision Tree, Random Forest, or XGBoost — to be added later).

---

## 🧩 Features & Workflow

### 1️⃣ Data Preparation
- Handle missing values with `SimpleImputer`
- Encode categorical features using `OneHotEncoder`
- Scale numerical features using `StandardScaler`
- Combine transformations via `ColumnTransformer`

### 2️⃣ Model Training
Implements and evaluates multiple **regularized Logistic Regression** variants:
- **Logistic Regression (L2 penalty)** – Ridge regularization  
- **Logistic Regression (L1 penalty)** – Lasso regularization  
- **ElasticNet Logistic Regression** – Hybrid (L1 + L2)  

Each model is trained and evaluated through a unified helper function.

### 3️⃣ Evaluation Metrics
For both train and test datasets:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Visual output:
- Confusion matrix for test predictions

### 4️⃣ Model Selection Logic
The key performance indicator (KPI) for model acceptance is the **F1-score**, ensuring a balance between precision and recall, with a target threshold of **F1 ≥ 0.80** on the test set.

---

## 🧮 Key Components

| File | Description |
|------|--------------|
| `Loan_approval_full_pipeline.py` | Main script containing data preprocessing, model training, and evaluation |
| `evaluate_model()` | Helper function for standardized training and scoring |
| `plot_conf_mat()` | Confusion matrix visualization helper |
| `results` | List storing model performance metrics for later comparison |

---

## 📊 Sample Output

