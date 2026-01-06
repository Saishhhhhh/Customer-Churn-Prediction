# 📊 Customer Churn Prediction

A machine learning project focused on predicting customer churn for a telecommunications company. This project demonstrates proper handling of imbalanced classification problems with business-aligned evaluation metrics and threshold optimization.

---

## 📋 Table of Contents

1. [Problem Context](#1-problem-context)
2. [Dataset Overview](#2-dataset-overview)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Training & Optimization](#5-model-training--optimization)
6. [Results Summary](#6-results-summary)
7. [Key Insights](#7-key-insights)
8. [Final Model Selection](#8-final-model-selection)
9. [Project Structure](#9-project-structure)

---

## 1️⃣ Problem Context

This project focuses on **customer churn prediction**, where the primary business objective is to **identify customers likely to churn (Class 1)**.

**Business Requirements:**
- **False Negatives (missing a churner)** are more costly than False Positives
- **Recall for churners is prioritized over accuracy**
- Model must be interpretable and deployable

**Solution Approach:**
- Decision threshold tuning using **Optuna** instead of default 0.5 threshold
- F2-score optimization (beta=2) to prioritize recall
- Comprehensive comparison of multiple models with and without SMOTE

---

## 2️⃣ Dataset Overview

- **Total Records:** 7,043 customers
- **Features:** 21 columns (demographics, services, billing, contract details)
- **Target Variable:** Churn (Yes/No)
- **Class Distribution:** Imbalanced (~27% churn rate)
- **Missing Values:** Found in `TotalCharges` column (handled during preprocessing)

**Key Features:**
- Demographics: SeniorCitizen, Partner, Dependents
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.
- Billing: MonthlyCharges, TotalCharges, PaymentMethod, PaperlessBilling
- Contract: Contract type, tenure

---

## 3️⃣ Exploratory Data Analysis (EDA)

### Data Quality Checks
- Verified data types and missing values
- Identified `TotalCharges` as object type (converted to numeric)
- Removed 11 records with missing `TotalCharges` values

### Class Distribution Analysis
- **Churn Rate:** ~27% (imbalanced dataset)
- Visualized churn distribution using count plots

### Feature Analysis
- **Categorical Features:** Analyzed relationship between all categorical predictors and churn using count plots
- **Numerical Features:** 
  - Created KDE plots for `MonthlyCharges` and `TotalCharges` by churn status
  - Identified patterns showing churners tend to have different charge distributions

### Key Observations
- Clear class imbalance requiring special handling
- Multiple categorical features with "No service" categories that needed standardization
- Numerical features required scaling for model training

---

## 4️⃣ Data Preprocessing

### 4.1 Feature Removal
- Dropped `customerID` (unique identifier, not predictive)
- Dropped `gender` (removed to avoid potential bias)

### 4.2 Data Cleaning
- **Standardized "No service" categories:**
  - Replaced `"No phone service"` → `"No"` in `MultipleLines`
  - Replaced `"No internet service"` → `"No"` in: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

### 4.3 Missing Value Handling
- Converted `TotalCharges` from object to numeric (using `pd.to_numeric` with `errors='coerce'`)
- Dropped rows with null values (11 records removed, final dataset: 7,032 records)

### 4.4 Feature Encoding & Scaling

Used `ColumnTransformer` with three transformation pipelines:

1. **OrdinalEncoder** (Binary categorical features):
   - `Partner`, `Dependents`, `PaperlessBilling`, `PhoneService`
   - `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`
   - `TechSupport`, `StreamingTV`, `StreamingMovies`, `Churn`

2. **OneHotEncoder** (Multi-category features, drop='first'):
   - `InternetService` → 3 categories (Fiber optic, No, DSL)
   - `PaymentMethod` → 4 categories (Credit card, Electronic check, Mailed check, Bank transfer)
   - `Contract` → 3 categories (Month-to-month, One year, Two year)

3. **StandardScaler** (Numerical features):
   - `MonthlyCharges`, `TotalCharges`, `tenure`

**Final Preprocessed Dataset:**
- 23 features (after encoding)
- All features standardized and ready for modeling
- Saved as `preprocessed_data.csv`

### 4.5 Outlier Detection
- Applied 3-sigma rule for outlier detection in `TotalCharges`
- No significant outliers found that required removal

---

## 5️⃣ Model Training & Optimization

### 5.1 Data Splitting
- **Train-Test Split:** 80/20 (random_state=42, stratified)
- **Training Set:** 5,625 samples
- **Test Set:** 1,407 samples

### 5.2 Handling Class Imbalance
- **SMOTE (Synthetic Minority Oversampling Technique):** Applied to training data for comparison
- Created balanced training sets for models with SMOTE
- Compared performance with and without SMOTE

### 5.3 Models Evaluated

Six model configurations were tested:

1. **Logistic Regression** (without SMOTE)
2. **Logistic Regression** (with SMOTE)
3. **Random Forest** (without SMOTE)
4. **Random Forest** (with SMOTE)
5. **XGBoost** (without SMOTE)
6. **XGBoost** (with SMOTE)

### 5.4 Hyperparameter Optimization

**Framework:** Optuna (Bayesian optimization)

**Optimization Strategy:**
- **Objective Function:** F2-score (beta=2) to prioritize recall
- **Trials:** 50 trials per model configuration
- **Optimized Parameters:**
  - **Logistic Regression:** `max_iter`, `tol`, `C`, `threshold`
  - **Random Forest:** `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`, `threshold`
  - **XGBoost:** `n_estimators`, `max_depth`, `subsample`, `threshold`
- **Threshold Range:** Varied by model (0.2-0.8) to optimize recall-precision trade-off

### 5.5 Evaluation Metrics

Primary metrics (focused on churn class):
- **Recall (Sensitivity):** Most important - ability to catch churners
- **F1-Score:** Balance between precision and recall
- **F2-Score:** Used for optimization (weights recall 2x more than precision)

Secondary metrics:
- **Accuracy:** Overall correctness (less important for imbalanced data)
- **Precision:** Minimize false positives

---

## 6️⃣ Results Summary

### Performance Comparison (Churn Class - Class 1)

| Model               | SMOTE | Recall (Churn) | F1 (Churn) | Accuracy | Optimal Threshold |
| ------------------- | ----- | -------------- | ---------- | -------- | ----------------- |
| **Logistic Regression** | ❌     | **0.82**       | **0.62**   | **0.73** | 0.2583            |
| Logistic Regression | ✅     | **0.92**       | 0.59       | 0.67     | 0.3526            |
| Random Forest       | ❌     | 0.92           | 0.57       | 0.63     | 0.3001            |
| Random Forest       | ✅     | 0.89           | 0.60       | 0.68     | 0.3679            |
| XGBoost             | ❌     | 0.75           | 0.57       | 0.69     | 0.2057            |
| XGBoost             | ✅     | 0.88           | 0.56       | 0.63     | 0.4186            |

**Note:** Bold values indicate the best performing model configuration.

---

## 7️⃣ Key Insights

### 🔹 1. Logistic Regression Outperformed Complex Models

Despite being a simpler model, **Logistic Regression consistently achieved the best balance between recall and precision** for churners after threshold tuning.

**Implications:**
- The dataset has **mostly linear decision boundaries**
- Well-engineered features + calibration matter more than model complexity
- Simpler models are easier to interpret and deploy

### 🔹 2. SMOTE Increased Recall but Hurt Overall Stability

**SMOTE Benefits:**
- Significantly increased churn recall (up to 0.92)

**SMOTE Drawbacks:**
- Reduced overall accuracy
- Increased false positives
- Degraded probability calibration

**Conclusion:** In this dataset, **SMOTE did not provide a clear net benefit** over threshold tuning alone.

### 🔹 3. Threshold Tuning Was More Impactful Than SMOTE

Adjusting the decision threshold:
- Improved recall without introducing synthetic noise
- Allowed alignment with business objectives
- Proved more effective than aggressive resampling techniques

**Key Learning:** **Post-model decision optimization is as important as model selection.**

### 🔹 4. Accuracy Is a Misleading Metric for Churn

Models with higher accuracy often:
- Missed more churners (lower recall)
- Performed worse from a business perspective

**Takeaway:** **Accuracy should not be the primary metric in imbalanced, cost-sensitive problems like churn.**

### 🔹 5. Optimal Thresholds Varied Significantly

- Thresholds ranged from **0.21 to 0.42** (far from default 0.5)
- Lower thresholds generally improved recall for churn class
- Demonstrates the importance of threshold optimization for business alignment

---

## 8️⃣ Final Model Selection

### ✅ Chosen Model

**Logistic Regression without SMOTE + Optuna-optimized threshold (0.2583)**

### ✅ Performance Metrics

- **Recall (Churn):** 0.82
- **F1-Score (Churn):** 0.62 (best among all models)
- **Accuracy:** 0.73
- **Precision (Churn):** 0.50

### ✅ Reasoning

1. **Best F1-score** for churn class (0.62)
2. **High recall** (0.82) - catches most churners
3. **Stable and interpretable** - easy to explain to stakeholders
4. **No synthetic data dependency** - uses real data only
5. **Business-aligned performance** - optimized for recall while maintaining reasonable precision
6. **Optimal threshold** (0.2583) aligns with business cost structure

### ✅ Model Artifacts

- **Saved Model:** `churn_lr_model.pkl`
- **Contains:** Trained model, optimal threshold, feature names
- **Ready for deployment**

---

## 9️⃣ Project Structure

```
Customer-Churn-Prediction/
│
├── 01_EDA.ipynb                 # Exploratory Data Analysis
├── 02_preprocessing.ipynb        # Data Preprocessing Pipeline
├── 03_Model_Training.ipynb      # Model Training & Comparison
├── 04_Model.ipynb               # Final Model Selection & Saving
│
├── raw_data.csv                 # Original dataset
├── preprocessed_data.csv        # Preprocessed dataset
├── churn_lr_model.pkl          # Saved final model
│
└── README.md                    # Project documentation
```

### Notebook Workflow

1. **01_EDA.ipynb:** Data exploration, visualization, and initial insights
2. **02_preprocessing.ipynb:** Data cleaning, encoding, scaling, and feature engineering
3. **03_Model_Training.ipynb:** Model comparison, hyperparameter tuning, and evaluation
4. **04_Model.ipynb:** Final model selection, training, and serialization

---

## 🎯 Business Takeaway

> In churn prediction, simpler and well-calibrated models with optimized decision thresholds can outperform complex ensemble models. Prioritizing recall and aligning evaluation with business cost leads to more practical and deployable solutions.

---

## 📚 What This Project Demonstrates

✅ Proper handling of imbalanced classification problems  
✅ Strategic use of threshold tuning for business alignment  
✅ Critical evaluation beyond accuracy metrics  
✅ Understanding of business-driven ML decisions  
✅ Clean experimental comparison of multiple models  
✅ Feature engineering and preprocessing best practices  
✅ Hyperparameter optimization with Optuna  
✅ Model interpretability and deployment readiness  

---

> **Key Principle:** Model selection should be driven by **problem context and business impact**, not algorithm complexity.

---

## 🔧 Technologies Used

- **Python 3.x**
- **Libraries:**
  - `pandas`, `numpy` - Data manipulation
  - `scikit-learn` - Machine learning models and preprocessing
  - `xgboost` - Gradient boosting classifier
  - `imbalanced-learn` - SMOTE for oversampling
  - `optuna` - Hyperparameter optimization
  - `matplotlib`, `seaborn` - Data visualization
  - `joblib` - Model serialization

---

## 📝 Notes

- All models were evaluated using stratified train-test split to maintain class distribution
- Threshold optimization was performed using F2-score to prioritize recall
- Final model uses Logistic Regression without SMOTE for better stability and interpretability
- Model is ready for deployment with saved artifacts

---
