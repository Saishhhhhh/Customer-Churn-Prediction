# 📌 Project Conclusion: Customer Churn Prediction

## 1️⃣ Problem Context

This project focuses on **customer churn prediction**, where the primary business objective is to **identify customers likely to churn (Class 1)**.
In such problems:

* **False Negatives (missing a churner)** are more costly than False Positives
* Therefore, **Recall for churners is prioritized over accuracy**

To align the model with business needs, **decision threshold tuning using Optuna** was applied instead of relying on the default 0.5 threshold.

---

## 2️⃣ Models & Techniques Evaluated

The following models and configurations were compared:

* Logistic Regression (with & without SMOTE)
* Random Forest (with & without SMOTE)
* XGBoost (with & without SMOTE)
* Threshold optimization using Optuna for all models

Evaluation was done using:

* Precision, Recall, F1-score (especially for churn class)
* Accuracy used only as a secondary metric

---

## 3️⃣ Key Results Summary (Churn Class – Class 1)

| Model               | SMOTE | Recall (Churn) | F1 (Churn) | Accuracy |
| ------------------- | ----- | -------------- | ---------- | -------- |
| Logistic Regression | ❌     | **0.82**       | **0.62**   | **0.73** |
| Logistic Regression | ✅     | **0.92**       | 0.59       | 0.67     |
| Random Forest       | ❌     | 0.92           | 0.57       | 0.63     |
| Random Forest       | ✅     | 0.89           | 0.60       | 0.68     |
| XGBoost             | ❌     | 0.75           | 0.57       | 0.69     |
| XGBoost             | ✅     | 0.88           | 0.56       | 0.63     |

---

## 4️⃣ Key Insights & Learnings

### 🔹 1. Logistic Regression Outperformed Complex Models

Despite being a simpler model, **Logistic Regression consistently achieved the best balance between recall and precision** for churners after threshold tuning.

This indicates that:

* The dataset has **mostly linear decision boundaries**
* Well-engineered features + calibration matter more than model complexity

---

### 🔹 2. SMOTE Increased Recall but Hurt Overall Stability

* SMOTE significantly increased churn recall
* However, it also:

  * Reduced accuracy
  * Increased false positives
  * Degraded probability calibration

In this dataset, **SMOTE did not provide a clear net benefit** over threshold tuning alone.

---

### 🔹 3. Threshold Tuning Was More Impactful Than SMOTE

Adjusting the decision threshold:

* Improved recall without introducing synthetic noise
* Allowed alignment with business objectives
* Proved more effective than aggressive resampling techniques

This highlights that **post-model decision optimization is as important as model selection**.

---

### 🔹 4. Accuracy Is a Misleading Metric for Churn

Models with higher accuracy often:

* Missed more churners
* Performed worse from a business perspective

This project reinforces that **accuracy should not be the primary metric in imbalanced, cost-sensitive problems like churn**.

---

## 5️⃣ Final Model Selection

### ✅ Chosen Model

**Logistic Regression without SMOTE + Optuna-optimized threshold**

### ✅ Reasoning

* High churn recall (0.92)
* Best F1-score for churn class
* Stable and interpretable
* No synthetic data dependency
* Business-aligned performance

---

## 6️⃣ Business Takeaway

> In churn prediction, simpler and well-calibrated models with optimized decision thresholds can outperform complex ensemble models. Prioritizing recall and aligning evaluation with business cost leads to more practical and deployable solutions.

---

## 7️⃣ What This Project Demonstrates

* Handling imbalanced data correctly
* Proper use of threshold tuning
* Critical evaluation beyond accuracy
* Understanding of business-driven ML decisions
* Clean experimental comparison of models

---

> This project emphasizes that **model selection should be driven by problem context and business impact, not algorithm complexity**.

---

