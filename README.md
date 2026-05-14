# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project to predict customer churn and provide actionable business insights using explainable AI.

---

## 🚀 Project Overview

Customer churn is a major challenge for telecom companies. This project builds a predictive model to identify customers likely to leave and explains the reasons behind their decisions.

---

## 📂 Dataset

- IBM Telco Customer Churn Dataset
- ~7,000 customers
- 20+ features including:
  - Demographics
  - Account information
  - Services subscribed

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- SHAP (Explainable AI)

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked missing values and data types
- Analyzed churn distribution
- Identified patterns in:
  - Contract type
  - Monthly charges
  - Tenure

📌 Key Insight:
Customers with month-to-month contracts have a higher churn rate.

---

## 🌳 Baseline Model — Decision Tree

- Built a simple Decision Tree classifier
- Fully interpretable model
- Visualized decision rules

📊 Evaluation:
- Confusion Matrix
- Precision, Recall, F1-score

---

## 🌲 Advanced Model — Random Forest Pipeline

- Built a full ML pipeline using:
  - ColumnTransformer
  - OneHotEncoding (categorical)
  - StandardScaler (numerical)
- Used RandomForestClassifier with class balancing

📊 Evaluation:
- Stratified 5-Fold Cross Validation
- ROC-AUC Score
- F1 Score

---

## 🧠 Model Explainability (SHAP)

Used SHAP values to understand model predictions:

- Global Explainability:
  - Feature importance (summary plot)
- Local Explainability:
  - Individual prediction breakdown (waterfall plot)

💡 Key Insight:
- High monthly charges → higher churn risk
- Long tenure → lower churn risk

---

## 📈 Model Comparison

| Model          | ROC-AUC | F1 Score |
|---------------|--------|---------|
| Decision Tree | 0.XX   | 0.XX    |
| Random Forest | 0.XX   | 0.XX    |

---

## 💼 Business Impact

This model helps businesses:

- Identify high-risk customers
- Design targeted retention strategies
- Reduce customer churn and increase revenue

---

## 📁 Project Structure
