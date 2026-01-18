# 📊 Customer Churn Analysis & Prediction System using EDA and ML

An end-to-end **Data Science and Machine Learning project** that analyzes customer churn behavior in a telecom dataset.  
The project follows a complete pipeline including **Exploratory Data Analysis (EDA), data cleaning, feature engineering, training multiple ML models, evaluating them using standard metrics, and predicting churn risk for new customers**.

**Churn** refers to the rate at which customers stop using a company’s service over a given period and is a key business metric in subscription-based industries.

---

## 📌 Project Objective

The main objectives of this project are to:
- Understand customer churn patterns using data analysis
- Identify features that strongly influence churn
- Perform data cleaning and feature engineering
- Train **multiple machine learning models**
- Evaluate and compare model performance using standard metrics
- Predict churn probability and classify customer risk
- Save trained models for reuse or deployment

---

## 📁 Dataset Overview

The dataset contains customer-level telecom information including:

- **Demographics**: gender, partner, dependents  
- **Service usage**: phone service, internet service, streaming services  
- **Account information**: tenure, contract type, payment method  
- **Billing details**: monthly charges, total charges  
- **Target variable**: `Churn` (Yes / No)

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was conducted to understand data distribution, detect patterns, and gain business insights using visualizations and summary statistics.

### Churn Distribution  
Shows the proportion of customers who churned versus those who were retained.

![Churn Distribution](EDA_Graphs/churn_distribution.png)

---

### Tenure vs Churn  
Customers with lower tenure are significantly more likely to churn, indicating early-stage customer dissatisfaction.

![Tenure vs Churn](EDA_Graphs/tenure_vs_churn.png)

---

### Monthly Charges vs Churn  
Higher monthly charges are associated with higher churn probability, suggesting pricing sensitivity.

![Monthly Charges vs Churn](EDA_Graphs/Monthly_Charges_vs_Churn.png)

---

### Monthly Charges Distribution  
Displays the overall distribution of monthly charges across all customers.

![Monthly Charges Distribution](EDA_Graphs/monthly_charges_distribution.png)

---

### Churn by Contract Type  
Month-to-month contracts exhibit the highest churn, while long-term contracts improve retention.

![Churn by Contract Type](EDA_Graphs/churn_by_contract_type.png)

---

### Churn Rate by Contract Type  
A stacked bar chart showing the proportion of churned vs retained customers across contract types.

![Churn Rate by Contract](EDA_Graphs/churn_rate_by_contract_type.png)

---

### Churn by Internet Service  
Fiber optic customers churn more frequently compared to DSL customers.

![Churn by Internet Service](EDA_Graphs/churn_by_internet_service.png)

---

### Churn by Tech Support  
Lack of technical support strongly correlates with higher churn.

![Churn by Tech Support](EDA_Graphs/churn_by_tech_support.png)

---

### Correlation Heatmap  
Shows correlations between numerical variables such as tenure, monthly charges, and total charges.

![Correlation Heatmap](EDA_Graphs/correlational_heatmap.png)

---

## 🧹 Data Cleaning & Feature Engineering

The raw dataset required multiple preprocessing steps before modeling.

### Data Cleaning
- Converted `TotalCharges` from string to numeric
- Handled missing values using **median imputation**
- Dropped non-informative column (`customerID`)
- Converted target variable (`Churn`) to binary (1 = Yes, 0 = No)

### Feature Encoding
- Binary encoding for:
  - gender, partner, dependents, phone service, paperless billing
- One-hot encoding for multi-category variables:
  - contract type, payment method, internet-related services

### Feature Engineering
- **AvgMonthlySpend** = `TotalCharges / (tenure + 1)`
- **HighValueCustomer** = 1 if `MonthlyCharges > 70`, else 0

### Feature Scaling
- Applied **StandardScaler**
- Scaled features:
  - tenure
  - MonthlyCharges
  - TotalCharges
  - AvgMonthlySpend

The cleaned dataset was saved as:
data/processed/churn_cleaned.csv


---

## 🤖 Machine Learning Models

Two machine learning models were trained and compared.

### 1️⃣ Logistic Regression (Baseline Model)
- Simple and interpretable
- Serves as a baseline for comparison
- Outputs probability estimates for churn

### 2️⃣ Random Forest Classifier (Final Model)
- Ensemble-based model
- Captures non-linear relationships
- Handles feature interactions effectively
- Provides robust probability predictions

---

## 📊 Model Evaluation & Metrics

Models were evaluated using standard classification metrics.

### Confusion Matrix  
Shows true positives, true negatives, false positives, and false negatives.

![Confusion Matrix](Graphs_model_building/confusion_matrix.png)

---

### Evaluation Metrics Explained
- **Accuracy**: Overall correctness of the model
- **Precision**: How many predicted churns were correct
- **Recall**: Ability to correctly identify churn customers
- **F1-score**: Balance between precision and recall

![Model Performance](Graphs_model_building/model_performance.png)

---

### ROC Curve  
Shows the trade-off between True Positive Rate and False Positive Rate.  
A higher AUC indicates better model performance.

![ROC Curve](Graphs_model_building/roc_curve.png)

---

### Precision–Recall Curve  
Important for imbalanced datasets like churn prediction.

![Precision Recall Curve](Graphs_model_building/precision_recall_curve.png)

---

## 🏆 Model Selection

After comparing both models:
- Random Forest achieved higher recall and ROC-AUC
- Reduced false negatives (critical for churn prediction)
- Selected as the **final production model**

---

## 🔮 Churn Prediction & Risk Classification

The final model predicts churn probability and assigns risk levels:

| Probability | Risk Level |
|-----------|-----------|
| ≥ 0.6 | High Risk |
| 0.3 – 0.6 | Medium Risk |
| < 0.3 | Low Risk |

### Example Prediction Output

![Churn Probability](Graphs_model_building/churn_probability.png)

---

### Churn vs Monthly Charges (Prediction Insight)

![Churn vs Monthly Charges](Graphs_model_building/churn_vs_monthly_charges.png)

---

## 💾 Model Persistence

The trained model and scaler were saved using `joblib`:


These can be reused for:
- APIs
- Dashboards
- Real-time prediction systems

---

## ✅ Final Conclusion

This project demonstrates a **complete real-world data science workflow**, including:
- Business problem understanding
- Deep EDA with visual insights
- Robust data preprocessing
- Feature engineering
- Training and comparison of ML models
- Model evaluation using industry-standard metrics
- Risk-based churn prediction
- Model persistence for deployment

---

## 🚀 Skills Demonstrated

- Python (Pandas, NumPy)
- Data Visualization (Matplotlib, Seaborn)
- Feature Engineering
- Machine Learning (Scikit-learn)
- Model Comparison & Evaluation
- Probability-Based Risk Prediction
- Business Insight Generation
