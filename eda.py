# ================================
# CUSTOMER CHURN - FULL EDA SCRIPT
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("data/raw/churn_data.csv")

# Basic checks
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# Target variable analysis
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))

sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Tenure vs Churn
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Contract Type vs Churn
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()

# Tech Support vs Churn
sns.countplot(x='TechSupport', hue='Churn', data=df)
plt.title("Churn by Tech Support")
plt.show()

# Internet Service vs Churn
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title("Churn by Internet Service")
plt.show()

# Correlation analysis
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
corr = df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Churn rate by contract
churn_contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
churn_contract.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Contract Type")
plt.show()

# Final insights
print("FINAL INSIGHTS")
print("1. Low tenure customers churn the most")
print("2. Month-to-month contracts have highest churn")
print("3. High monthly charges increase churn")
print("4. Lack of tech support increases churn")
