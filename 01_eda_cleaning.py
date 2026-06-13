# 01_eda_cleaning.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diabetes.csv")
print(df.shape)         # (768, 9)
print(df.isnull().sum()) # No NaN — but zeros are fake NaN!

# Columns where 0 is medically impossible
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zeros with NaN, then impute per Outcome group
df[zero_cols] = df[zero_cols].replace(0, np.nan)

df[zero_cols] = df.groupby('Outcome')[zero_cols].transform(
    lambda x: x.fillna(x.median())
)

print(f"Missing after imputation: {df.isnull().sum().sum()}")  # 0

# Class balance
df['Outcome'].value_counts().plot(kind='bar', color=['#1D9E75', '#D85A30'])
plt.title('Class distribution')
plt.xticks([0, 1], ['No diabetes', 'Diabetes'], rotation=0)
plt.savefig('class_balance.png', bbox_inches='tight')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature correlations')
plt.savefig('correlation_heatmap.png', bbox_inches='tight')

df.to_csv("diabetes_cleaned.csv", index=False)
print("Saved cleaned dataset.")