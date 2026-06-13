# 03_shap_explainability.py
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

best_model = joblib.load('best_model.pkl')
df = pd.read_csv("diabetes_cleaned.csv")

# Re-create features (same as modelling script)
df['BMI_category'] = pd.cut(df['BMI'], bins=[0,18.5,25,30,100],
                             labels=['Underweight','Normal','Overweight','Obese'])
df['Age_group'] = pd.cut(df['Age'], bins=[0,30,45,60,100],
                          labels=['Young','Middle','Senior','Elderly'])
df['Glucose_risk'] = pd.cut(df['Glucose'], bins=[0,100,126,300],
                             labels=['Normal','Pre-diabetic','Diabetic'])
df = pd.get_dummies(df, columns=['BMI_category','Age_group','Glucose_risk'], drop_first=True)

X = df.drop('Outcome', axis=1)

# SHAP only works on the final estimator, NOT the full pipeline
# Extract the classifier step
clf = best_model.named_steps['clf']

# Transform X through the pipeline up to (not including) the classifier
X_transformed = best_model[:-1].transform(X)
X_transformed_df = pd.DataFrame(X_transformed, columns=X.columns)

# Compute SHAP values
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_transformed_df)

# 1. Global beeswarm plot
plt.figure()
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title('SHAP feature importance (global)')
plt.tight_layout()
plt.savefig('shap_beeswarm.png', bbox_inches='tight')
plt.close()

# 2. Bar summary
plt.figure()
shap.plots.bar(shap_values, max_display=12, show=False)
plt.title('Mean |SHAP value| per feature')
plt.tight_layout()
plt.savefig('shap_bar.png', bbox_inches='tight')
plt.close()

# 3. Waterfall for a single high-risk patient
high_risk_idx = shap_values.values[:, 0].argmax()  # patient with highest Glucose SHAP
plt.figure()
shap.plots.waterfall(shap_values[high_risk_idx], show=False)
plt.title(f'Why patient #{high_risk_idx} is high risk')
plt.tight_layout()
plt.savefig('shap_waterfall_sample.png', bbox_inches='tight')
plt.close()

# 4. Dependence plot: Glucose vs BMI
shap.dependence_plot('Glucose', shap_values.values, X_transformed_df,
                     interaction_index='BMI', show=False)
plt.title('SHAP dependence: Glucose (coloured by BMI)')
plt.savefig('shap_dependence_glucose.png', bbox_inches='tight')
plt.close()

print("SHAP plots saved.")