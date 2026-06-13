# 02_modelling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                              RocCurveDisplay, ConfusionMatrixDisplay)
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("diabetes_cleaned.csv")

# Feature engineering
df['BMI_category'] = pd.cut(df['BMI'], bins=[0,18.5,25,30,100],
                             labels=['Underweight','Normal','Overweight','Obese'])
df['Age_group'] = pd.cut(df['Age'], bins=[0,30,45,60,100],
                          labels=['Young','Middle','Senior','Elderly'])
df['Glucose_risk'] = pd.cut(df['Glucose'], bins=[0,100,126,300],
                             labels=['Normal','Pre-diabetic','Diabetic'])

# Encode new categoricals
df = pd.get_dummies(df, columns=['BMI_category','Age_group','Glucose_risk'], drop_first=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipelines
def make_pipeline(clf):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     clf)
    ])

models = {
    'Logistic Regression': make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    'Random Forest':        make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)),
    'XGBoost':              make_pipeline(xgb.XGBClassifier(use_label_encoder=False,
                                          eval_metric='logloss', random_state=42))
}

results = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    results[name] = {'pipeline': pipeline, 'auc': auc, 'proba': y_pred_prob}
    print(f"{name:25s}  AUC: {auc:.4f}")

# Tune best model (XGBoost usually wins)
param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth':    [3, 5, 7],
    'clf__learning_rate':[0.05, 0.1, 0.2],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(models['XGBoost'], param_grid, cv=cv,
                  scoring='roc_auc', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
y_prob_best = best_model.predict_proba(X_test)[:, 1]
print(f"\nBest XGBoost AUC: {roc_auc_score(y_test, y_prob_best):.4f}")
print(classification_report(y_test, best_model.predict(X_test)))

# Save model + feature names
joblib.dump(best_model, 'best_model.pkl')
pd.Series(X_train.columns.tolist()).to_csv('feature_names.csv', index=False)

# ROC curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[0])
axes[0].set_title('ROC Curve — XGBoost (tuned)')
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test,
                                       display_labels=['No Diabetes','Diabetes'], ax=axes[1])
axes[1].set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('model_evaluation.png', bbox_inches='tight')