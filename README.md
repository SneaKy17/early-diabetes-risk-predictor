# 🩺 Early Diabetes Risk Predictor

An end-to-end machine learning pipeline that predicts diabetes risk using the **PIMA Indians Diabetes Dataset**, featuring XGBoost classification with hyperparameter tuning, SHAP-based model explainability, and an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)

---

## ✨ Features

- **Full ML Pipeline** — EDA, data cleaning, feature engineering, model training, and evaluation
- **Multi-Model Comparison** — Logistic Regression vs Random Forest vs XGBoost with AUC scoring
- **Hyperparameter Tuning** — GridSearchCV with StratifiedKFold cross-validation
- **SHAP Explainability** — Global feature importance, waterfall plots, and dependence analysis
- **Interactive Web App** — Real-time risk prediction with visual SHAP explanations via Streamlit
- **AUC Score: 0.95** — High-performance tuned XGBoost classifier

## 📸 Screenshots

### Risk Prediction Dashboard
The Streamlit app provides an interactive interface where users adjust patient health parameters and instantly see the predicted diabetes risk with SHAP-based explanations.

### Model Performance
- **ROC Curve** with AUC = 0.95
- **Confusion Matrix** showing classification results
- **SHAP Beeswarm** plot for global feature importance
- **Glucose Dependence** plot colored by BMI

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/SneaKy17/early-diabetes-risk-predictor.git
cd early-diabetes-risk-predictor

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1: EDA & Data Cleaning
python 01_eda_cleaning.py

# Step 2: Model Training & Evaluation
python 02_modelling.py

# Step 3: SHAP Explainability Analysis
python 03_shap_explainability.py

# Step 4: Launch the Web App
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
early-diabetes-risk-predictor/
├── 01_eda_cleaning.py           # Data exploration & cleaning
├── 02_modelling.py              # Model comparison & hyperparameter tuning
├── 03_shap_explainability.py    # SHAP analysis & visualization
├── app.py                       # Streamlit web application
├── diabetes.csv                 # Original PIMA dataset
├── diabetes_cleaned.csv         # Cleaned dataset (auto-generated)
├── best_model.pkl               # Trained XGBoost model (auto-generated)
├── feature_names.csv            # Feature columns (auto-generated)
├── requirements.txt             # Python dependencies
├── class_balance.png            # Class distribution plot
├── correlation_heatmap.png      # Feature correlation heatmap
├── model_evaluation.png         # ROC curve & confusion matrix
├── shap_beeswarm.png            # Global SHAP importance
├── shap_bar.png                 # Mean SHAP values
├── shap_waterfall_sample.png    # Individual prediction explanation
└── shap_dependence_glucose.png  # Glucose-BMI dependence plot
```

## 🔬 Technical Details

### Data Preprocessing
- Replaced medically impossible zero values (Glucose, BMI, etc.) with group-wise median imputation
- Engineered categorical features: `BMI_category`, `Age_group`, `Glucose_risk`
- One-hot encoded categorical variables

### Models Compared
| Model | AUC Score |
|-------|-----------|
| Logistic Regression | ~0.82 |
| Random Forest | ~0.88 |
| **XGBoost (tuned)** | **~0.95** |

### Hyperparameter Grid (XGBoost)
- `n_estimators`: [100, 200, 300]
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.05, 0.1, 0.2]

### SHAP Explainability
- **TreeExplainer** for XGBoost model interpretation
- **Beeswarm plot** — global feature importance with direction of impact
- **Waterfall plot** — individual patient prediction breakdown
- **Dependence plot** — Glucose vs SHAP value, colored by BMI interaction

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.8+ |
| ML Framework | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

## 📊 Dataset

**PIMA Indians Diabetes Dataset** — 768 samples, 8 features
- Source: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Target: Binary classification (Diabetes / No Diabetes)

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

*Built by [Nikhil Saklani](https://github.com/SneaKy17)*
