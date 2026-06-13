# app.py  — crash-safe version
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

try:
    import joblib, numpy as np, pandas as pd, shap, matplotlib.pyplot as plt

    @st.cache_resource
    def load_model():
        return joblib.load("best_model.pkl")

    model = load_model()
    train_cols = pd.read_csv("feature_names.csv").iloc[:, 0].tolist()

    st.title("🩺 Early Diabetes Risk Predictor")
    st.caption("PIMA Indians Diabetes Dataset · XGBoost + SHAP explainability")

    st.sidebar.header("Patient health inputs")
    pregnancies    = st.sidebar.slider("Pregnancies",               0,   17,   1)
    glucose        = st.sidebar.slider("Glucose (mg/dL)",          50,  200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)",   40,  130,  72)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)",       5,   60,  23)
    insulin        = st.sidebar.slider("Insulin (μU/mL)",          10,  500,  79)
    bmi            = st.sidebar.slider("BMI",                      15.0, 60.0, 28.0, step=0.1)
    dpf            = st.sidebar.slider("Diabetes Pedigree Function",0.05, 2.5, 0.47, step=0.01)
    age            = st.sidebar.slider("Age",                      18,   80,  33)

    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]],
                             columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                      'Insulin','BMI','DiabetesPedigreeFunction','Age'])

    # Feature engineering — must match training
    input_df['BMI_category'] = pd.cut(input_df['BMI'],
                                bins=[0,18.5,25,30,100],
                                labels=['Underweight','Normal','Overweight','Obese'])
    input_df['Age_group']    = pd.cut(input_df['Age'],
                                bins=[0,30,45,60,100],
                                labels=['Young','Middle','Senior','Elderly'])
    input_df['Glucose_risk'] = pd.cut(input_df['Glucose'],
                                bins=[0,100,126,300],
                                labels=['Normal','Pre-diabetic','Diabetic'])
    input_df = pd.get_dummies(input_df,
                  columns=['BMI_category','Age_group','Glucose_risk'], drop_first=True)
    input_df = input_df.reindex(columns=train_cols, fill_value=0)

    prob = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Risk Score")
        color = "red" if prob > 0.5 else "orange" if prob > 0.3 else "green"
        st.markdown(f"<h1 style='color:{color};font-size:56px'>{prob*100:.1f}%</h1>",
                    unsafe_allow_html=True)
        risk_label = "🔴 High Risk" if prob > 0.5 else "🟠 Moderate Risk" if prob > 0.3 else "🟢 Low Risk"
        st.markdown(f"**{risk_label}**")
        st.progress(float(prob))

    with col2:
        st.subheader("Why this prediction? (SHAP)")
        clf   = model.named_steps['clf']
        X_t   = model[:-1].transform(input_df)
        X_t_df = pd.DataFrame(X_t, columns=train_cols)
        explainer = shap.TreeExplainer(clf)
        sv    = explainer(X_t_df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.divider()

    import os
    if os.path.exists("model_evaluation.png"):
        st.subheader("Model Performance")
        st.image("model_evaluation.png", use_container_width=True)

    col3, col4 = st.columns(2)
    if os.path.exists("shap_beeswarm.png"):
        col3.subheader("Global Feature Importance")
        col3.image("shap_beeswarm.png", use_container_width=True)
    if os.path.exists("shap_dependence_glucose.png"):
        col4.subheader("Glucose Dependence Plot")
        col4.image("shap_dependence_glucose.png", use_container_width=True)

except Exception as e:
    st.error(f"❌ Something went wrong:\n\n`{e}`")
    st.info("👉 Make sure you ran 01_eda_cleaning.py → 02_modelling.py → 03_shap_explainability.py first, and all output files are in the same folder as app.py")