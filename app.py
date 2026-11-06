import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2 - App", layout="wide")

st.title("M.Tech AIML: Machine Learning Assignment 2")
st.markdown("**Interactive Model Evaluation & Deployment Dashboard**")

# Sidebar configurations
st.sidebar.header("Control Panel")
model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

@st.cache_resource
def load_artifacts(model_name):
    filename = model_name.lower().replace(" ", "_") + ".pkl"
    model = joblib.load(f"model/{filename}")
    try:
        scaler = joblib.load("model/scaler.pkl")
    except:
        scaler = None
    return model, scaler

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(test_data.head())
    
    if 'target' in test_data.columns:
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        model, scaler = load_artifacts(model_choice)
        
        # Apply scaling if required
        if scaler and model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test
            
        # Predict using saved model
        preds = model.predict(X_test_processed)
        try:
            probs = model.predict_proba(X_test_processed)[:, 1]
        except:
            probs = np.zeros_like(preds, dtype=float)
            
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) == 2 else 0.0
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        mcc = matthews_corrcoef(y_test, preds)
        
        st.header(f"Results for: {model_choice}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        st.subheader("Classification Report & Confusion Matrix")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.text(classification_report(y_test, preds))
            
        with col_b:
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
    else:
        st.error("Uploaded test CSV file must contain the 'target' column.")
else:
    st.info("Please upload your `test_data.csv` file using the sidebar panel to evaluate models.")