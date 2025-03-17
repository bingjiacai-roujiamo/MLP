import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Load model
@st.cache_resource
def load_model():
    model_dict = joblib.load('mlp_final_model.pkl')
    return model_dict['model'], model_dict['preprocessor'], model_dict['features']

model, preprocessor, features = load_model()

# Create SHAP explainer
@st.cache_resource
def create_explainer():
    background = shap.utils.sample(model.coefs_[0], 50)  # Adjust based on your model
    return shap.KernelExplainer(model.predict_proba, background)

explainer = create_explainer()

# App title
st.title("HBsAg Clearance Prediction Model")
st.markdown("**For Research Use Only** - This tool is intended for academic research purposes only.")

# Input form
st.header("Patient Input Parameters")
with st.form("prediction_form"):
    hbsag_12w = st.number_input("HBsAg at 12 weeks (IU/mL)", min_value=0.0, format="%.2f")
    plt_val = st.number_input("Platelet Count (×10^9/L)", min_value=0, step=1)
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Create input DataFrame
    input_df = pd.DataFrame([[hbsag_12w, plt_val]], columns=features)
    
    # Preprocess
    processed_input = preprocessor.transform(input_df)
    
    # Predict
    proba = model.predict_proba(processed_input)[0]
    prediction = 1 if proba[1] >= 0.5 else 0  # Using 0.5 threshold
    
    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Outcome", 
                 value="Clearance" if prediction == 1 else "No Clearance")
    with col2:
        st.metric(label="Probability of Clearance", 
                 value=f"{proba[1]:.1%}")
    
    # SHAP explanation
    st.subheader("Explanation of Prediction")
    shap_values = explainer.shap_values(processed_input)
    
    # Select appropriate SHAP class
    class_idx = prediction  # 0 or 1
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[class_idx][0], 
                        max_display=10,
                        show=False)
    plt.tight_layout()
    st.pyplot(fig)

# Add disclaimer
st.markdown("---")
st.markdown("""
**Disclaimer:**  
This predictive model is provided for research purposes only. Clinical decisions should not be based solely on this tool. Always consult with qualified healthcare professionals and consider individual patient circumstances.
""")
