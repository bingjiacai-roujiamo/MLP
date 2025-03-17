import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.neural_network import MLPClassifier

# Page configuration (MUST be first command)
st.set_page_config(
    page_title="HBsAg Seroconversion Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model_dict = joblib.load('mlp_final_model.pkl')
        return (
            model_dict['model'],
            model_dict['preprocessor'],
            model_dict['features']
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, preprocessor, features = load_model()

# SHAP initialization with proper background samples
@st.cache_resource
def create_explainer():
    try:
        # Generate background samples matching model input dimensions
        dummy_data = pd.DataFrame(
            np.zeros((1, len(features))),
            columns=features
        )
        processed_dummy = preprocessor.transform(dummy_data)
        background = shap.utils.sample(processed_dummy, 50)
        return shap.KernelExplainer(model.predict_proba, background)
    except Exception as e:
        st.error(f"Error initializing SHAP: {str(e)}")
        st.stop()

explainer = create_explainer()

# --- App Content ---
st.title("HBsAg Seroconversion Prediction Model")
st.markdown("""
**For Research Use Only**  
*This tool is intended for academic research purposes only.*
""")

# Input form
with st.form("prediction_form"):
    st.header("Patient Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        hbsag_12w = st.number_input(
            "HBsAg at 12 weeks (IU/mL)", 
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
            step=0.1,
            format="%.1f"
        )
    
    with col2:
        plt_val = st.number_input(
            "Platelet Count (×10⁹/L)", 
            min_value=0,
            max_value=1000,
            value=150,
            step=1
        )
    
    submitted = st.form_submit_button("Predict", type="primary")

# Prediction logic
if submitted:
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([[hbsag_12w, plt_val]], columns=features)
        
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        
        # Validate input shape
        if processed_input.shape[1] != model.coefs_[0].shape[0]:
            st.error("Input dimension mismatch with model")
            st.stop()
            
        # Get prediction
        proba = model.predict_proba(processed_input)[0]
        prediction = 1 if proba[1] >= 0.5 else 0
        
        # Display results
        st.subheader("Results")
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric(
                label="Predicted Outcome",
                value="Seroconversion ✅" if prediction else "No Seroconversion ❌"
            )
            
        with result_col2:
            st.metric(
                label="Probability of Seroconversion",
                value=f"{proba[1]:.1%}",
                delta=f"{(proba[1]-0.5):+.1%}"  # Show difference from 50% threshold
            )
        
        # SHAP explanation
        st.subheader("Explanation")
        with st.spinner("Generating explanation..."):
            try:
                shap_values = explainer.shap_values(processed_input)
                
                # Select explanation based on predicted class
                class_idx = 1 if prediction else 0
                
                fig, ax = plt.subplots()
                shap.plots.waterfall(
                    shap_values[class_idx][0],
                    max_display=5,
                    show=False
                )
                plt.title(f"SHAP Explanation for Class {class_idx}")
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Could not generate explanation: {str(e)}")
                
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:**  
This predictive model is provided for research purposes only. Clinical decisions should 
not be based solely on this tool. Always consult with qualified healthcare professionals 
and consider individual patient circumstances.
""")
