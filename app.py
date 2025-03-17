import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Load model components
@st.cache_resource
def load_model_components():
    model_data = joblib.load('mlp_final_model.pkl')
    return model_data['model'], model_data['preprocessor'], model_data['features']

model, preprocessor, features = load_model_components()

# Create SHAP explainer with proper background
@st.cache_resource
def create_shap_explainer():
    # Generate synthetic background data matching preprocessed shape
    background = shap.utils.sample(np.zeros((1, len(features))), 50)  # Adjusted for feature count
    return shap.KernelExplainer(model.predict_proba, background)

explainer = create_shap_explainer()

# Streamlit interface
st.title("HBsAg Clearance Prediction")
st.markdown("""
**Clinical Decision Support Tool**  
Predicts likelihood of HBsAg seroclearance based on treatment response.
""")

# Input Section
st.header("Patient Parameters")
input_data = {}

col1, col2 = st.columns(2)
with col1:
    input_data['HBsAg12w'] = st.number_input(
        '12-week HBsAg (IU/mL)', 
        min_value=0.0,
        max_value=50000.0,
        value=100.0,
        step=0.1
    )
with col2:
    input_data['PLT'] = st.number_input(
        'Platelet Count (×10⁹/L)', 
        min_value=0,
        max_value=1000,
        value=150,
        step=1
    )

# Prediction Logic
if st.button('Predict Outcomes'):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        proba = model.predict_proba(processed_input)[0]
        prediction = proba.argmax()
        
        # Display results
        st.subheader("Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Predicted Outcome", 
                     "High Clearance Probability" if prediction == 1 else "Low Clearance Probability")
        with result_col2:
            st.metric("Confidence Score", 
                     f"{proba[prediction]:.1%}")
        
        # SHAP Explanation
        st.subheader("Feature Contribution Analysis")
        
        # Get SHAP values
        shap_values = explainer.shap_values(processed_input)
        
        # Select explanation based on prediction class
        class_idx = 1 if prediction == 1 else 0
        feature_names = preprocessor.get_feature_names_out()
        
        # Create force plot
        plt.figure(figsize=(10, 4))
        shap.force_plot(
            explainer.expected_value[class_idx],
            shap_values[class_idx][0],
            processed_input[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Add clinical disclaimer
st.markdown("""
---
**Clinical Note**: This prediction tool provides probabilistic estimates based on machine learning models. 
Clinical decisions should always incorporate comprehensive patient evaluation and professional judgment.
""")
