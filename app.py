import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import os

# Set page title
st.set_page_config(page_title="HBsAg Seroconversion Prediction System", layout="wide")

# Title and description
st.title("HBsAg Seroconversion Prediction System")
st.markdown("This system predicts the probability of HBsAg seroconversion within 48 weeks based on HBsAg12w and PLT values.")

# Sidebar - Input parameters
st.sidebar.header("Patient Parameters")

# Create input fields
HBsAg12w = st.sidebar.number_input("HBsAg12w Value", min_value=0.0, max_value=25000.0, value=100.0, step=0.1)
PLT = st.sidebar.number_input("PLT Value (×10^9/L)", min_value=0, max_value=1000, value=200, step=1)

# Load model
@st.cache_resource
def load_model():
    try:
        loaded = joblib.load('mlp_final_model.pkl')
        return loaded
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Get model and preprocessor
loaded_data = load_model()
if loaded_data:
    model = loaded_data['model']
    preprocessor = loaded_data['preprocessor']
    selected_features = loaded_data['features']

    # Prediction function
    def predict(HBsAg12w, PLT):
        # Create input data DataFrame
        input_data = pd.DataFrame({
            'HBsAg12w': [HBsAg12w],
            'PLT': [PLT]
        })
        
        # Ensure DataFrame contains all selected features
        for feature in selected_features:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Transform data using preprocessor
        processed = preprocessor.transform(input_data[selected_features])
        
        # Predict probability
        prob = model.predict_proba(processed)[0, 1]
        return prob

    # Calculate SHAP values with proper handling of standardized features
    @st.cache_resource
    def get_shap_explainer():
        # Create a wrapper function that transforms the data before prediction
        def model_wrapper(X_raw):
            # Create a DataFrame with the right column names
            X_df = pd.DataFrame(X_raw, columns=selected_features)
            # Apply the same preprocessing as in the model training
            X_processed = preprocessor.transform(X_df)
            # Get prediction probabilities
            return model.predict_proba(X_processed)[:, 1]
        
        # Create a background dataset with raw (non-standardized) values
        # Using realistic value ranges for each feature
        background_data = pd.DataFrame({
            'HBsAg12w': [10, 50, 100, 500, 1000],
            'PLT': [100, 150, 200, 250, 300]
        })
        
        # Create SHAP explainer using raw data
        return shap.KernelExplainer(model_wrapper, background_data.values, feature_names=selected_features)

    # Main content area
    if st.sidebar.button("Predict"):
        # Perform prediction
        probability = predict(HBsAg12w, PLT)
        prediction = 1 if probability > 0.5 else 0
        
        # Display results
        st.header("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Seroconversion Probability", 
                value=f"{probability:.2%}"
            )
            
            if prediction == 1:
                st.success("Prediction: High probability of HBsAg seroconversion within 48 weeks")
            else:
                st.error("Prediction: Low probability of HBsAg seroconversion within 48 weeks, further examination and treatment recommended")
                
        # SHAP explanation
        with col2:
            st.subheader("SHAP Value Explanation")
            
            try:
                # Get SHAP explainer
                explainer = get_shap_explainer()
                
                # Create input data
                input_data = pd.DataFrame({
                    'HBsAg12w': [HBsAg12w],
                    'PLT': [PLT]
                })
                
                # Ensure all selected features are included
                for feature in selected_features:
                    if feature not in input_data.columns:
                        input_data[feature] = 0
                
                # Preprocess input data
                processed_input = preprocessor.transform(input_data[selected_features])
                
                # Calculate SHAP values
                shap_values = explainer(processed_input)
                
                # Create SHAP force plot
                fig, ax = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
                
                # Feature importance explanation
                st.write("Chart Explanation:")
                st.write("- Red indicates the feature value increases the probability of seroconversion")
                st.write("- Blue indicates the feature value decreases the probability of seroconversion")
                st.write("- Bar length represents the impact of the feature on the prediction result")
                
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {e}")
        
        # Add patient parameter summary
        st.header("Patient Parameter Summary")
        data = {
            'Parameter': ['HBsAg12w', 'PLT'],
            'Value': [HBsAg12w, PLT]
        }
        st.table(pd.DataFrame(data))
        
else:
    st.error("Unable to load model. Please ensure the model file is in the correct location.")

# Add footer
st.markdown("---")
st.markdown("© 2025 HBsAg Seroconversion Prediction System")
