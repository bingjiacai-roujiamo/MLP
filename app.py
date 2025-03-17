import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="HBsAg Seroconversion Prediction",
    page_icon="🧬",
    layout="wide"
)

# Main title
st.title("HBsAg Seroconversion Prediction Tool")
st.markdown("Predict the probability of HBsAg seroconversion based on clinical parameters.")

# Load the model
@st.cache_resource
def load_model():
    try:
        loaded = joblib.load('mlp_final_model.pkl')
        model = loaded['model']
        preprocessor = loaded['preprocessor']
        features = loaded['features']
        return model, preprocessor, features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, preprocessor, features = load_model()

# Create a function for making predictions
def predict(input_data):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure the input has all required features
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Default value (adjust as needed)
    
    # Preprocess the input data
    processed_data = preprocessor.transform(input_df[features])
    
    # Get prediction probability
    prediction_prob = model.predict_proba(processed_data)[0, 1]
    prediction_class = 1 if prediction_prob >= 0.5 else 0
    
    return prediction_class, prediction_prob, processed_data

# Create a function to generate SHAP explanation
def generate_shap_explanation(processed_data, prediction_class):
    # Get background data (use a subset of existing data)
    explainer = shap.KernelExplainer(
        model.predict_proba, 
        shap.sample(pd.DataFrame(processed_data, columns=features), 50, random_state=42)
    )
    
    # Get SHAP values - class 0 for negative prediction, class 1 for positive prediction
    shap_values = explainer.shap_values(processed_data)
    
    # Get the correct class based on prediction
    class_idx = prediction_class
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        explainer.expected_value[class_idx], 
        shap_values[class_idx][0], 
        feature_names=features,
        show=False
    )
    plt.title(f"SHAP Explanation for {'Positive' if prediction_class == 1 else 'Negative'} Prediction")
    plt.tight_layout()
    
    # Convert plot to image
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Create the input form
st.header("Patient Parameters")

# Use two columns for better layout
col1, col2 = st.columns(2)

with col1:
    hbsag12w = st.number_input("HBsAg at 12 weeks (IU/mL)", 
                               min_value=0.0, 
                               max_value=100000.0, 
                               value=1000.0,
                               help="Hepatitis B surface antigen level at 12 weeks")

with col2:
    plt_value = st.number_input("PLT (×10^9/L)", 
                                min_value=0.0, 
                                max_value=1000.0, 
                                value=200.0,
                                help="Platelet count")

# Create input data dictionary
input_data = {
    "HBsAg12w": hbsag12w,
    "PLT": plt_value
}

# Prediction button
if st.button("Predict Seroconversion"):
    if model is not None and preprocessor is not None:
        # Get prediction
        prediction_class, prediction_prob, processed_data = predict(input_data)
        
        # Display prediction
        st.header("Prediction Results")
        
        # Use columns to display results
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.subheader("Classification")
            if prediction_class == 1:
                st.success("✅ Positive: HBsAg Seroconversion Predicted")
            else:
                st.error("❌ Negative: HBsAg Seroconversion Not Predicted")
                
            st.subheader("Probability")
            st.metric("Seroconversion Probability", f"{prediction_prob:.2%}")
        
        with result_col2:
            st.subheader("SHAP Explanation")
            shap_plot = generate_shap_explanation(processed_data, prediction_class)
            st.image(shap_plot, use_column_width=True)
        
        # Additional information
        st.header("About this Model")
        st.markdown("""
        This model predicts the probability of HBsAg seroconversion based on:
        - **HBsAg12w**: Hepatitis B surface antigen level at 12 weeks
        - **PLT**: Platelet count
        
        The model was developed using a Neural Network (MLP) approach with SHAP-based feature selection.
        
        *Note: This tool is for research purposes only and should not replace clinical judgment.*
        """)
    else:
        st.error("Model failed to load. Please check the model files.")
