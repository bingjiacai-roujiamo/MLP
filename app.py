import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="HBV Surface Antigen Clearance Prediction",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .result-positive {
        font-size: 1.8rem;
        color: #2E7D32;
        font-weight: bold;
    }
    .result-negative {
        font-size: 1.8rem;
        color: #C62828;
        font-weight: bold;
    }
    .description {
        font-size: 1rem;
        line-height: 1.6;
    }
    .probability {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>HBV Surface Antigen Clearance Prediction</h1>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/mlp_final_model.pkl')

try:
    model_data = load_model()
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    features = model_data['features']
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
<p class='description'>
This app predicts the probability of HBV surface antigen clearance based on two key features:
HBsAg12w (HBsAg at 12 weeks) and PLT (Platelet count).
</p>
<p class='description'>
The model was trained using a Neural Network (Multi-Layer Perceptron) classifier.
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h2 class='sub-header'>Instructions</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
<p class='description'>
1. Enter the patient's HBsAg12w value and PLT value
2. Click the 'Predict' button
3. View the prediction result and SHAP explanation
</p>
""", unsafe_allow_html=True)

# Main content
st.markdown("<h2 class='sub-header'>Patient Data Input</h2>", unsafe_allow_html=True)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    hbsag12w = st.number_input(
        "HBsAg12w (IU/ml)",
        min_value=0.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
        help="HBsAg level at 12 weeks"
    )

with col2:
    plt_value = st.number_input(
        "PLT (×10^9/L)",
        min_value=0.0,
        max_value=1000.0,
        value=150.0,
        step=10.0,
        help="Platelet count"
    )

# Create a prediction function
def predict_clearance(hbsag12w, plt_value):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'HBsAg12w': [hbsag12w],
        'PLT': [plt_value]
    })
    
    # Save original values for SHAP explanation
    original_values = input_data.copy()
    
    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict_proba(processed_data)[0, 1]
    prediction = 1 if prediction_proba >= 0.5 else 0
    
    return prediction, prediction_proba, original_values, processed_data

# Create a function to explain predictions with SHAP
def explain_prediction(processed_data, original_values, prediction):
    # Create background data for SHAP explainer
    feature_names = preprocessor.get_feature_names_out()
    cleaned_feature_names = [name.split('__')[-1].replace('_1', '') for name in feature_names]
    
    # Create DataFrame with processed data and cleaned feature names
    processed_df = pd.DataFrame(processed_data, columns=cleaned_feature_names)
    
    # Create SHAP explainer
    background_samples = shap.sample(processed_df, 50, random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background_samples)
    
    # Get SHAP values for the current prediction
    # For class 1 (clearance) if prediction is 1, otherwise for class 0
    class_idx = 1 if prediction == 1 else 0
    shap_values = explainer.shap_values(processed_df)[class_idx]
    
    # Create a force plot
    fig = plt.figure(figsize=(10, 3))
    force_plot = shap.force_plot(
        base_value=explainer.expected_value[class_idx],
        shap_values=shap_values,
        features=original_values,
        feature_names=cleaned_feature_names,
        matplotlib=True,
        show=False
    )
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Prediction button
if st.button("Predict", type="primary"):
    # Add a spinner during prediction
    with st.spinner("Generating prediction..."):
        # Make prediction
        prediction, probability, original_values, processed_data = predict_clearance(hbsag12w, plt_value)
        
        # Display results in a nice box
        st.markdown("---")
        st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
        
        result_container = st.container()
        
        with result_container:
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown("<p class='result-positive'>HBsAg Clearance Expected</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='probability'>Probability: {probability:.2%}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='result-negative'>HBsAg Clearance Not Expected</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='probability'>Probability: {probability:.2%}</p>", unsafe_allow_html=True)
            
            with col2:
                # Input parameters summary
                st.markdown("<h3>Input Parameters:</h3>", unsafe_allow_html=True)
                st.write(f"**HBsAg12w:** {hbsag12w:.2f} IU/ml")
                st.write(f"**PLT:** {plt_value:.2f} ×10^9/L")
        
        # Generate SHAP explanation
        with st.spinner("Generating SHAP explanation..."):
            explanation_img = explain_prediction(processed_data, original_values, prediction)
            
            # Display SHAP explanation
            st.markdown("<h2 class='sub-header'>Prediction Explanation</h2>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("The following SHAP force plot shows how each feature contributes to predicting **HBsAg clearance** (class 1):")
            else:
                st.markdown("The following SHAP force plot shows how each feature contributes to predicting **no HBsAg clearance** (class 0):")
            
            # Display the SHAP force plot
            st.image(explanation_img, use_column_width=True)
            
            # SHAP explanation text
            st.markdown("<h3>Interpretation:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - **Red features** push the prediction higher (toward clearance)
            - **Blue features** push the prediction lower (against clearance)
            - **Feature size** shows the magnitude of each feature's impact
            """)
            
            st.markdown("""
            <p class='description'>
            The force plot shows how the model arrived at its prediction by showing how each input 
            feature pushed the prediction away from the base value (average prediction across all patients).
            </p>
            """, unsafe_allow_html=True)
            
            # Additional interpretation based on medical knowledge
            st.markdown("<h3>Clinical Interpretation:</h3>", unsafe_allow_html=True)
            
            if hbsag12w < 100:
                st.markdown("- **Lower HBsAg12w values** are typically associated with higher clearance rates")
            else:
                st.markdown("- **Higher HBsAg12w values** typically indicate lower chances of clearance")
                
            if plt_value > 150:
                st.markdown("- **Higher platelet counts (PLT)** may indicate better liver function and potentially better response")
            else:
                st.markdown("- **Lower platelet counts (PLT)** may indicate compromised liver function")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
This app is for research and educational purposes only. Always consult with a healthcare professional for medical advice.
</p>
""", unsafe_allow_html=True)
