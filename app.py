import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import os
import base64
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="HBsAg Seroconversion Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
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
    .disclaimer {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .prediction-positive {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .prediction-negative {
        background-color: #fbe9e7;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .probability {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessor."""
    loaded = joblib.load('mlp_final_model.pkl')
    return loaded['model'], loaded['preprocessor'], loaded['features']

def get_shap_explainer(model, X_processed):
    """Create a SHAP explainer for the model."""
    # Using fewer background samples for efficiency
    background_samples = shap.sample(X_processed, 50, random_state=42)
    return shap.KernelExplainer(model.predict_proba, background_samples)

def plot_shap_waterfall(explainer, X_processed, features, class_idx=1):
    """Generate SHAP waterfall plot for the given instance."""
    shap_values = explainer.shap_values(X_processed)
    
    plt.figure(figsize=(10, 6))
    # Create a DataFrame with the original feature names for better visualization
    X_display = pd.DataFrame([X_processed[0]], columns=features)
    
    # Create a waterfall plot
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[class_idx][0], 
            base_values=explainer.expected_value[class_idx],
            data=X_display.iloc[0].values,
            feature_names=features
        ),
        show=False
    )
    
    plt.tight_layout()
    return plt

def main():
    # Load model and components
    model, preprocessor, features = load_model()
    
    # Header
    st.markdown("<h1 class='main-header'>HBsAg Seroconversion Prediction Model</h1>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(
        "<div class='disclaimer'>"
        "<strong>DISCLAIMER:</strong> This application is designed SOLELY FOR RESEARCH PURPOSES. "
        "The predictions generated should not be used for clinical decision-making without proper "
        "medical consultation. The model's predictions are based on limited data and may not account "
        "for all clinical factors relevant to individual patient cases."
        "</div>",
        unsafe_allow_html=True
    )
    
    # About section
    with st.expander("About this Application"):
        st.write("""
        This application uses a Neural Network model trained on clinical data to predict the probability 
        of HBsAg seroconversion in chronic hepatitis B patients. The model was developed using features 
        selected through SHAP (SHapley Additive exPlanations) values to identify the most important 
        predictors of seroconversion.
        
        **Key Features Used in Prediction:**
        - HBsAg12w: HBsAg levels at 12 weeks
        - PLT: Platelet count
        
        The model provides the probability of seroconversion and a SHAP waterfall plot to explain 
        the factors contributing to the prediction for the individual patient.
        """)
    
    # Input form
    st.markdown("<h2 class='sub-header'>Patient Data Input</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        hbsag12w = st.number_input("HBsAg at 12 weeks (IU/mL)", 
                                  min_value=0.0, 
                                  max_value=25000.0, 
                                  value=1000.0,
                                  help="Enter the HBsAg level at 12 weeks after treatment initiation")
    
    with col2:
        plt_value = st.number_input("Platelet count (×10^9/L)", 
                                   min_value=50, 
                                   max_value=500, 
                                   value=200,
                                   help="Enter the platelet count")
    
    # Create a dictionary with input data
    input_data = {
        'HBsAg12w': hbsag12w,
        'PLT': plt_value
    }
    
    # Create a DataFrame from the input
    input_df = pd.DataFrame([input_data])
    
    # Prediction section
    predict_btn = st.button("Predict Seroconversion", type="primary")
    
    if predict_btn:
        # Apply preprocessing
        processed_input = preprocessor.transform(input_df[features])
        
        # Make prediction
        probability = model.predict_proba(processed_input)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Display results
        st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(
                f"<div class='prediction-positive'>Prediction: <span style='color:green;'>Likely to achieve HBsAg seroconversion</span></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='prediction-negative'>Prediction: <span style='color:red;'>Unlikely to achieve HBsAg seroconversion</span></div>",
                unsafe_allow_html=True
            )
        
        st.markdown(f"<p class='probability'>Probability: {probability:.2%}</p>", unsafe_allow_html=True)
        
        # Generate SHAP explanation
        explainer = get_shap_explainer(model, processed_input)
        
        # Show waterfall plot based on prediction
        class_to_explain = 1 if prediction == 1 else 0
        explanation_text = "Positive" if class_to_explain == 1 else "Negative"
        
        st.markdown(f"<h3>SHAP Explanation for {explanation_text} Prediction</h3>", unsafe_allow_html=True)
        st.write("""
        The waterfall plot below shows how each feature contributes to the prediction. 
        Red bars push the prediction higher, while blue bars push it lower.
        """)
        
        fig = plot_shap_waterfall(explainer, processed_input, features, class_to_explain)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
