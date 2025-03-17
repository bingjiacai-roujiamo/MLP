import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import os
import sys
import traceback
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="HBsAg Seroconversion Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Debug information
st.sidebar.title("Debug Information")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"Current working directory: {os.getcwd()}")
st.sidebar.write(f"Directory contents: {os.listdir('.')}")

# Try to list model directory contents
try:
    if os.path.exists('models'):
        st.sidebar.write(f"Models directory contents: {os.listdir('models')}")
    else:
        st.sidebar.write("Models directory does not exist!")
except Exception as e:
    st.sidebar.write(f"Error checking models directory: {str(e)}")

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
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model and preprocessor with error handling."""
    try:
        st.sidebar.write("Attempting to load model...")
        model_path = 'mlp_final_model.pkl'
        
        if not os.path.exists(model_path):
            st.sidebar.error(f"Model file not found at: {model_path}")
            return None, None, None
            
        loaded = joblib.load(model_path)
        st.sidebar.success("Model loaded successfully!")
        
        # Verify model structure
        if not isinstance(loaded, dict):
            st.sidebar.error("Model file does not contain a dictionary structure!")
            return None, None, None
            
        if 'model' not in loaded or 'preprocessor' not in loaded or 'features' not in loaded:
            st.sidebar.error("Model dictionary missing required keys!")
            keys = loaded.keys() if hasattr(loaded, 'keys') else "Not a dictionary"
            st.sidebar.write(f"Available keys: {keys}")
            return None, None, None
            
        return loaded['model'], loaded['preprocessor'], loaded['features']
        
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.sidebar.write(f"Traceback: {traceback.format_exc()}")
        return None, None, None

def get_shap_explainer(model, X_processed):
    """Create a SHAP explainer for the model with error handling."""
    try:
        # Using fewer background samples for efficiency
        background_samples = shap.sample(X_processed, 10, random_state=42)  # Using just 10 samples for speed
        return shap.KernelExplainer(model.predict_proba, background_samples)
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {str(e)}")
        return None

def plot_shap_waterfall(explainer, X_processed, features, class_idx=1):
    """Generate SHAP waterfall plot with error handling."""
    try:
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
    except Exception as e:
        st.error(f"Error creating SHAP waterfall plot: {str(e)}")
        st.write(f"Traceback: {traceback.format_exc()}")
        return None

def simple_prediction_display(probability, prediction):
    """A simpler display method without SHAP explanations."""
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

def main():
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
    
    # Load model and components
    model, preprocessor, features = load_model()
    
    if model is None or preprocessor is None or features is None:
        st.markdown(
            "<div class='error-message'>"
            "<strong>Error:</strong> Failed to load the prediction model. "
            "Please check the debug information in the sidebar."
            "</div>",
            unsafe_allow_html=True
        )
        return
    
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
        try:
            # Apply preprocessing
            st.sidebar.write(f"Input features: {input_df[features]}")
            processed_input = preprocessor.transform(input_df[features])
            st.sidebar.write(f"Processed input shape: {processed_input.shape}")
            
            # Make prediction
            probability = model.predict_proba(processed_input)[0, 1]
            prediction = 1 if probability >= 0.5 else 0
            
            # Display results
            st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
            simple_prediction_display(probability, prediction)
            
            try:
                # Generate SHAP explanation
                st.write("Generating SHAP explanation (this may take a moment)...")
                explainer = get_shap_explainer(model, processed_input)
                
                if explainer is not None:
                    # Show waterfall plot based on prediction
                    class_to_explain = 1 if prediction == 1 else 0
                    explanation_text = "Positive" if class_to_explain == 1 else "Negative"
                    
                    st.markdown(f"<h3>SHAP Explanation for {explanation_text} Prediction</h3>", unsafe_allow_html=True)
                    st.write("""
                    The waterfall plot below shows how each feature contributes to the prediction. 
                    Red bars push the prediction higher, while blue bars push it lower.
                    """)
                    
                    fig = plot_shap_waterfall(explainer, processed_input, features, class_to_explain)
                    if fig is not None:
                        st.pyplot(fig)
                else:
                    st.warning("SHAP explainer could not be created. Showing prediction results only.")
            except Exception as shap_error:
                st.warning(f"Could not generate SHAP explanation: {str(shap_error)}")
                st.write("Displaying prediction results without SHAP explanation.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
