import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="HBsAg Seroconversion Prediction",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .explanation {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .positive-prediction {
        background-color: #c8e6c9;
    }
    .negative-prediction {
        background-color: #ffcdd2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and preprocessor"""
    return joblib.load('mlp_final_model.pkl')

def preprocess_input(input_data, preprocessor, features):
    """Preprocess the input data using the saved preprocessor"""
    # Convert to DataFrame with the right format
    df = pd.DataFrame([input_data], columns=features)
    # Apply preprocessing
    return preprocessor.transform(df)

def get_shap_plot(model, processed_data, original_data, features, class_idx):
    """Generate SHAP force plot for explanation"""
    # Create a background dataset (using empty DataFrame with feature names)
    background_data = pd.DataFrame(columns=features)
    
    # Initialize the SHAP explainer
    explainer = shap.KernelExplainer(
        model.predict_proba, 
        shap.sample(processed_data, 1)  # Use minimal background
    )
    
    # Get SHAP values for the processed input
    shap_values = explainer.shap_values(processed_data)
    
    # Convert processed_data to DataFrame for better visualization
    processed_df = pd.DataFrame(processed_data, columns=features)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot SHAP force plot
    shap.force_plot(
        base_value=explainer.expected_value[class_idx],
        shap_values=shap_values[class_idx][0],
        features=original_data,
        feature_names=features,
        matplotlib=True,
        show=False,
        axis=ax
    )
    
    # Adjust layout
    fig.tight_layout()
    return fig

def main():
    # Load model components
    loaded_data = load_model()
    model = loaded_data['model']
    preprocessor = loaded_data['preprocessor']
    features = loaded_data['features']
    
    # Header
    st.title("HBsAg Seroconversion Prediction Tool")
    
    # Information section
    with st.expander("About this app", expanded=False):
        st.markdown("""
        This application predicts the probability of HBsAg seroconversion based on two key features:
        
        - **HBsAg12w**: HBsAg level at week 12
        - **PLT**: Platelet count
        
        The model was built using a Neural Network (MLP) trained on clinical data.
        Enter your values below to get a prediction and see the explanation.
        """)
    
    # Input form
    st.header("Patient Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hbsag12w = st.number_input(
            "HBsAg at Week 12 (IU/mL)", 
            min_value=0.0, 
            value=100.0,
            help="Enter the HBsAg level at week 12 of treatment"
        )
    
    with col2:
        plt_value = st.number_input(
            "Platelet Count (×10^9/L)",
            min_value=0.0,
            value=150.0,
            help="Enter the platelet count"
        )
    
    # Create input data dictionary
    input_data = {
        'HBsAg12w': hbsag12w,
        'PLT': plt_value
    }
    
    # Prediction section
    st.header("Prediction")
    
    if st.button("Predict Seroconversion"):
        # Process input
        input_df = pd.DataFrame([input_data])
        processed_input = preprocessor.transform(input_df[features])
        
        # Get prediction and probability
        prediction = model.predict(processed_input)[0]
        probabilities = model.predict_proba(processed_input)[0]
        
        # Display prediction
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box positive-prediction">
                <h2>Likely to achieve HBsAg seroconversion</h2>
                <h3>Probability: {probabilities[1]:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box negative-prediction">
                <h2>Unlikely to achieve HBsAg seroconversion</h2>
                <h3>Probability: {probabilities[0]:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)

        # Display SHAP explanation
        st.header("Prediction Explanation")
        
        # Get SHAP visualization
        class_to_explain = int(prediction)  # 0 or 1 based on prediction
        fig = get_shap_plot(model, processed_input, input_data, features, class_to_explain)
        
        # Display the plot
        st.pyplot(fig)
        
        # Add explanation text
        st.markdown("""
        <div class="explanation">
            <p><strong>How to interpret:</strong> The SHAP force plot above shows how each feature contributed to the prediction.
            Red arrows push the prediction higher (toward seroconversion), while blue arrows push it lower (against seroconversion).
            The width of each arrow indicates the magnitude of that feature's impact.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display raw input and processed values for reference
        with st.expander("Technical Details"):
            st.subheader("Original Input Values")
            st.write(input_data)
            
            st.subheader("Processed Input Values (after preprocessing)")
            st.write(pd.DataFrame(processed_input, columns=preprocessor.get_feature_names_out()))
            
            st.subheader("Prediction Probabilities")
            st.write({
                "No Seroconversion (Class 0)": f"{probabilities[0]:.4f}",
                "Seroconversion (Class 1)": f"{probabilities[1]:.4f}"
            })

if __name__ == "__main__":
    main()
