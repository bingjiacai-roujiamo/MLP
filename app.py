import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
    .feature-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .feature-positive {
        background-color: rgba(46, 125, 50, 0.1);
        border-left: 5px solid #2E7D32;
    }
    .feature-negative {
        background-color: rgba(198, 40, 40, 0.1);
        border-left: 5px solid #C62828;
    }
    .feature-name {
        font-weight: bold;
    }
    .feature-value {
        font-size: 1.1rem;
    }
    .feature-impact {
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>HBV Surface Antigen Clearance Prediction</h1>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('mlp_final_model.pkl')

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
3. View the prediction result and feature interpretation
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
    
    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict_proba(processed_data)[0, 1]
    prediction = 1 if prediction_proba >= 0.5 else 0
    
    return prediction, prediction_proba

# Prediction button
if st.button("Predict", type="primary"):
    # Make prediction
    prediction, probability = predict_clearance(hbsag12w, plt_value)
    
    # Display results
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
    
    # Feature interpretation
    st.markdown("<h2 class='sub-header'>Feature Interpretation</h2>", unsafe_allow_html=True)
    
    # HBsAg12w interpretation
    hbsag_class = "feature-positive" if hbsag12w < 100 else "feature-negative"
    hbsag_impact = "Favorable for clearance" if hbsag12w < 100 else "Less favorable for clearance"
    
    st.markdown(f"""
    <div class="feature-box {hbsag_class}">
        <p class="feature-name">HBsAg12w: <span class="feature-value">{hbsag12w:.2f} IU/ml</span></p>
        <p class="feature-impact">{hbsag_impact}</p>
        <p>Lower HBsAg levels at 12 weeks (<100 IU/ml) are generally associated with higher chances of clearance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # PLT interpretation
    plt_class = "feature-positive" if plt_value > 150 else "feature-negative"
    plt_impact = "Favorable for clearance" if plt_value > 150 else "Less favorable for clearance"
    
    st.markdown(f"""
    <div class="feature-box {plt_class}">
        <p class="feature-name">PLT: <span class="feature-value">{plt_value:.2f} ×10^9/L</span></p>
        <p class="feature-impact">{plt_impact}</p>
        <p>Higher platelet counts (>150 ×10^9/L) may indicate better liver function and potentially better response to treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical recommendation (generic)
    st.markdown("<h3>Clinical Note:</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p>This prediction is based on a model trained on historical data. The model considers both HBsAg12w and PLT values 
    to estimate the likelihood of HBV surface antigen clearance. Always consult with a hepatologist for clinical decisions.</p>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
This app is for research and educational purposes only. Always consult with a healthcare professional for medical advice.
</p>
""", unsafe_allow_html=True)
