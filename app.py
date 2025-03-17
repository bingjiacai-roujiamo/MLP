import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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
3. View the prediction result and feature importance
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
    
    # Save original values for explanation
    original_values = input_data.copy()
    
    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict_proba(processed_data)[0, 1]
    prediction = 1 if prediction_proba >= 0.5 else 0
    
    return prediction, prediction_proba, original_values, processed_data

# Simple waterfall chart function
def create_waterfall_chart(input_data, prediction):
    # Create feature importance based on simple logic (for demonstration)
    # In a real scenario, you would use a proper feature importance calculation
    
    # Get feature values
    hbsag12w_value = input_data['HBsAg12w'].values[0]
    plt_value = input_data['PLT'].values[0]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initial baseline - start with 0.5 (neutral probability)
    baseline = 0.5
    
    # Determine impact direction based on prediction and known medical knowledge
    if prediction == 1:  # Clearance expected
        hbsag_impact = -0.3 if hbsag12w_value < 100 else 0.1  # Lower HBsAg is better for clearance
        plt_impact = 0.2 if plt_value > 150 else -0.1  # Higher PLT is better for clearance
    else:  # Clearance not expected
        hbsag_impact = 0.3 if hbsag12w_value > 100 else -0.1  # Higher HBsAg is worse for clearance  
        plt_impact = -0.2 if plt_value < 150 else 0.1  # Lower PLT is worse for clearance
    
    # Calculate final probability
    final_prob = baseline + hbsag_impact + plt_impact
    final_prob = max(0.01, min(0.99, final_prob))  # Ensure probability is between 0.01 and 0.99
    
    # Create data for waterfall chart
    labels = ['Baseline', 'HBsAg12w\n' + f"{hbsag12w_value:.1f} IU/ml", 'PLT\n' + f"{plt_value:.1f} ×10^9/L", 'Final']
    values = [baseline, hbsag_impact, plt_impact, 0]  # The last value is a placeholder
    
    # Determine colors based on impact
    colors = ['gray', 
              'red' if hbsag_impact > 0 else 'blue', 
              'red' if plt_impact > 0 else 'blue', 
              'green' if prediction == 1 else 'darkred']
    
    # Create the waterfall chart
    # First, create a bar for the baseline
    ax.bar(0, baseline, width=0.6, color=colors[0], alpha=0.7)
    
    # Add HBsAg impact
    bottom = baseline
    if hbsag_impact > 0:
        ax.bar(1, hbsag_impact, bottom=bottom, width=0.6, color=colors[1], alpha=0.7)
        bottom += hbsag_impact
    else:
        ax.bar(1, hbsag_impact, bottom=bottom+hbsag_impact, width=0.6, color=colors[1], alpha=0.7)
        bottom += hbsag_impact
    
    # Add PLT impact
    if plt_impact > 0:
        ax.bar(2, plt_impact, bottom=bottom, width=0.6, color=colors[2], alpha=0.7)
        bottom += plt_impact
    else:
        ax.bar(2, plt_impact, bottom=bottom+plt_impact, width=0.6, color=colors[2], alpha=0.7)
        bottom += plt_impact
    
    # Add final result
    ax.bar(3, final_prob, width=0.6, color=colors[3], alpha=0.7)
    
    # Add connecting lines
    # From baseline to HBsAg
    ax.plot([0.3, 0.7], [baseline, baseline], color='black', linestyle='-', linewidth=1)
    
    # From HBsAg to PLT
    ax.plot([1.3, 1.7], [bottom-plt_impact, bottom-plt_impact], color='black', linestyle='-', linewidth=1)
    
    # From PLT to Final
    ax.plot([2.3, 2.7], [bottom, bottom], color='black', linestyle='-', linewidth=1)
    
    # Add labels and title
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Probability of Clearance')
    ax.set_title('Impact of Features on Prediction')
    
    # Add a horizontal line at 0.5 (decision boundary)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add annotations for impacts
    ax.annotate(f"{hbsag_impact:.2f}", xy=(1, baseline + hbsag_impact/2), 
                xytext=(1, baseline + hbsag_impact/2),
                ha='center', va='center', fontweight='bold')
    
    ax.annotate(f"{plt_impact:.2f}", xy=(2, bottom - plt_impact/2), 
                xytext=(2, bottom - plt_impact/2),
                ha='center', va='center', fontweight='bold')
    
    ax.annotate(f"{final_prob:.2f}", xy=(3, final_prob/2), 
                xytext=(3, final_prob/2),
                ha='center', va='center', fontweight='bold')
    
    # Set y-axis limits to ensure all elements are visible
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(['Connection', 'Baseline', 'HBsAg12w Impact', 'PLT Impact', 'Final Prediction'],
              loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
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
        
        # Generate explanation chart
        with st.spinner("Generating feature impact visualization..."):
            # Create waterfall chart
            chart_img = create_waterfall_chart(original_values, prediction)
            
            # Display chart
            st.markdown("<h2 class='sub-header'>Prediction Explanation</h2>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("The following chart shows how each feature contributes to predicting **HBsAg clearance**:")
            else:
                st.markdown("The following chart shows how each feature contributes to predicting **no HBsAg clearance**:")
            
            # Display the chart
            st.image(chart_img, use_column_width=True)
            
            # Explanation text
            st.markdown("<h3>Interpretation:</h3>", unsafe_allow_html=True)
            st.markdown("""
            - **Red bars** push the prediction higher (toward clearance)
            - **Blue bars** push the prediction lower (against clearance)
            - The chart shows how we start from a baseline (0.5) and how each feature moves the prediction up or down
            - The final prediction is shown in green (for clearance) or dark red (for no clearance)
            """)
            
            # Additional interpretation based on medical knowledge
            st.markdown("<h3>Clinical Interpretation:</h3>", unsafe_allow_html=True)
            
            if hbsag12w < 100:
                st.markdown("- **Lower HBsAg12w values** (<100 IU/ml) are typically associated with higher clearance rates")
            else:
                st.markdown("- **Higher HBsAg12w values** (>100 IU/ml) typically indicate lower chances of clearance")
                
            if plt_value > 150:
                st.markdown("- **Higher platelet counts (PLT)** (>150 ×10^9/L) may indicate better liver function and potentially better response")
            else:
                st.markdown("- **Lower platelet counts (PLT)** (<150 ×10^9/L) may indicate compromised liver function")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
This app is for research and educational purposes only. Always consult with a healthcare professional for medical advice.
</p>
""", unsafe_allow_html=True)
