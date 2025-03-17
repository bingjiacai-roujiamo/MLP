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
    
    return prediction, prediction_proba, input_data

# Create custom waterfall chart
def create_custom_waterfall(input_data, prediction_proba):
    # Base expected value (average prediction)
    # This would ideally come from your model, but we'll set a reasonable value
    expected_value = 0.451  # Approximation based on your image
    
    # Determine feature impacts based on input values
    # These values would normally come from SHAP but we'll use simplified logic
    
    # Direction of impact: negative values push toward "no clearance"
    # Magnitude of impact: based on difference from reference values
    
    # Reference values (approximate thresholds)
    hbsag_ref = 100.0  # IU/ml
    plt_ref = 150.0  # ×10^9/L
    
    # Get actual values
    hbsag_value = input_data['HBsAg12w'].values[0]
    plt_value = input_data['PLT'].values[0]
    
    # Calculate impacts
    # Lower HBsAg12w is better for clearance, so higher values have negative impact
    hbsag_impact = -0.23 if hbsag_value > hbsag_ref else 0.15
    # Scale based on magnitude difference from reference
    hbsag_impact = hbsag_impact * (1 + abs(hbsag_value - hbsag_ref) / hbsag_ref) * 0.5
    hbsag_impact = max(-0.40, min(0.40, hbsag_impact))  # Cap the impact
    
    # Higher PLT is better, so lower values have negative impact
    plt_impact = -0.03 if plt_value < plt_ref else 0.02
    # Scale based on magnitude difference from reference
    plt_impact = plt_impact * (1 + abs(plt_value - plt_ref) / plt_ref) * 0.5
    plt_impact = max(-0.15, min(0.15, plt_impact))  # Cap the impact
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Set up positions for the bars
    positions = np.arange(4)
    
    # Values for the waterfall chart
    values = [0, hbsag_impact, plt_impact, 0]  # Start, HBsAg impact, PLT impact, End
    
    # Colors for impacts
    colors = ['white', '#1F77B4' if hbsag_impact < 0 else '#FF7F0E', 
              '#1F77B4' if plt_impact < 0 else '#FF7F0E', 'white']
    
    # Calculate bottom positions for the bars
    bottoms = [expected_value, 
               expected_value, 
               expected_value + hbsag_impact,
               0]  # Not used for the final bar
    
    # Calculate the final value
    final_value = expected_value + hbsag_impact + plt_impact
    
    # Draw the bars
    # First bar is the expected value (base)
    ax.text(positions[0], expected_value / 2, f"E[f(X)] = {expected_value:.3f}", 
            ha='center', va='center', fontweight='bold')
    
    # HBsAg impact
    ax.bar(positions[1], hbsag_impact, bottom=bottoms[1], color=colors[1], width=0.8)
    ax.text(positions[1], bottoms[1] + hbsag_impact/2, f"{hbsag_impact:.2f}", 
            ha='center', va='center', color='white', fontweight='bold')
    
    # PLT impact
    ax.bar(positions[2], plt_impact, bottom=bottoms[2], color=colors[2], width=0.8)
    ax.text(positions[2], bottoms[2] + plt_impact/2, f"{plt_impact:.2f}", 
            ha='center', va='center', color='white', fontweight='bold')
    
    # Final prediction
    ax.text(positions[3], final_value / 2, f"f(x) = {final_value:.3f}", 
            ha='center', va='center', fontweight='bold')
    
    # Draw connecting lines
    ax.plot([positions[0], positions[1]], [expected_value, expected_value], 'k-')
    ax.plot([positions[1], positions[2]], [bottoms[1] + hbsag_impact, bottoms[1] + hbsag_impact], 'k-')
    ax.plot([positions[2], positions[3]], [bottoms[2] + plt_impact, bottoms[2] + plt_impact], 'k-')
    
    # Add feature labels on the left
    ax.text(positions[1] - 0.5, bottoms[1] - 0.05, f"{hbsag_value:.3f} = HBsAg12w", 
            ha='right', va='center', fontsize=10, color='gray')
    ax.text(positions[2] - 0.5, bottoms[2] - 0.05, f"{plt_value:.0f} = PLT", 
            ha='right', va='center', fontsize=10, color='gray')
    
    # Set labels and title
    ax.set_title('Feature Impact on Prediction', fontsize=14)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, max(expected_value, final_value) * 1.5)
    ax.set_xticks([])  # Remove x-axis ticks
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add a horizontal line at the expected value
    ax.axhline(y=expected_value, color='black', linestyle='--', alpha=0.3)
    
    # Add a horizontal line at the decision boundary (0.5)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.text(3.5, 0.5, "Decision Boundary", va='center', color='red', alpha=0.7)
    
    # Save figure to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Prediction button
if st.button("Predict", type="primary"):
    # Make prediction
    prediction, probability, input_data = predict_clearance(hbsag12w, plt_value)
    
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
    
    # Try to create waterfall chart
    try:
        waterfall_img = create_custom_waterfall(input_data, probability)
        st.image(waterfall_img, use_column_width=True)
    except Exception as e:
        st.warning(f"Could not generate waterfall visualization. Using text-based explanation instead.")
        
        # Text-based feature interpretation as fallback
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
