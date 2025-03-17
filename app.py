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
    """Generate SHAP waterfall plot for explanation"""
    # Create a background dataset for the explainer
    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, class_idx], 
        shap.sample(processed_data, 5)  # Use minimal background
    )
    
    # Calculate SHAP values for the processed input
    shap_values = explainer.shap_values(processed_data)
    
    # Create matplotlib figure for waterfall plot (more reliable than force plot in Streamlit)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a waterfall plot using matplotlib
    feature_names = features
    feature_values = list(original_data.values())
    
    # Sort SHAP values and corresponding features by absolute magnitude
    indices = np.argsort(np.abs(shap_values))
    sorted_shap_values = np.array(shap_values)[indices]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_feature_values = [feature_values[i] for i in indices]
    
    # Calculate cumulative SHAP values for waterfall
    base_value = explainer.expected_value
    cumulative = np.cumsum(sorted_shap_values)
    total_shap = np.sum(shap_values)
    
    # Colors for positive and negative contributions
    colors = ['#ff4d4d' if v > 0 else '#1e88e5' for v in sorted_shap_values]
    
    # Plot bars
    y_pos = np.arange(len(sorted_feature_names) + 1)
    
    # Plot horizontal lines
    for i in range(len(sorted_shap_values)):
        plt.plot([base_value + cumulative[i] - sorted_shap_values[i], 
                  base_value + cumulative[i]], 
                 [y_pos[i], y_pos[i]], 
                 'k-', alpha=0.3)
    
    # Plot vertical lines
    for i in range(len(sorted_shap_values)):
        plt.plot([base_value + cumulative[i], base_value + cumulative[i]], 
                 [y_pos[i], y_pos[i+1]], 
                 'k-', alpha=0.3)
    
    # Add final prediction line
    plt.plot([base_value + total_shap, base_value + total_shap], 
             [y_pos[-2], y_pos[-1]], 
             'k-', alpha=0.3)
    
    # Plot bars for SHAP values
    barlist = plt.barh(y_pos[:-1], sorted_shap_values, align='center', alpha=0.7)
    for i, bar in enumerate(barlist):
        bar.set_color(colors[i])
    
    # Add feature names and values as y-tick labels
    labels = [f"{name} = {value:.3g}" for name, value in zip(sorted_feature_names, sorted_feature_values)]
    labels.append("Prediction")
    plt.yticks(y_pos, labels)
    
    # Add base value and prediction
    plt.axvline(x=base_value, color='black', linestyle='-', alpha=0.3)
    plt.text(base_value, len(sorted_feature_names) + 0.5, f'Base value: {base_value:.3f}', 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Final prediction value
    prediction_value = base_value + total_shap
    plt.text(prediction_value, len(sorted_feature_names) + 0.5, 
             f'Final prediction: {prediction_value:.3f}', 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add title and labels
    plt.title(f'SHAP Explanation for Class {class_idx} Prediction')
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.xlim(min(base_value - abs(base_value)*0.5, base_value + min(0, total_shap) - 0.1),
             max(base_value + abs(base_value)*0.5, base_value + max(0, total_shap) + 0.1))
    
    plt.tight_layout()
    return fig

# Alternative simpler SHAP visualization function as backup
def get_simple_shap_plot(model, processed_data, features, class_idx):
    """Generate a simpler SHAP bar plot for explanation (backup option)"""
    # Create background and explainer
    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, class_idx],
        shap.sample(processed_data, 5)
    )
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(processed_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot bar chart of SHAP values
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, shap_values, align='center')
    
    # Color bars based on contribution direction
    for i, bar in enumerate(bars):
        bar.set_color('#ff4d4d' if shap_values[i] > 0 else '#1e88e5')
    
    # Add feature names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    
    # Add labels and title
    ax.set_xlabel('SHAP Value (Impact on Prediction)')
    ax.set_title(f'Feature Importance for Class {class_idx} Prediction')
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
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
        
        # Get SHAP visualization - try the custom waterfall plot first
        class_to_explain = int(prediction)  # 0 or 1 based on prediction
        
        try:
            # Try the custom waterfall SHAP plot first
            fig = get_shap_plot(model, processed_input, input_data, features, class_to_explain)
            # Display the plot
            st.pyplot(fig)
        except Exception as e:
            st.warning("Advanced visualization unavailable. Showing simplified explanation.")
            # If that fails, use the simpler bar plot
            try:
                fig = get_simple_shap_plot(model, processed_input, features, class_to_explain)
                st.pyplot(fig)
            except Exception as e2:
                st.error("Unable to generate explanation visualization.")
                st.write(f"Error details: {str(e2)}")
        
        # Add explanation text
        st.markdown("""
        <div class="explanation">
            <p><strong>How to interpret:</strong> The SHAP plot above shows how each feature contributed to the prediction.
            Red bars push the prediction higher (toward seroconversion), while blue bars push it lower (against seroconversion).
            The length of each bar indicates the magnitude of that feature's impact.</p>
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
