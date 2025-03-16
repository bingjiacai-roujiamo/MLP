# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Disable warning for pyplot global use
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load model components
@st.cache_resource
def load_components():
    model_dict = joblib.load('models/mlp_final_model.pkl')
    return (
        model_dict['model'], 
        model_dict['preprocessor'],
        model_dict['features']
    )

model, preprocessor, feature_names = load_components()

# Initialize SHAP explainer
@st.cache_resource
def create_explainer():
    # Create dummy data matching preprocessing requirements
    dummy_data = pd.DataFrame(columns=feature_names)
    processed_dummy = preprocessor.transform(dummy_data)
    return shap.KernelExplainer(model.predict_proba, processed_dummy)

explainer = create_explainer()

# Web Interface
st.title('HBsAg Seroclearance Prediction System')
st.markdown("""
**Clinical Decision Support Tool for Hepatitis B Surface Antigen Clearance**
""")

# Input Section
with st.sidebar:
    st.header('Patient Parameters')
    hbsag = st.number_input(
        'HBsAg at 12 weeks (IU/mL)',
        min_value=0.0,
        max_value=50000.0,
        value=100.0,
        help="Measured serum HBsAg level at week 12"
    )
    plt_val = st.number_input(
        'Platelet Count (×10^9/L)',
        min_value=0.0,
        max_value=1000.0,
        value=150.0,
        help="Current platelet count measurement"
    )

# Create input dataframe
input_df = pd.DataFrame([[hbsag, plt_val]], columns=feature_names)

# Prediction Logic
if st.button('Calculate Probability'):
    try:
        # Preprocessing
        processed_input = preprocessor.transform(input_df)
        
        # Prediction
        probability = model.predict_proba(processed_input)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Display Results
        st.subheader('Prediction Outcome')
        outcome_text = f"**Predicted Probability:** <span style='color:{"green" if prediction else "red"};font-size:20px'>{probability:.2%}</span>"
        st.markdown(outcome_text, unsafe_allow_html=True)
        
        # Clinical interpretation
        if prediction:
            st.success('High probability of HBsAg seroclearance within 48 weeks (≥50% probability)')
        else:
            st.error('Low probability of seroclearance. Recommend further clinical evaluation.')
        
        # SHAP Explanation
        st.subheader('Model Interpretation')
        st.markdown("**Feature Contribution Analysis**")
        
        # Compute SHAP values
        shap_values = explainer.shap_values(preprocessor.transform(input_df))
        
        # Generate force plot
        plt.figure()
        shap.force_plot(
            base_value=explainer.expected_value[1],
            shap_values=shap_values[1][0],
            features=processed_input[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        st.pyplot(bbox_inches='tight')
        plt.clf()
        
    except Exception as e:
        st.error(f'Prediction error: {str(e)}')

# Documentation Section
with st.expander("Usage Guidelines"):
    st.markdown("""
    1. Input laboratory values in left sidebar
    2. Click 'Calculate Probability' 
    3. Results show:
       - Probability percentage
       - Clinical interpretation
       - Feature contribution diagram
    4. Decision threshold: 50% probability
    """)

# Footer
st.markdown("---")
st.markdown("**CLINICAL DECISION SUPPORT SYSTEM | FOR MEDICAL USE ONLY**")
