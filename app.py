import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved model
def load_model():
    model_data = joblib.load('mlp_final_model.pkl')
    return model_data['model'], model_data['preprocessor']

model, preprocessor = load_model()

# Configure page
st.set_page_config(
    page_title="HBsAg Clearance Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Disclaimer
st.markdown("""
**Disclaimer:**  
This prediction tool is intended for research use only. It should not be used for clinical decision-making. The results do not constitute medical advice.
""")

# Title
st.title("HBsAg Clearance Prediction Tool")
st.markdown("""
This tool predicts the probability of HBsAg seroclearance based on clinical parameters.
""")

# Input form
with st.form("prediction_form"):
    st.header("Patient Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hbsag_12w = st.number_input(
            "HBsAg at 12 weeks (IU/mL)",
            min_value=0.0,
            max_value=50000.0,
            value=100.0,
            step=0.1
        )
        
    with col2:
        plt_count = st.number_input(
            "Platelet Count (×10^9/L)",
            min_value=0,
            max_value=1000,
            value=150,
            step=1
        )
    
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    try:
        # Create input DataFrame
        input_data = pd.DataFrame([[hbsag_12w, plt_count]],
                                columns=['HBsAg12w', 'PLT'])
        
        # Preprocess input
        processed_input = preprocessor.transform(input_data)
        
        # Make prediction
        prob = model.predict_proba(processed_input)[0][1]
        prediction = "Probable Seroclearance" if prob >= 0.5 else "Unlikely Seroclearance"
        
        # Display results
        st.subheader("Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric(label="Prediction", value=prediction)
            
        with result_col2:
            st.metric(label="Probability", value=f"{prob:.2%}")
        
        # Interpretation guide
        st.markdown("""
        **Interpretation:**
        - Probability ≥50%: Suggestive of potential HBsAg seroclearance
        - Probability <50%: Low likelihood of seroclearance
        """)
        
        # SHAP explanation (optional)
        with st.expander("Explanation of Prediction"):
            st.markdown("""
            The prediction is based on a neural network model considering:
            - **HBsAg levels at 12 weeks** (log-transformed)
            - **Platelet count**
            
            Higher HBsAg levels generally indicate lower clearance probability,
            while higher platelet counts may suggest better liver function.
            """)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Technical Notes:**
- Model type: Multilayer Perceptron (MLP) Neural Network
- Training data: Retrospective clinical cohort
- Validation AUC: 0.85 (internal), 0.82 (external)
- Intended for research use only
""")
