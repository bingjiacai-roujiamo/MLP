import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load model and components
@st.cache_resource
def load_model():
    model_data = joblib.load('mlp_final_model.pkl')
    return model_data['model'], model_data['preprocessor'], model_data['features']

model, preprocessor, selected_features = load_model()

# Create SHAP explainer
@st.cache_resource
def create_explainer():
    background = shap.utils.sample(model.coefs_[0].shape[1], 50)  # Adjust based on your model
    return shap.KernelExplainer(model.predict_proba, background)

explainer = create_explainer()

# Streamlit app
st.title('HBsAg Clearance Prediction with MLP')
st.header('Input Features')

# Create input fields
input_data = {}
col1, col2 = st.columns(2)
with col1:
    input_data['HBsAg12w'] = st.number_input('HBsAg12w (IU/mL)', min_value=0.0, format="%.2f")
with col2:
    input_data['PLT'] = st.number_input('PLT (×10^9/L)', min_value=0, step=1)

# Create DataFrame from inputs
input_df = pd.DataFrame([input_data])

# Preprocess input
processed_input = preprocessor.transform(input_df)

# Make prediction
if st.button('Predict'):
    # Get probabilities
    proba = model.predict_proba(processed_input)[0]
    prediction = proba.argmax()
    
    # Display results
    st.subheader('Prediction Results')
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric(label="Prediction", 
                value="Clearance (HBsAg-)" if prediction == 1 else "No Clearance (HBsAg+)")
    with result_col2:
        st.metric(label="Probability", 
                value=f"{proba[prediction]:.2%}")
    
    # SHAP explanation
    st.subheader('SHAP Explanation')
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(processed_input)
    
    # Select explanation based on prediction
    if prediction == 1:
        shap_value = shap_values[1][0]
        expected_value = explainer.expected_value[1]
    else:
        shap_value = shap_values[0][0]
        expected_value = explainer.expected_value[0]
    
    # Plot SHAP force plot
    fig, ax = plt.subplots()
    shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value,
        features=processed_input[0],
        feature_names=preprocessor.get_feature_names_out(),
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
