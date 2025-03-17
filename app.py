import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the saved model
saved_model = joblib.load('models/mlp_final_model.pkl')
final_model = saved_model['model']
final_preprocessor = saved_model['preprocessor']
selected_features = saved_model['features']

# Set page title and research-only statement
st.title('Neural Network Model for Surface Antigen Seroconversion Prediction')
st.markdown("This application is for research purposes only. The results are not intended for clinical diagnosis.")

# Create input fields for each feature
input_data = {}
for feature in selected_features:
    input_value = st.number_input(f"Enter value for {feature}", value=0.0)
    input_data[feature] = [input_value]

# Create a prediction button
if st.button('Predict'):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Preprocess the input data
    processed_input = final_preprocessor.transform(input_df[selected_features])
    
    # Make a prediction
    prediction_proba = final_model.predict_proba(processed_input)[:, 1][0]
    prediction = 1 if prediction_proba >= 0.5 else 0
    prediction_text = 'Surface Antigen Seroconversion' if prediction == 1 else 'No Surface Antigen Seroconversion'
    
    # Display the prediction results
    st.subheader('Prediction Result')
    st.write(f'Predicted Classification: {prediction_text}')
    st.write(f'Prediction Probability: {prediction_proba:.4f}')

    # Calculate SHAP values
    background_samples = shap.sample(pd.DataFrame(processed_input, columns=final_preprocessor.get_feature_names_out()), 50, random_state=42)
    explainer = shap.KernelExplainer(final_model.predict_proba, background_samples)
    shap_values = explainer.shap_values(processed_input)

    # Select the appropriate SHAP values based on the prediction
    shap_values_to_display = shap_values[prediction]

    # Create a DataFrame for the original input data
    input_original_df = pd.DataFrame(input_data)

    # Plot SHAP waterfall plot with original values
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(values=shap_values_to_display[0], base_values=explainer.expected_value[prediction], data=input_original_df.values[0], feature_names=selected_features), show=False)
    st.pyplot(fig)
