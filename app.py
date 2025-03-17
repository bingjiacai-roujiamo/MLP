import streamlit as st
import joblib
import pandas as pd

# Load the saved model
saved_model = joblib.load('mlp_final_model.pkl')
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
    prediction = 'Surface Antigen Seroconversion' if prediction_proba >= 0.5 else 'No Surface Antigen Seroconversion'
    
    # Display the prediction results
    st.subheader('Prediction Result')
    st.write(f'Predicted Classification: {prediction}')
    st.write(f'Prediction Probability: {prediction_proba:.4f}')
