import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Load model components
@st.cache_resource
def get_feature_names():
    return [name.split('__')[-1] for name in preprocessor.get_feature_names_out()]
def load_model_components():
    model_data = joblib.load('mlp_final_model.pkl')
    return model_data['model'], model_data['preprocessor'], model_data['features']

model, preprocessor, features = load_model_components()

# Create SHAP explainer with proper background
@st.cache_resource
def create_shap_explainer():
    # Generate synthetic background data matching preprocessed shape
    background = shap.utils.sample(np.zeros((1, len(features))), 50)  # Adjusted for feature count
    return shap.KernelExplainer(model.predict_proba, background)

explainer = create_shap_explainer()

# Streamlit interface
st.title("HBsAg Clearance Prediction")
st.markdown("""
**Clinical Decision Support Tool**  
Predicts likelihood of HBsAg seroclearance based on treatment response.
""")

# Input Section
st.header("Patient Parameters")
input_data = {}

col1, col2 = st.columns(2)
with col1:
    input_data['HBsAg12w'] = st.number_input(
        '12-week HBsAg (IU/mL)', 
        min_value=0.0,
        max_value=50000.0,
        value=100.0,
        step=0.1
    )
with col2:
    input_data['PLT'] = st.number_input(
        'Platelet Count (×10⁹/L)', 
        min_value=0,
        max_value=1000,
        value=150,
        step=1
    )

# Prediction Logic
if st.button('Prediction'):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        proba = model.predict_proba(processed_input)[0]
        prediction = proba.argmax()
        
        # Display results
        st.subheader("Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Predicted Outcome", 
                     "High Clearance Probability" if prediction == 1 else "Low Clearance Probability")
        with result_col2:
            st.metric("Confidence Score", 
                     f"{proba[prediction]:.1%}")
        
        # SHAP Explanation
        st.subheader("SHAP Force Plot Interpretation")
        
        # 获取原始输入值（未标准化的）
        raw_values = input_df.iloc[0].values
        
        # 获取特征名称
        feature_names = [name.split('__')[-1] for name in preprocessor.get_feature_names_out()]
        
        # 动态选择SHAP解释类别
        class_idx = 1 if prediction == 1 else 0
        
        # 计算SHAP值
        shap_values = explainer.shap_values(processed_input)
        
        # 创建解释图
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.plots.force(
            base_value=explainer.expected_value[class_idx],
            shap_values=shap_values[class_idx][0],
            features=raw_values,  # 使用原始输入值
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=15
        )
        
        # 添加自定义说明
        plt.title(f"SHAP Explanation for {'Positive' if prediction == 1 else 'Negative'} Class", 
                 fontsize=12, pad=20)
        plt.xlabel("Feature Impact Value", fontsize=10)
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")

# Add clinical disclaimer
st.markdown("""
---
**Clinical Note**: This prediction tool provides probabilistic estimates based on machine learning models. 
Clinical decisions should always incorporate comprehensive patient evaluation and professional judgment.
""")
