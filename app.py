import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer

# Load model components
@st.cache_resource
def load_model_components():
    model_data = joblib.load('mlp_final_model.pkl')
    return model_data['model'], model_data['preprocessor'], model_data['features']

model, preprocessor, features = load_model_components()

# Create SHAP explainer
@st.cache_resource
def create_shap_explainer():
    # Generate representative background data
    background = np.zeros((50, len(features)))  # 50 samples x 2 features
    background[:, 0] = np.random.uniform(0, 1000, 50)  # HBsAg12w range
    background[:, 1] = np.random.randint(50, 300, 50)  # PLT range
    return shap.KernelExplainer(model.predict_proba, background)

explainer = create_shap_explainer()

# Streamlit界面
st.set_page_config(layout="wide")
st.title("HBsAg Seroclearance Prediction System")
st.markdown("""
**Clinical Decision Support Tool**  
Predicts likelihood of HBsAg seroclearance based on treatment response parameters.
""")

# 输入部分
with st.container():
    st.header("Patient Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        hbsag = st.number_input(
            '12-week HBsAg (IU/mL)', 
            min_value=0.0,
            max_value=10000.0,
            value=200.0,
            step=0.1,
            format="%.1f"
        )
    
    with col2:
        plt_count = st.number_input(
            'Platelet Count (×10⁹/L)', 
            min_value=0,
            max_value=1000,
            value=150,
            step=1
        )

# 预测逻辑
if st.button('Run Prediction Analysis'):
    try:
        # 创建输入数据
        input_df = pd.DataFrame([[hbsag, plt_count]], columns=features)
        
        # 预处理
        processed_input = preprocessor.transform(input_df)
        
        # 预测
        proba = model.predict_proba(processed_input)[0]
        prediction = proba.argmax()
        
        # 显示结果
        with st.container():
            st.subheader("Prediction Results")
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                outcome = "High Clearance (HBsAg-)" if prediction == 1 else "Low Clearance (HBsAg+)"
                st.metric("Predicted Outcome", outcome)
            
            with result_col2:
                confidence = f"{proba[prediction]:.1%}"
                st.metric("Confidence Score", confidence,
                        delta=f"{(proba[prediction]-0.5):+.1%} from decision threshold")
        
        # SHAP解释
        with st.expander("Detailed Feature Impact Analysis"):
            class_idx = 1 if prediction == 1 else 0
            shap_values = explainer.shap_values(processed_input)
            
            # 创建解释图
            fig, ax = plt.subplots(figsize=(12, 4))
            shap.force_plot(
                explainer.expected_value[class_idx],
                shap_values[class_idx][0],
                input_df.values[0],
                feature_names=features,
                matplotlib=True,
                show=False
            )
            
            # 添加自定义样式
            plt.title(f"Feature Impact Analysis | Predicted Class: {class_idx}", fontsize=14)
            plt.xlabel("Cumulative Effect on Prediction", fontsize=10)
            st.pyplot(fig)
            plt.close()
            
            # 添加数值解释
            st.markdown(f"""
            **Interpretation Guide:**
            - Baseline Value: {explainer.expected_value[class_idx]:.2f} (Average model output)
            - Current Prediction: {proba[prediction]:.2f}
            - Feature Impacts: 
              - HBsAg12w: {shap_values[class_idx][0][0]:.2f}
              - PLT: {shap_values[class_idx][0][1]:.2f}
            """)
    
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

# 临床声明
st.markdown("""
---
**Clinical Disclaimer:**  
This AI prediction tool provides probabilistic estimates based on historical data analysis. Clinical decisions should always be made in conjunction with:  
- Comprehensive patient evaluation  
- Laboratory findings verification  
- Professional clinical judgment  
- Latest clinical guidelines  
""")
