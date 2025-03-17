import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="HBV Surface Antigen Clearance Prediction",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* ...原有CSS样式保持不变... */
    .debug-info {
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ... [保持原有标题和样式部分不变] ...

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

# ... [保持侧边栏内容不变] ...

# 修改后的预测函数
def predict_clearance(hbsag12w, plt_value):
    # 创建DataFrame时应用log转换
    input_data = pd.DataFrame({
        'HBsAg12w': [np.log10(hbsag12w)],  # 关键修改：应用log转换
        'PLT': [plt_value]
    })
    
    # 预处理数据
    try:
        processed_data = preprocessor.transform(input_data)
    except Exception as e:
        st.error(f"预处理错误: {str(e)}")
        raise
    
    # 获取预测结果
    prediction_proba = model.predict_proba(processed_data)[0, 1]
    prediction = 1 if prediction_proba >= 0.5 else 0
    
    # 返回处理后的数据用于调试
    return prediction, prediction_proba, processed_data, input_data

# 修改后的预测按钮逻辑
if st.button("Predict", type="primary"):
    try:
        # 执行预测
        prediction, probability, processed_data, input_data = predict_clearance(hbsag12w, plt_value)
        
        # 显示结果
        st.markdown("---")
        st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
        
        # ... [保持原有结果显示部分不变] ...

        # 新增调试信息
        with st.expander("调试信息"):
            st.markdown("### 原始输入值:")
            st.write(f"HBsAg12w (原始): {hbsag12w:.2f} IU/ml")
            st.write(f"PLT (原始): {plt_value:.2f} ×10^9/L")
            
            st.markdown("### 预处理后值:")
            st.write(f"Log10(HBsAg12w): {input_data['HBsAg12w'].values[0]:.4f}")
            
            # 获取特征名称
            numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            binary_features = preprocessor.named_transformers_['binary'].get_feature_names_out()
            all_features = list(numeric_features) + list(binary_features)
            
            st.write("标准化后的值:")
            debug_df = pd.DataFrame(processed_data, columns=all_features)
            st.dataframe(debug_df.style.format("{:.4f}"))

    except Exception as e:
        st.error(f"预测失败: {str(e)}")

# ... [保持特征解释和页脚部分不变] ...
