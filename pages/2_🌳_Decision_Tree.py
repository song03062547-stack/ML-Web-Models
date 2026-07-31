# -*- coding: utf-8 -*-
"""
Wine Classifier Web App - Streamlit (Decision Tree)
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ===== 1. ตั้งค่าหน้าเว็บ =====
st.set_page_config(page_title="Decision Tree Model", page_icon="🌳", layout="wide")


# ===== 2. Sidebar (รวมข้อมูลผู้พัฒนา + ข้อมูลโมเดลไว้ที่เดียว) =====
with st.sidebar:
    # ค้นหารูปโปรไฟล์จากนามสกุลไฟล์ต่างๆ ในโฟลเดอร์ assets
    profile_image_path = None
    for ext in ["jpg", "jpeg", "png", "JPG", "PNG"]:
        path_to_check = f"assets/profile.{ext}"
        if os.path.exists(path_to_check):
            profile_image_path = path_to_check
            break
            
    if profile_image_path:
        st.image(profile_image_path, width=130)
    else:
        st.warning("⚠️ ไม่พบรูปโปรไฟล์ใน assets/")

    st.markdown("### 👨‍💻 ข้อมูลผู้พัฒนา")
    st.markdown("""
    **ชื่อ-นามสกุล:** นายปฐมพงศ์ ชัยสรรค์ 
    **ชื่อเล่น:** ซองค์
    **รหัสนักศึกษา:** 664245039  
    **หมู่เรียน:** 66/44
    """)
    st.divider()

    st.markdown("### 🍷 เกี่ยวกับโมเดล Wine Classifier")
    st.markdown("""
    **แอปพลิเคชันจำแนกประเภทไวน์**  
    ใช้โมเดล Decision Tree ในการวิเคราะห์คุณสมบัติทางเคมีของไวน์
    
    - **Dataset:** Wine Dataset
    - **Model:** Decision Tree
    - **Features:** 13 คุณสมบัติ
    - **Classes:** 3 ประเภท
    """)
    st.divider()


# ===== 3. Custom CSS สำหรับความสวยงาม =====
st.markdown(
    """
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #2D3748 0%, #1A202C 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
        color: #E2E8F0;
    }
    
    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-box h2 {
        margin: 0;
        font-size: 1.8rem;
        color: white;
    }
    
    /* Button Style */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ===== 4. โหลดโมเดล =====
@st.cache_resource
def load_model():
    """โหลดโมเดลและ scaler จากไฟล์"""
    try:
        # แก้ไข Path ชี้ไปยังโฟลเดอร์ model_files/
        model = joblib.load("model_files/dt_model.pkl")
        scaler = joblib.load("model_files/scaler.pkl")
        features = joblib.load("model_files/feature_names.pkl")
        return model, scaler, features
    except FileNotFoundError:
        st.warning(
            "⚠️ ไม่พบไฟล์โมเดลในโฟลเดอร์ 'model_files/' (กำลังใช้ข้อมูลตัวอย่างเพื่อแสดงผล UI)"
        )
        default_features = [
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280/od315_of_diluted_wines",
            "proline",
        ]
        return None, None, default_features


model, scaler, features = load_model()

# ชื่อคลาสไวน์
WINE_NAMES = {
    0: "🍷 Class 0 - ไวน์ชนิดที่ 1",
    1: "🍷 Class 1 - ไวน์ชนิดที่ 2",
    2: "🍷 Class 2 - ไวน์ชนิดที่ 3",
}

# ===== 5. Header หลัก =====
st.markdown(
    """
<div class="main-header">
    <h1>🌳 Decision Tree: Wine Quality Classifier</h1>
    <p>ระบบจำแนกประเภทไวน์ด้วยเทคนิค Decision Tree Machine Learning</p>
</div>
""",
    unsafe_allow_html=True,
)

# ===== 6. Input Form =====
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔬 ป้อนข้อมูลคุณสมบัติทางเคมี")

    input_data = {}

    half = len(features) // 2 + (1 if len(features) % 2 != 0 else 0)
    left_features = features[:half]
    right_features = features[half:]

    col_left, col_right = st.columns(2)

    feature_ranges = {
        "alcohol": (10.0, 15.0, 12.5),
        "malic_acid": (0.5, 5.0, 2.5),
        "ash": (1.3, 3.5, 2.3),
        "alcalinity_of_ash": (10.0, 30.0, 19.0),
        "magnesium": (70.0, 160.0, 100.0),
        "total_phenols": (0.8, 4.0, 2.3),
        "flavanoids": (0.2, 5.0, 2.0),
        "nonflavanoid_phenols": (0.1, 1.0, 0.4),
        "proanthocyanins": (0.4, 4.0, 1.6),
        "color_intensity": (1.5, 17.0, 5.0),
        "hue": (0.3, 1.7, 0.95),
        "od280/od315_of_diluted_wines": (1.2, 4.0, 2.6),
        "proline": (250.0, 1700.0, 750.0),
    }

    with col_left:
        for feat in left_features:
            min_v, max_v, default = feature_ranges.get(feat, (0.0, 10.0, 5.0))
            input_data[feat] = st.number_input(
                feat.replace("_", " ").title(),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default),
                step=0.1,
                key=f"in_{feat}",
            )

    with col_right:
        for feat in right_features:
            min_v, max_v, default = feature_ranges.get(feat, (0.0, 10.0, 5.0))
            input_data[feat] = st.number_input(
                feat.replace("_", " ").title(),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default),
                step=0.1,
                key=f"in_{feat}",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 ทำนายผล", use_container_width=True)

with col2:
    st.markdown("### 📋 ข้อมูลที่ป้อน")
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df.T.rename(columns={0: "ค่า"}), use_container_width=True)

    if st.button("🔄 รีเซ็ตค่า", use_container_width=True):
        st.rerun()

# ===== 7. Prediction & Output =====
if predict_button:
    if model is not None and scaler is not None:
        with st.spinner("🤖 กำลังวิเคราะห์ข้อมูล..."):
            input_array = np.array([list(input_data.values())])
            input_scaled = scaler.transform(input_array)

            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]

            st.markdown("---")
            st.markdown(
                f"""
            <div class="prediction-box">
                <h2>{WINE_NAMES.get(prediction, f"Class {prediction}")}</h2>
                <p style="margin-top: 0.5rem; font-size: 1.2rem;">
                    ความมั่นใจ: {probabilities[prediction]*100:.2f}%
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("### 📊 ความน่าจะเป็นของแต่ละคลาส")
            prob_df = pd.DataFrame({
                "คลาส": [WINE_NAMES.get(i, f"Class {i}") for i in range(3)],
                "ความน่าจะเป็น (%)": [p * 100 for p in probabilities],
            })

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            with col_b:
                chart_df = pd.DataFrame(
                    probabilities * 100,
                    index=["Class 0", "Class 1", "Class 2"],
                    columns=["ความน่าจะเป็น (%)"],
                )
                st.bar_chart(chart_df, color="#4F46E5")

            # Feature Importance
            st.markdown("---")
            st.markdown("### 🎯 Feature Importance")

            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True)

            importance_df = importance_df[importance_df["Importance"] > 0]

            if len(importance_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(
                    importance_df["Feature"],
                    importance_df["Importance"],
                    color="#4F46E5",
                    edgecolor="white",
                )
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance จากโมเดล Decision Tree")
                st.pyplot(fig)
    else:
        st.error(
            "❌ ไม่สามารถทำนายผลได้เนื่องจากยังไม่มีไฟล์โมเดลในโฟลเดอร์ 'model_files/'"
        )

# ===== Footer =====
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎓 พัฒนาเพื่อการศึกษา | Decision Tree Classifier with Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)