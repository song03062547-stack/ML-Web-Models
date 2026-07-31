# -*- coding: utf-8 -*-
"""
Gold Price Prediction Web App - Streamlit (Support Vector Regression - SVR)
"""

import os
import pickle
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ===== 1. ตั้งค่าหน้าเว็บ =====
st.set_page_config(page_title="SVM Model", page_icon="📈", layout="wide")

# ===== 2. Sidebar (ข้อมูลผู้พัฒนา) =====
with st.sidebar:
    from pathlib import Path
    
    # คำนวณหาตำแหน่งโฟลเดอร์หลัก (ML-Web-Models) จากตำแหน่งไฟล์ปัจจุบัน
    base_dir = Path(__file__).resolve().parent.parent # ถอยหลังจาก pages/ ออกมา 1 ชั้น
    
    profile_image_path = None
    for ext in ["jpg", "jpeg", "png", "JPG", "PNG"]:
        img_path = base_dir / "assets" / f"profile.{ext}"
        if img_path.exists():
            profile_image_path = str(img_path)
            break
            
    if profile_image_path:
        st.image(profile_image_path, width=130)
    else:
        st.warning("⚠️ ไม่พบรูปโปรไฟล์ใน assets/")

    st.markdown("### 👨‍💻 ข้อมูลผู้พัฒนา")
    st.markdown(
        """
    **ชื่อ-นามสกุล:** นายปฐมพงศ์ ชัยสรรค์ 
    **ชื่อเล่น:** ซองค์
    **รหัสนักศึกษา:** 664245039  
    **หมู่เรียน:** 66/44 
    """
    )
    st.divider()

    st.markdown("### 📈 เกี่ยวกับโมเดล SVR")
    st.markdown(
        """
    **ระบบวิเคราะห์และทำนายราคาทองคำ (GLD)**
    - **Model:** Support Vector Regression (SVR)
    - **Features:** 14 มิติ (Market Indices, Moving Average, Returns, Lags)
    """
    )
    st.divider()

# ===== 3. Custom CSS สำหรับตกแต่ง UI =====
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%);
        color: white;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.2);
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(14, 165, 233, 0.3);
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ===== 4. โหลดโมเดล =====
@st.cache_resource
def load_model():
    """โหลดโมเดล SVR จากไฟล์ที่กำหนด รองรับโครงสร้างโฟลเดอร์ใหม่"""
    possible_paths = [
        Path("model_files/gold/gld_svm_model.pkl"),
    ]

    for path in possible_paths:
        if path.exists():
            try:
                # ลองโหลดด้วย pickle
                with open(path, "rb") as f:
                    data = pickle.load(f)
                return data
            except Exception:
                try:
                    # ถ้า pickle ไม่ผ่าน ให้ลอง joblib
                    data = joblib.load(path)
                    return data
                except Exception:
                    continue
    return None


loaded_data = load_model()

if loaded_data is not None and isinstance(loaded_data, dict):
    model = loaded_data.get("model")
    scaler = loaded_data.get("scaler")
    features = loaded_data.get("features")
    metrics = loaded_data.get("metrics", {})
elif loaded_data is not None:
    # กรณีบันทึกเฉพาะวัตถุโมเดลอย่างเดียว
    model = loaded_data
    scaler = None
    features = None
    metrics = {}
else:
    model = None
    scaler = None
    features = [
        "SPX",
        "USO",
        "SLV",
        "EUR/USD",
        "GLD_MA5",
        "GLD_MA10",
        "SPX_MA5",
        "GLD_Return",
        "SPX_Return",
        "USO_Return",
        "SLV_Return",
        "GLD_Lag1",
        "GLD_Lag2",
        "SPX_Lag1",
    ]
    metrics = {}

# ===== 5. Header หลัก =====
st.markdown(
    '<p class="main-title">💰 Gold Price Prediction Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">ระบบวิเคราะห์และทำนายราคาทองคำ (GLD) ด้วย Machine Learning (Support Vector Regression)</p>',
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        "⚠️ ไม่พบไฟล์โมเดลในโฟลเดอร์ 'model_files/svm/' หรือ 'model_files/' (กำลังแสดงผลหน้าจอ UI แบบจำลอง)"
    )

# ===== 6. โครงสร้างหน้าจอหลัก (กรอกข้อมูล 60% : แสดงผล 40%) =====
main_col1, main_col2 = st.columns([3, 2], gap="large")

with main_col1:
    st.markdown("### 📊 ป้อนข้อมูลดัชนีชี้วัดตลาด")

    # ครอบ Form เพื่อป้องการลืม State ระหว่างกรอกตัวเลขใน Tabs
    with st.form("svm_predict_form"):
        tab1, tab2, tab3 = st.tabs([
            "📈 Market Indices (ดัชนีหลัก)",
            "🔄 Moving Average & Returns (ค่าเฉลี่ย/ผลตอบแทน)",
            "⏳ Lag Features (ค่าย้อนหลัง)",
        ])

        with tab1:
            st.caption("ระบุราคาและอัตราแลกเปลี่ยนปัจจุบันในตลาด")
            c1, c2 = st.columns(2)
            with c1:
                spx = st.number_input(
                    "SPX Index (ดัชนี S&P 500)",
                    value=2700.0,
                    step=10.0,
                    format="%.2f",
                    key="spx",
                )
                uso = st.number_input(
                    "USO (ราคาน้ำมัน)",
                    value=14.50,
                    step=0.1,
                    format="%.2f",
                    key="uso",
                )
            with c2:
                slv = st.number_input(
                    "SLV (ราคาแร่เงิน)",
                    value=15.50,
                    step=0.1,
                    format="%.2f",
                    key="slv",
                )
                eur_usd = st.number_input(
                    "EUR/USD (อัตราแลกเปลี่ยน)",
                    value=1.18,
                    step=0.01,
                    format="%.4f",
                    key="eur_usd",
                )

        with tab2:
            st.caption("ระบุค่าทางสถิติและผลตอบแทนเคลื่อนที่")
            c3, c4 = st.columns(2)
            with c3:
                gld_ma5 = st.number_input(
                    "GLD_MA5 (เฉลี่ยทองคำ 5 วัน)",
                    value=122.50,
                    step=0.5,
                    format="%.2f",
                    key="gld_ma5",
                )
                gld_ma10 = st.number_input(
                    "GLD_MA10 (เฉลี่ยทองคำ 10 วัน)",
                    value=121.80,
                    step=0.5,
                    format="%.2f",
                    key="gld_ma10",
                )
                spx_ma5 = st.number_input(
                    "SPX_MA5 (เฉลี่ย S&P 5 วัน)",
                    value=2695.0,
                    step=10.0,
                    format="%.2f",
                    key="spx_ma5",
                )
            with c4:
                gld_return = st.number_input(
                    "GLD_Return (ผลตอบแทนทองคำรายวัน)",
                    value=0.0050,
                    step=0.0001,
                    format="%.5f",
                    key="gld_return",
                )
                spx_return = st.number_input(
                    "SPX_Return (ผลตอบแทน S&P รายวัน)",
                    value=0.0030,
                    step=0.0001,
                    format="%.5f",
                    key="spx_return",
                )
                uso_return = st.number_input(
                    "USO_Return (ผลตอบแทนน้ำมันรายวัน)",
                    value=-0.0100,
                    step=0.0001,
                    format="%.5f",
                    key="uso_return",
                )
                slv_return = st.number_input(
                    "SLV_Return (ผลตอบแทนแร่เงินรายวัน)",
                    value=0.0080,
                    step=0.0001,
                    format="%.5f",
                    key="slv_return",
                )

        with tab3:
            st.caption("ระบุค่าราคาปิดของวันก่อนหน้า")
            c5, c6 = st.columns(2)
            with c5:
                gld_lag1 = st.number_input(
                    "GLD_Lag1 (ราคาทองคำย้อนหลัง 1 วัน)",
                    value=122.30,
                    step=0.5,
                    format="%.2f",
                    key="gld_lag1",
                )
                gld_lag2 = st.number_input(
                    "GLD_Lag2 (ราคาทองคำย้อนหลัง 2 วัน)",
                    value=121.90,
                    step=0.5,
                    format="%.2f",
                    key="gld_lag2",
                )
            with c6:
                spx_lag1 = st.number_input(
                    "SPX_Lag1 (ดัชนี S&P ย้อนหลัง 1 วัน)",
                    value=2690.0,
                    step=10.0,
                    format="%.2f",
                    key="spx_lag1",
                )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.form_submit_button(
            "🚀 ประมวลผลและทำนายราคาทองคำ", use_container_width=True
        )

# ===== 7. ส่วนแสดงผลการประมวลผล =====
with main_col2:
    st.markdown("### 🔮 ผลการวิเคราะห์")

    with st.container():
        if predict_btn:
            if model is not None:
                with st.spinner("🤖 กำลังคำนวณราคาทองคำ..."):
                    input_data = {
                        "SPX": spx,
                        "USO": uso,
                        "SLV": slv,
                        "EUR/USD": eur_usd,
                        "GLD_MA5": gld_ma5,
                        "GLD_MA10": gld_ma10,
                        "SPX_MA5": spx_ma5,
                        "GLD_Return": gld_return,
                        "SPX_Return": spx_return,
                        "USO_Return": uso_return,
                        "SLV_Return": slv_return,
                        "GLD_Lag1": gld_lag1,
                        "GLD_Lag2": gld_lag2,
                        "SPX_Lag1": spx_lag1,
                    }

                    new_df = pd.DataFrame([input_data])
                    if features:
                        new_df = new_df[features]

                    # ทำการ Scale ข้อมูลกรณีมี Scaler ติดมากับไฟล์โมเดล
                    if scaler is not None:
                        new_scaled = scaler.transform(new_df)
                        prediction = model.predict(new_scaled)
                    else:
                        prediction = model.predict(new_df)

                    st.markdown(
                        '<div class="metric-card">', unsafe_allow_html=True
                    )
                    st.metric(
                        label="💰 ราคา GLD คาดการณ์ (Estimated Price)",
                        value=f"${prediction[0]:,.2f} USD",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.success("✨ คำนวณราคาปิดตามสภาวะตลาดปัจจุบันเสร็จสิ้น")
            else:
                st.error("❌ ไม่สามารถคำนวณได้เนื่องจากยังไม่ได้โหลดโมเดล")
        else:
            st.info(
                "💡 กรุณากรอกข้อมูลในแท็บซ้ายมือให้เรียบร้อย แล้วกดปุ่มเพื่อให้ระบบ Machine Learning คำนวณราคาทองคำ"
            )

    if metrics:
        st.markdown("---")
        with st.expander(
            "📊 ตรวจสอบประสิทธิภาพของโมเดล (Model Performance Information)",
            expanded=True,
        ):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(
                    label="R² Score (ความแม่นยำ)",
                    value=f"{metrics.get('r2', 0):.4f}",
                )
            with col_m2:
                st.metric(
                    label="RMSE (ค่าความคลาดเคลื่อน)",
                    value=f"{metrics.get('rmse', 0):.4f}",
                )
            st.caption(
                "ℹ️ โมเดลนี้ขับเคลื่อนด้วย Support Vector Regression (SVR) ผ่านกระบวนการ Feature Engineering 14 มิติ"
            )

# ===== Footer =====
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎓 พัฒนาเพื่อการศึกษา | Support Vector Regression (SVR) Dashboard with Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)