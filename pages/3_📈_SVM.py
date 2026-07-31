import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SVM Model", page_icon="📈", layout="wide")

# Sidebar ข้อมูลผู้พัฒนา (แก้ไขข้อมูลให้ถูกต้อง)
with st.sidebar:
    if os.path.exists("assets/profile.jpg"):
        st.image("assets/profile.jpg", width=130)
    st.markdown("### 👨‍💻 ข้อมูลผู้พัฒนา")
    st.markdown("""
    **ชื่อ-นามสกุล:** นายปฐมพงศ์ ชัยสรรค์ 
    **รหัสนักศึกษา:** 664245039  
    **หมู่เรียน:** 66/44 
    """)
    st.divider()

st.title("📈 Support Vector Machine (SVM) Model")
st.write("ระบบทำนายผลด้วยโมเดล SVR (Support Vector Regression)")

# ปรับแต่งสไตล์ด้วย CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    div.stButton > button:first-child {
        background-color: #0EA5E9;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.2);
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #0284C7;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.3);
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


# 2. โหลดโมเดลที่บันทึกไว้ (แก้ path ชี้ไปที่โฟลเดอร์ models/)
@st.cache_resource
def load_model():
    model_path = "models/gld_svm_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        # ลองค้นหาชื่อไฟล์สั้น กรณีวางไว้นอกโฟลเดอร์
        alt_path = "models/svm_model.pkl"
        if os.path.exists(alt_path):
            with open(alt_path, "rb") as f:
                data = pickle.load(f)
            return data
    return None


loaded_data = load_model()

if loaded_data is not None:
    model = loaded_data.get("model")
    scaler = loaded_data.get("scaler")
    features = loaded_data.get("features")
    metrics = loaded_data.get("metrics", {})
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

# Header ส่วนหัวเว็บ
st.markdown(
    '<p class="main-title">💰 Gold Price Prediction Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">ระบบวิเคราะห์และทำนายราคาทองคำ (GLD) ด้วย Machine Learning (SVR)</p>',
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        "⚠️ ไม่พบไฟล์โมเดลในโฟลเดอร์ 'models/' (แสดงผลหน้าจอแบบจำลองเพื่อทดสอบ UI)"
    )

# แบ่งโครงสร้างหน้าจอหลักออกเป็น 2 ฝั่ง (ฝั่งกรอกข้อมูล 60% : ฝั่งแสดงผล 40%)
main_col1, main_col2 = st.columns([3, 2], gap="large")

with main_col1:
    st.markdown("### 📊 ป้อนข้อมูลดัชนีชี้วัดตลาด")

    # 🌟 ใช้ Tabs แยกหมวดหมู่ข้อมูล
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
            )
            uso = st.number_input(
                "USO (ราคาน้ำมัน)", value=14.50, step=0.1, format="%.2f"
            )
        with c2:
            slv = st.number_input(
                "SLV (ราคาแร่เงิน)", value=15.50, step=0.1, format="%.2f"
            )
            eur_usd = st.number_input(
                "EUR/USD (อัตราแลกเปลี่ยน)",
                value=1.18,
                step=0.01,
                format="%.4f",
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
            )
            gld_ma10 = st.number_input(
                "GLD_MA10 (เฉลี่ยทองคำ 10 วัน)",
                value=121.80,
                step=0.5,
                format="%.2f",
            )
            spx_ma5 = st.number_input(
                "SPX_MA5 (เฉลี่ย S&P 5 วัน)",
                value=2695.0,
                step=10.0,
                format="%.2f",
            )
        with c4:
            gld_return = st.number_input(
                "GLD_Return (ผลตอบแทนทองคำรายวัน)",
                value=0.0050,
                step=0.0001,
                format="%.5f",
            )
            spx_return = st.number_input(
                "SPX_Return (ผลตอบแทน S&P รายวัน)",
                value=0.0030,
                step=0.0001,
                format="%.5f",
            )
            uso_return = st.number_input(
                "USO_Return (ผลตอบแทนน้ำมันรายวัน)",
                value=-0.0100,
                step=0.0001,
                format="%.5f",
            )
            slv_return = st.number_input(
                "SLV_Return (ผลตอบแทนแร่เงินรายวัน)",
                value=0.0080,
                step=0.0001,
                format="%.5f",
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
            )
            gld_lag2 = st.number_input(
                "GLD_Lag2 (ราคาทองคำย้อนหลัง 2 วัน)",
                value=121.90,
                step=0.5,
                format="%.2f",
            )
        with c6:
            spx_lag1 = st.number_input(
                "SPX_Lag1 (ดัชนี S&P ย้อนหลัง 1 วัน)",
                value=2690.0,
                step=10.0,
                format="%.2f",
            )

    st.markdown("##")
    predict_btn = st.button(
        "🚀 ประมวลผลและทำนายราคาทองคำ", use_container_width=True
    )

with main_col2:
    st.markdown("### 🔮 ผลการวิเคราะห์")

    with st.container():
        if predict_btn:
            if model is not None and scaler is not None:
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
                new_df = new_df[features]

                new_scaled = scaler.transform(new_df)
                prediction = model.predict(new_scaled)

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
        st.markdown("##")
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