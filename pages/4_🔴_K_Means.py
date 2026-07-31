# -*- coding: utf-8 -*-
"""
K-Means Clustering Web Application
"""

import os
import pickle
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ===== 1. ตั้งค่าหน้าเว็บ =====
st.set_page_config(page_title="K-Means Model", page_icon="🔴", layout="wide")

# ===== 2. Custom CSS สำหรับตกแต่ง UI =====
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        color: #1e293b;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1.5rem 0;
    }
    .cluster-number {
        font-size: 3.8rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #0d47a1;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ===== 3. ฟังก์ชันโหลดหรือสร้างโมเดล =====
@st.cache_resource
def load_or_create_model():
    """โหลดโมเดล K-Means จากไฟล์ หรือสร้างโมเดลเริ่มต้นถ้าไม่พบไฟล์"""
    possible_paths = [
        Path("model_files/kmeans/kmeans_model.pkl"),
        Path("model_files/kmeans/feature_names.pkl"),
        Path("model_files/kmeans/scaler.pkl"),
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    return (
                        data.get("model"),
                        data.get("scaler"),
                        data.get("features"),
                    )
                return data, None, None
            except Exception:
                try:
                    data = joblib.load(path)
                    if isinstance(data, dict):
                        return (
                            data.get("model"),
                            data.get("scaler"),
                            data.get("features"),
                        )
                    return data, None, None
                except Exception:
                    continue

    # หากไม่พบไฟล์โมเดล ให้สร้างโมเดล Iris Dataset เป็นค่าเริ่มต้น
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)

    return kmeans, scaler, list(df.columns)


# ===== 4. Sidebar (ข้อมูลผู้พัฒนา) =====
with st.sidebar:
    # ✅ ใช้โค้ดชุดนี้แทน จะช่วยสแกนหาโฟลเดอร์ assets ให้เจอชัวร์ๆ
    from pathlib import Path
    
    profile_image_path = None
    paths_to_try = [
        Path("assets/profile.jpg"),
        Path("assets/profile.png"),
        Path("../assets/profile.jpg"),
        Path("../assets/profile.png"),
    ]
    if "__file__" in locals():
        paths_to_try.extend([
            Path(__file__).parent / "assets/profile.jpg",
            Path(__file__).parent / "assets/profile.png",
            Path(__file__).parent.parent / "assets/profile.jpg",
            Path(__file__).parent.parent / "assets/profile.png",
        ])

    for p in paths_to_try:
        if p.exists():
            profile_image_path = str(p)
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

    st.markdown("## 📋 เกี่ยวกับแอปพลิเคชัน")
    st.info(
        """
    ระบบนี้ใช้โมเดล **K-Means Clustering** ในการจัดกลุ่มข้อมูลตามคุณลักษณะ (Features)
    
    **รายละเอียดโมเดล:**
    - **อัลกอริทึม:** K-Means Clustering
    - **ชุดข้อมูล:** Iris Dataset (Default)
    - **จำนวนฟีเจอร์:** 4 ฟีเจอร์
    """
    )

    st.markdown("---")
    st.markdown("## 🎯 วิธีการใช้งาน")
    st.markdown(
        """
    1. **Manual Prediction**: ปรับค่าฟีเจอร์ผ่าน Slider
    2. **Batch Prediction**: อัปโหลดไฟล์ CSV เพื่อจัดกลุ่มแบบชุดข้อมูล
    3. **Model Information**: ตรวจสอบรายละเอียดและ Cluster Centers
    """
    )

    st.markdown("---")
    if st.button("🔄 รีเซ็ตระบบ (Reset All)"):
        st.rerun()

# ===== 5. Main Header =====
st.markdown(
    """
<div class="main-header">
    <h1>🔴 K-Means Clustering App</h1>
    <p>ระบบจัดกลุ่มข้อมูลและวิเคราะห์กลุ่มข้อมูลแบบปฏิสัมพันธ์ (Interactive)</p>
</div>
""",
    unsafe_allow_html=True,
)

# ===== 6. โหลดโมเดลและประมวลผลหน้าจอหลัก =====
try:
    model, scaler, feature_names = load_or_create_model()

    if feature_names is None:
        feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

    tab1, tab2, tab3 = st.tabs([
        "📝 Manual Prediction (ทำนายรายรายการ)",
        "📁 Batch Prediction (ประมวลผลไฟล์ CSV)",
        "ℹ️ Model Information (ข้อมูลโมเดล)",
    ])

    # ------------------ Tab 1: Manual Prediction ------------------
    with tab1:
        st.markdown("### 🎯 กำหนดค่าฟีเจอร์สำหรับทดสอบ")

        with st.form("kmeans_manual_form"):
            col1, col2 = st.columns(2)

            with col1:
                sepal_length = st.slider(
                    "Sepal Length (cm)",
                    min_value=4.0,
                    max_value=8.0,
                    value=5.5,
                    step=0.1,
                )
                sepal_width = st.slider(
                    "Sepal Width (cm)",
                    min_value=2.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                )

            with col2:
                petal_length = st.slider(
                    "Petal Length (cm)",
                    min_value=1.0,
                    max_value=7.0,
                    value=4.0,
                    step=0.1,
                )
                petal_width = st.slider(
                    "Petal Width (cm)",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                )

            submit_btn = st.form_submit_button(
                "🔮 ทำนายกลุ่มข้อมูล (Predict Cluster)"
            )

        if submit_btn:
            input_data = np.array([[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
            ]])

            if scaler is not None:
                input_scaled = scaler.transform(input_data)
            else:
                input_scaled = input_data

            cluster = model.predict(input_scaled)[0]
            distances = np.linalg.norm(
                model.cluster_centers_ - input_scaled, axis=1
            )
            closest_distance = distances[cluster]

            st.markdown(
                f"""
            <div class="result-card">
                <h2>🎉 ผลการจัดกลุ่ม (Prediction Result)</h2>
                <div class="cluster-number">Cluster {cluster}</div>
                <p>ระยะห่างจากจุดศูนย์กลางกลุ่ม (Distance to Cluster Center): <strong>{closest_distance:.4f}</strong></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h4>จำนวนฟีเจอร์นำเข้า</h4>
                    <p style="font-size: 1.5rem; font-weight: bold;">{len(input_data[0])} ค่า</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with c_m2:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h4>กลุ่มที่ได้รับ (Cluster)</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: #667eea;">Cluster {cluster}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with c_m3:
                confidence_score = 1 / (1 + closest_distance)
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h4>ความใกล้เคียงศูนย์กลาง</h4>
                    <p style="font-size: 1.5rem; font-weight: bold;">{confidence_score:.2%}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "### 📏 ระยะห่างไปยังจุดศูนย์กลางของทุกกลุ่ม (Distances to All Cluster Centers)"
            )
            distance_df = pd.DataFrame({
                "Cluster": [
                    f"Cluster {i}" for i in range(len(distances))
                ],
                "Distance": distances,
                "Is Nearest": [
                    "✅ กลุ่มที่สังกัด" if i == cluster else "-"
                    for i in range(len(distances))
                ],
            })
            st.dataframe(
                distance_df, use_container_width=True, hide_index=True
            )

            st.markdown("### 📊 การเปรียบเทียบฟีเจอร์ (Feature Radar Chart)")
            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=input_scaled[0].tolist() + [input_scaled[0][0]],
                    theta=feature_names + [feature_names[0]],
                    fill="toself",
                    name="Input Sample",
                    line_color="rgb(102, 126, 234)",
                )
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=model.cluster_centers_[cluster].tolist()
                    + [model.cluster_centers_[cluster][0]],
                    theta=feature_names + [feature_names[0]],
                    fill="toself",
                    name=f"Cluster {cluster} Center",
                    line_color="rgb(255, 99, 132)",
                )
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------ Tab 2: Batch Prediction ------------------
    with tab2:
        st.markdown("### 📁 อัปโหลดไฟล์ CSV เพื่อจัดกลุ่มแบบชุดข้อมูล")
        st.info(
            "ข้อแนะนำ: ไฟล์ CSV ต้องมีคอลัมน์ฟีเจอร์ครบถ้วนตามชื่อดังนี้: "
            + ", ".join(feature_names)
        )

        uploaded_file = st.file_uploader(
            "เลือกไฟล์ CSV สำหรับประมวลผล", type=["csv"]
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown("### 📊 ตัวอย่างข้อมูลที่อัปโหลด (Preview)")
                st.dataframe(df.head(), use_container_width=True)

                required_cols = set(feature_names)
                actual_cols = set(df.columns)

                if required_cols.issubset(actual_cols):
                    X_batch = df[feature_names].values
                    if scaler is not None:
                        X_batch_scaled = scaler.transform(X_batch)
                    else:
                        X_batch_scaled = X_batch

                    predictions = model.predict(X_batch_scaled)
                    df["Predicted_Cluster"] = predictions

                    st.markdown(
                        "### ✅ ประมวลผลและทำนายผลการจัดกลุ่มเรียบร้อย"
                    )
                    st.dataframe(df, use_container_width=True)

                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ (Download CSV)",
                        data=csv_data,
                        file_name="kmeans_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.markdown(
                        "### 📊 สัดส่วนและปริมาณข้อมูลในแต่ละ Cluster"
                    )
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        cluster_counts = (
                            df["Predicted_Cluster"]
                            .value_counts()
                            .sort_index()
                        )
                        fig_pie = px.pie(
                            values=cluster_counts.values,
                            names=[
                                f"Cluster {i}" for i in cluster_counts.index
                            ],
                            title="สัดส่วนจำนวนข้อมูล (Cluster Ratio)",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_b2:
                        fig_bar = px.bar(
                            x=[f"Cluster {i}" for i in cluster_counts.index],
                            y=cluster_counts.values,
                            title="จำนวนแถวข้อมูลในแต่ละกลุ่ม (Sample Count)",
                            labels={"x": "Cluster", "y": "Count"},
                            color=cluster_counts.values,
                            color_continuous_scale="Viridis",
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown("### 📈 แผนภาพการกระจายตัว (2D Scatter Plot)")
                    fig_2d = px.scatter(
                        df,
                        x=feature_names[0],
                        y=feature_names[1],
                        color=df["Predicted_Cluster"].astype(str),
                        title=f"การกระจายตัว: {feature_names[0]} vs {feature_names[1]}",
                        color_discrete_sequence=px.colors.qualitative.Set1,
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
                else:
                    st.error(
                        f"❌ **โครงสร้างคอลัมน์ไม่ถูกต้อง!**\n\nคอลัมน์ที่ต้องการ: {', '.join(required_cols)}\n\nคอลัมน์ที่พบในไฟล์: {', '.join(actual_cols)}"
                    )
            except Exception as e:
                st.error(
                    f"❌ เกิดข้อผิดพลาดในการอ่านหรือประมวลผลไฟล์: {str(e)}"
                )

    # ------------------ Tab 3: Model Information ------------------
    with tab3:
        st.markdown("### ℹ️ รายละเอียดและโครงสร้างของโมเดล")
        col_i1, col_i2 = st.columns(2)

        with col_i1:
            st.markdown("#### 🔧 พารามิเตอร์โมเดล (Parameters)")
            st.markdown(f"""
            - **Algorithm**: K-Means Clustering
            - **Number of Clusters (K)**: {model.n_clusters}
            - **Max Iterations**: {model.max_iter}
            - **Random State**: {model.random_state if hasattr(model, 'random_state') else 'N/A'}
            - **N Init**: {model.n_init if hasattr(model, 'n_init') else 'N/A'}
            """)

        with col_i2:
            st.markdown("#### 📊 ค่าสถิติของโมเดล (Statistics)")
            st.markdown(f"""
            - **Inertia (Sum of Squared Errors)**: {model.inertia_:.4f}
            - **จำนวนฟีเจอร์**: {len(feature_names)}
            - **รายชื่อฟีเจอร์**: {', '.join(feature_names)}
            """)

        st.markdown("---")
        st.markdown("### 📍 พิกัดจุดศูนย์กลางกลุ่ม (Cluster Centers)")
        centers_df = pd.DataFrame(
            model.cluster_centers_,
            columns=feature_names,
            index=[f"Cluster {i}" for i in range(model.n_clusters)],
        )
        st.dataframe(centers_df, use_container_width=True)

        st.markdown("### 🗺️ แผนภาพความร้อนแสดงค่า Center (Heatmap Visual)")
        fig_centers = px.imshow(
            model.cluster_centers_,
            labels=dict(x="Features", y="Clusters", color="Value"),
            x=feature_names,
            y=[f"Cluster {i}" for i in range(model.n_clusters)],
            color_continuous_scale="Viridis",
            aspect="auto",
        )
        st.plotly_chart(fig_centers, use_container_width=True)

        st.markdown(
            """
        <div class="info-box">
            <strong>💡 หลักการทำงานของ K-Means Clustering:</strong>
            <ol>
                <li><strong>Initialization:</strong> กำหนดจุดศูนย์กลางของกลุ่ม (Centroids) จำนวน K จุดแบบสุ่ม</li>
                <li><strong>Assignment:</strong> จัดแต่ละจุดข้อมูลไปยัง Centroid ที่อยู่ใกล้ที่สุดด้วยระยะห่างยูคลิด (Euclidean Distance)</li>
                <li><strong>Update:</strong> คำนวณตำแหน่ง Centroid ใหม่จากการหาค่าเฉลี่ย (Mean) ของจุดข้อมูลทั้งหมดในกลุ่มนั้น</li>
                <li><strong>Iteration:</strong> ทำซ้ำขั้นตอนที่ 2 และ 3 จนกว่าตำแหน่ง Centroid จะไม่เปลี่ยนแปลง</li>
            </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดกับระบบโมเดล: {str(e)}")

# ===== Footer =====
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1.5rem;'>
    <p>🎓 <strong>Machine Learning for Python Programming Course</strong></p>
    <p>Built with ❤️ using Streamlit | K-Means Dashboard</p>
</div>
""",
    unsafe_allow_html=True,
)