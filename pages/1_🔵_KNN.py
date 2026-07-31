import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="KNN Model", page_icon="🔵", layout="wide")

# 2. Sidebar ข้อมูลผู้พัฒนา
with st.sidebar:
    # เช็คตรงๆ ทั้งแบบอยู่ที่ Root และอยู่ใน pages/
    img_paths = [
        "assets/profile.jpg",
        "assets/profile.png",
        "../assets/profile.jpg",
        "../assets/profile.png"
    ]
    
    profile_image_path = None
    for p in img_paths:
        if os.path.exists(p):
            profile_image_path = p
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

# 3. หัวข้อหลัก
st.title("🔵 K-Nearest Neighbor (KNN) Model")
st.write("การทำนายข้อมูลโรคหัวใจด้วยเทคนิค K-Nearest Neighbor")

# แสดงรูปภาพ (ใส่ try-except เผื่อกรณีหาไฟล์รูปไม่พบ เว็บจะได้ไม่ล่ม)
col1, col2 = st.columns(2)
with col1:
    st.header("")
    if os.path.exists("assets/heart1.jpg"):
        st.image("assets/heart1.jpg")

with col2:
    st.header("")
    if os.path.exists("assets/heart2.jpg"):
        st.image("assets/heart2.jpg")

# แบนเนอร์หัวข้อ
html_7 = """
<div style="background-color:#33beff;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h4>ข้อมูลโรคหัวใจสำหรับทำนาย</h4></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

# 4. โหลดข้อมูล CSV
csv_path = "data/Heart3.csv"
if os.path.exists(csv_path):
    dt = pd.read_csv(csv_path)

    st.subheader("📋 ข้อมูลส่วนแรก 10 แถว")
    st.write(dt.head(10))

    st.subheader("📋 ข้อมูลส่วนสุดท้าย 10 แถว")
    st.write(dt.tail(10))

    # สถิติพื้นฐาน
    st.subheader("📈 สถิติพื้นฐานของข้อมูล")
    st.write(dt.describe())

    # แสดงกราฟ Boxplot
    st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
    feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

    st.write(f"### 🎯 Boxplot: {feature} แยกตามชนิดของโรคหัวใจ")
    fig, ax = plt.subplots()
    sns.boxplot(data=dt, x="HeartDisease", y=feature, ax=ax)
    st.pyplot(fig)

    # แสดง Pairplot
    if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
        st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
        fig2 = sns.pairplot(dt, hue="HeartDisease")
        st.pyplot(fig2)

    # 5. ส่วนฟอร์มทำนายข้อมูล
    html_8 = """
    <div style="background-color:#6BD5DA;padding:15px;border-radius:15px;border-style:solid;border-color:black">
    <center><h5>ทำนายข้อมูล</h5></center>
    </div>
    """
    st.markdown(html_8, unsafe_allow_html=True)
    st.markdown("")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        A1 = st.number_input("กรุณาเลือกข้อมูล A1", value=0.0)
        A2 = st.number_input("กรุณาเลือกข้อมูล A2", value=0.0)
        A3 = st.number_input("กรุณาเลือกข้อมูล A3", value=0.0)
        A4 = st.number_input("กรุณาเลือกข้อมูล A4", value=0.0)
        A5 = st.number_input("กรุณาเลือกข้อมูล A5", value=0.0)
        A6 = st.number_input("กรุณาเลือกข้อมูล A6", value=0.0)

    with col_in2:
        A7 = st.number_input("กรุณาเลือกข้อมูล A7", value=0.0)
        A8 = st.number_input("กรุณาเลือกข้อมูล A8", value=0.0)
        A9 = st.number_input("กรุณาเลือกข้อมูล A9", value=0.0)
        A10 = st.number_input("กรุณาเลือกข้อมูล A10", value=0.0)
        A11 = st.number_input("กรุณาเลือกข้อมูล A11", value=0.0)

    if st.button("ทำการทำนายผล"):
        X = dt.drop("HeartDisease", axis=1)
        y = dt.HeartDisease

        Knn_model = KNeighborsClassifier(n_neighbors=3)
        Knn_model.fit(X, y)

        x_input = np.array([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]])
        out = Knn_model.predict(x_input)

        st.subheader("ผลการทำนาย:")
        if out[0] == 1:
            st.error("⚠️ ผลลัพธ์: พบความเสี่ยงเป็นโรคหัวใจ (Class 1)")
            if os.path.exists("assets/heart1.jpg"):
                st.image("assets/heart1.jpg", width=300)
        else:
            st.success("✅ ผลลัพธ์: ปกติ / ไม่พบความเสี่ยง (Class 0)")
            if os.path.exists("assets/heart2.jpg"):
                st.image("assets/heart2.jpg", width=300)
else:
    st.error(f"❌ ไม่พบไฟล์ข้อมูลที่พาท '{csv_path}' กรุณาเช็กว่ามีไฟล์ Heart3.csv ในโฟลเดอร์ data/ หรือยัง")