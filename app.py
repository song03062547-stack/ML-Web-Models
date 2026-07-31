import os
import streamlit as st
from PIL import Image

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ML Models Dashboard | CS NPRU", page_icon="🤖", layout="wide"
)

# 2. ฟังก์ชันแสดง Sidebar ข้อมูลผู้พัฒนา
def show_developer_sidebar():
    with st.sidebar:
        profile_path = "assets/profile.jpg"
        if os.path.exists(profile_path):
            st.image(profile_path, width=130)

        st.markdown("### 👨‍💻 ข้อมูลผู้พัฒนา")
        st.markdown("""
        **ชื่อ-นามสกุล:** นายพฐมพงษ์ ชัยสาร  
        **ชื่อเล่น:** ซองค์  
        **รหัสนักศึกษา:** 654230029  
        **หมู่เรียน:** 65/42  
        **สาขา:** วิทยาการคอมพิวเตอร์  
        **สถาบัน:** มหาวิทยาลัยราชภัฏนครปฐม
        """)
        st.divider()


# แสดง Sidebar
show_developer_sidebar()

# 3. เนื้อหาหน้าหลัก (Home Page)
st.title("🤖 Machine Learning Models Demonstration Dashboard")
st.subheader("ระบบรวมผลงานและสาธิตการประมวลผลโมเดล Machine Learning")

st.markdown("""
---
### 📋 รายการโมเดลที่เปิดให้ทดสอบใช้งาน (เลือกเมนูจาก Sidebar ด้านซ้าย):

1. **1. K-nearest neighbor (KNN):** จำแนกกลุ่มด้วยความใกล้เคียงของข้อมูล
2. **2. Decision Tree:** การจำแนกประเภทข้อมูลด้วยต้นไม้ตัดสินใจ
3. **3. Support Vector Machine (SVM):** การหาเส้นขอบเขตในการแบ่งแยกกลุ่มข้อมูล
4. **4. K-means:** การจัดกลุ่มข้อมูล (Clustering) แบบ Unsupervised Learning
5. **5. Regression Analysis:** การทำนายและพยากรณ์ค่าเชิงตัวเลขแบบต่อเนื่อง
6. **6. Ensemble (Random Forest):** โมเดลสุ่มสร้างต้นไม้ตัดสินใจหลายต้นเพื่อความแม่นยำสูง
---
""")
st.info("👈 คลิกเลือกรายการโมเดลที่ต้องการทดสอบจาก **Sidebar เมนูด้านซ้าย** ได้เลยครับ")
