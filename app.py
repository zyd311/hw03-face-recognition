import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 页面配置
st.set_page_config(page_title="人脸检测系统", layout="wide")
st.title("📸 人脸检测系统（HW03）")

# 加载人脸检测器：同时加载正面和侧脸模型
@st.cache_resource
def load_detectors():
    # 正面人脸检测器
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # 侧脸人脸检测器（专门用来识别侧脸）
    profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    return face_detector, profile_detector

face_detector, profile_detector = load_detectors()

# 上传图片
uploaded_file = st.file_uploader("上传一张包含人脸的图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读取图片
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # 转灰度图（OpenCV需要）
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 1. 先用正面模型检测
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    # 2. 如果正面模型没检测到，用侧脸模型再检测一次
    if len(faces) == 0:
        faces = profile_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
    
    # 3. 再试一次水平翻转后的侧脸检测（部分侧脸模型对反向脸识别更好）
    if len(faces) == 0:
        flipped_gray = cv2.flip(gray, 1)
        faces = profile_detector.detectMultiScale(
            flipped_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
        # 把翻转后的坐标还原回去
        faces = [(img_np.shape[1] - (x + w), y, w, h) for (x, y, w, h) in faces]
    
    # 在原图上画框
    result_img = img_np.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_img, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # 分栏展示（同时修复之前的警告）
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图片")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("检测结果")
        st.image(result_img, caption=f"检测到 {len(faces)} 张人脸", use_container_width=True)
    
    st.success(f"✅ 处理完成！共检测到 {len(faces)} 张人脸")

else:
    st.info("请上传一张包含人脸的图片")

# 侧边栏说明
with st.sidebar:
    st.header("作业说明")
    st.markdown("""
    1. 上传一张包含人脸的图片
    2. 系统会自动检测人脸（支持正面+侧脸），并用绿色框标注
    3. 检测结果会显示人脸数量和位置
    """)