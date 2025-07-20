import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title=" Obstacle Detection App", layout="centered")

# Title & Sidebar
st.title("Obstacle Detection in Drone image")
st.markdown("Detect cars, bikes, trucks, and more in images using a custom-trained YOLOv8 model.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    st.markdown("Upload a road image below to detect vehicles.")

# Load model (cached)
@st.cache_resource
def load_model():
    model = YOLO("models/best.pt")
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=" Uploaded Image", use_column_width=True)

    if st.button(" Detect Obstacles"):
        with st.spinner("Running YOLOv8 model..."):
            results = model.predict(image, conf=confidence)
            result_img = results[0].plot()  # Returns annotated image (np array)

            st.success(" Detection Complete!")
            st.image(result_img, caption=" Detected Obstacles", use_column_width=True)

            # Optional download
            result_pil = Image.fromarray(result_img)
            st.download_button(" Download Result Image",
                               data=result_pil.tobytes(),
                               file_name="detected_image.jpg",
                               mime="image/jpeg")
else:
    st.info(" Please upload an image to get started.")
