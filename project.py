# 🌿 Plant Disease Detection from Leaf Images using CNN + Streamlit

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# ---------- Step 1: Load Model ----------
model_path = "plant_disease_model.h5"
if not os.path.exists(model_path):
    st.error("⚠ Model file not found. Please train the model first.")
    st.stop()

model = load_model(model_path)

# ---------- Step 2: Class Labels (Update as per your classes) ----------
class_names = ['Potato__Early_blight', 'Potato_Healthy', 'Potato__Late_blight',
               'Tomato__Early_blight', 'Tomato_Healthy', 'Tomato__Late_blight']

# ---------- Step 3: Streamlit App ----------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image to predict the disease using a trained CNN model.")

uploaded_file = st.file_uploader("Choose a leaf image (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((128, 128))
    st.image(img, caption="Uploaded Leaf", use_column_width=True)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🩺 Predicted Disease: *{predicted_class}*")
    st.info(f"🔍 Confidence: {confidence:.2f}%")
