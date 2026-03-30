import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# 🔥 Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# 🔥 Load model safely (FIXED)
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model(
            'models/brain_tumor_model.h5',
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# Image size
IMAGE_SIZE = 128

# Class names (ensure same order as training)
class_names = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

# 🔥 Prediction function
def predict_image(image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = float(preds[0][class_idx])

    return class_names[class_idx], confidence

# 🔥 UI
st.title("🧠 Brain Tumor Detection App")

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        if model is None:
            st.error("Model failed to load. Please check deployment.")
        else:
            with st.spinner("Analyzing image..."):
                prediction, confidence = predict_image(image)

                st.success(f"🧾 Prediction: {prediction}")
                st.info(f"📊 Confidence: {confidence:.2f}")
