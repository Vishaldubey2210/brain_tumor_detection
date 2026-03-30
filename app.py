import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load your trained model (adjust path if needed)
model = load_model('models/brain_tumor_model.h5')

# Define image size and class names consistent with training
IMAGE_SIZE = 128
class_names = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']  # Use your actual classes

def predict_image(image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    return class_names[class_idx], confidence

st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            prediction, confidence = predict_image(image)
            st.success(f"Prediction: {prediction}")
            st.info(f"Confidence: {confidence:.2f}")
