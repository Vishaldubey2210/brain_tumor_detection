# Prediction logic goes here
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = 128
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

model = load_model('models/brain_tumor_model.h5')

def predict_brain_tumor(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]

    return CLASS_NAMES[class_idx], confidence
