import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 128

def load_data(data_dir):
    classes = os.listdir(data_dir)
    images = []
    labels = []
    
    for idx, cls in enumerate(classes):
        cls_folder = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img = img_to_array(img) / 255.0  # Normalize
            images.append(img)
            labels.append(idx)
    
    images = np.array(images)
    labels = to_categorical(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
