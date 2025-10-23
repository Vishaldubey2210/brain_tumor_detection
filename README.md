
# Brain Tumor Detection Using Machine Learning

## 🚀 Project Overview

This project demonstrates training a **Convolutional Neural Network (CNN)** to detect brain tumors from MRI images using the publicly available Kaggle dataset. The dataset consists of 3 classes of tumors:

- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  

The goal is to classify MRI brain scans into one of these tumor types with high accuracy, aiding early diagnosis.


## 📂 Project Structure

```
brain_tumor_ml/
│
├── data/                     # Raw dataset images (Training + Testing folders)
├── models/                   # Saved trained model (.h5 format)
├── notebooks/                # Jupyter notebook for training, evaluation & EDA
│   └── train_brain_tumor_model.ipynb
├── src/                      # Modular Python scripts
│   ├── dataloader.py         # Data loading & preprocessing
│   ├── model.py              # Model definition & training functions
│   ├── predict.py            # Prediction utility functions
│   └── utils.py              # Helper functions like visualization
├── app.py                    # Streamlit web app for easy predictions
├── requirements.txt          # Required packages for easy installation
└── README.md                 # This file - project documentation
```

---

## ⚙️ How This Project Works

### Data Loading & Preprocessing

- MRI images organized by tumor type in `data/Training` and `data/Testing`.
- Each image resized to 128x128 pixels.
- Pixel intensities normalized to [0,1].
- Labels converted to one-hot categorical encoding for multiclass classification.

### Model Architecture

- CNN composed of 3 convolutional layers with max pooling after each.
- Dense layers with ReLU activations follow, with dropout to prevent overfitting.
- Output layer uses softmax for multi-class classification.

### Training Process

- Model trains for 15 epochs using the Adam optimizer.
- Uses categorical cross-entropy loss.
- Tracks both training and validation accuracy each epoch.
- Best model is saved in `models/brain_tumor_model.h5` for future use.

### Evaluation Metrics

- Reports test accuracy and loss.
- Generates confusion matrix and classification report detailing precision, recall, and F1-score per tumor class.
- These metrics confirm the model's reliability on unseen data.

---

## 📈 Results

- Training accuracy surpasses 90% within 15 epochs.
- Validation accuracy closely tracks training, indicating good generalization.
- Confusion matrix and detailed classification report provide insights on class-wise performance.

---

## 🏃 How to Run This Project

### 1. Install Required Python Packages

```
pip install -r requirements.txt
```

### 2. Setup Dataset

Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and unzip into `data/` maintaining this folder structure:

```
data/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── pituitary_tumor/
    └── no_tumor/
```

### 3. Train the Model (Optional)

Run the Jupyter notebook:

```
jupyter notebook notebooks/train_brain_tumor_model.ipynb
```

This handles data loading, preprocessing, model training, evaluation, and saves the best model.

### 4. Run Streamlit App for Inference

For interactive predictions, run the webapp:

```
streamlit run app.py
```

Upload an MRI image and get instant brain tumor type classification with confidence scores.

---

## 🔮 Future Enhancements

- Apply data augmentation techniques to improve model robustness.
- Use transfer learning with pre-trained models like ResNet or EfficientNet for better performance.
- Add Grad-CAM or similar explainability methods to visualize tumor regions.
- Deploy as a cloud-based web service or via APIs for real-world usage.

---


