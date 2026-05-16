# AI-Powered Plant Disease Detection System 

## Overview

This project is a Deep Learning based Plant Disease Detection System developed using Computer Vision techniques. The model predicts whether a plant leaf is healthy or affected by a disease using image classification.

The system was trained on the PlantVillage dataset and uses both a custom CNN model and Transfer Learning with MobileNetV2 for better accuracy and performance.

A Streamlit web application is also integrated for easy image upload and prediction.

---

## Features

- Plant leaf disease classification
- CNN baseline model
- Transfer Learning using MobileNetV2
- Real-world image testing
- Batch image prediction
- Unknown disease detection
- Confidence score display
- Remedy suggestions
- Streamlit web application

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

---

## Dataset

Dataset Used:
PlantVillage Dataset

Classes Included:
- Tomato Diseases
- Potato Diseases
- Pepper Healthy Leaves

Dataset contains healthy and diseased leaf images used for training and evaluation.

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Data Augmentation
4. CNN Model Training
5. Transfer Learning
6. Model Evaluation
7. Real-world Testing
8. Streamlit Deployment

---

## Model Performance

### Final Test Accuracy
90.52%

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Real-World Testing

The model was tested on:
- Real leaf images
- Low-light images
- Blurry images
- Complex background images

Threshold-based unknown detection was added to reduce incorrect predictions on unrelated plant images.

---

## Project Structure

```bash
AI-Plant-Disease-Detection/
│
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── streamlit_app/
├── README.md
├── requirements.txt
└── .gitignore