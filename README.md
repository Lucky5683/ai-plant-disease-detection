# AI Plant Disease Detection

A deep learning project for detecting plant diseases from leaf images using image classification.

The project uses the PlantVillage dataset and compares a custom CNN model with transfer learning using MobileNetV2. A Streamlit application is also included so that users can upload a leaf image and get a prediction.

## What This Project Does

The model takes an image of a plant leaf and predicts the corresponding class.

The project includes:

- CNN-based image classification
- Transfer learning using MobileNetV2
- Image preprocessing and augmentation
- Model evaluation using multiple metrics
- Testing with real-world leaf images
- Confidence-based prediction
- Unknown image detection
- A Streamlit web application
- Batch image prediction

## Dataset

The project uses the **PlantVillage dataset**.

The classes used in this project include:

- Tomato diseases
- Potato diseases
- Pepper healthy leaves

The dataset contains both healthy and diseased plant leaf images for training and evaluation.

## Models

Two approaches were used in the project:

### Custom CNN

A CNN model was developed as a baseline for the classification task.

### MobileNetV2 Transfer Learning

MobileNetV2 was used as a pretrained model and fine-tuned for plant disease classification.

Transfer learning provided better performance compared with the baseline CNN and was used as the final model.

## Project Workflow

```text
PlantVillage Dataset
        ↓
Data Exploration
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
CNN Baseline
        ↓
MobileNetV2 Transfer Learning
        ↓
Model Evaluation
        ↓
Real-World Testing
        ↓
Streamlit Application
