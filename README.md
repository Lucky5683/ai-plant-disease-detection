# AI Plant Disease Detection
````markdown


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
````

## Model Performance

The final model achieved a test accuracy of:

**90.52%**

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve

## Real-World Testing

In addition to the dataset images, the model was tested with different types of images to understand how it behaves outside the training dataset.

Testing included:

* Real leaf images
* Low-light images
* Blurry images
* Images with complex backgrounds

An unknown-detection threshold was also added to reduce incorrect predictions when an image does not belong to the expected classes.

## Streamlit Application

A Streamlit application is included for testing the trained model through a simple web interface.

The application allows users to:

1. Upload a plant leaf image
2. Process the image
3. Run the trained model
4. View the predicted class
5. View the confidence score
6. Get a suggested remedy where available

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit

## Project Structure

```text
AI-Plant-Disease-Detection/
│
├── archive/
│   └── raw/
│       └── PlantVillage/
│
├── data/
│   └── processed/
│
├── models/
│   └── transfer_learning/
│
├── notebooks/
│
├── reports/
│
├── screenshots/
│
├── src/
│
├── streamlit_app/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Screenshots

### Dataset Exploration

![Dataset Exploration](./screenshots/dataset-exploration.png)

### Class Distribution

![Class Distribution](./screenshots/class-distribution.png)

### Sample Leaf Images

![Sample Leaf Images](screenshots/sample-leaf-images.png)

### Model Evaluation

![Model Evaluation](screenshots/model-evaluation-metrics.png)

### Training Loss

![Training Loss](Screenshots/training-loss.png)

### Test Accuracy

![Test Accuracy](screenshots/test-accuracy.png)

### ROC Curve

![ROC Curve](screenshots/roc-curve.png)

### Prediction Result

![Prediction Result](screenshots/prediction-result.png)

## Running the Project

Clone the repository:

```bash
git clone https://github.com/Lucky5683/ai-plant-disease-detection.git
cd ai-plant-disease-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app/app.py
```

## Project Purpose

The main purpose of this project was to understand how computer vision and deep learning can be applied to image classification problems.

Through this project, I worked with image preprocessing, CNNs, transfer learning, model evaluation, real-world testing, and deployment through Streamlit.

## Author

**Dinesh Kumar**

B.Tech in Computer Science Engineering
Specialization: Artificial Intelligence & Data Science

GitHub: [https://github.com/Lucky5683](https://github.com/Lucky5683)

```
