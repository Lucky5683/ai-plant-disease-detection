import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st

import numpy as np

import cv2

from PIL import Image

from tensorflow.keras.models import load_model

from src.deployment.remedies import remedies


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(

    page_title="AI Plant Disease Detection",

    layout="centered"
)


# ============================================
# LOAD MODEL
# ============================================

model = load_model(

    r"X:\AI-Plant-Disease-Detection\models\transfer_learning\final_mobilenet_model.h5"
)


# ============================================
# CLASS NAMES
# ============================================

class_names = [

    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


# ============================================
# THRESHOLD
# ============================================

THRESHOLD = 0.95


# ============================================
# TITLE
# ============================================

st.title("AI Plant Disease Detection System")

st.write(
    "Upload a plant leaf image for disease prediction."
)


# ============================================
# FILE UPLOADER
# ============================================

uploaded_file = st.file_uploader(

    "Choose a leaf image",

    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================
# PREDICTION
# ============================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    image_np = np.array(image)


    st.image(

        image,

        caption="Uploaded Image",

        use_container_width=True
    )


    # ========================================
    # PREPROCESS
    # ========================================

    img_resized = cv2.resize(

        image_np,

        (224,224)
    )

    img_normalized = img_resized / 255.0

    img_input = np.expand_dims(

        img_normalized,

        axis=0
    )


    # ========================================
    # PREDICT
    # ========================================

    prediction = model.predict(

        img_input,

        verbose=0
    )


    top_3_idx = prediction[0].argsort()[-3:][::-1]

    predicted_class = top_3_idx[0]

    confidence = prediction[0][predicted_class]


    # ========================================
    # UNKNOWN DETECTION
    # ========================================

    if confidence < THRESHOLD:

        st.error(
            "Unknown Plant or Disease"
        )

        st.warning(
            f"Confidence: {confidence*100:.2f}%"
        )

    else:

        disease_name = class_names[predicted_class]


        st.success(
            f"Prediction: {disease_name}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )


        # ====================================
        # REMEDY
        # ====================================

        remedy = remedies.get(

            disease_name,

            "No remedy available."
        )

        st.warning(
            f"Suggested Remedy: {remedy}"
        )


    # ========================================
    # TOP 3 PREDICTIONS
    # ========================================

    st.subheader("Top 3 Predictions")


    for idx in top_3_idx:

        st.write(

            f"{class_names[idx]} : "

            f"{prediction[0][idx]*100:.2f}%"
        )