import os

import numpy as np

import cv2

from tensorflow.keras.models import load_model


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
# FOLDER PATH
# ============================================

folder_path = r"X:\AI-Plant-Disease-Detection\data\processed\test_real_world"


# ============================================
# CONFIDENCE THRESHOLD
# ============================================

THRESHOLD = 0.95


# ============================================
# VALID IMAGE EXTENSIONS
# ============================================

valid_extensions = (

    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ============================================
# LOOP THROUGH IMAGES
# ============================================

for filename in os.listdir(folder_path):

    if filename.lower().endswith(valid_extensions):

        image_path = os.path.join(

            folder_path,

            filename
        )

        print("\n===================================")

        print(f"Processing: {filename}")


        # ====================================
        # READ IMAGE
        # ====================================

        img = cv2.imread(image_path)


        if img is None:

            print("Error loading image!")

            continue


        # ====================================
        # PREPROCESS IMAGE
        # ====================================

        img_resized = cv2.resize(

            img,

            (224,224)
        )

        img_normalized = img_resized / 255.0

        img_input = np.expand_dims(

            img_normalized,

            axis=0
        )


        # ====================================
        # PREDICT
        # ====================================

        prediction = model.predict(

            img_input,

            verbose=0
        )


        # ====================================
        # TOP 3 PREDICTIONS
        # ====================================

        top_3_idx = prediction[0].argsort()[-3:][::-1]


        # ====================================
        # BEST PREDICTION
        # ====================================

        predicted_class = top_3_idx[0]

        confidence = prediction[0][predicted_class]


        # ====================================
        # OUTPUT
        # ====================================

        if confidence < THRESHOLD:

            print("\nPrediction: Unknown Plant/Disease")

            print(f"Confidence: {confidence*100:.2f}%")

        else:

            print(

                f"\nPrediction: {class_names[predicted_class]}"
            )

            print(

                f"Confidence: {confidence*100:.2f}%"
            )


        # ====================================
        # SHOW TOP 3
        # ====================================

        print("\nTop 3 Predictions:\n")


        for idx in top_3_idx:

            print(

                f"{class_names[idx]} --> "

                f"{prediction[0][idx]*100:.2f}%"
            )


print("\n===================================")

print("Batch Prediction Completed!")