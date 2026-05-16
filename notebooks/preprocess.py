
# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================
# 2. DEFINE DATASET PATH
# ============================================

dataset_path = dataset_path = r"X:\AI-Plant-Disease-Detection\archive\raw\PlantVillage\PlantVillage"

print(os.path.exists(dataset_path))
# ============================================
# 3. GET CLASS NAMES
# ============================================

classes = os.listdir(dataset_path)

print("Total Classes:", len(classes))


# ============================================
# 4. COUNT IMAGES PER CLASS
# ============================================

class_counts = {}

for cls in classes:
    
    class_path = os.path.join(dataset_path, cls)
    
    class_counts[cls] = len(os.listdir(class_path))


# ============================================
# 5. CREATE DATAFRAME
# ============================================

df = pd.DataFrame({
    "Class": class_counts.keys(),
    "Image_Count": class_counts.values()
})

print(df.head())


# ============================================
# 6. PLOT CLASS DISTRIBUTION
# ============================================

plt.figure(figsize=(15,8))

plt.bar(df["Class"], df["Image_Count"])

plt.xticks(rotation=90)

plt.title("Class Distribution")

plt.xlabel("Classes")

plt.ylabel("Number of Images")

plt.show()


# ============================================
# DISPLAY SAMPLE IMAGES
# ============================================

plt.figure(figsize=(15,10))

for i, cls in enumerate(classes[:9]):

    class_folder = os.path.join(dataset_path, cls)

    image_name = os.listdir(class_folder)[0]

    image_path = os.path.join(class_folder, image_name)

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(3,3,i+1)

    plt.imshow(img)

    plt.title(cls)

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================
# 8. CHECK IMAGE DIMENSIONS
# ============================================

sample_class = classes[0]

sample_image_path = os.path.join(
    dataset_path,
    sample_class,
    os.listdir(os.path.join(dataset_path, sample_class))[0]
)

img = cv2.imread(sample_image_path)

print("Original Image Shape:", img.shape)


# ============================================
# 9. DEFINE PARAMETERS
# ============================================

IMG_SIZE = 224

BATCH_SIZE = 32


# ============================================
# 10. CREATE IMAGE DATA GENERATOR
# ============================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True,

    brightness_range=[0.8,1.2],

    validation_split=0.15
)


# ============================================
# 11. CREATE TRAIN GENERATOR
# ============================================

train_generator = train_datagen.flow_from_directory(

    dataset_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='training'
)


# ============================================
# 12. CREATE VALIDATION GENERATOR
# ============================================

val_generator = train_datagen.flow_from_directory(

    dataset_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='validation'
)


# ============================================
# 13. CHECK CLASS INDICES
# ============================================

print(train_generator.class_indices)


# ============================================
# 14. CHECK BATCH SHAPES
# ============================================

images, labels = next(train_generator)

print("Image Batch Shape:", images.shape)

print("Label Batch Shape:", labels.shape)


# ============================================
# 15. VISUALIZE AUGMENTED IMAGES
# ============================================

plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(images[i])

    plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================
# 16. CREATE TEST DATA GENERATOR
# ============================================

test_datagen = ImageDataGenerator(
    rescale=1./255
)


# ============================================
# 17. FINAL SUMMARY
# ============================================

print("\nPreprocessing Pipeline Completed Successfully!")

print(f"\nTotal Classes: {len(classes)}")

print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")

print(f"Batch Size: {BATCH_SIZE}")

print("\nTrain Generator Ready")

print("Validation Generator Ready")

print("Data Augmentation Applied")

print("Normalization Applied")
