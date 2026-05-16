# ============================================
# DATASET SPLITTING SCRIPT
# ============================================

# Import Libraries
import os
import shutil
import random

from sklearn.model_selection import train_test_split


# ============================================
# DEFINE PATHS
# ============================================

dataset_path = r"X:\AI-Plant-Disease-Detection\archive\raw\PlantVillage\PlantVillage"

output_path = r"X:\AI-Plant-Disease-Detection\data\processed"


# ============================================
# CREATE TRAIN / VAL / TEST FOLDERS
# ============================================

splits = ['train', 'val', 'test']

for split in splits:

    split_path = os.path.join(output_path, split)

    os.makedirs(split_path, exist_ok=True)

print("Train / Validation / Test folders created successfully!")


# ============================================
# GET CLASSES
# ============================================

classes = os.listdir(dataset_path)

print(f"Total Classes Found: {len(classes)}")


# ============================================
# SPLIT DATASET
# ============================================

for cls in classes:

    class_path = os.path.join(dataset_path, cls)

    images = os.listdir(class_path)

    random.shuffle(images)


    # 70% Train
    # 30% Temp

    train_imgs, temp_imgs = train_test_split(

        images,

        test_size=0.30,

        random_state=42
    )


    # Remaining 30%
    # Split into 15% Validation
    # and 15% Test

    val_imgs, test_imgs = train_test_split(

        temp_imgs,

        test_size=0.50,

        random_state=42
    )


    split_data = {

        'train': train_imgs,

        'val': val_imgs,

        'test': test_imgs
    }


    # ============================================
    # COPY IMAGES TO NEW FOLDERS
    # ============================================

    for split_name, split_images in split_data.items():

        split_class_path = os.path.join(

            output_path,

            split_name,

            cls
        )

        os.makedirs(split_class_path, exist_ok=True)


        for img_name in split_images:

            src_path = os.path.join(class_path, img_name)

            dst_path = os.path.join(split_class_path, img_name)

            shutil.copy(src_path, dst_path)


    print(f"{cls} split completed!")


# ============================================
# FINAL MESSAGE
# ============================================

print("\nDataset splitting completed successfully!")

print("\nDataset Structure:")

print("data/processed/train")
print("data/processed/val")
print("data/processed/test")