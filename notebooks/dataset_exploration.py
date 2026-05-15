import os
import cv2
import random
import matplotlib.pyplot as plt
from collections import Counter

# Dataset path
DATASET_PATH = r"X:\AI-Plant-Disease-Detection\archive\raw\PlantVillage"

# Get class folders
classes = [folder for folder in os.listdir(DATASET_PATH)
           if os.path.isdir(os.path.join(DATASET_PATH, folder))]

print(f"\nTotal Classes: {len(classes)}\n")

# Count images per class
image_counts = {}

for cls in classes:
    class_path = os.path.join(DATASET_PATH, cls)

    images = [img for img in os.listdir(class_path)
              if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    image_counts[cls] = len(images)

# Print counts
print("Images Per Class:\n")

for cls, count in image_counts.items():
    print(f"{cls}: {count}")

# Check imbalance
print("\nDataset Balance Check:")
print(f"Maximum Images: {max(image_counts.values())}")
print(f"Minimum Images: {min(image_counts.values())}")

# Show sample images
plt.figure(figsize=(12, 8))

valid_extensions = ('.jpg', '.jpeg', '.png')

plot_index = 1

for cls in classes[:6]:

    class_path = os.path.join(DATASET_PATH, cls)

    images = [img for img in os.listdir(class_path)
              if img.lower().endswith(valid_extensions)]

    if len(images) == 0:
        continue

    img_path = os.path.join(class_path, random.choice(images))

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, plot_index)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

    plot_index += 1

plt.tight_layout()
plt.show()

# Check image sizes
print("\nChecking Image Sizes...\n")

sample_sizes = []

for cls in classes[:5]:

    class_path = os.path.join(DATASET_PATH, cls)

    images = os.listdir(class_path)

    img_path = os.path.join(class_path, images[0])

    img = cv2.imread(img_path)

    if img is not None:
        sample_sizes.append(img.shape)

print("Sample Image Sizes:")
for size in sample_sizes:
    print(size)