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

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from src.evaluation.metrics import (

    evaluate_model,

    print_classification_report,

    plot_confusion_matrix
)


# ============================================
# LOAD MODEL
# ============================================

model = load_model(

    r"X:\AI-Plant-Disease-Detection\models\transfer_learning\final_mobilenet_model.h5"
)


# ============================================
# DATA PATH
# ============================================

test_path = r"X:\AI-Plant-Disease-Detection\data\processed\test"


# ============================================
# TEST GENERATOR
# ============================================

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(

    test_path,

    target_size=(224,224),

    batch_size=32,

    class_mode='categorical',

    shuffle=False
)


# ============================================
# PREDICTIONS
# ============================================

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)

y_true = test_generator.classes

class_names = list(
    test_generator.class_indices.keys()
)


# ============================================
# EVALUATION METRICS
# ============================================

evaluate_model(y_true, y_pred)


# ============================================
# CLASSIFICATION REPORT
# ============================================

print_classification_report(

    y_true,

    y_pred,

    class_names
)


# ============================================
# CONFUSION MATRIX
# ============================================

plot_confusion_matrix(

    y_true,

    y_pred,

    class_names
)


# ============================================
# ROC CURVE
# ============================================

y_true_bin = label_binarize(

    y_true,

    classes=range(len(class_names))
)


plt.figure(figsize=(10,8))

for i in range(len(class_names)):

    fpr, tpr, _ = roc_curve(

        y_true_bin[:, i],

        predictions[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(

        fpr,

        tpr,

        label=f"{class_names[i]} AUC={roc_auc:.2f}"
    )


plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curves")

plt.legend(loc="lower right", fontsize=8)

plt.show()


# ============================================
# MISCLASSIFIED IMAGES
# ============================================

misclassified = np.where(y_true != y_pred)[0]

print(f"\nTotal Misclassified Images: {len(misclassified)}")


# Load all test images

full_test_generator = test_datagen.flow_from_directory(

    test_path,

    target_size=(224,224),

    batch_size=len(y_true),

    class_mode='categorical',

    shuffle=False
)

test_images, test_labels = next(full_test_generator)


# Display misclassified images

if len(misclassified) > 0:

    plt.figure(figsize=(15,10))

    for i, idx in enumerate(misclassified[:9]):

        plt.subplot(3,3,i+1)

        plt.imshow(test_images[idx])

        plt.title(
            f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()