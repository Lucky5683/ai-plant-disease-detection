from sklearn.metrics import (

    classification_report,

    confusion_matrix,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score
)

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np


# ============================================
# PRINT METRICS
# ============================================

def evaluate_model(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_true,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average='weighted'
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall: {recall:.4f}")

    print(f"F1-Score: {f1:.4f}")


# ============================================
# CLASSIFICATION REPORT
# ============================================

def print_classification_report(y_true, y_pred, class_names):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print("\nClassification Report:\n")

    print(report)


# ============================================
# CONFUSION MATRIX
# ============================================

def plot_confusion_matrix(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(15,12))

    sns.heatmap(

        cm,

        annot=False,

        cmap='Blues',

        xticklabels=class_names,

        yticklabels=class_names
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.xticks(rotation=90)

    plt.yticks(rotation=0)

    plt.show()