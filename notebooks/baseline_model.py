# ============================================
# BASELINE CNN TRAINING
# ============================================

# IMPORT LIBRARIES
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

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (

    EarlyStopping,

    ModelCheckpoint
)



from src.training.cnn_model import build_cnn_model


# ============================================
# DATA PATHS
# ============================================

train_path = r"X:\AI-Plant-Disease-Detection\data\processed\train"

val_path = r"X:\AI-Plant-Disease-Detection\data\processed\val"

test_path = r"X:\AI-Plant-Disease-Detection\data\processed\test"


# ============================================
# PARAMETERS
# ============================================

IMG_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10


# ============================================
# DATA GENERATORS
# ============================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True,

    brightness_range=[0.8,1.2]
)

test_val_datagen = ImageDataGenerator(
    rescale=1./255
)


# ============================================
# TRAIN GENERATOR
# ============================================

train_generator = train_datagen.flow_from_directory(

    train_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical'
)


# ============================================
# VALIDATION GENERATOR
# ============================================

val_generator = test_val_datagen.flow_from_directory(

    val_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical'
)


# ============================================
# TEST GENERATOR
# ============================================

test_generator = test_val_datagen.flow_from_directory(

    test_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=False
)


# ============================================
# BUILD MODEL
# ============================================

model = build_cnn_model(

    train_generator.num_classes
)


# ============================================
# MODEL SUMMARY
# ============================================

model.summary()


# ============================================
# COMPILE MODEL
# ============================================

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)


# ============================================
# CALLBACKS
# ============================================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=3,

    restore_best_weights=True
)

checkpoint = ModelCheckpoint(

    r"X:\AI-Plant-Disease-Detection\models\baseline_cnn.h5",

    save_best_only=True
)


# ============================================
# TRAIN MODEL
# ============================================

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint]
)


# ============================================
# PLOT ACCURACY
# ============================================

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.show()


# ============================================
# PLOT LOSS
# ============================================

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train", "Validation"])

plt.show()


# ============================================
# EVALUATE MODEL
# ============================================

loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")


# ============================================
# SAVE FINAL MODEL
# ============================================

model.save(

    r"X:\AI-Plant-Disease-Detection\models\final_baseline_model.h5"
)

print("\nBaseline CNN Model Saved Successfully!")