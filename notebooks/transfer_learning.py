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
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam

from src.training.mobilenet_model import build_mobilenet_model


# ============================================
# PATHS
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

LEARNING_RATE = 0.001


# ============================================
# DATA AUGMENTATION
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
# DATA LOADERS
# ============================================

train_generator = train_datagen.flow_from_directory(

    train_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical'
)

val_generator = test_val_datagen.flow_from_directory(

    val_path,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical'
)

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

model, base_model = build_mobilenet_model(

    train_generator.num_classes
)


# ============================================
# COMPILE MODEL
# ============================================

model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

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

    r"X:\AI-Plant-Disease-Detection\models\transfer_learning\mobilenet_best.h5",

    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.2,

    patience=2,

    verbose=1
)


# ============================================
# INITIAL TRAINING
# ============================================

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint, reduce_lr]
)


# ============================================
# FINE-TUNING
# ============================================

base_model.trainable = True


# Freeze earlier layers

for layer in base_model.layers[:-20]:

    layer.trainable = False


# Recompile

model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)


# ============================================
# FINE-TUNE TRAINING
# ============================================

fine_tune_history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=5
)


# ============================================
# EVALUATE MODEL
# ============================================

loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")


# ============================================
# PLOT ACCURACY
# ============================================

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("MobileNetV2 Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.show()


# ============================================
# PLOT LOSS
# ============================================

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("MobileNetV2 Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train", "Validation"])

plt.show()


# ============================================
# SAVE FINAL MODEL
# ============================================

model.save(

    r"X:\AI-Plant-Disease-Detection\models\transfer_learning\final_mobilenet_model.h5"
)

print("\nTransfer Learning Model Saved Successfully!")