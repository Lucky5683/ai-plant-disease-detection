# ============================================
# CNN MODEL ARCHITECTURE
# File: cnn_model.py
# ============================================

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    Conv2D,

    MaxPooling2D,

    Flatten,

    Dense,

    Dropout
)


def build_cnn_model(num_classes):

    model = Sequential([

        # ====================================
        # FIRST CONVOLUTION BLOCK
        # ====================================

        Conv2D(

            32,

            (3,3),

            activation='relu',

            input_shape=(224,224,3)
        ),

        MaxPooling2D(2,2),


        # ====================================
        # SECOND CONVOLUTION BLOCK
        # ====================================

        Conv2D(

            64,

            (3,3),

            activation='relu'
        ),

        MaxPooling2D(2,2),


        # ====================================
        # THIRD CONVOLUTION BLOCK
        # ====================================

        Conv2D(

            128,

            (3,3),

            activation='relu'
        ),

        MaxPooling2D(2,2),


        # ====================================
        # FLATTEN
        # ====================================

        Flatten(),


        # ====================================
        # DENSE LAYERS
        # ====================================

        Dense(

            128,

            activation='relu'
        ),

        Dropout(0.5),


        # ====================================
        # OUTPUT LAYER
        # ====================================

        Dense(

            num_classes,

            activation='softmax'
        )
    ])

    return model