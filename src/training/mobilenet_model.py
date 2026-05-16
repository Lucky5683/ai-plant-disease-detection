from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)


def build_mobilenet_model(num_classes):

    # Load pretrained MobileNetV2

    base_model = MobileNetV2(

        weights='imagenet',

        include_top=False,

        input_shape=(224,224,3)
    )


    # Freeze pretrained layers

    base_model.trainable = False


    # Custom classification head

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        128,
        activation='relu'
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        num_classes,
        activation='softmax'
    )(x)


    model = Model(
        inputs=base_model.input,
        outputs=outputs
    )

    return model, base_model