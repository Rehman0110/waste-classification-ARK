# =============================================================
# model_builder.py (Clean Version)
# MobileNetV2 for Binary Classification
# =============================================================

from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = (224, 224, 3)


def build_mobilenet_model():

    # Load pretrained MobileNetV2 (without top layer)
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE,
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model
    base_model.trainable = False

    # Add custom classification head
    inputs = Input(shape=IMAGE_SIZE)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    # Binary output (2 classes)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


# ── Optional Fine-Tuning ─────────────────────────────────────

def unfreeze_for_fine_tuning(model, base_model, fine_tune_at=100):

    base_model.trainable = True

    # Freeze early layers, train deeper layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model