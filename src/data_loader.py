# =============================================================
# utils/data_loader.py (Final Clean Version)
# =============================================================

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array


# ── Constants ────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ── Data Generators ──────────────────────────────────────────

def get_data_generators(data_dir: str, val_split: float = 0.2):

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=val_split
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split
    )

    # ── Train Generator ──────────────────────────────────────
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",      # ✅ 2 classes
        subset="training",
        shuffle=True,
        seed=42
    )

    # ── Validation Generator ─────────────────────────────────
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",      # ✅ must match train
        subset="validation",
        shuffle=False,            # ✅ IMPORTANT
        seed=42
    )

    # ── Dynamic Class Info ───────────────────────────────────
    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    print(f"\n✅ Dataset loaded from: {data_dir}")
    print(f"   Classes       : {train_gen.class_indices}")
    print(f"   Num Classes   : {num_classes}")
    print(f"   Train Samples : {train_gen.samples}")
    print(f"   Val Samples   : {val_gen.samples}\n")

    return train_gen, val_gen, class_names


# ── Single Image Preprocessing ───────────────────────────────

def preprocess_image(image_path: str):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_pil_image(pil_image):
    img = pil_image.resize(IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)