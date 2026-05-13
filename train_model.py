"""
Plant Disease Detection - Model Training Script
================================================
Dataset: Custom cleaned PlantVillage dataset

Run:
python train_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import matplotlib.pyplot as plt
import numpy as np
import random
import json
import os

# ============================================================
# CPU PERFORMANCE OPTIMIZATION
# ============================================================

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224

BATCH_SIZE = 32

INITIAL_EPOCHS = 10

FINE_TUNE_EPOCHS = 5

DATA_DIR = "data/custom_dataset"

MODEL_SAVE_PATH = "model/plant_model.keras"

LABELS_SAVE_PATH = "model/labels.json"

# ============================================================
# VALIDATE DATASET
# ============================================================

if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

# ============================================================
# DATA PREPARATION
# ============================================================

print("Loading dataset...")

# Balanced augmentation
train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=10,

    width_shift_range=0.05,

    height_shift_range=0.05,

    zoom_range=0.05,

    horizontal_flip=True,

    validation_split=0.2
)

val_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(

    DATA_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True,

    seed=SEED
)

val_generator = val_datagen.flow_from_directory(

    DATA_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False,

    seed=SEED
)

# ============================================================
# SAVE LABELS
# ============================================================

class_labels = {

    str(v): k

    for k, v in sorted(
        train_generator.class_indices.items(),
        key=lambda x: x[1]
    )
}

with open(LABELS_SAVE_PATH, "w", encoding="utf-8") as f:

    json.dump(
        class_labels,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Found {train_generator.samples} training images")
print(f"Found {val_generator.samples} validation images")
print(f"Classes: {len(class_labels)}")

# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(

    input_shape=(IMG_SIZE, IMG_SIZE, 3),

    include_top=False,

    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.2),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.1),

    layers.Dense(
        len(class_labels),
        activation="softmax"
    )
])

# ============================================================
# COMPILE INITIAL MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=4,

        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-6,

        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_SAVE_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    )
]

# ============================================================
# INITIAL TRAINING
# ============================================================

print("\nStarting initial training...")

history = model.fit(

    train_generator,

    epochs=INITIAL_EPOCHS,

    validation_data=val_generator,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# FINE-TUNING
# ============================================================

print("\nStarting fine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-30]:

    layer.trainable = False

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

history_fine = model.fit(

    train_generator,

    epochs=FINE_TUNE_EPOCHS,

    validation_data=val_generator,

    callbacks=callbacks,

    verbose=1,

)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_SAVE_PATH)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# ============================================================
# COMBINE HISTORY
# ============================================================

train_acc = (
    history.history["accuracy"] +
    history_fine.history["accuracy"]
)

val_acc = (
    history.history["val_accuracy"] +
    history_fine.history["val_accuracy"]
)

train_loss = (
    history.history["loss"] +
    history_fine.history["loss"]
)

val_loss = (
    history.history["val_loss"] +
    history_fine.history["val_loss"]
)

# ============================================================
# PLOT HISTORY
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

# Accuracy

ax1.plot(train_acc, label="Train Accuracy")
ax1.plot(val_acc, label="Validation Accuracy")

ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

# Loss

ax2.plot(train_loss, label="Train Loss")
ax2.plot(val_loss, label="Validation Loss")

ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()

plt.savefig(

    "model/training_history.png",

    dpi=150,

    bbox_inches="tight"
)

print("Training plot saved to: model/training_history.png")

# ============================================================
# FINAL EVALUATION
# ============================================================

val_loss, val_acc = model.evaluate(
    val_generator,
    verbose=0
)

print("\n==============================")
print("FINAL MODEL EVALUATION")
print("==============================")

print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

print("\nTraining completed successfully.")