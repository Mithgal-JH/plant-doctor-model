"""
Plant Disease Detection - Model Training Script
================================================
Dataset: PlantVillage (available on Kaggle)
Link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Instructions:
1. Download dataset from Kaggle
2. Extract to: plant_doctor/data/plantvillage/
3. Run: python train_model.py
4. Model will be saved to: plant_doctor/model/plant_model.h5
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "data/custom_dataset"


MODEL_SAVE_PATH = "model/plant_model.h5"
LABELS_SAVE_PATH = "model/labels.json"

# ============================================================
# DATA PREPARATION
# ============================================================
print("📂 Loading dataset...")

# Data augmentation for training (helps model generalize)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # 80% train, 20% validation
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open(LABELS_SAVE_PATH, 'w') as f:
    json.dump(class_labels, f, ensure_ascii=False, indent=2)

print(f"✅ Found {train_generator.samples} training images")
print(f"✅ Found {val_generator.samples} validation images")
print(f"✅ Classes: {len(class_labels)}")

# ============================================================
# MODEL ARCHITECTURE (Transfer Learning with MobileNetV2)
# ============================================================
print("\n🧠 Building model...")

# Use pretrained MobileNetV2 as base (trained on ImageNet)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Remove original classifier
    weights='imagenet'
)

# Freeze base model layers (we only train our custom layers)
base_model.trainable = False

# Build complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(len(class_labels), activation='softmax')  # Output layer
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# TRAINING
# ============================================================
print("\n🚀 Starting training...")

callbacks = [
    # Stop early if no improvement
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    # Reduce learning rate when stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    ),
    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# FINE-TUNING (Unfreeze last layers for better accuracy)
# ============================================================
print("\n🔧 Fine-tuning...")

# Unfreeze last 20 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# SAVE & EVALUATE
# ============================================================
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.savefig('model/training_history.png', dpi=150, bbox_inches='tight')
print("✅ Training plot saved to: model/training_history.png")

# Final evaluation
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\n📊 Final Validation Accuracy: {val_acc*100:.2f}%")
print(f"📊 Final Validation Loss: {val_loss:.4f}")
