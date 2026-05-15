"""
Plant Disease Detection - Model Training Script
================================================
Dataset: PlantVillage (available on Kaggle)
Link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Instructions:
1. Download dataset from Kaggle
2. Extract to: plant_doctor/data/plantvillage/
3. Run: python train_model.py
4. Model will be saved to: plant_doctor/model/plant_model.keras
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "data/custom_dataset"


MODEL_SAVE_PATH = "model/plant_model.keras"
LABELS_SAVE_PATH = "model/labels.json"

# ============================================================
# DATA PREPARATION
# ============================================================
print("📂 Loading dataset...")

# Data augmentation for training (helps model generalize)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # 80% train, 20% validation
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
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

# Freeze BatchNorm layers
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

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

# Create reports directory
reports_dir = "model/reports"
os.makedirs(reports_dir, exist_ok=True)

# ── Training History Graph ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(reports_dir, 'training_history.png'), dpi=100, bbox_inches='tight')
print(f"✅ Training history plot saved to: {reports_dir}/training_history.png")
plt.close()

# ── Confusion Matrix ──
print("\n📊 Generating confusion matrix...")
val_predictions = model.predict(val_generator, verbose=0)
val_true = val_generator.classes
val_pred = np.argmax(val_predictions, axis=1)

cm = confusion_matrix(val_true, val_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(class_labels.keys()), 
            yticklabels=list(class_labels.keys()),
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Validation Data')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
print(f"✅ Confusion matrix saved to: {reports_dir}/confusion_matrix.png")
plt.close()

# ── Classification Report ──
print("📋 Generating classification report...")
report = classification_report(val_true, val_pred, target_names=list(class_labels.keys()), digits=4)
with open(os.path.join(reports_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✅ Classification report saved to: {reports_dir}/classification_report.txt")

# Final evaluation
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\n📊 Final Validation Accuracy: {val_acc*100:.2f}%")
print(f"📊 Final Validation Loss: {val_loss:.4f}")
