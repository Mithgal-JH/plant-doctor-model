"""
Plant Disease Predictor
=======================
Loads trained model and predicts disease from image.
Also contains disease information database (Arabic + English).
"""

import numpy as np
import json
import os
from PIL import Image
import io
import base64

class PlantDiseasePredictor:
    def __init__(
        self, model_path="model/plant_model.keras", labels_path="model/labels.json"
    ):
        self.model = None
        self.labels = None
        self.model_path = model_path
        self.labels_path = labels_path
        self._load_model()

    def _load_model(self):
        """Load model if it exists"""
        try:
            import tensorflow as tf

            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                with open(self.labels_path, "r") as f:
                    self.labels = json.load(f)
                print("✅ Model loaded successfully!")
            else:
                print("⚠️  Model not found. Run train_model.py first.")
        except ImportError:
            print("TensorFlow not installed.")

    def preprocess_image(self, image_data):
        """Convert base64 image to numpy array for model"""
        # Decode base64
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img).astype("float32")
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)

      
      
    def predict(self, image_data):
        """
        Predict plant disease from base64 image.
        Returns disease info dict.
        """
        if self.model is None:
          return {
              "success": False,
              "error": "Model not loaded"
          }

        try:
            img = self.preprocess_image(image_data)
            predictions = self.model.predict(img, verbose=0)
            confidence = float(np.max(predictions))
            if confidence < 0.40:
              return {
                  "success": False,
                  "error": "Unable to confidently identify the disease",
                  "confidence": round(confidence * 100, 1)
              }
            class_idx = str(np.argmax(predictions))
            class_name = self.labels.get(class_idx, "Unknown")

            

            return {
              "success": True,
              "class_name": class_name,
              "confidence": round(confidence * 100, 1),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Standalone test
if __name__ == "__main__":

    predictor = PlantDiseasePredictor()

    if predictor.model is not None:
        print("Model loaded successfully")
    else:
        print("Model not loaded")
