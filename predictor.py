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

# ============================================================
# DISEASE INFORMATION DATABASE
# ============================================================
KNOWLEDGE_BASE = {}

kb_path = "knowledge_base/disease_json"

if os.path.exists(kb_path):

    for file_name in os.listdir(kb_path):

        if file_name.endswith(".json"):

            file_path = os.path.join(kb_path, file_name)

            try:

                with open(file_path, "r", encoding="utf-8") as f:

                    class_name = file_name.replace(".json", "")

                    KNOWLEDGE_BASE[class_name] = json.load(f)

            except Exception as e:

                print(f"Error loading {file_name}: {e}")



SEVERITY_COLORS = {
    "none": "#22c55e",
    "low": "#eab308",
    "medium": "#f97316",
    "high": "#ef4444",
}

SEVERITY_LABELS = {"none": "سليم", "low": "خفيف", "medium": "متوسط", "high": "خطير"}


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


    def get_disease_info(self, class_name):

      raw = KNOWLEDGE_BASE.get(class_name, {})

      return {
          "name_ar": raw.get("arabic_name", "غير معروف"),

          "name_en": raw.get("disease", class_name),

          "severity": raw.get("severity", "medium"),

          "cause": raw.get("cause", "غير معروف"),

          "symptoms_ar": "، ".join(raw.get("symptoms", [])),

          "treatment_ar": "، ".join(raw.get("treatments", [])),

          "prevention_ar": "، ".join(raw.get("prevention", [])),

          "local_pesticides": "، ".join(raw.get("local_pesticides", [])),

          "local_advice": "، ".join(raw.get("local_advice", [])),

          "risk_season": "، ".join(raw.get("palestine_risk_season", [])),

          "sources": "، ".join(raw.get("sources", [])),

          "emoji": raw.get("emoji", "🌿")
      }
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
            if confidence < 0.60:
              return {
                  "success": False,
                  "error": "Unable to confidently identify the disease",
                  "confidence": round(confidence * 100, 1)
              }
            class_idx = str(np.argmax(predictions))
            class_name = self.labels.get(class_idx, "Unknown")

            disease_info = self.get_disease_info(class_name)

            return {
              "success": True,
              "class_name": class_name,
              "confidence": round(confidence * 100, 1),
              **disease_info,
              "severity_color": SEVERITY_COLORS.get(
                  disease_info.get("severity", "medium"), "#f97316"
              ),
              "severity_label": SEVERITY_LABELS.get(
                  disease_info.get("severity", "medium"), "متوسط"
              ),
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
