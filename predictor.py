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
DISEASE_INFO = {
    # =========================
    # APPLE DISEASES
    # =========================
    "Apple___Apple_scab": {
        "name_ar": "جرب التفاح",
        "name_en": "Apple Scab",
        "severity": "medium",
        "cause": "فطر Venturia inaequalis",
        "symptoms_ar": "بقع زيتونية داكنة على الأوراق والثمار، وقد تتشقق الثمار المصابة.",
        "treatment_ar": "استخدم مبيداً فطرياً مناسباً وأزل الأوراق المصابة من التربة.",
        "prevention_ar": "تحسين التهوية وتقليل الرطوبة وتقليم الأشجار بانتظام.",
        "emoji": "🟤",
    },
    "Apple___Black_rot": {
        "name_ar": "العفن الأسود للتفاح",
        "name_en": "Apple Black Rot",
        "severity": "high",
        "cause": "فطر Botryosphaeria obtusa",
        "symptoms_ar": "بقع سوداء وتعفن على الأوراق والثمار مع جفاف الأغصان.",
        "treatment_ar": "إزالة الأجزاء المصابة واستخدام مبيد فطري نحاسي.",
        "prevention_ar": "تنظيف الأشجار والتخلص من الثمار المتعفنة وتحسين التهوية.",
        "emoji": "⚫",
    },
    "Apple___Cedar_apple_rust": {
        "name_ar": "صدأ التفاح",
        "name_en": "Cedar Apple Rust",
        "severity": "medium",
        "cause": "فطر Gymnosporangium juniperi-virginianae",
        "symptoms_ar": "بقع صفراء أو برتقالية على الأوراق مع تشوهات خفيفة.",
        "treatment_ar": "رش مبيد فطري مناسب خلال بداية الموسم.",
        "prevention_ar": "إزالة النباتات العائلة القريبة وتحسين التهوية.",
        "emoji": "🟠",
    },
    "Apple___healthy": {
        "name_ar": "تفاح سليم ✅",
        "name_en": "Healthy Apple",
        "severity": "none",
        "cause": "لا يوجد مرض",
        "symptoms_ar": "النبات يبدو بصحة جيدة!",
        "treatment_ar": "لا حاجة لأي علاج.",
        "prevention_ar": "استمر في الري المنتظم والتسميد المتوازن.",
        "emoji": "💚",
    },
    # =========================
    # TOMATO DISEASES
    # =========================
    "Tomato___Early_blight": {
        "name_ar": "اللفحة المبكرة للطماطم",
        "name_en": "Tomato Early Blight",
        "severity": "medium",
        "cause": "فطر Alternaria solani",
        "symptoms_ar": "بقع بنية داكنة على الأوراق مع حلقات متحدة المركز تشبه هدف الرماية.",
        "treatment_ar": "رش مبيد فطري يحتوي على Chlorothalonil أو Mancozeb.",
        "prevention_ar": "تجنب الري الرأسي ودوّر المحاصيل.",
        "emoji": "🟤",
    },
    "Tomato___Late_blight": {
        "name_ar": "اللفحة المتأخرة للطماطم",
        "name_en": "Tomato Late Blight",
        "severity": "high",
        "cause": "فطر Phytophthora infestans",
        "symptoms_ar": "بقع مائية بنية تنتشر بسرعة على الأوراق.",
        "treatment_ar": "رش مبيد نحاسي أو Metalaxyl فوراً.",
        "prevention_ar": "تقليل الرطوبة وتحسين التهوية.",
        "emoji": "🔵",
    },
    "Tomato___Leaf_Mold": {
        "name_ar": "عفن أوراق الطماطم",
        "name_en": "Tomato Leaf Mold",
        "severity": "medium",
        "cause": "فطر Passalora fulva",
        "symptoms_ar": "بقع صفراء وطبقة زيتونية تحت الأوراق.",
        "treatment_ar": "تقليل الرطوبة واستخدام Chlorothalonil.",
        "prevention_ar": "توفير تهوية جيدة بين النباتات.",
        "emoji": "🟢",
    },
    "Tomato___Bacterial_spot": {
        "name_ar": "البقعة البكتيرية للطماطم",
        "name_en": "Tomato Bacterial Spot",
        "severity": "high",
        "cause": "بكتيريا Xanthomonas",
        "symptoms_ar": "بقع صغيرة داكنة مع هالة صفراء.",
        "treatment_ar": "استخدام مركبات النحاس الزراعية.",
        "prevention_ar": "تعقيم الأدوات والبذور.",
        "emoji": "🔴",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name_ar": "العنكبوت الأحمر",
        "name_en": "Tomato Spider Mites",
        "severity": "medium",
        "cause": "حشرة Tetranychus urticae",
        "symptoms_ar": "اصفرار الأوراق مع خيوط عنكبوتية دقيقة.",
        "treatment_ar": "رش بزيت النيم أو Abamectin.",
        "prevention_ar": "الحفاظ على رطوبة مناسبة.",
        "emoji": "🕷️",
    },
    "Tomato___healthy": {
        "name_ar": "طماطم سليمة ✅",
        "name_en": "Healthy Tomato",
        "severity": "none",
        "cause": "لا يوجد مرض",
        "symptoms_ar": "النبات يبدو بصحة جيدة!",
        "treatment_ar": "لا حاجة لأي علاج.",
        "prevention_ar": "استمر في العناية الجيدة بالنبات.",
        "emoji": "💚",
    },
    "Tomato___Target_Spot": {
        "name_ar": "بقعة الهدف للطماطم",
        "name_en": "Tomato Target Spot",
        "severity": "medium",
        "cause": "فطر Corynespora cassiicola",
        "symptoms_ar": "بقع دائرية بنية مع حلقات متداخلة على الأوراق.",
        "treatment_ar": "استخدام مبيد فطري مناسب وتقليل الرطوبة.",
        "prevention_ar": "تحسين التهوية وتجنب الري الزائد.",
        "emoji": "🎯",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name_ar": "فيروس تجعد واصفرار أوراق الطماطم",
        "name_en": "Tomato Yellow Leaf Curl Virus",
        "severity": "high",
        "cause": "فيروس ينتقل بواسطة الذبابة البيضاء",
        "symptoms_ar": "اصفرار وتجعد شديد للأوراق مع ضعف النمو.",
        "treatment_ar": "لا يوجد علاج مباشر، يجب إزالة النباتات المصابة.",
        "prevention_ar": "مكافحة الذبابة البيضاء واستخدام شتلات سليمة.",
        "emoji": "🟡",
    },
    "Tomato___Tomato_mosaic_virus": {
        "name_ar": "فيروس موزاييك الطماطم",
        "name_en": "Tomato Mosaic Virus",
        "severity": "high",
        "cause": "فيروس Tomato Mosaic Virus",
        "symptoms_ar": "تغير لون الأوراق إلى نمط موزاييك مع تشوهات بالنمو.",
        "treatment_ar": "إزالة النباتات المصابة وتعقيم الأدوات الزراعية.",
        "prevention_ar": "استخدام بذور سليمة وتجنب لمس النباتات المصابة.",
        "emoji": "🦠",
    },
    "Tomato___Septoria_leaf_spot": {
        "name_ar": "تبقع أوراق الطماطم السبتوري",
        "name_en": "Tomato Septoria Leaf Spot",
        "severity": "medium",
        "cause": "فطر Septoria lycopersici",
        "symptoms_ar": "بقع صغيرة رمادية مع حواف داكنة على الأوراق.",
        "treatment_ar": "استخدام مبيد فطري وإزالة الأوراق المصابة.",
        "prevention_ar": "تجنب الري على الأوراق وتحسين التهوية.",
        "emoji": "🍂",
    },
    # =========================
    # POTATO DISEASES
    # =========================
    "Potato___Early_blight": {
        "name_ar": "اللفحة المبكرة للبطاطا",
        "name_en": "Potato Early Blight",
        "severity": "medium",
        "cause": "فطر Alternaria solani",
        "symptoms_ar": "بقع داكنة على الأوراق القديمة.",
        "treatment_ar": "رش Mancozeb أو Chlorothalonil.",
        "prevention_ar": "إزالة بقايا النباتات المصابة.",
        "emoji": "🟤",
    },
    "Potato___Late_blight": {
        "name_ar": "اللفحة المتأخرة للبطاطا",
        "name_en": "Potato Late Blight",
        "severity": "high",
        "cause": "فطر Phytophthora infestans",
        "symptoms_ar": "بقع مائية سريعة الانتشار على الأوراق.",
        "treatment_ar": "رش مبيد نحاسي أو Metalaxyl.",
        "prevention_ar": "تقليل الرطوبة واستخدام أصناف مقاومة.",
        "emoji": "🔵",
    },
    "Potato___healthy": {
        "name_ar": "بطاطا سليمة ✅",
        "name_en": "Healthy Potato",
        "severity": "none",
        "cause": "لا يوجد مرض",
        "symptoms_ar": "النبات يبدو بصحة جيدة!",
        "treatment_ar": "لا حاجة لأي علاج.",
        "prevention_ar": "استمر في العناية المنتظمة.",
        "emoji": "💚",
    },
    # =========================
    # PEPPER DISEASES
    # =========================
    "Pepper,_bell___Bacterial_spot": {
        "name_ar": "البقعة البكتيرية للفلفل",
        "name_en": "Pepper Bacterial Spot",
        "severity": "high",
        "cause": "بكتيريا Xanthomonas campestris",
        "symptoms_ar": "بقع مائية صغيرة تتحول إلى بنية.",
        "treatment_ar": "رش بالنحاس الزراعي مع Mancozeb.",
        "prevention_ar": "استخدام بذور معقمة وتقليل الرطوبة.",
        "emoji": "🔴",
    },
    "Pepper,_bell___healthy": {
        "name_ar": "فلفل سليم ✅",
        "name_en": "Healthy Pepper",
        "severity": "none",
        "cause": "لا يوجد مرض",
        "symptoms_ar": "النبات يبدو بصحة جيدة!",
        "treatment_ar": "لا حاجة لأي علاج.",
        "prevention_ar": "استمر في العناية المنتظمة.",
        "emoji": "💚",
    },
    "Grape___Black_rot": {
        "name_ar": "العفن الأسود للعنب",
        "name_en": "Grape Black Rot",
        "severity": "high",
        "cause": "فطر Guignardia bidwellii",
        "symptoms_ar": "بقع بنية دائرية على الأوراق وتعفن داكن على الثمار.",
        "treatment_ar": "استخدام مبيد فطري مناسب وإزالة الأجزاء المصابة.",
        "prevention_ar": "تحسين التهوية والتقليم المنتظم.",
        "emoji": "⚫",
    },
    # =========================
    # GRAPE DISEASES
    # =========================
    "Grape___Esca_(Black_Measles)": {
        "name_ar": "إيسكا العنب",
        "name_en": "Grape Esca (Black Measles)",
        "severity": "high",
        "cause": "عدوى فطرية تصيب خشب العنب",
        "symptoms_ar": "بقع صفراء وبنية على الأوراق مع جفاف تدريجي للأغصان.",
        "treatment_ar": "إزالة الأجزاء المصابة وتقليل الإجهاد على النبات.",
        "prevention_ar": "تعقيم أدوات التقليم وتحسين إدارة الري.",
        "emoji": "🟤",
    },
    "Grape___healthy": {
        "name_ar": "عنب سليم ✅",
        "name_en": "Healthy Grape",
        "severity": "none",
        "cause": "لا يوجد مرض",
        "symptoms_ar": "النبات يبدو بصحة جيدة!",
        "treatment_ar": "لا حاجة لأي علاج.",
        "prevention_ar": "الاستمرار بالعناية الجيدة والتقليم المنتظم.",
        "emoji": "💚",
    },
}
SEVERITY_COLORS = {
    "none": "#22c55e",
    "low": "#eab308",
    "medium": "#f97316",
    "high": "#ef4444",
}

SEVERITY_LABELS = {"none": "سليم", "low": "خفيف", "medium": "متوسط", "high": "خطير"}


class PlantDiseasePredictor:
    def __init__(
        self, model_path="model/plant_model.h5", labels_path="model/labels.json"
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
            print("⚠️  TensorFlow not installed. Using demo mode.")

    def preprocess_image(self, image_data):
        """Convert base64 image to numpy array for model"""
        # Decode base64
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def get_disease_info(self, class_name):

      # 1. Try online API first
      try:
          import requests

          response = requests.get(
              f"http://localhost:8000/disease/{class_name}",
              timeout=2
          )

          if response.status_code == 200:
              return response.json()

      except:
          print("⚠️ Offline mode activated")

      # 2. Fallback to local database
      return DISEASE_INFO.get(class_name, {
          "name_ar": "غير معروف",
          "name_en": class_name,
          "severity": "medium",
          "cause": "غير معروف",
          "symptoms_ar": "لا توجد معلومات متوفرة.",
          "treatment_ar": "يرجى استشارة مختص زراعي.",
          "prevention_ar": "حافظ على العناية الجيدة بالنبات.",
          "emoji": "❓"
      })
    
    def predict(self, image_data):
        """
        Predict plant disease from base64 image.
        Returns disease info dict.
        """
        if self.model is None:
            # DEMO MODE: return sample result for testing
            return self._demo_predict()

        try:
            img = self.preprocess_image(image_data)
            predictions = self.model.predict(img, verbose=0)
            confidence = float(np.max(predictions))
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

    def _demo_predict(self):
        """Demo prediction for testing without real model"""
        import random

        diseases = list(DISEASE_INFO.keys())
        chosen = random.choice(diseases)
        info = DISEASE_INFO[chosen]
        return {
            "success": True,
            "class_name": chosen,
            "confidence": round(random.uniform(75, 98), 1),
            **info,
            "severity_color": SEVERITY_COLORS.get(
                info.get("severity", "medium"), "#f97316"
            ),
            "severity_label": SEVERITY_LABELS.get(
                info.get("severity", "medium"), "متوسط"
            ),
            "demo_mode": True,
        }


# Standalone test
if __name__ == "__main__":
    predictor = PlantDiseasePredictor()
    result = predictor._demo_predict()
    print("\n🌿 Test Prediction:")
    print(f"Disease: {result['name_ar']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Severity: {result['severity_label']}")
    print(f"Treatment: {result['treatment_ar']}")
