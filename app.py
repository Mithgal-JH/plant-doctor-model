"""
Plant Doctor - Main Flask Application
======================================
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify
import json
import os
import sys
import traceback
import requests

sys.path.insert(0, os.path.dirname(__file__))

from predictor import PlantDiseasePredictor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")


app = Flask(__name__, static_folder="static")

app.config["JSON_AS_ASCII"] = False

# Initialize predictor
predictor = PlantDiseasePredictor()

# =========================
# Load Local Knowledge Base
# =========================

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

print(f"Loaded {len(KNOWLEDGE_BASE)} local knowledge files")


# =========================
# Routes
# =========================

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        data = request.json
        image_data = data.get("image", "")

        if not image_data:

            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400

        result = predictor.predict(image_data)

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.json

        messages = data.get("messages", [])
        disease_context = data.get("disease_context", {})

        user_message = (
            messages[-1].get("content", "")
            if messages else ""
        )

        # Current disease class
        class_name = disease_context.get("class_name", "")

        # Load local KB
        local_knowledge = KNOWLEDGE_BASE.get(class_name, {})

        # Build AI system prompt
        prompt = f"""
                    أنت مساعد زراعي ذكي اسمه "طبيب النبات".

                    مهمتك:
                    - مساعدة المزارعين
                    - شرح الأمراض النباتية
                    - إعطاء نصائح علاج ووقاية
                    - الاعتماد على المعلومات المحلية
                    - الإجابة بالعربية فقط

                    معلومات التشخيص الحالية:
                    {json.dumps(disease_context, ensure_ascii=False, indent=2)}

                    المعلومات الزراعية المحلية:
                    {json.dumps(local_knowledge, ensure_ascii=False, indent=2)}

                    قواعد:
                    - أجب بالعربية فقط
                    - لا تتجاوز 120 كلمة
                    - كن عملياً ومختصراً
                    - لا تخترع أسماء مبيدات غير موجودة
                    - اعتمد على المعلومات المحلية المعطاة
                    - إذا لم توجد معلومة، قل ذلك بوضوح
                    - إذا كان النبات سليماً، طمّن المستخدم
                    """

        # Build real multi-turn conversation
        conversation = [
            {
                "role": "system",
                "content": prompt
            }
        ]


        # Keep last 6 messages
        for msg in messages[-6:]:

            role = msg.get("role", "user")

            if role not in ["user", "assistant"]:
                role = "user"

            conversation.append({
                "role": role,
                "content": msg.get("content", "")
            })

        # Send request to OpenRouter
        response = requests.post(

            url="https://openrouter.ai/api/v1/chat/completions",

            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },

            json={
                "model": "deepseek/deepseek-chat",
                "messages": conversation
            },

            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        if app.debug:
            print("OpenRouter response received")

        # Handle API errors safely
        if "choices" not in result:

            return jsonify({
                "success": False,
                "error": result
            })

        reply = result["choices"][0]["message"]["content"]

        return jsonify({
            "success": True,
            "reply": reply
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "success": True,
            "reply": "تعذر الاتصال بالمساعد الذكي حالياً. حاول مرة أخرى بعد قليل."
        })


@app.route("/diseases")
def get_diseases():
    return jsonify(KNOWLEDGE_BASE)


# =========================
# Test Knowledge Base Route
# =========================

@app.route("/kb/<disease>")
def test_kb(disease):

    return jsonify(
        KNOWLEDGE_BASE.get(disease, {})
    )


# =========================
# Run App
# =========================

if __name__ == "__main__":

    print("Plant Doctor - http://localhost:5000")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )