# 🌿 Plant Doctor | طبيب النبات
## مشروع AIO 2026 - كشف أمراض النباتات بالذكاء الاصطناعي

---

## 📋 وصف المشروع
تطبيق ويب يستخدم الذكاء الاصطناعي (Computer Vision + Machine Learning) لتشخيص أمراض النباتات من الصور، مع شات بوت زراعي ذكي.

## 🛠️ التقنيات المستخدمة
- **Python** + **Flask** (الخادم)
- **TensorFlow / Keras** (نموذج الذكاء الاصطناعي)
- **MobileNetV2** (Transfer Learning)
- **Claude AI** (الشات بوت)
- **HTML/CSS/JavaScript** (الواجهة)

## 📁 هيكل المشروع
```
plant_doctor/
├── app.py              # الخادم الرئيسي
├── predictor.py        # نموذج التشخيص + قاعدة بيانات الأمراض
├── train_model.py      # كود تدريب النموذج
├── requirements.txt    # المكتبات المطلوبة
├── static/
│   └── index.html      # واجهة المستخدم
├── model/
│   ├── plant_model.h5  # النموذج المدرب (بعد التدريب)
│   └── labels.json     # أسماء الأمراض
└── data/
    └── plantvillage/   # ضع الداتا هنا
```

---

## 🚀 تشغيل المشروع

### الخطوة 1: تثبيت المكتبات
```bash
pip install -r requirements.txt
```

### الخطوة 2: تحميل الداتا (للتدريب)
1. اذهب إلى: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. حمّل الداتا وفك الضغط في مجلد `data/plantvillage/`

### الخطوة 3: تدريب النموذج
```bash
python train_model.py
```
⏱️ مدة التدريب: 30-60 دقيقة (حسب قوة الجهاز)

### الخطوة 4: تشغيل التطبيق
```bash
python app.py
```
ثم افتح المتصفح على: http://localhost:5000

---

## 🤖 وضع التجربة (بدون تدريب)
التطبيق يعمل بوضع تجريبي حتى لو لم تدرب النموذج بعد!
فقط شغّل `python app.py` وجرّب.

---

## 📊 Dataset
- **PlantVillage Dataset**: 54,305 صورة
- **38 مرض** في **14 نوع نبات**
- متاح مجاناً على Kaggle

## 🌱 الأمراض المكتشفة
- طماطم: اللفحة المبكرة، المتأخرة، عفن الأوراق، البقعة البكتيرية، حلم العنكبوت
- بطاطا: اللفحة المبكرة والمتأخرة
- فلفل: البقعة البكتيرية
- وأكثر...

---

## 💡 مميزات إضافية
- ✅ شات بوت زراعي ذكي (Claude AI)
- ✅ قاعدة بيانات أمراض باللغة العربية
- ✅ معلومات العلاج والوقاية
- ✅ واجهة عربية كاملة
- ✅ دعم السحب والإفلات للصور

---

## 👥 الفريق
- **المشروع**: AIO 2026 - STEAM Center
- **الموضوع**: Eco Innovation

---

## 📚 المراجع
1. Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health. arXiv:1511.08060
2. Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks. arXiv:1704.04861
3. PlantVillage Dataset - Penn State University
