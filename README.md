# 🥗 AI Diet Analyzer – Your Smart Virtual Dietician

The **AI Diet Analyzer** is an intelligent system that analyzes a **photo of your meal** and provides an **instant nutritional breakdown** — including calories, macronutrients, and essential vitamins & minerals.  
It also gives **personalized health warnings and dietary advice**, acting like your **virtual dietician**.

---

## 💡 Overview

In today’s fast-paced lifestyle, keeping track of what we eat is tough.  
This project combines **Computer Vision** and **Nutrition Science** to help users **make healthier food choices** effortlessly.

Simply **take a picture of your meal**, and the system:
1. Detects food items in the image.
2. Estimates portion sizes and nutrients.
3. Warns about unhealthy components (e.g., high sugar, excess oil, processed food).
4. Suggests diet improvements for a balanced lifestyle.

---

## 🚀 Features
- 🍛 **Food Detection:** Identifies multiple food items from a single image.
- 🔬 **Nutrient Estimation:** Calculates calories, protein, carbs, fats, vitamins, and minerals.
- ⚕️ **Health Warnings:** Alerts if the meal is high in sugar, fat, or sodium.
- 🧠 **AI Dietician Mode:** Offers dietary suggestions based on health goals (e.g., weight loss, muscle gain).
- 📱 **User-Friendly Interface:** Easy-to-use frontend for image upload and results display.

---

## 🧩 System Architecture
1. **Image Input:** User uploads a food image.  
2. **Food Recognition Model:** Uses CNN-based model (e.g., ResNet / EfficientNet) to detect and classify food items.  
3. **Portion & Nutrient Estimator:** Matches detected food with nutritional databases (e.g., USDA / FDC).  
4. **Diet Analysis Module:** Evaluates total nutrient content and checks dietary balance.  
5. **Recommendation Engine:** Generates personalized health feedback and warnings.

---

## 🧠 Tech Stack
- **Frontend:** React / Streamlit  
- **Backend:** Flask / FastAPI  
- **ML Models:** CNN (ResNet / MobileNet) for food recognition  
- **Database:** USDA FoodData Central API for nutrient data  
- **Languages:** Python, JavaScript  
- **Libraries:** TensorFlow / PyTorch, OpenCV, NumPy, Pandas

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/priyanshugautam1234/ai-diet-analyzer.git
cd ai-diet-analyzer

# Install dependencies
pip install -r requirements.txt
