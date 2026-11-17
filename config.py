"""
Global Configuration for AI Diet Analyzer
Defines API endpoints, deterministic guardrail thresholds, portion estimation constants,
and food density reference mappings.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# USDA FoodData Central allows 'DEMO_KEY' for testing (up to 30 requests/hour, 50/day)
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")
USDA_API_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# SQLite Cache Path
SQLITE_CACHE_PATH = CACHE_DIR / "nutrition_cache.db"

# Gemini & Vision Model Defaults
# Note: 'gemini-pro-vision' is legacy/deprecated; default to 'gemini-1.5-flash' with fallbacks
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
FALLBACK_GEMINI_MODELS = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]

# -------------------------------------------------------------
# 4. Deterministic Threshold Guardrails (Python-computed limits)
# -------------------------------------------------------------
GUARDRAILS = {
    "sodium_mg": {
        "warning_threshold": 800.0,
        "critical_threshold": 1200.0,
        "label": "High Sodium (Salt)",
        "unit": "mg",
        "guideline": "AHA / WHO recommended single-meal budget is <800mg to manage hypertension and cardiovascular strain."
    },
    "added_sugar_g": {
        "warning_threshold": 25.0,
        "critical_threshold": 36.0,
        "label": "High Added Sugar",
        "unit": "g",
        "guideline": "WHO recommends limiting free sugars to under 25g (approx. 6 tsp) per day for glycemic control."
    },
    "saturated_fat_g": {
        "warning_threshold": 15.0,
        "critical_threshold": 22.0,
        "label": "High Saturated Fat",
        "unit": "g",
        "guideline": "AHA suggests keeping saturated fat under 13-15g per meal to protect LDL cholesterol profiles."
    },
    "calories_kcal": {
        "warning_threshold": 800.0,
        "critical_threshold": 1050.0,
        "label": "High Calorie Density",
        "unit": "kcal",
        "guideline": "Exceeds standard 500-750 kcal single-meal maintenance targets for sedentary/moderate adults."
    },
    "protein_g": {
        "min_threshold": 15.0,
        "optimal_threshold": 25.0,
        "label": "Low Protein Intake",
        "unit": "g",
        "guideline": "A balanced main meal should ideally deliver at least 15-25g of protein to promote satiety and muscle preservation."
    },
    "carb_calories_percent": {
        "warning_threshold": 65.0,
        "label": "High Carbohydrate Ratio",
        "unit": "%",
        "guideline": "Over 65% of meal calories come from carbohydrates. Consider adding dietary fiber, legumes, or lean protein."
    }
}

# Explicit Non-Medical Grounding Disclaimer
DISCLAIMER_TEXT = (
    "⚠️ CLINICAL & NON-MEDICAL DISCLAIMER: This nutritional analysis, portion estimation, "
    "and dietary feedback are generated algorithmically for educational and lifestyle wellness "
    "guidance only. It is not medical advice, clinical diagnosis, or a therapeutic prescription. "
    "Always consult a registered dietitian, physician, or qualified healthcare professional "
    "before starting any medical dietary protocol."
)

# -------------------------------------------------------------
# 2. Explicit Portion Estimation Reference Constants
# -------------------------------------------------------------
# Standard reference plate diameter in centimeters (standard dining plate)
STANDARD_PLATE_DIAMETER_CM = 27.0
# Standard serving grams baseline for full plate area occupancy (approx. 350g)
STANDARD_PLATE_SERVING_GRAMS = 350.0

# Density factors for common food categories (density relative to water/standard dish)
DEFAULT_FOOD_DENSITY = 0.85
FOOD_DENSITY_FACTORS = {
    # Staples & Grains
    "rice": 0.85,
    "steamed rice": 0.85,
    "biryani": 0.95,
    "pulao": 0.90,
    "roti": 0.65,
    "chapati": 0.65,
    "naan": 0.70,
    "paratha": 0.85,
    "bread": 0.45,
    "poha": 0.60,
    "idli": 0.75,
    "dosa": 0.65,
    "upma": 0.85,
    "khichdi": 0.95,
    "pasta": 0.75,
    "noodles": 0.75,
    
    # Lentils, Pulses & Curries (gravy-heavy)
    "dal": 1.05,
    "dal tadka": 1.05,
    "dal makhani": 1.10,
    "sambar": 1.02,
    "chole": 1.05,
    "rajma": 1.05,
    "curry": 1.05,
    "soup": 1.00,
    "rasam": 1.00,
    
    # Vegetables & Sabzi
    "sabzi": 0.80,
    "paneer": 1.10,
    "paneer butter masala": 1.12,
    "palak paneer": 1.05,
    "aloo gobi": 0.85,
    "bhindi": 0.70,
    "mixed veg": 0.80,
    "salad": 0.35,
    "lettuce": 0.25,
    "cucumber": 0.40,
    
    # Meats & Seafood
    "chicken": 1.05,
    "chicken curry": 1.10,
    "fish": 0.95,
    "egg": 0.90,
    "omelette": 0.80,
    
    # Dairy & Accompaniments
    "curd": 1.03,
    "yogurt": 1.03,
    "raita": 1.02,
    "paneer bhurji": 0.95,
    
    # Snacks & Desserts
    "samosa": 0.90,
    "pakora": 0.75,
    "gulab jamun": 1.25,
    "jalebi": 1.15,
    "halwa": 1.20,
    "kheer": 1.10
}
