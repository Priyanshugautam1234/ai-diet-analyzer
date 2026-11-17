"""
Hybrid Nutrition Lookup Module
Prioritizes Indian Food Composition Tables (IFCT / curated local index),
falls back to USDA FoodData Central API, and uses a local SQLite database
for persistent caching to prevent redundant API calls and rate-limiting.
"""

import json
import sqlite3
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import requests

from config import (
    DATA_DIR,
    SQLITE_CACHE_PATH,
    USDA_API_KEY,
    USDA_API_BASE_URL,
    DEFAULT_FOOD_DENSITY
)

# Load curated IFCT dataset
IFCT_DB_PATH = DATA_DIR / "ifct_food_db.json"

def _load_ifct_db() -> Dict[str, Any]:
    if IFCT_DB_PATH.exists():
        with open(IFCT_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

IFCT_DATABASE = _load_ifct_db()


class NutritionLookupCache:
    """SQLite-backed cache for nutrition queries."""
    
    def __init__(self, db_path=SQLITE_CACHE_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS food_nutrition_cache (
                    query_key TEXT PRIMARY KEY,
                    source TEXT,
                    food_name TEXT,
                    calories_kcal REAL,
                    protein_g REAL,
                    carbohydrates_g REAL,
                    fat_g REAL,
                    fiber_g REAL,
                    sodium_mg REAL,
                    saturated_fat_g REAL,
                    added_sugar_g REAL,
                    iron_mg REAL,
                    calcium_mg REAL,
                    potassium_mg REAL,
                    vitamin_c_mg REAL,
                    density_factor REAL,
                    default_serving_grams REAL,
                    hit_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get(self, query_key: str) -> Optional[Dict[str, Any]]:
        norm_key = query_key.strip().lower()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM food_nutrition_cache WHERE query_key = ?", (norm_key,))
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    "UPDATE food_nutrition_cache SET hit_count = hit_count + 1 WHERE query_key = ?", 
                    (norm_key,)
                )
                conn.commit()
                res = dict(row)
                res["source"] = f"SQLite Cache (cached from {res.get('source', 'Unknown')})"
                return res
        return None

    def set(self, query_key: str, data: Dict[str, Any], source: str):
        norm_key = query_key.strip().lower()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO food_nutrition_cache (
                    query_key, source, food_name, calories_kcal, protein_g,
                    carbohydrates_g, fat_g, fiber_g, sodium_mg, saturated_fat_g,
                    added_sugar_g, iron_mg, calcium_mg, potassium_mg, vitamin_c_mg,
                    density_factor, default_serving_grams, hit_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                norm_key,
                source,
                data.get("name", query_key.title()),
                float(data.get("calories_kcal", 0.0)),
                float(data.get("protein_g", 0.0)),
                float(data.get("carbohydrates_g", 0.0)),
                float(data.get("fat_g", 0.0)),
                float(data.get("fiber_g", 0.0)),
                float(data.get("sodium_mg", 0.0)),
                float(data.get("saturated_fat_g", 0.0)),
                float(data.get("added_sugar_g", 0.0)),
                float(data.get("iron_mg", 0.0)),
                float(data.get("calcium_mg", 0.0)),
                float(data.get("potassium_mg", 0.0)),
                float(data.get("vitamin_c_mg", 0.0)),
                float(data.get("density_factor", DEFAULT_FOOD_DENSITY)),
                float(data.get("default_serving_grams", 150.0)),
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM food_nutrition_cache")
            total_items, total_hits = cursor.fetchone()
            return {
                "total_cached_foods": total_items,
                "total_cache_hits": total_hits
            }


cache = NutritionLookupCache()


def _normalize_name(name: str) -> str:
    """Normalize food query string for canonical lookup."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", name).strip().lower()
    return clean


def _lookup_ifct(query: str) -> Optional[Dict[str, Any]]:
    """Look up food in curated Indian Food Composition Tables (IFCT)."""
    norm = _normalize_name(query)
    
    # Direct match in keys
    if norm in IFCT_DATABASE:
        res = dict(IFCT_DATABASE[norm])
        res["source"] = "IFCT (Indian Food Composition Tables - Local)"
        return res

    # Match aliases or partial keys
    for key, data in IFCT_DATABASE.items():
        if key in norm or norm in key:
            res = dict(data)
            res["source"] = "IFCT (Indian Food Composition Tables - Local)"
            return res
        for alias in data.get("aliases", []):
            if alias in norm or norm in alias:
                res = dict(data)
                res["source"] = "IFCT (Indian Food Composition Tables - Local)"
                return res

    return None


def _lookup_usda(query: str) -> Optional[Dict[str, Any]]:
    """
    Look up nutrition via USDA FoodData Central API.
    Returns normalized 100g nutrient profile.
    """
    if not USDA_API_KEY:
        return None

    try:
        url = f"{USDA_API_BASE_URL}/foods/search"
        params = {
            "query": query,
            "api_key": USDA_API_KEY,
            "pageSize": 3,
            "dataType": ["Survey (FNDDS)", "Foundation", "SR Legacy", "Branded"]
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json()
        foods = data.get("foods", [])
        if not foods:
            return None

        # Pick top relevant food item
        top_food = foods[0]
        food_name = top_food.get("description", query.title())
        nutrients = {n.get("nutrientName", "").lower(): n.get("value", 0.0) 
                     for n in top_food.get("foodNutrients", [])}

        # Helper to find nutrient by common names
        def get_nutrient(patterns):
            for pattern in patterns:
                for k, v in nutrients.items():
                    if pattern in k:
                        return float(v)
            return 0.0

        calories = get_nutrient(["energy", "calorie"])
        protein = get_nutrient(["protein"])
        fat = get_nutrient(["total lipid", "fat"])
        carbs = get_nutrient(["carbohydrate"])
        fiber = get_nutrient(["fiber"])
        sodium = get_nutrient(["sodium"])
        sat_fat = get_nutrient(["fatty acids, total saturated", "saturated fat"])
        sugars = get_nutrient(["sugars, total", "sugars, added"])
        iron = get_nutrient(["iron"])
        calcium = get_nutrient(["calcium"])
        potassium = get_nutrient(["potassium"])
        vit_c = get_nutrient(["vitamin c"])

        result = {
            "name": food_name,
            "category": top_food.get("foodCategory", "General"),
            "calories_kcal": round(calories, 1),
            "protein_g": round(protein, 1),
            "carbohydrates_g": round(carbs, 1),
            "fat_g": round(fat, 1),
            "fiber_g": round(fiber, 1),
            "sodium_mg": round(sodium, 1),
            "saturated_fat_g": round(sat_fat, 1),
            "added_sugar_g": round(sugars, 1),
            "iron_mg": round(iron, 1),
            "calcium_mg": round(calcium, 1),
            "potassium_mg": round(potassium, 1),
            "vitamin_c_mg": round(vit_c, 1),
            "density_factor": DEFAULT_FOOD_DENSITY,
            "default_serving_grams": 150.0,
            "source": "USDA FoodData Central (FDC API)"
        }
        return result
    except Exception as e:
        # Fallback gracefully if network/API fails
        return None


def get_nutrition_profile_100g(food_query: str) -> Dict[str, Any]:
    """
    Hybrid lookup:
    1. Check SQLite Cache
    2. Check IFCT Local Database
    3. Fallback to USDA FoodData Central API
    4. Safe baseline fallback if completely unknown
    """
    clean_query = _normalize_name(food_query)

    # 1. Check SQLite cache
    cached_val = cache.get(clean_query)
    if cached_val:
        return cached_val

    # 2. Check IFCT (curated Indian dataset)
    ifct_result = _lookup_ifct(clean_query)
    if ifct_result:
        cache.set(clean_query, ifct_result, source="IFCT Local Database")
        return ifct_result

    # 3. Check USDA API
    usda_result = _lookup_usda(food_query)
    if usda_result:
        cache.set(clean_query, usda_result, source="USDA FoodData Central API")
        return usda_result

    # 4. Graceful default fallback for unidentified foods
    default_profile = {
        "name": food_query.title(),
        "category": "Mixed Dish",
        "calories_kcal": 150.0,
        "protein_g": 5.0,
        "carbohydrates_g": 20.0,
        "fat_g": 6.0,
        "fiber_g": 2.0,
        "sodium_mg": 250.0,
        "saturated_fat_g": 1.5,
        "added_sugar_g": 0.5,
        "iron_mg": 1.0,
        "calcium_mg": 30.0,
        "potassium_mg": 150.0,
        "vitamin_c_mg": 2.0,
        "density_factor": DEFAULT_FOOD_DENSITY,
        "default_serving_grams": 150.0,
        "source": "Estimated Standard Profile (Generic Fallback)"
    }
    cache.set(clean_query, default_profile, source="Generic Fallback")
    return default_profile


def calculate_portion_nutrition(food_name: str, grams: float) -> Dict[str, Any]:
    """
    Calculates exact macro and micronutrient breakdown for a specified portion in grams.
    """
    base_100g = get_nutrition_profile_100g(food_name)
    factor = max(grams, 0.0) / 100.0

    return {
        "food_name": base_100g.get("name", food_name.title()),
        "query": food_name,
        "portion_grams": round(grams, 1),
        "source": base_100g.get("source", "IFCT"),
        "density_factor": base_100g.get("density_factor", DEFAULT_FOOD_DENSITY),
        "calories_kcal": round(base_100g.get("calories_kcal", 0.0) * factor, 1),
        "protein_g": round(base_100g.get("protein_g", 0.0) * factor, 1),
        "carbohydrates_g": round(base_100g.get("carbohydrates_g", 0.0) * factor, 1),
        "fat_g": round(base_100g.get("fat_g", 0.0) * factor, 1),
        "fiber_g": round(base_100g.get("fiber_g", 0.0) * factor, 1),
        "sodium_mg": round(base_100g.get("sodium_mg", 0.0) * factor, 1),
        "saturated_fat_g": round(base_100g.get("saturated_fat_g", 0.0) * factor, 1),
        "added_sugar_g": round(base_100g.get("added_sugar_g", 0.0) * factor, 1),
        "iron_mg": round(base_100g.get("iron_mg", 0.0) * factor, 2),
        "calcium_mg": round(base_100g.get("calcium_mg", 0.0) * factor, 1),
        "potassium_mg": round(base_100g.get("potassium_mg", 0.0) * factor, 1),
        "vitamin_c_mg": round(base_100g.get("vitamin_c_mg", 0.0) * factor, 1)
    }
