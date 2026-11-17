"""
Unit tests for Hybrid Nutrition Lookup (IFCT + USDA + SQLite Cache)
"""

import os
from pathlib import Path
import pytest
from core.nutrition_lookup import (
    get_nutrition_profile_100g,
    calculate_portion_nutrition,
    cache
)


def test_ifct_lookup_indian_staples():
    # Test common Indian dishes in IFCT
    staples = ["dal tadka", "roti", "rice", "paneer bhurji", "poha", "biryani"]
    for dish in staples:
        profile = get_nutrition_profile_100g(dish)
        assert profile is not None
        assert profile["calories_kcal"] > 0
        assert profile["protein_g"] > 0
        assert "IFCT" in profile["source"] or "Cache" in profile["source"]


def test_sqlite_caching():
    query = "test_custom_dish"
    test_data = {
        "name": "Test Custom Dish",
        "calories_kcal": 210.0,
        "protein_g": 12.0,
        "carbohydrates_g": 25.0,
        "fat_g": 8.0,
        "fiber_g": 3.0,
        "sodium_mg": 350.0,
        "saturated_fat_g": 2.5,
        "added_sugar_g": 1.0,
        "iron_mg": 1.5,
        "calcium_mg": 40.0,
        "potassium_mg": 160.0,
        "vitamin_c_mg": 5.0,
        "density_factor": 0.85,
        "default_serving_grams": 150.0
    }
    
    # Store in cache
    cache.set(query, test_data, source="Unit Test Mock")
    
    # First fetch from cache
    cached = cache.get(query)
    assert cached is not None
    assert cached["food_name"] == "Test Custom Dish"
    assert cached["calories_kcal"] == 210.0
    assert "SQLite Cache" in cached["source"]
    
    # Verify hit counter increments
    second_fetch = cache.get(query)
    assert second_fetch["hit_count"] >= 2


def test_portion_scaling():
    # 200g should have exactly double the nutrients of 100g
    nutr_100g = calculate_portion_nutrition("dal tadka", 100.0)
    nutr_200g = calculate_portion_nutrition("dal tadka", 200.0)
    
    assert pytest.approx(nutr_200g["calories_kcal"], 0.2) == nutr_100g["calories_kcal"] * 2
    assert pytest.approx(nutr_200g["protein_g"], 0.2) == nutr_100g["protein_g"] * 2
    assert pytest.approx(nutr_200g["sodium_mg"], 0.2) == nutr_100g["sodium_mg"] * 2
