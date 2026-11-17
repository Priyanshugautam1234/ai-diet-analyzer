"""
Unit tests for Deterministic Threshold Guardrails & Grounded Advice
"""

import pytest
from core.guardrails import evaluate_meal_guardrails, generate_grounded_advice
from config import DISCLAIMER_TEXT


def test_high_sodium_guardrail():
    # Sodium = 950mg (above warning 800mg)
    totals = {
        "calories_kcal": 550.0,
        "protein_g": 20.0,
        "carbohydrates_g": 60.0,
        "fat_g": 18.0,
        "fiber_g": 6.0,
        "sodium_mg": 950.0,
        "saturated_fat_g": 8.0,
        "added_sugar_g": 4.0
    }
    res = evaluate_meal_guardrails(totals)
    alert_codes = [a["code"] for a in res["alerts"]]
    assert "GUARDRAIL_HIGH_SODIUM" in alert_codes
    assert res["status"] == "MODERATE_ADJUSTMENT_RECOMMENDED"


def test_critical_sodium_guardrail():
    # Sodium = 1350mg (above critical 1200mg)
    totals = {
        "calories_kcal": 650.0,
        "protein_g": 22.0,
        "carbohydrates_g": 70.0,
        "fat_g": 20.0,
        "fiber_g": 5.0,
        "sodium_mg": 1350.0,
        "saturated_fat_g": 10.0,
        "added_sugar_g": 5.0
    }
    res = evaluate_meal_guardrails(totals)
    alert_codes = [a["code"] for a in res["alerts"]]
    assert "GUARDRAIL_CRITICAL_SODIUM" in alert_codes
    assert res["status"] == "CRITICAL_ATTENTION_NEEDED"


def test_sugar_and_saturated_fat_guardrails():
    # Sugar = 30g (>25g), Saturated Fat = 18g (>15g)
    totals = {
        "calories_kcal": 750.0,
        "protein_g": 12.0,
        "carbohydrates_g": 90.0,
        "fat_g": 32.0,
        "fiber_g": 2.0,
        "sodium_mg": 450.0,
        "saturated_fat_g": 18.0,
        "added_sugar_g": 30.0
    }
    res = evaluate_meal_guardrails(totals)
    alert_codes = [a["code"] for a in res["alerts"]]
    assert "GUARDRAIL_HIGH_SUGAR" in alert_codes
    assert "GUARDRAIL_HIGH_SAT_FAT" in alert_codes
    assert "GUARDRAIL_LOW_PROTEIN" in alert_codes


def test_well_balanced_meal():
    totals = {
        "calories_kcal": 520.0,
        "protein_g": 26.0,
        "carbohydrates_g": 55.0,
        "fat_g": 14.0,
        "fiber_g": 8.0,
        "sodium_mg": 450.0,
        "saturated_fat_g": 4.0,
        "added_sugar_g": 2.0
    }
    res = evaluate_meal_guardrails(totals)
    assert res["status"] == "WELL_BALANCED"
    assert len(res["alerts"]) == 0


def test_grounded_advice_contains_disclaimer():
    totals = {
        "calories_kcal": 600.0,
        "protein_g": 18.0,
        "carbohydrates_g": 75.0,
        "fat_g": 20.0,
        "fiber_g": 4.0,
        "sodium_mg": 920.0,
        "saturated_fat_g": 16.0,
        "added_sugar_g": 5.0
    }
    guardrails = evaluate_meal_guardrails(totals)
    items = [{"name": "Dal", "final_grams": 150.0}, {"name": "Rice", "final_grams": 150.0}]
    
    advice = generate_grounded_advice(items, totals, guardrails, user_goal="Weight Loss")
    assert DISCLAIMER_TEXT in advice
    assert "Deterministic Guardrail Assessment" in advice
