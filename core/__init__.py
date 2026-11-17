"""
Core AI Diet Analyzer Engine Modules
"""

from core.detection import detect_multi_items_vlm, detect_multi_items_yolo, SingleItemBaselineClassifier
from core.portion_estimator import detect_plate_reference, estimate_portions, draw_portion_overlay
from core.nutrition_lookup import get_nutrition_profile_100g, calculate_portion_nutrition, cache
from core.guardrails import evaluate_meal_guardrails, generate_grounded_advice
from core.benchmark import get_benchmark_comparison_matrix, run_live_image_benchmark

__all__ = [
    "detect_multi_items_vlm",
    "detect_multi_items_yolo",
    "SingleItemBaselineClassifier",
    "detect_plate_reference",
    "estimate_portions",
    "draw_portion_overlay",
    "get_nutrition_profile_100g",
    "calculate_portion_nutrition",
    "cache",
    "evaluate_meal_guardrails",
    "generate_grounded_advice",
    "get_benchmark_comparison_matrix",
    "run_live_image_benchmark"
]
