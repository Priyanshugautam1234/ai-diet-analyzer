"""
Unit tests for Explicit Portion Estimation Formula & Plate Reference Detection
"""

import math
import numpy as np
from PIL import Image, ImageDraw
import pytest

from core.portion_estimator import (
    detect_plate_reference,
    estimate_portions,
    get_density_factor,
    draw_portion_overlay
)
from config import STANDARD_PLATE_SERVING_GRAMS


def create_synthetic_meal_image():
    """Generates a synthetic plate with food items for testing."""
    img = Image.new("RGB", (600, 600), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw circular plate
    draw.ellipse([50, 50, 550, 550], fill=(220, 220, 220), outline=(100, 100, 100), width=4)
    # Draw food zones
    draw.ellipse([100, 200, 300, 450], fill=(255, 255, 230), outline=(180, 180, 0), width=2) # Rice
    draw.ellipse([320, 120, 480, 280], fill=(230, 150, 50), outline=(200, 100, 0), width=2)  # Dal
    return img


def test_density_factor_mapping():
    assert get_density_factor("dal tadka") == 1.05
    assert get_density_factor("steamed rice") == 0.85
    assert get_density_factor("roti") == 0.65
    assert get_density_factor("salad") == 0.35


def test_plate_reference_detection():
    img = create_synthetic_meal_image()
    img_np = np.array(img)
    plate_info = detect_plate_reference(img_np)
    
    assert "plate_area" in plate_info
    assert plate_info["plate_area"] > 0
    assert "radius" in plate_info
    assert "center" in plate_info


def test_portion_formula_calculation():
    image_shape = (600, 600)
    plate_radius = 250.0
    plate_area = math.pi * (plate_radius ** 2)
    plate_info = {
        "found": True,
        "center": (300, 300),
        "radius": plate_radius,
        "plate_area": plate_area,
        "method": "HoughCircles"
    }

    # An item occupying 200x200 bounding box
    bw, bh = 200, 200
    expected_item_area = (bw * bh) * (math.pi / 4.0)
    expected_ratio = expected_item_area / plate_area
    density = 0.85  # rice
    expected_grams = expected_ratio * STANDARD_PLATE_SERVING_GRAMS * density

    items = [
        {"name": "Steamed Rice", "box": [100, 200, 300, 400], "density_factor": density}
    ]

    results = estimate_portions(items, image_shape, plate_info=plate_info)
    assert len(results) == 1
    res = results[0]
    
    assert math.isclose(res["computed_grams"], round(expected_grams, 1), abs_tol=0.2)
    assert res["final_grams"] == res["computed_grams"]
    assert res["is_user_override"] is False
    assert "formula_repr" in res


def test_portion_user_gram_overrides():
    image_shape = (600, 600)
    items = [
        {"name": "Steamed Rice", "box": [100, 200, 300, 400]},
        {"name": "Dal Tadka", "box": [320, 120, 480, 280]}
    ]
    
    # User overrides Dal Tadka to 180 grams
    overrides = {"Dal Tadka": 180.0}
    results = estimate_portions(items, image_shape, user_overrides=overrides)
    
    assert results[1]["final_grams"] == 180.0
    assert results[1]["is_user_override"] is True
    assert results[0]["is_user_override"] is False


def test_draw_portion_overlay():
    img = create_synthetic_meal_image()
    plate_info = {"center": (300, 300), "radius": 250, "method": "Test"}
    items = [
        {"name": "Rice", "box": [100, 200, 300, 400], "final_grams": 150.0},
        {"name": "Dal", "box": [320, 120, 480, 280], "final_grams": 130.0}
    ]
    annotated = draw_portion_overlay(img, plate_info, items)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == img.size
