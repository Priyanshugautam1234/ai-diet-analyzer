"""
Unit tests for Multi-Item Detection & Single-Item Baseline
"""

from PIL import Image
from core.detection import SingleItemBaselineClassifier, detect_multi_items_yolo
from core.benchmark import get_benchmark_comparison_matrix


def test_single_item_baseline_classifier():
    classifier = SingleItemBaselineClassifier(architecture="ResNet-50 Baseline")
    img = Image.new("RGB", (400, 400), color=(200, 200, 200))
    res = classifier.predict(img)

    assert res["items_detected_count"] == 1
    assert "detected_label" in res
    assert "limitation_note" in res
    assert len(res["detected_items"]) == 1


def test_multi_item_yolo_detection():
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    items = detect_multi_items_yolo(img)

    assert isinstance(items, list)
    assert len(items) >= 1
    for it in items:
        assert "name" in it
        assert "box" in it
        assert len(it["box"]) == 4


def test_benchmark_matrix_structure():
    df = get_benchmark_comparison_matrix()
    assert not df.empty
    assert "Feature / Metric" in df.columns
    assert "Single-Item Baseline (ResNet-50 / EfficientNet)" in df.columns
    assert "YOLOv8 Multi-Item Detector" in df.columns
    assert "Multi-Item VLM (Gemini Structured Vision)" in df.columns
