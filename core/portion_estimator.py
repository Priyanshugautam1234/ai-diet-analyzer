"""
Explicit Portion Estimation Module
Implements the Plate-Reference Bounding-Box Ratio algorithm using OpenCV:
- Detects the reference plate / bowl contour via HoughCircles and contour analysis.
- Computes (item_area / plate_area * standard_serving_grams * food_density_factor).
- Supports user gram overrides and provides visual OpenCV overlays.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image

# Import cv2 gracefully if available
try:
    import cv2
except ImportError:
    cv2 = None

from config import (
    STANDARD_PLATE_SERVING_GRAMS,
    DEFAULT_FOOD_DENSITY,
    FOOD_DENSITY_FACTORS
)


def get_density_factor(food_name: str) -> float:
    """Retrieve food density factor for a given food name with fuzzy matching."""
    norm = food_name.strip().lower()
    if norm in FOOD_DENSITY_FACTORS:
        return FOOD_DENSITY_FACTORS[norm]
    for k, v in FOOD_DENSITY_FACTORS.items():
        if k in norm or norm in k:
            return v
    return DEFAULT_FOOD_DENSITY


def detect_plate_reference(image_np: np.ndarray) -> Dict[str, Any]:
    """
    Detect the reference dining plate or thali boundary in the image.
    Uses OpenCV HoughCircles with morphological fallback.
    
    Returns:
        dict containing:
            - found: bool
            - center: (x, y)
            - radius: float
            - plate_area: float (pixels^2)
            - method: str ('HoughCircles' | 'ContourEllipse' | 'ImageAreaFallback')
    """
    h, w = image_np.shape[:2]
    total_img_area = float(w * h)

    if cv2 is None:
        # Fallback without OpenCV
        est_radius = min(w, h) * 0.42
        est_area = math.pi * (est_radius ** 2)
        return {
            "found": False,
            "center": (int(w / 2), int(h / 2)),
            "radius": est_radius,
            "plate_area": est_area,
            "method": "ImageAreaFallback"
        }

    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY if image_np.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    # Preprocessing: bilateral filter / Gaussian blur to reduce noise while keeping edges
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 1. Strategy: HoughCircles
    min_r = int(min(w, h) * 0.22)
    max_r = int(max(w, h) * 0.55)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(w, h) * 0.4),
        param1=50,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Find the largest circle closest to the center of the image
        img_center = np.array([w / 2.0, h / 2.0])
        best_circle = None
        best_score = float("-inf")

        for (cx, cy, r) in circles:
            dist_to_center = np.linalg.norm(np.array([cx, cy]) - img_center)
            # Preference for larger radius and proximity to frame center
            score = r - (0.5 * dist_to_center)
            if score > best_score:
                best_score = score
                best_circle = (cx, cy, r)

        if best_circle is not None:
            cx, cy, r = best_circle
            plate_area = math.pi * (r ** 2)
            return {
                "found": True,
                "center": (int(cx), int(cy)),
                "radius": float(r),
                "plate_area": float(plate_area),
                "method": "HoughCircles"
            }

    # 2. Strategy: Contour detection with Convex Hull / MinEnclosingCircle
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_c_area = 0.0

    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area > (total_img_area * 0.15) and c_area > max_c_area:
            max_c_area = c_area
            largest_contour = c

    if largest_contour is not None:
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        if radius >= min_r:
            plate_area = math.pi * (radius ** 2)
            return {
                "found": True,
                "center": (int(cx), int(cy)),
                "radius": float(radius),
                "plate_area": float(plate_area),
                "method": "ContourMinEnclosingCircle"
            }

    # 3. Default fallback: Canonical dinner plate assumed occupying ~65% width/height
    est_radius = min(w, h) * 0.44
    est_area = math.pi * (est_radius ** 2)
    return {
        "found": False,
        "center": (int(w / 2), int(h / 2)),
        "radius": est_radius,
        "plate_area": est_area,
        "method": "CanonicalPlateAssumption"
    }


def estimate_portions(
    detected_items: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    plate_info: Optional[Dict[str, Any]] = None,
    standard_serving_grams: float = STANDARD_PLATE_SERVING_GRAMS,
    user_overrides: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Computes portion estimation for each detected bounding box using:
    
    Formula:
        item_area = (box_width * box_height) * elliptical_correction (approx pi/4 = 0.785)
        area_ratio = item_area / plate_area
        estimated_grams = area_ratio * standard_serving_grams * food_density_factor

    Args:
        detected_items: List of dicts with 'name', 'box' [xmin, ymin, xmax, ymax]
        image_shape: (height, width)
        plate_info: Output of detect_plate_reference
        standard_serving_grams: Standard plate serving baseline
        user_overrides: Optional dict mapping food name or index to custom gram weight

    Returns:
        List of enriched items with portion details, formula variables, and final grams.
    """
    h, w = image_shape[:2]
    if plate_info is None:
        est_r = min(w, h) * 0.44
        plate_area = math.pi * (est_r ** 2)
    else:
        plate_area = max(plate_info.get("plate_area", 1.0), 100.0)

    user_overrides = user_overrides or {}
    results = []

    for idx, item in enumerate(detected_items):
        name = item.get("name", f"Item-{idx+1}").strip()
        box = item.get("box", [0, 0, w, h])  # [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = box

        bw = max(xmax - xmin, 1)
        bh = max(ymax - ymin, 1)

        # Approximate food footprint area with elliptical factor (food isn't a sharp rectangle)
        item_area = float(bw * bh) * (math.pi / 4.0)
        area_ratio = item_area / plate_area

        # Get density factor
        density = item.get("density_factor") or get_density_factor(name)

        # Calculate raw estimated grams
        computed_grams = area_ratio * standard_serving_grams * density

        # Reasonable bounding: minimum 20g, maximum 650g
        bounded_grams = max(min(computed_grams, 650.0), 20.0)
        bounded_grams = round(bounded_grams, 1)

        # Check for user overrides (by exact name or by index string "0", "1", etc.)
        is_overridden = False
        override_key_idx = str(idx)
        final_grams = bounded_grams

        if name in user_overrides:
            final_grams = float(user_overrides[name])
            is_overridden = True
        elif override_key_idx in user_overrides:
            final_grams = float(user_overrides[override_key_idx])
            is_overridden = True

        results.append({
            **item,
            "index": idx,
            "name": name,
            "density_factor": round(density, 2),
            "item_area_px": round(item_area, 1),
            "plate_area_px": round(plate_area, 1),
            "area_ratio": round(area_ratio, 3),
            "computed_grams": bounded_grams,
            "final_grams": round(final_grams, 1),
            "is_user_override": is_overridden,
            "formula_repr": (
                f"({round(item_area, 0)} px² / {round(plate_area, 0)} px²) × "
                f"{standard_serving_grams}g × {round(density, 2)} = {bounded_grams}g"
            )
        })

    return results


def draw_portion_overlay(
    image: Image.Image,
    plate_info: Dict[str, Any],
    items_with_portions: List[Dict[str, Any]]
) -> Image.Image:
    """
    Renders visual bounding boxes, OpenCV plate contour, and portion badges on the image.
    """
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]

    if cv2 is None:
        return image

    overlay = img_np.copy()

    # Draw detected plate contour / circle
    if plate_info and "center" in plate_info and "radius" in plate_info:
        cx, cy = plate_info["center"]
        r = int(plate_info["radius"])
        # Green / Teal ring for plate boundary
        cv2.circle(overlay, (cx, cy), r, (0, 230, 180), 3)
        cv2.circle(overlay, (cx, cy), 5, (0, 230, 180), -1)
        method = plate_info.get("method", "Detected")
        cv2.putText(
            overlay,
            f"Reference Plate ({method})",
            (max(cx - 100, 10), max(cy - r - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 230, 180),
            2,
            cv2.LINE_AA
        )

    # Color palette for food items
    colors = [
        (255, 87, 34),   # Deep Orange
        (33, 150, 243),  # Blue
        (156, 39, 176),  # Purple
        (76, 175, 80),   # Green
        (255, 193, 7),   # Amber
        (233, 30, 99)    # Pink
    ]

    for idx, item in enumerate(items_with_portions):
        box = item.get("box", [0, 0, w, h])
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        color = colors[idx % len(colors)]

        # Draw item bounding box
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)

        # Label banner
        name = item.get("name", "Food")
        grams = item.get("final_grams", 100.0)
        label = f"{name}: {grams}g"
        if item.get("is_user_override"):
            label += " (Custom)"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(overlay, (xmin, max(ymin - 24, 0)), (xmin + tw + 10, max(ymin, 24)), color, -1)
        cv2.putText(
            overlay,
            label,
            (xmin + 5, max(ymin - 6, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Semi-transparent blending for smooth look
    blended = cv2.addWeighted(overlay, 0.9, img_np, 0.1, 0)
    return Image.fromarray(blended)
