"""
AI Diet Analyzer - Main Entry Point
Supports all four technical upgrades:
1. Multi-Item Detection instead of Single-Label Classification
2. Explicit Portion Estimation Formula (OpenCV Plate-Reference Bounding-Box Ratio)
3. Hybrid Nutrition Lookup (IFCT Local + USDA FoodData Central + SQLite Cache)
4. Deterministic Threshold Guardrails + Grounded Advice & Non-Medical Disclaimer
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import and execute the modern app UI
from app import *
