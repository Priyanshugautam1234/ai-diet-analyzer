"""
AI Diet Analyzer Pro - Modern Multi-Item Nutrition & Diet Analysis App
Implements the 4 Technical Upgrades:
1. Multi-Item Object Detection (YOLOv8 + Gemini VLM with Pydantic JSON schema vs. ResNet-50 Single-Item Baseline)
2. Explicit Portion Estimation Formula (OpenCV Plate-Reference Bounding-Box Ratio + Overrides)
3. Hybrid Nutrition Lookup (IFCT Local Index + USDA FoodData Central + SQLite Persistent Cache)
4. Deterministic Threshold Guardrails + Grounded Advice & Non-Medical Disclaimer
"""

import os
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Core Engine Imports
from config import (
    GOOGLE_API_KEY,
    USDA_API_KEY,
    STANDARD_PLATE_DIAMETER_CM,
    STANDARD_PLATE_SERVING_GRAMS,
    DEFAULT_FOOD_DENSITY,
    DISCLAIMER_TEXT,
    DATA_DIR
)
from core.detection import (
    detect_multi_items_vlm,
    detect_multi_items_yolo,
    SingleItemBaselineClassifier
)
from core.portion_estimator import (
    detect_plate_reference,
    estimate_portions,
    draw_portion_overlay,
    get_density_factor
)
from core.nutrition_lookup import (
    get_nutrition_profile_100g,
    calculate_portion_nutrition,
    cache,
    IFCT_DATABASE
)
from core.guardrails import (
    evaluate_meal_guardrails,
    generate_grounded_advice
)
from core.benchmark import (
    get_benchmark_comparison_matrix,
    run_live_image_benchmark
)

# Page Configuration
st.set_page_config(
    page_title="AI Diet Analyzer Pro",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-ifct {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-usda {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-cache {
        background-color: #F3E8FF;
        color: #6B21A8;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400&q=80", use_container_width=True)
    st.title("⚙️ Engine Settings")

    st.subheader("1. Detection Paradigm")
    detection_engine = st.selectbox(
        "Detection Model",
        options=[
            "Multi-Item VLM (Gemini Structured Vision)",
            "Multi-Item YOLOv8 (Local Object Detection)",
            "Single-Item Baseline (ResNet-50 Benchmark)"
        ],
        index=0,
        help="Compare Multi-Item bounding box localization vs. Single-label ResNet baseline."
    )

    st.subheader("2. Health & Wellness Goal")
    health_goal = st.selectbox(
        "Target Goal",
        options=[
            "General Wellness & Maintenance",
            "Weight Loss / Caloric Deficit",
            "Muscle Hypertrophy / High Protein",
            "Heart-Healthy / Low Sodium (DASH)",
            "Glycemic Control / Low Added Sugar"
        ]
    )

    st.subheader("3. Portion Calibration")
    plate_diameter = st.number_input(
        "Plate Diameter (cm)",
        min_value=15.0,
        max_value=40.0,
        value=float(STANDARD_PLATE_DIAMETER_CM),
        step=0.5,
        help="Standard reference plate diameter used for HoughCircles calibration."
    )
    standard_plate_grams = st.number_input(
        "Plate Full Capacity (grams)",
        min_value=200.0,
        max_value=700.0,
        value=float(STANDARD_PLATE_SERVING_GRAMS),
        step=25.0,
        help="Standard baseline food mass when plate is 100% full."
    )

    st.subheader("4. API & Cache Status")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=GOOGLE_API_KEY,
        type="password",
        help="Required for cloud VLM detection and LLM grounded feedback."
    )

    # Cache Stats
    cache_stats = cache.get_stats()
    st.caption(
        f"💾 **SQLite Cache:** {cache_stats['total_cached_foods']} items stored | "
        f"{cache_stats['total_cache_hits']} cache hits"
    )

    st.markdown("---")
    st.info("💡 **Tip:** Switch to the Benchmark tab to compare ResNet vs. YOLOv8 & VLM.")


# ---------------------------------------------------------------------------
# Main Content & Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🥗 AI Diet Analyzer Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Item Detection • Plate-Reference Portion Estimation • '
    'Hybrid IFCT/USDA Nutrition • Deterministic Guardrails</div>',
    unsafe_allow_html=True
)

tab_analyzer, tab_benchmark, tab_database, tab_formula = st.tabs([
    "🍽️ Meal Analysis & Overrides",
    "⚖️ Baseline vs Multi-Item Benchmark",
    "🔍 Hybrid Nutrition & Cache DB",
    "📐 Explicit Portion Formula"
])


# ===========================================================================
# TAB 1: MEAL ANALYSIS & USER OVERRIDES
# ===========================================================================
with tab_analyzer:
    st.markdown("### Step 1: Provide Meal Image")
    col_input1, col_input2 = st.columns([1.5, 1])

    with col_input1:
        uploaded_file = st.file_uploader(
            "Upload Meal Photo (JPEG/PNG)",
            type=["jpg", "jpeg", "png"]
        )

    with col_input2:
        st.markdown("**Or choose a pre-loaded sample meal:**")
        sample_choice = st.radio(
            "Sample Plate",
            options=["None (Use Upload)", "Traditional Indian Thali", "South Indian Breakfast"],
            index=1 if uploaded_file is None else 0
        )

    # Load Image
    active_image = None
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
    elif sample_choice == "Traditional Indian Thali":
        sample_path = DATA_DIR / "sample_meals" / "indian_thali.png"
        if sample_path.exists():
            active_image = Image.open(sample_path)
    elif sample_choice == "South Indian Breakfast":
        sample_path = DATA_DIR / "sample_meals" / "breakfast_plate.png"
        if sample_path.exists():
            active_image = Image.open(sample_path)

    if active_image is not None:
        st.markdown("---")
        col_img, col_results = st.columns([1, 1.2])

        # Step 1: Plate Reference Detection (OpenCV HoughCircles / Contours)
        img_np = np.array(active_image.convert("RGB"))
        plate_info = detect_plate_reference(img_np)

        # Step 2: Food Items Detection
        with st.spinner("Analyzing plate composition..."):
            if "Single-Item Baseline" in detection_engine:
                baseline = SingleItemBaselineClassifier()
                base_res = baseline.predict(active_image)
                raw_items = base_res["detected_items"]
                st.warning(base_res["limitation_note"])
            elif "YOLOv8" in detection_engine:
                raw_items = detect_multi_items_yolo(active_image)
            else:
                raw_items = detect_multi_items_vlm(active_image, api_key=api_key_input)

        # Initialize session state for user gram overrides
        if "gram_overrides" not in st.session_state:
            st.session_state["gram_overrides"] = {}

        # Step 3: Compute portions
        items_with_portions = estimate_portions(
            detected_items=raw_items,
            image_shape=img_np.shape[:2],
            plate_info=plate_info,
            standard_serving_grams=standard_plate_grams,
            user_overrides=st.session_state["gram_overrides"]
        )

        # Render OpenCV Visual Overlay
        annotated_img = draw_portion_overlay(active_image, plate_info, items_with_portions)

        with col_img:
            st.markdown("#### 📷 Plate Localization Overlay")
            st.image(annotated_img, use_container_width=True)
            method = plate_info.get("method", "Standard")
            found = plate_info.get("found", False)
            status_text = f"✅ Plate Boundary Localized ({method})" if found else f"ℹ️ Reference Frame ({method})"
            st.caption(f"{status_text} | Area: {round(plate_info.get('plate_area', 0))} px²")

        with col_results:
            st.markdown("#### ⚖️ Detected Items & Portion Gram Overrides")
            st.caption("Review calculated grams below. You can override any gram value:")

            # Interactive User Gram Overrides Form
            new_overrides = {}
            for idx, item in enumerate(items_with_portions):
                c_name, c_grams, c_override = st.columns([1.8, 1.2, 1])
                name = item["name"]
                calc_g = item["computed_grams"]
                current_g = item["final_grams"]

                with c_name:
                    st.write(f"**{idx+1}. {name}**")
                    st.caption(f"Category: {item.get('category', 'Dish')} | Density: {item['density_factor']}")

                with c_grams:
                    val = st.number_input(
                        f"Grams for item {idx+1}",
                        min_value=10.0,
                        max_value=600.0,
                        value=float(current_g),
                        step=5.0,
                        key=f"item_override_{idx}",
                        label_visibility="collapsed"
                    )
                    if abs(val - calc_g) > 0.5:
                        new_overrides[name] = val

                with c_override:
                    if name in new_overrides:
                        st.markdown("<span class='badge-cache'>Custom</span>", unsafe_allow_html=True)
                    else:
                        st.caption(f"Calc: {calc_g}g")

            if new_overrides != st.session_state["gram_overrides"]:
                st.session_state["gram_overrides"] = new_overrides
                st.rerun()

        # Step 4: Hybrid Nutrition Lookup (IFCT -> USDA -> SQLite Cache)
        st.markdown("---")
        st.markdown("### 📊 Nutritional Breakdown & Guardrails")

        item_nutrients = []
        meal_totals = {
            "calories_kcal": 0.0,
            "protein_g": 0.0,
            "carbohydrates_g": 0.0,
            "fat_g": 0.0,
            "fiber_g": 0.0,
            "sodium_mg": 0.0,
            "saturated_fat_g": 0.0,
            "added_sugar_g": 0.0,
            "iron_mg": 0.0,
            "calcium_mg": 0.0,
            "potassium_mg": 0.0,
            "vitamin_c_mg": 0.0
        }

        for it in items_with_portions:
            nutr = calculate_portion_nutrition(it["name"], it["final_grams"])
            item_nutrients.append(nutr)
            for k in meal_totals:
                meal_totals[k] += nutr.get(k, 0.0)

        # Round totals
        for k in meal_totals:
            meal_totals[k] = round(meal_totals[k], 1)

        # Step 5: Deterministic Threshold Guardrails Evaluation
        guardrail_eval = evaluate_meal_guardrails(meal_totals)

        # Summary Metric Cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Total Energy", f"{meal_totals['calories_kcal']} kcal")
        col_m2.metric(
            "Protein", 
            f"{meal_totals['protein_g']} g",
            f"{guardrail_eval['macro_breakdown']['protein_pct']}% kcal"
        )
        col_m3.metric(
            "Carbohydrates", 
            f"{meal_totals['carbohydrates_g']} g",
            f"{guardrail_eval['macro_breakdown']['carbs_pct']}% kcal"
        )
        col_m4.metric(
            "Total Fat", 
            f"{meal_totals['fat_g']} g",
            f"{guardrail_eval['macro_breakdown']['fat_pct']}% kcal"
        )
        col_m5.metric(
            "Sodium", 
            f"{meal_totals['sodium_mg']} mg",
            delta="-High" if meal_totals['sodium_mg'] > 800 else "Normal",
            delta_color="inverse"
        )

        # Deterministic Guardrail Alert Box
        st.markdown("#### 🛡️ Deterministic Guardrail Assessment")
        alerts = guardrail_eval.get("alerts", [])
        if alerts:
            for alert in alerts:
                if alert["severity"] == "CRITICAL":
                    st.error(f"🚨 **{alert['title']}**: {alert['message']}  \n*Guideline:* {alert['guideline']}")
                elif alert["severity"] == "WARNING":
                    st.warning(f"⚠️ **{alert['title']}**: {alert['message']}  \n*Guideline:* {alert['guideline']}")
                else:
                    st.info(f"ℹ️ **{alert['title']}**: {alert['message']}")
        else:
            st.success("✅ **Balanced Profile:** All critical thresholds (Sodium <800mg, Free Sugar <25g, Saturated Fat <15g) are strictly satisfied.")

        # Itemized Nutrition Table with Data Source
        st.markdown("#### 📋 Itemized Breakdown (Hybrid Lookup)")
        table_rows = []
        for n in item_nutrients:
            source_tag = "IFCT Local" if "IFCT" in n["source"] else ("SQLite Cache" if "Cache" in n["source"] else "USDA FDC")
            table_rows.append({
                "Food Item": n["food_name"],
                "Portion": f"{n['portion_grams']}g",
                "Calories": f"{n['calories_kcal']} kcal",
                "Protein": f"{n['protein_g']}g",
                "Carbs": f"{n['carbohydrates_g']}g",
                "Fat": f"{n['fat_g']}g",
                "Sat. Fat": f"{n['saturated_fat_g']}g",
                "Sodium": f"{n['sodium_mg']}mg",
                "Sugar": f"{n['added_sugar_g']}g",
                "Fiber": f"{n['fiber_g']}g",
                "Source Database": source_tag
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # Grounded AI Advice Section
        st.markdown("#### 🧠 Grounded Advice & Dietician Review")
        with st.spinner("Generating grounded advice..."):
            advice = generate_grounded_advice(
                detected_items=items_with_portions,
                totals=meal_totals,
                guardrail_results=guardrail_eval,
                user_goal=health_goal,
                api_key=api_key_input
            )
        st.markdown(advice)

        # Mandatory Non-Medical Disclaimer
        st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 2: BENCHMARK EVALUATION (BASELINE vs MULTI-ITEM)
# ===========================================================================
with tab_benchmark:
    st.markdown("### ⚖️ Architectural & Performance Benchmark Matrix")
    st.markdown(
        "Standard ResNet/EfficientNet classifiers predict exactly **one label** for the whole image. "
        "The comparison table below details why multi-item detection (YOLOv8 / VLM with Pydantic JSON schema) "
        "is critical for real-world composite plates and thalis."
    )

    df_matrix = get_benchmark_comparison_matrix()
    st.dataframe(df_matrix, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧪 Run Live Benchmark Comparison on Current Meal")

    if active_image is not None:
        if st.button("Run Side-by-Side Model Evaluation"):
            with st.spinner("Benchmarking Baseline Classifier vs. Multi-Item Detector..."):
                bench_res = run_live_image_benchmark(active_image, api_key=api_key_input)

            c_b1, c_b2 = st.columns(2)
            with c_b1:
                st.markdown("#### 🔴 Single-Item Baseline (ResNet-50)")
                st.error(f"**Items Detected:** {bench_res['baseline_items_found']}")
                st.write(f"**Predicted Label:** {bench_res['baseline_item_names']}")
                st.caption("Fails to localize individual dishes. Cannot perform portion estimation.")

            with c_b2:
                st.markdown("#### 🟢 Multi-Item Upgrade (VLM / YOLOv8)")
                st.success(f"**Items Detected:** {bench_res['multi_items_found']}")
                st.write(f"**Detected Dishes:** {', '.join(bench_res['multi_item_names'])}")
                st.caption("Individual bounding boxes enable exact plate-ratio portion estimation.")

            st.info(f"**Summary Analysis:** {bench_res['comparison_summary']}")
    else:
        st.info("Select or upload an image in Tab 1 to run the live benchmark.")


# ===========================================================================
# TAB 3: HYBRID NUTRITION & CACHE EXPLORER
# ===========================================================================
with tab_database:
    st.markdown("### 🔍 Hybrid Nutrition Database (IFCT + USDA FoodData Central)")
    st.markdown(
        "Queries local **Indian Food Composition Tables (IFCT)** first for staples like dal tadka, "
        "roti, and paneer bhurji, falling back to **USDA FoodData Central API** with local **SQLite caching**."
    )

    search_query = st.text_input("Search Food Item in Hybrid DB", value="dal tadka")
    if search_query:
        profile = get_nutrition_profile_100g(search_query)
        st.markdown(f"#### Results for **{profile.get('name', search_query.title())}** (per 100g)")
        
        c_src1, c_src2 = st.columns([1, 3])
        with c_src1:
            st.metric("Calories", f"{profile.get('calories_kcal', 0)} kcal")
            st.caption(f"**Source:** {profile.get('source', 'IFCT')}")
            st.caption(f"**Density Factor:** {profile.get('density_factor', 0.85)}")
        with c_src2:
            st.write(
                f"- **Protein:** {profile.get('protein_g', 0)}g  \n"
                f"- **Carbohydrates:** {profile.get('carbohydrates_g', 0)}g  \n"
                f"- **Fat:** {profile.get('fat_g', 0)}g (Saturated: {profile.get('saturated_fat_g', 0)}g)  \n"
                f"- **Sodium:** {profile.get('sodium_mg', 0)}mg  \n"
                f"- **Added Sugar:** {profile.get('added_sugar_g', 0)}g  \n"
                f"- **Dietary Fiber:** {profile.get('fiber_g', 0)}g"
            )

    st.markdown("---")
    st.markdown("#### 📂 Curated Indian Food Composition Tables (IFCT) Index")
    ifct_list = []
    for k, v in IFCT_DATABASE.items():
        ifct_list.append({
            "Food Key": k,
            "Item Name": v.get("name", k),
            "Category": v.get("category", "General"),
            "Energy (kcal/100g)": v.get("calories_kcal", 0),
            "Protein (g)": v.get("protein_g", 0),
            "Carbs (g)": v.get("carbohydrates_g", 0),
            "Fat (g)": v.get("fat_g", 0),
            "Sodium (mg)": v.get("sodium_mg", 0),
            "Density": v.get("density_factor", 0.85)
        })
    st.dataframe(pd.DataFrame(ifct_list), use_container_width=True)


# ===========================================================================
# TAB 4: EXPLICIT PORTION ESTIMATION FORMULA
# ===========================================================================
with tab_formula:
    st.markdown("### 📐 Plate-Reference Bounding-Box Ratio Formula")
    st.markdown(
        """
        Since 2D RGB images lack depth information, converting bounding-box pixels into physical mass (grams) 
        requires geometric normalization against a known spatial reference—the **plate contour**:
        """
    )

    st.latex(r"""
    \text{Area Ratio} = \frac{\text{Item Footprint Area (pixels)}}{\text{Reference Plate Area (pixels)}}
    """)

    st.latex(r"""
    \text{Estimated Grams} = \text{Area Ratio} \times \text{Standard Plate Full Mass (g)} \times \text{Food Density Factor} (\rho)
    """)

    st.markdown(
        """
        #### Mathematical Derivation:
        1. **Plate Contour Detection:**
           OpenCV `cv2.HoughCircles` detects the circular boundary of the dining plate $(x_c, y_c, r)$, yielding:
           $$A_{\text{plate}} = \pi \cdot r^2$$
        2. **Item Footprint Approximation:**
           Bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ enclose food heaps. 
           To account for rounded pile contours rather than sharp rectangles, an elliptical correction is applied:
           $$A_{\text{item}} = (x_{\max} - x_{\min}) \cdot (y_{\max} - y_{\min}) \times \frac{\pi}{4}$$
        3. **Food Density Scaling $(\rho)$:**
           Foods possess distinct mass densities relative to standard meal mixtures:
           - **Curries / Dals:** $\rho \approx 1.05$ (gravy and liquid lentils)
           - **Steamed Rice:** $\rho \approx 0.85$ (aerated cooked grains)
           - **Dry Roti / Flatbread:** $\rho \approx 0.65$ (thin rolled bread)
           - **Fresh Kachumber Salad:** $\rho \approx 0.35$ (low-density greens)
        4. **User Override Mechanism:**
           Allows deterministic manual adjustments to override algorithmic estimation whenever the user knows the exact gram intake.
        """
    )
