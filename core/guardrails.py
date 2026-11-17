"""
Deterministic Threshold Guardrails & Grounded Advice Generator
Calculates hard macro and micronutrient alerts strictly in Python:
- Sodium > 800mg (Warning) / 1200mg (Critical)
- Added Sugar > 25g (Warning) / 36g (Critical)
- Saturated Fat > 15g (Warning) / 22g (Critical)
- Calorie Density > 800 kcal (Warning)
- Protein < 15g (Warning: Low Protein)
- Carbohydrate ratio > 65% of meal calories

Feeds verified flags into grounded advice with a prominent non-medical disclaimer.
"""

from typing import Dict, Any, List, Optional
import os

from config import GUARDRAILS, DISCLAIMER_TEXT, DEFAULT_GEMINI_MODEL


def evaluate_meal_guardrails(totals: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes hard macro/micronutrient alerts deterministically in Python.
    No hallucination: every warning is backed by arithmetic threshold validation.
    """
    alerts = []
    
    cal = totals.get("calories_kcal", 0.0)
    sodium = totals.get("sodium_mg", 0.0)
    sugar = totals.get("added_sugar_g", 0.0)
    sat_fat = totals.get("saturated_fat_g", 0.0)
    protein = totals.get("protein_g", 0.0)
    carbs = totals.get("carbohydrates_g", 0.0)
    fat = totals.get("fat_g", 0.0)
    fiber = totals.get("fiber_g", 0.0)

    # 1. Sodium Check
    s_cfg = GUARDRAILS["sodium_mg"]
    if sodium >= s_cfg["critical_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_CRITICAL_SODIUM",
            "severity": "CRITICAL",
            "nutrient": "Sodium",
            "value": round(sodium, 1),
            "unit": "mg",
            "threshold": s_cfg["critical_threshold"],
            "title": "Severe Sodium Spike Alert",
            "message": f"Sodium content is {round(sodium, 1)}mg (Limit: {s_cfg['critical_threshold']}mg). High risk for arterial pressure & water retention.",
            "guideline": s_cfg["guideline"]
        })
    elif sodium >= s_cfg["warning_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_HIGH_SODIUM",
            "severity": "WARNING",
            "nutrient": "Sodium",
            "value": round(sodium, 1),
            "unit": "mg",
            "threshold": s_cfg["warning_threshold"],
            "title": "High Sodium Flag",
            "message": f"Sodium is {round(sodium, 1)}mg, which exceeds the single-meal safe threshold of {s_cfg['warning_threshold']}mg.",
            "guideline": s_cfg["guideline"]
        })

    # 2. Added Sugar Check
    sug_cfg = GUARDRAILS["added_sugar_g"]
    if sugar >= sug_cfg["critical_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_CRITICAL_SUGAR",
            "severity": "CRITICAL",
            "nutrient": "Added Sugar",
            "value": round(sugar, 1),
            "unit": "g",
            "threshold": sug_cfg["critical_threshold"],
            "title": "Excessive Free Sugar Alert",
            "message": f"Added sugar is {round(sugar, 1)}g (Over whole-day recommended cap of {sug_cfg['warning_threshold']}g). Causes rapid glycemic spike.",
            "guideline": sug_cfg["guideline"]
        })
    elif sugar >= sug_cfg["warning_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_HIGH_SUGAR",
            "severity": "WARNING",
            "nutrient": "Added Sugar",
            "value": round(sugar, 1),
            "unit": "g",
            "threshold": sug_cfg["warning_threshold"],
            "title": "Elevated Sugar Warning",
            "message": f"Added sugar reached {round(sugar, 1)}g (Limit: {sug_cfg['warning_threshold']}g). Watch for dessert or sweetened sauces.",
            "guideline": sug_cfg["guideline"]
        })

    # 3. Saturated Fat Check
    fat_cfg = GUARDRAILS["saturated_fat_g"]
    if sat_fat >= fat_cfg["critical_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_CRITICAL_SAT_FAT",
            "severity": "CRITICAL",
            "nutrient": "Saturated Fat",
            "value": round(sat_fat, 1),
            "unit": "g",
            "threshold": fat_cfg["critical_threshold"],
            "title": "Heavy Saturated Fat Content",
            "message": f"Saturated fat is {round(sat_fat, 1)}g (Limit: {fat_cfg['warning_threshold']}g). Heavy reliance on ghee, butter, or fried items.",
            "guideline": fat_cfg["guideline"]
        })
    elif sat_fat >= fat_cfg["warning_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_HIGH_SAT_FAT",
            "severity": "WARNING",
            "nutrient": "Saturated Fat",
            "value": round(sat_fat, 1),
            "unit": "g",
            "threshold": fat_cfg["warning_threshold"],
            "title": "High Saturated Fat Flag",
            "message": f"Saturated fat is {round(sat_fat, 1)}g (Limit: {fat_cfg['warning_threshold']}g).",
            "guideline": fat_cfg["guideline"]
        })

    # 4. Total Calories Check
    c_cfg = GUARDRAILS["calories_kcal"]
    if cal >= c_cfg["critical_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_CRITICAL_CALORIES",
            "severity": "CRITICAL",
            "nutrient": "Energy",
            "value": round(cal, 1),
            "unit": "kcal",
            "threshold": c_cfg["critical_threshold"],
            "title": "Excessive Caloric Surplus",
            "message": f"Meal total is {round(cal, 1)} kcal. Substantially above single-meal energetic budget.",
            "guideline": c_cfg["guideline"]
        })
    elif cal >= c_cfg["warning_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_HIGH_CALORIES",
            "severity": "WARNING",
            "nutrient": "Energy",
            "value": round(cal, 1),
            "unit": "kcal",
            "threshold": c_cfg["warning_threshold"],
            "title": "High Calorie Meal",
            "message": f"Meal energy is {round(cal, 1)} kcal (Warning boundary: {c_cfg['warning_threshold']} kcal).",
            "guideline": c_cfg["guideline"]
        })

    # 5. Protein Inadequacy Check (only if meal is substantial > 250 kcal)
    p_cfg = GUARDRAILS["protein_g"]
    if cal >= 250.0 and protein < p_cfg["min_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_LOW_PROTEIN",
            "severity": "WARNING",
            "nutrient": "Protein",
            "value": round(protein, 1),
            "unit": "g",
            "threshold": p_cfg["min_threshold"],
            "title": "Low Protein Balance",
            "message": f"Only {round(protein, 1)}g protein in a {round(cal, 0)} kcal meal (Target: ≥{p_cfg['min_threshold']}g). May reduce fullness and satiety.",
            "guideline": p_cfg["guideline"]
        })

    # 6. Carbohydrate Caloric Ratio Check
    carb_kcal = carbs * 4.0
    carb_ratio = (carb_kcal / cal * 100.0) if cal > 0 else 0.0
    if cal >= 250.0 and carb_ratio > GUARDRAILS["carb_calories_percent"]["warning_threshold"]:
        alerts.append({
            "code": "GUARDRAIL_HIGH_CARB_RATIO",
            "severity": "INFO",
            "nutrient": "Carbohydrates",
            "value": round(carb_ratio, 1),
            "unit": "%",
            "threshold": GUARDRAILS["carb_calories_percent"]["warning_threshold"],
            "title": "High Carbohydrate Proportion",
            "message": f"{round(carb_ratio, 1)}% of total energy comes from carbohydrates (Reference cap: 65%).",
            "guideline": GUARDRAILS["carb_calories_percent"]["guideline"]
        })

    # Overall meal compliance status
    critical_count = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    warning_count = sum(1 for a in alerts if a["severity"] == "WARNING")

    if critical_count > 0:
        overall_status = "CRITICAL_ATTENTION_NEEDED"
    elif warning_count > 0:
        overall_status = "MODERATE_ADJUSTMENT_RECOMMENDED"
    else:
        overall_status = "WELL_BALANCED"

    # Macro distribution percentages
    protein_pct = round((protein * 4.0 / cal * 100.0) if cal > 0 else 0.0, 1)
    fat_pct = round((fat * 9.0 / cal * 100.0) if cal > 0 else 0.0, 1)
    carbs_pct = round((carbs * 4.0 / cal * 100.0) if cal > 0 else 0.0, 1)

    return {
        "status": overall_status,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "alerts": alerts,
        "macro_breakdown": {
            "protein_pct": protein_pct,
            "carbs_pct": carbs_pct,
            "fat_pct": fat_pct
        },
        "disclaimer": DISCLAIMER_TEXT
    }


def generate_grounded_advice(
    detected_items: List[Dict[str, Any]],
    totals: Dict[str, float],
    guardrail_results: Dict[str, Any],
    user_goal: str = "General Wellness",
    api_key: Optional[str] = None
) -> str:
    """
    Generates grounded lifestyle advice by feeding the deterministically verified
    alerts and macronutrient figures into Gemini (or a deterministic fallback engine).
    Stops the LLM from inventing or hallucinating medical diagnostics.
    """
    alerts = guardrail_results.get("alerts", [])
    status = guardrail_results.get("status", "WELL_BALANCED")
    macro_pct = guardrail_results.get("macro_breakdown", {})

    # Itemized summary text
    items_summary = ", ".join([f"{item.get('name', 'Food')} ({item.get('final_grams', 0)}g)" for item in detected_items])
    
    alert_lines = []
    for a in alerts:
        alert_lines.append(f"- [{a['severity']}] {a['title']}: {a['message']}")
    alerts_text = "\n".join(alert_lines) if alert_lines else "None! All primary nutrients fall within standard recommended meal bounds."

    # Try Gemini LLM if key is configured
    key = api_key or os.getenv("GOOGLE_API_KEY", "")
    if key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            
            prompt = f"""
You are a professional nutrition science advisor. Review the verified plate analysis below and provide concise, grounded, actionable dietary feedback.

STRICT CONSTRAINTS:
1. ONLY reference the verified deterministic nutritional calculations and guardrail alerts listed below.
2. DO NOT invent alternative calorie, sodium, sugar, or fat numbers.
3. DO NOT diagnose diseases or prescribe medications (hypertension, diabetes, etc.).
4. Tailor practical meal adjustments for the user's stated goal: "{user_goal}".
5. ALWAYS conclude with the mandatory disclaimer provided below.

PLATE DATA:
- Detected Items: {items_summary}
- Total Calories: {totals.get('calories_kcal', 0)} kcal
- Protein: {totals.get('protein_g', 0)}g ({macro_pct.get('protein_pct', 0)}% of kcal)
- Carbohydrates: {totals.get('carbohydrates_g', 0)}g ({macro_pct.get('carbs_pct', 0)}% of kcal)
- Fat: {totals.get('fat_g', 0)}g ({macro_pct.get('fat_pct', 0)}% of kcal)
- Saturated Fat: {totals.get('saturated_fat_g', 0)}g
- Sodium: {totals.get('sodium_mg', 0)}mg
- Added Sugar: {totals.get('added_sugar_g', 0)}g
- Fiber: {totals.get('fiber_g', 0)}g

DETERMINISTIC GUARDRAIL ALERTS:
{alerts_text}

MANDATORY DISCLAIMER:
{DISCLAIMER_TEXT}

Format your response in crisp Markdown with:
### 1. Meal Evaluation
### 2. Guardrail Breakdown & Risk Mitigation
### 3. Actionable Swaps & Portion Adjustments
### 4. Non-Medical Disclaimer
"""
            model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
            resp = model.generate_content(prompt)
            if resp and resp.text:
                return resp.text.strip()
        except Exception as e:
            # Fall through to deterministic advice engine
            pass

    # Deterministic Rule-Based Fallback Engine (Runs locally without external API)
    advice_sections = []
    advice_sections.append(f"### 🥗 Meal Quality Score: **{status.replace('_', ' ').title()}**")
    advice_sections.append(
        f"**Meal Composition:** {items_summary} totaling **{totals.get('calories_kcal', 0)} kcal** "
        f"(Protein: {totals.get('protein_g', 0)}g | Carbs: {totals.get('carbohydrates_g', 0)}g | Fat: {totals.get('fat_g', 0)}g)."
    )

    advice_sections.append("\n### 🛡️ Deterministic Guardrail Assessment")
    if alerts:
        for a in alerts:
            badge = "🔴" if a["severity"] == "CRITICAL" else ("🟡" if a["severity"] == "WARNING" else "🔵")
            advice_sections.append(f"{badge} **{a['title']}**: {a['message']}\n  *Guideline reference:* {a['guideline']}")
    else:
        advice_sections.append("✅ **All Clear:** Sodium, free sugars, saturated fat, and caloric density are within single-meal guidelines.")

    advice_sections.append("\n### 💡 Tailored Action Steps")
    if totals.get("protein_g", 0) < 15.0:
        advice_sections.append("- **Boost Protein:** Incorporate dal tadka, paneer bhurji, boiled eggs, or greek yogurt/curd to reach 20-25g protein for sustained satiety.")
    if totals.get("sodium_mg", 0) > 800.0:
        advice_sections.append("- **Reduce Sodium:** Limit pickled items, papad, packaged gravy mixes, or extra table salt. Use lemon juice and fresh herbs for zest.")
    if totals.get("saturated_fat_g", 0) > 15.0:
        advice_sections.append("- **Moderate Heavy Fats:** Substitute rich cream/butter bases with tomato-onion bases or strained curd; use non-stick dry roasting.")
    if totals.get("added_sugar_g", 0) > 25.0:
        advice_sections.append("- **Sugar Cap:** Swap sweets or sweetened beverages with whole fresh fruit or buttermilk (chaas).")
    if totals.get("fiber_g", 0) < 3.0:
        advice_sections.append("- **Increase Dietary Fiber:** Add a side bowl of cucumber-tomato kachumber salad or steamed greens to improve digestive speed.")

    advice_sections.append(f"\n---\n{DISCLAIMER_TEXT}")

    return "\n\n".join(advice_sections)
