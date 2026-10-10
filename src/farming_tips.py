"""
farming_tips.py — Curated sustainable farming tips for AgriSmart AI.

Tips are organized by category and can optionally be filtered by crop name.
All advice is clearly labelled as general educational information.
"""

from __future__ import annotations

GENERAL_TIPS: dict[str, list[str]] = {
    "Soil Health & Testing": [
        "Test your soil at least once every two to three years to understand its nutrient levels, pH and organic matter content.",
        "Add compost or well-rotted manure to improve soil structure and increase beneficial microbial activity.",
        "Avoid compacting the soil by minimising heavy machinery passes, especially when the soil is wet.",
        "Maintain soil cover with mulch or cover crops to reduce erosion and retain moisture.",
        "Monitor soil pH regularly — most crops grow best between pH 6.0 and 7.5. Lime can raise pH; sulfur can lower it.",
    ],
    "Efficient Irrigation & Water Conservation": [
        "Use drip irrigation or micro-sprinklers to deliver water directly to the root zone and reduce evaporation losses.",
        "Water early in the morning to minimise evaporation and reduce the risk of leaf diseases.",
        "Install soil moisture sensors or use simple tensiometers to irrigate only when the crop actually needs water.",
        "Collect and store rainwater in farm ponds or tanks for use during dry periods.",
        "Check for leaks in irrigation pipes and emitters regularly — even small leaks waste significant water over a season.",
    ],
    "Crop Rotation & Diversity": [
        "Rotate crops each season to break pest and disease cycles and improve soil fertility naturally.",
        "Include legumes (beans, lentils, chickpeas) in the rotation — they fix atmospheric nitrogen and reduce fertilizer needs.",
        "Avoid planting the same crop family in the same field for at least two to three consecutive seasons.",
        "Growing diverse crops reduces the risk of total crop failure and supports soil biodiversity.",
        "Intercropping compatible crops (e.g., maize and beans) can increase total yield and reduce weed pressure.",
    ],
    "Responsible Fertilizer Use": [
        "Apply fertilizers based on a soil test result — over-fertilizing wastes money and can pollute water bodies.",
        "Split fertilizer applications into two or three doses rather than applying everything at once, to match crop demand.",
        "Use slow-release or organic fertilizers where possible to reduce nutrient runoff into streams and groundwater.",
        "Apply fertilizers when rain is not expected in the next 24 hours to prevent runoff losses.",
        "Keep records of what you apply, when and how much — this helps you improve efficiency each season.",
    ],
    "Sustainable Agricultural Practices": [
        "Integrated Pest Management (IPM) combines biological, cultural and chemical controls to manage pests with minimal environmental impact.",
        "Preserve beneficial insects such as bees and ladybirds by reducing broad-spectrum pesticide use and planting flowering borders.",
        "Keep field boundaries, hedgerows and tree windbreaks intact — they support wildlife and protect crops from wind erosion.",
        "Reduce tillage where possible. Minimum tillage preserves soil structure, retains moisture and reduces carbon release.",
        "Keep a farm diary recording planting dates, inputs, weather and yields — this data will help you make better decisions each year.",
    ],
}

# Per-crop specific tips (additive — shown alongside general tips)
CROP_SPECIFIC_TIPS: dict[str, list[str]] = {
    "rice": [
        "Rice grown in flooded paddies emits methane. Consider alternate wetting and drying (AWD) to reduce emissions and save water.",
        "Transplant seedlings at the right spacing (20×20 cm) to improve air circulation and reduce fungal diseases.",
    ],
    "maize": [
        "Maize is a heavy nitrogen feeder. Split N applications: a third at planting, the rest as a top-dress at the 6-leaf stage.",
        "Plant maize in rows oriented north–south to maximise sunlight interception.",
    ],
    "wheat": [
        "Timely sowing is critical for wheat. Late sowing significantly reduces yields due to heat stress at grain fill.",
        "Monitor for rust diseases (yellow, brown, black) and apply fungicide at first signs.",
    ],
    "chickpea": [
        "Chickpeas fix nitrogen effectively — inoculate seeds with Rhizobium before planting to maximise this benefit.",
        "Chickpeas are drought-tolerant; avoid over-irrigation which encourages root rot.",
    ],
    "cotton": [
        "Cotton requires well-drained soils. Waterlogged conditions cause root rot and reduce fibre quality.",
        "Monitor for bollworm and whitefly. Use pheromone traps for early detection.",
    ],
    "sugarcane": [
        "Sugarcane ratoons can be productive for 3–4 cycles. Maintain good drainage and fertilize ratoons carefully.",
        "Trash mulching after harvest conserves soil moisture and reduces weed growth.",
    ],
    "banana": [
        "Bananas are heavy potassium users. Ensure adequate K supply especially during bunch development.",
        "Remove and destroy infected leaves at the first sign of Sigatoka or Panama wilt disease.",
    ],
    "coffee": [
        "Coffee grows best under partial shade. Shade trees also help maintain soil moisture and reduce erosion.",
        "Selective hand-picking of ripe cherries produces higher-quality coffee than strip-picking.",
    ],
    "mango": [
        "Avoid heavy nitrogen after flowering — it promotes vegetative growth at the expense of fruit.",
        "Prune after harvest to open the canopy, improve air circulation and stimulate new flowering wood.",
    ],
    "grapes": [
        "Grapes need careful canopy management. Maintain leaf-to-fruit ratio for good sugar and colour development.",
        "Monitor soil moisture carefully during veraison (colour change) — water stress or excess at this stage affects quality.",
    ],
}


def get_tips_for_crop(crop: str | None = None) -> dict[str, list[str]]:
    """
    Return all general tips, plus crop-specific tips appended under a new category.

    Parameters
    ----------
    crop : crop name (lowercase), or None for general tips only.

    Returns
    -------
    dict category → list of tip strings.
    """
    result = dict(GENERAL_TIPS)

    if crop:
        specific = CROP_SPECIFIC_TIPS.get(crop.lower(), [])
        if specific:
            result[f"Tips for {crop.title()}"] = specific

    return result
