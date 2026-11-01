"""
Rule-Based Recycling and Disposal Fallback Module for WasteWise.

Provides deterministic, expert-curated disposal, composting, recycling,
and reuse recommendations when the Gemini Generative AI API is offline,
unconfigured, or experiencing quota/network issues.
"""

from typing import Dict, Any, Optional


FALLBACK_DISCLAIMER = (
    "Note: Municipal recycling capabilities and composting regulations differ by municipality. "
    "Always consult your local waste management guidelines for specific local sorting rules."
)

RULE_BASED_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "Organic": {
        "title": "Organic Waste Guidance",
        "explanation": (
            "Organic waste consists of biodegradable organic matter--primarily food scraps, vegetable "
            "and fruit peels, coffee grounds, garden clippings, and non-treated natural materials. "
            "These decompose naturally through microbial action into nutrient-rich compost."
        ),
        "disposal_instructions": (
            "1. Dispose in your designated green or brown organic/compost bin.\n"
            "2. Do not use conventional non-biodegradable plastic bags; use certified compostable liners or newspaper.\n"
            "3. Drain excess liquids before disposal to prevent odors and leakage.\n"
            "4. Keep cooked meats, grease, and dairy separate if your local municipal system only accepts plant-based organics."
        ),
        "recycling_composting": (
            "- Home Composting: Excellent for backyard composting bins or vermiculture (worm composting).\n"
            "- Brown/Green Balance: Balance moist green materials (fruit scraps, greens) with dry carbon-rich brown materials "
            "(dry leaves, unbleached cardboard).\n"
            "- Municipal Composting: Industrial composting facilities process items at high heat, allowing them to accept "
            "tougher plant fibers and food-soiled paper napkins."
        ),
        "reuse_suggestions": (
            "- Vegetable broth: Collect clean vegetable trimmings (carrots, onions, celery) in a freezer bag to boil into homemade broth.\n"
            "- Citrus cleaner: Infuse citrus peels in white vinegar for 2 weeks to make a natural household cleaning spray.\n"
            "- Coffee grounds: Use spent grounds as natural soil amendment for acid-loving garden plants (tomatoes, blueberries)."
        ),
        "environmental_benefits": (
            "Diverting organic waste from anaerobic landfills prevents the formation of potent methane gas "
            "(a greenhouse gas over 25 times more potent than carbon dioxide) and produces natural humus "
            "that restores soil fertility and water retention."
        ),
        "safety_tip": (
            "Avoid letting food waste stagnate indoors; seal compost caddies to deter pests and fruit flies. "
            "Always wash hands thoroughly after handling decaying organic material."
        ),
    },
    "Recyclable": {
        "title": "Recyclable Waste Guidance",
        "explanation": (
            "Recyclable waste comprises manufactured dry materials--such as rigid plastics (PETE #1, HDPE #2), "
            "aluminum and tin cans, clean cardboard, paper, and glass containers. These items can be re-melted, "
            "pulped, or reprocessed into new commercial goods."
        ),
        "disposal_instructions": (
            "1. Empty and lightly rinse all containers to remove food residues (oil and sauce ruin paper/plastic batches).\n"
            "2. Flatten cardboard boxes and plastic jugs to conserve space in collection trucks.\n"
            "3. Keep recyclables loose inside the blue/yellow recycling cart; never bag them in plastic trash bags.\n"
            "4. Check plastic resin codes: clean bottles and jugs (#1 & #2) are widely accepted; film plastics require special drop-off."
        ),
        "recycling_composting": (
            "- Closed-Loop Recycling: Clean aluminum cans and glass bottles can be recycled infinitely without loss of quality.\n"
            "- Paper & Cardboard: Clean cardboard and paper fibers can be recycled 5 to 7 times into packaging and newsprint.\n"
            "- Contamination Warning: Greasy pizza boxes, broken window glass, and soiled foil cannot be mechanically recycled."
        ),
        "reuse_suggestions": (
            "- Glass jars: Wash glass jars to reuse as airtight pantry storage for bulk grains, spices, or stationary.\n"
            "- Cardboard boxes: Reuse sturdy boxes for package shipping, storage organization, or sheet mulching in gardens.\n"
            "- Plastic jugs & tubs: Cut plastic bottles to create indoor seedling planters or garden watering scoops."
        ),
        "environmental_benefits": (
            "Recycling aluminum saves 95% of the energy needed to manufacture new aluminum from bauxite ore. "
            "Recycling 1 ton of paper saves approximately 17 mature trees and 7,000 gallons of water while reducing landfill load."
        ),
        "safety_tip": (
            "Be cautious of sharp metal edges when rinsing food cans. Never place broken glassware, hazardous chemical containers, "
            "or lithium batteries into standard curbside recycling bins."
        ),
    },
}


def get_fallback_recommendations(
    category: str,
    confidence: float,
    item_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate structured, rule-based recycling and disposal recommendations.

    Args:
        category: Predicted category ('Organic' or 'Recyclable').
        confidence: Confidence score of the classifier in range [0.0, 1.0].
        item_description: Optional user-provided description of the item.

    Returns:
        Dict[str, Any]: Structured dictionary with verified guidance sections,
                       clearly identified as a rule-based fallback response.
    """
    # Normalize category string
    norm_cat = category.strip().capitalize()
    if norm_cat not in RULE_BASED_KNOWLEDGE:
        norm_cat = "Organic" if "organ" in norm_cat.lower() else "Recyclable"

    rule_data = RULE_BASED_KNOWLEDGE[norm_cat]
    conf_pct = round(confidence * 100, 1)

    # Contextualize explanation if an item description is provided
    desc_context = ""
    if item_description and item_description.strip():
        clean_desc = item_description.strip()
        desc_context = f"For item: **{clean_desc}** (classified as {norm_cat} with {conf_pct}% confidence).\n\n"

    # Assemble structured text
    formatted_text = (
        f"### {rule_data['title']} (Rule-Based Fallback)\n\n"
        f"{desc_context}"
        f"**1. Category Overview:**\n{rule_data['explanation']}\n\n"
        f"**2. Disposal Instructions:**\n{rule_data['disposal_instructions']}\n\n"
        f"**3. Composting / Recycling Recommendations:**\n{rule_data['recycling_composting']}\n\n"
        f"**4. Practical Reuse Ideas:**\n{rule_data['reuse_suggestions']}\n\n"
        f"**5. Environmental Impact:**\n{rule_data['environmental_benefits']}\n\n"
        f"**6. Safe Handling Tip:**\n{rule_data['safety_tip']}\n\n"
        f"> **Local Variance Notice:** {FALLBACK_DISCLAIMER}"
    )

    return {
        "status": "success",
        "source": "Rule-Based Fallback System (Offline)",
        "is_fallback": True,
        "category": norm_cat,
        "confidence": float(confidence),
        "confidence_percent": conf_pct,
        "item_description": item_description.strip() if item_description else None,
        "sections": {
            "explanation": rule_data["explanation"],
            "disposal_instructions": rule_data["disposal_instructions"],
            "recycling_recommendations": rule_data["recycling_composting"],
            "reuse_suggestions": rule_data["reuse_suggestions"],
            "environmental_benefits": rule_data["environmental_benefits"],
            "safety_tip": rule_data["safety_tip"],
        },
        "disclaimer": FALLBACK_DISCLAIMER,
        "formatted_text": formatted_text,
    }


if __name__ == "__main__":
    print("Testing Organic Fallback:")
    res_o = get_fallback_recommendations("Organic", 0.95, "apple core")
    print(res_o["formatted_text"])
    print("\n" + "=" * 50 + "\n")
    print("Testing Recyclable Fallback:")
    res_r = get_fallback_recommendations("Recyclable", 0.92, "empty soda can")
    print(res_r["formatted_text"])
