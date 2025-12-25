from src.classification.prediction import predict
from src.segmentation.segment import segment_food
from src.segmentation.portion import estimation_portion
from src.nutrition.portion_to_grams import estimate_total_weight
from src.nutrition.standard_servings import STANDARD_SERVINGS
from src.nutrition.ingredient_weights import estimate_ingredient_weights
from src.nutrition.calculate_nutrition import calculate_nutrition


def pipeline(image_path):

    result = predict(image_path) #get prediction

    if result["status"] == "unknown": #if dish not recognized
        return {
            "status": "rejected",
            "reason": "Dish not recognized with sufficient confidence",
            "confidence": round(result["confidence"], 2),
            "top3_predictions": [
                {"dish": d, "confidence": round(c, 2)}
                for d, c in result["top3"]
            ]
        }

    dish = result["dish"] #get dish
    confidence = result["confidence"] #get confidence


    if dish not in STANDARD_SERVINGS:
        raise ValueError(
            f"Dish '{dish}' not supported. Please use a known Indian dish." #if dish not supported
        )
    _, mask = segment_food(image_path) #segment food
    portion_ratio = estimation_portion(mask) #get portion
    
    total_weight = estimate_total_weight(portion_ratio, dish) #get total weight
    ingredient_weights = estimate_ingredient_weights(total_weight, dish) #get ingredient weights

    nutrition = calculate_nutrition(ingredient_weights) #get nutrition

    return {
        "dish": dish,
        "portion_ratio": round(portion_ratio, 2),
        "estimated_weight_g": round(total_weight, 1),
        "nutrition": {
            "calories_kcal": round(nutrition["calories"], 1),
            "carbs_g": round(nutrition["carbs"], 1),
            "protein_g": round(nutrition["protein"], 1),
            "fat_g": round(nutrition["fat"], 1)
        }
    }

def format_result(result):

    if result["status"] == "rejected":
        output = f"""
Dish Not Recognized

Reason: {result.get("reason", "Low confidence prediction")}
Confidence: {int(result["confidence"] * 100)}%

Top predictions:
"""
        for pred in result.get("top3_predictions", []):
            output += f" - {pred['dish'].replace('_', ' ').title()} ({int(pred['confidence'] * 100)}%)\n"

        output += "\nPlease try an image of a supported Indian dish."
        return output

    dish_name = result["dish"].replace("_", " ").title()

    output = f"""
🍽 Dish: {dish_name}

📊 Portion Estimation:
- Portion of plate: {int(result["portion_ratio"] * 100)}%
- Estimated quantity: {result["estimated_weight_g"]} g

🔥 Nutritional Information:
- Calories: {result["nutrition"]["calories_kcal"]} kcal
- Carbohydrates: {result["nutrition"]["carbs_g"]} g
- Protein: {result["nutrition"]["protein_g"]} g
- Fat: {result["nutrition"]["fat_g"]} g
"""
    return output

 
result = pipeline("/Users/sukesh/Desktop/NutriVision/Dataset_3/Indian Food Images/boondi/0ad99f040e.jpg")
print(format_result(result))