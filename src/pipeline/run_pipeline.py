from src.classification.prediction import predict
from src.segmentation.segment import segment_food
from src.segmentation.portion import estimation_portion
from src.nutrition.portion_to_grams import estimate_total_weight
from src.nutrition.standard_servings import STANDARD_SERVINGS
from src.nutrition.ingredient_weights import estimate_ingredient_weights
from src.nutrition.calculate_nutrition import calculate_nutrition


def pipeline(image_path):

    result = predict(image_path) # Predict dish

    if result["status"] == "unknown": # If dish is unknown
        return {
            "status": "rejected", # Return status
            "reason": "Dish not recognized with sufficient confidence", # Return reason
            "confidence": round(result["confidence"], 2), # Return confidence
            "top3_predictions": [
                {"dish": d, "confidence": round(c, 2)}
                for d, c in result["top3"]
            ] # Return top 3 predictions
        }

    dish = result["dish"] # Get dish
    confidence = result["confidence"] # Get confidence

    if dish not in STANDARD_SERVINGS: # If dish is not supported
        return {
            "status": "rejected", # Return status
            "reason": f"Dish '{dish}' not supported for nutrition estimation", # Return reason
            "confidence": round(confidence, 2) # Return confidence
        }

    _, mask = segment_food(image_path) # Segment food
    portion_ratio = estimation_portion(mask) # Estimate portion

    total_weight = estimate_total_weight(portion_ratio, dish) # Estimate total weight
    ingredient_weights = estimate_ingredient_weights(total_weight, dish) # Estimate ingredient weights
    nutrition = calculate_nutrition(ingredient_weights) # Calculate nutrition

    return {
        "status": "ok", # Return status
        "dish": dish, # Return dish
        "confidence": round(confidence, 2), #  Return confidence
        "portion_ratio": round(portion_ratio, 2), # Return portion ratio
        "estimated_weight_g": round(total_weight, 1), # Return estimated weight
        "nutrition": {
            "calories_kcal": round(nutrition["calories"], 1), # Return nutrition
            "carbs_g": round(nutrition["carbs"], 1), # Return nutrition
            "protein_g": round(nutrition["protein"], 1), # Return nutrition
            "fat_g": round(nutrition["fat"], 1), # Return nutrition
        }
    }


def format_result(result): # Format result
    if result["status"] == "rejected": # If dish is rejected
        output = f"""
Dish Rejected
Reason: {result['reason']}
Confidence: {result['confidence']}
""" # Return output
        if "top3_predictions" in result:
            output += "\nTop Predictions:\n"
            for p in result["top3_predictions"]:
                output += f"- {p['dish']} ({p['confidence']})\n"
        return output

    output = f"""
🍽 Dish: {result['dish']}
Confidence: {result['confidence']}

📏 Portion Estimation:
- Portion of plate: {int(result['portion_ratio'] * 100)}%
- Estimated quantity: {result['estimated_weight_g']} g

🔥 Nutrition:
- Calories: {result['nutrition']['calories_kcal']} kcal
- Carbs: {result['nutrition']['carbs_g']} g
- Protein: {result['nutrition']['protein_g']} g
- Fat: {result['nutrition']['fat_g']} g
"""
    return output


 
result = pipeline("/Users/sukesh/Desktop/NutriVision/dataset/test/Vada_pav/00000139_resized.png") # Run pipeline
print(format_result(result)) # Print result