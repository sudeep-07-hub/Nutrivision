from src.classification.prediction import predict
from src.segmentation.segment import segment_food
from src.segmentation.portion import estimation_portion
from src.nutrition.portion_to_grams import estimate_total_weight
from src.nutrition.standard_servings import STANDARD_SERVINGS
from src.nutrition.ingredient_weights import estimate_ingredient_weights
from src.nutrition.calculate_nutrition import calculate_nutrition


def pipeline(image_path):

    dish = predict(image_path)

    if dish not in STANDARD_SERVINGS:
        raise ValueError(
            f"Dish '{dish}' not supported. Please use a known Indian dish."
        )
    _, mask = segment_food(image_path)
    portion_ratio = estimation_portion(mask)
    
    total_weight = estimate_total_weight(portion_ratio, dish)
    ingredient_weights = estimate_ingredient_weights(total_weight, dish)

    nutrition = calculate_nutrition(ingredient_weights)

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
    dish_name = result["dish"].replace("_", " ").title()

    output = f"""
🍽️ Dish Identified: {dish_name}

📏 Portion Estimation:
- Portion of plate: {int(result["portion_ratio"] * 100)}%
- Estimated quantity: {result["estimated_weight_g"]} g

🔥 Nutritional Information:
- Calories: {result["nutrition"]["calories_kcal"]} kcal
- Carbohydrates: {result["nutrition"]["carbs_g"]} g
- Protein: {result["nutrition"]["protein_g"]} g
- Fat: {result["nutrition"]["fat_g"]} g
"""
    return output



result = pipeline("/Users/sukesh/Desktop/NutriVision/Dataset/test/kadai_panner/0cbb1c68c2.jpg")
print(format_result(result))
