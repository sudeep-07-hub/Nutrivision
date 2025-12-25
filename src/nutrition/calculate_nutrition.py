from src.nutrition.ingredient_nutrition import INGREDIENT_NUTRITION

def calculate_nutrition(ingredient_weights):

    totals = {
        "calories" : 0,
        "carbs" : 0,
        "protein" : 0,
        "fat" : 0
    }

    for ingredient, grams in ingredient_weights.items():

        if ingredient not in INGREDIENT_NUTRITION: 
            continue

        nutr = INGREDIENT_NUTRITION[ingredient] # calories, carbs, protein, fat
        factor = grams / 100 # convert grams to kg

        totals["calories"] += nutr["calories"] * factor
        totals["carbs"] += nutr["carbs"] * factor 
        totals["protein"] += nutr["protein"] * factor
        totals["fat"] += nutr["fat"] * factor
    
    return totals 
