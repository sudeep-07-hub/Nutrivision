from src.nutrition.standard_servings import STANDARD_SERVINGS

def estimate_total_weight(portion_ratio,dish_name):
    

    standard_weight = STANDARD_SERVINGS[dish_name] #get standard weight
    estimated_weight = portion_ratio * standard_weight #estimate weight

    return estimated_weight