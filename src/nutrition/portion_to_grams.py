from standard_servings import STANDARD_SERVINGS

def estimate_total_weight(portion_ratio,dish_name):

    standard_weight = STANDARD_SERVINGS[dish_name]
    estimated_weight = portion_ratio * standard_weight

    return estimated_weight

portion_ratio = 0.4
dish = "butter_chicken"

weight = estimate_total_weight(portion_ratio, dish)
print(f"The estimated weight of {dish} is {weight} grams.")
