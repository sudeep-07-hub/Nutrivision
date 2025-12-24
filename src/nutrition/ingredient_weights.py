from src.nutrition.recipe_priors import RECIPE_PRIORS


def estimate_ingredient_weights(total_weight, dish_name):
    prios = RECIPE_PRIORS[dish_name]
    ingredient_weights = {}
    for ingredient, fraction in prios.items():
        ingredient_weights[ingredient] = fraction * total_weight

    return ingredient_weights