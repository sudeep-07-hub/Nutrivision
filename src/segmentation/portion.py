import numpy as np

def estimation_portion(mask):

    food_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]

    portion_ratio = food_pixels / total_pixels
    portion_ratio = max(0.05, min(portion_ratio, 0.7))

    return portion_ratio 