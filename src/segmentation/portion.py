import numpy as np

def estimation_portion(mask):

    food_pixels = np.sum(mask > 0) # Count the number of non-zero pixels
    total_pixels = mask.shape[0] * mask.shape[1] # Get the total number of pixels

    portion_ratio = food_pixels / total_pixels # Calculate the portion ratio
    portion_ratio = max(0.05, min(portion_ratio, 0.7)) # Limit the portion ratio between 5% and 70%

    return portion_ratio 