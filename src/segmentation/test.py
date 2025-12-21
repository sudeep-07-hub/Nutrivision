from segment import segment_food
from portion import estimation_portion
import cv2

image, mask = segment_food("/Users/sukesh/Desktop/NutriVision/Dataset/train/bhindi_masala/0ba897c635.jpg")
portion = estimation_portion(mask)

print(f"Estimated portion ratio: {portion:.2f}")

cv2.imshow("Food", image)
cv2.imshow("Mask", mask)
cv2.waitKey(0)