import cv2
import numpy as np

def segment_food(image_path):
    image = cv2.imread(image_path) #read image
    image = cv2.resize(image, (224,224)) #resize image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert image to hsv

    lower = np.array([0,30,30]) #create lower and upper bounds for mask
    upper = np.array([180,255,255])

    mask = cv2.inRange(hsv, lower, upper) #create mask

    kernal = np.ones((5,5), np.uint8) #kernel for morphological operations
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel = kernal) #close holes inside a segmented object
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel = kernal) #remove noise outside a segmented object

    return image, mask