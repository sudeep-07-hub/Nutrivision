import torch
from torchvision.transforms import transforms
from PIL import Image
from src.classification.model import get_model
import numpy as np

# check if GPU is available else use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.65

CLASSES = ["Besan_cheela", "Biryani", "Chapathi", "Chole_bature", "Dahl","Dhokla","Dosa", "Gulab_jamun", "Idli", "Jalebi","Pakoda","Pav_bhaji","Poha","Rolls","Samosa","Vada_pav"] #List of classes

model = get_model(num_classes=len(CLASSES))  # Get model
model.load_state_dict(torch.load("dish_classifier.pth"))  # Load model
model.to(DEVICE)  # Move model to device
model.eval()  # Set model to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)), #Resize image
    transforms.ToTensor(), #Convert image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) #Normalize image
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB") # Open image
    image = transform(image).unsqueeze(0).to(DEVICE) # Transform image

    with torch.no_grad():
        outputs = model(image) # Forward pass
        probs = torch.softmax(outputs, dim=1) # Compute probabilities

    top3_probs, top3_indices = torch.topk(probs, k=3, dim=1) # Get top 3 predictions

    top3_indices = top3_indices[0].tolist() #  Convert to list
    top3_probs = top3_probs[0].tolist() # Convert to list
    top3_dishes = [CLASSES[i] for i in top3_indices] # Convert indices to class names

    dish = top3_dishes[0] # Get dish
    confidence = top3_probs[0] # Get confidence

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "unknown", # Return status
            "confidence": confidence, # Return confidence
            "top3": list(zip(top3_dishes, top3_probs)) # Return top 3 predictions
        }

    return {
        "status": "ok", # Return status
        "dish": dish, # Return dish
        "confidence": confidence, # Return confidence
        "top3": list(zip(top3_dishes, top3_probs)) # Return top 3 predictions
    }
