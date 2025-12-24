import torch
from torchvision.transforms import transforms
from PIL import Image
from src.classification.model import get_model
import numpy as np

# check if GPU is available else use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["aloo_gobi", "aloo_methi", "bhindi_masala", "biryani", "butter_chicken","daal_puri","dal_makhani", "kadai_panner", "palak_panner", "panner_butter_masala"]  # List of classes

model = get_model(num_classes=len(CLASSES))  # Get model
model.load_state_dict(torch.load("dish_classifier.pth"))  # Load model
model.to(DEVICE)  # Move model to device
model.eval()  # Set model to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.topk(outputs, 3)

    return CLASSES[pred.item()]
