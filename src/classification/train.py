import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import get_model

# check if GPU is available else use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Train_data_path = "/Users/sukesh/Desktop/NutriVision/Dataset/train"
Test_data_path = "/Users/sukesh/Desktop/NutriVision/Dataset/test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.RandomHorizontalFlip(),  # Imge augmentation
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # Normalize image
])

train_data = ImageFolder(root=Train_data_path, transform=transform)
test_data = ImageFolder(root=Test_data_path, transform=transform)

train_data_loder = DataLoader(train_data, batch_size=16, shuffle=True)
test_data_loder = DataLoader(test_data, batch_size=16)

model = get_model(num_classes=len(test_data.classes))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Optimizer

for epoch in range(10): #Number of epochs

    model.train() # Set model to training mode
    running_loss = 0.0 # Variable to store running loss

    for images, lables in train_data_loder:
        images, lables = images.to(DEVICE), lables.to(DEVICE) # Move images and lables to device

        optimizer.zero_grad() # Zero the gradients accumulated in the previous batch
        outputs = model(images)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        running_loss += loss.items()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_data_loder):.4f}")

torch.save(model.state_dict(), "dish_classifier.pth")

