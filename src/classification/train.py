import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import get_model
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# check if GPU is available else use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Train_data_path = "/Users/sukesh/Desktop/NutriVision/Dataset/train" # Path to training data
Test_data_path = "/Users/sukesh/Desktop/NutriVision/Dataset/test" # Path to testing data

transform = transforms.Compose([
    transforms.Resize((224, 224)), #Resize image
    transforms.RandomHorizontalFlip(), #Flip image
    transforms.RandomRotation(20), #Rotate image
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2
    ), #Augment image
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), #Crop image
    transforms.ToTensor(), #Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) #Normalize image
])

train_data = ImageFolder(root=Train_data_path, transform=transform) # Load training data
test_data = ImageFolder(root=Test_data_path, transform=transform) # Load testing data

train_data_loder = DataLoader(train_data, batch_size=16, shuffle=True) # Create dataloader
test_data_loder = DataLoader(test_data, batch_size=16) # Create dataloader

model = get_model(num_classes=len(test_data.classes)) # Get model
model.to(DEVICE) # Move model to device

criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr= 3e-5, weight_decay= 1e-4)  # Optimizer

train_losses = []
val_losses = []

# best_val_loss = float("inf")
# patience = 3          # how many epochs to wait
# counter = 0

for epoch in range(10):  # Number of epochs

    model.train()  # Set model to training mode
    running_loss = 0.0  # Variable to store running loss

    for images, lables in train_data_loder:
        images, lables = images.to(DEVICE), lables.to(
            DEVICE)  # Move images and lables to device

        optimizer.zero_grad()  # Zero the gradients accumulated in the previous batch
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, lables)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Add loss to running loss

    avg_train_loss = running_loss / len(train_data_loder) # Compute average loss
    train_losses.append(avg_train_loss) # Add average loss to list

    model.eval() # Set model to evaluation mode
    running_val_loss = 0.0 # Variable to store running loss

    with torch.no_grad():
        for images, labels in test_data_loder:
            images, labels = images.to(DEVICE), labels.to(DEVICE) # Move images and lables to device
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            running_val_loss += loss.item() # Add loss to running loss

    avg_val_loss = running_val_loss / len(test_data_loder) # Compute average loss
    val_losses.append(avg_val_loss) # Add average loss to list

    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     torch.save(model.state_dict(), "best_model.pth")
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Early stopping triggered")
    #         break

    print(
        f"Epoch [{epoch+1}/10] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

torch.save(model.state_dict(), "dish_classifier.pth")  # Save model

model.eval() # Set model to evaluation mode

all_pred = []
all_lables = []

with torch.no_grad():
    for images, lables in test_data_loder:

        images, lables = images.to(DEVICE), lables.to(DEVICE) # Move images and lables to device

        outputs = model(images) # Forward pass
        _, pred = torch.max(outputs, 1) # Get predictions

        all_pred.extend(pred.cpu().numpy()) # Add predictions to list
        all_lables.extend(lables.cpu().numpy()) # Add lables to list

accuracy = accuracy_score(all_lables, all_pred) # Compute accuracy
print(f"Accuracy: {accuracy:.4f}") # Print accuracy

confusion = confusion_matrix(all_lables, all_pred) # Compute confusion matrix
print(confusion)