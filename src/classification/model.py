import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):

    model = models.resnet18(pretrained = True) #imports pretrained ResNet18 model

    for param in model.parameters():
        param.requires_grad = False #Freeze model parameters
    
    for param in model.layer4.parameters():
        param.requires_grad = True #Unfreeze last layer
    
    for param in model.fc.parameters():
        param.requires_grad = True #Unfreeze last layer

    model.fc = nn.Linear(model.fc.in_features, num_classes) #maps input features of CNN network to number of classes or dishes

    return model