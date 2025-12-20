# import torch
from torch.nn import nn
from torchvision import models

def get_model(num_classes):

    model = models.resnet18(pretrained = True) #imports pretrained ResNet18 model

    for parm in models.parameters():

        parm.requires_grad = False #freeze model parameters to avoid overfitting and training the network from scratch
    
    model.fc = nn.Linear(model.fc.in_features, num_classes) #maps input features of CNN network to number of classes or dishes

    return model