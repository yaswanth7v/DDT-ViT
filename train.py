import torch
import torch.nn as nn
import random
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import timeit
from tqdm import tqdm
import math
import json
from model import *
from config import get_config

config = get_config()

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PATCHES = (config["IMG_SIZE"] // config["PATCH_SIZE"]) ** 2 
model = ViT(config["IN_CHANNELS"], 
            config["EMBED_DIM"], 
            config["PATCH_SIZE"], NUM_PATCHES, 
            config["NUM_HEADS"], 
            config["NUM_ENCODERS"], config["NUM_CLASSES"], 
            config["ACTIVATION"], config["DROPOUT"],
            config["HIDDEN_DIM"]).to(device)

img_size = config["IMG_SIZE"]
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize to a fixed size
    transforms.ToTensor(),          # Convert PIL Image to tensor
    transforms.Normalize(mean=config["mean"], std=config["std"])  # Normalize
])

train_dataset = ImageFolder(root='/kaggle/input/state-farm-distracted-driver-detection/imgs/train', transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = config["BATCH_SIZE"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=config["ADAM_BETAS"], lr=config["LEARNING_RATE"], weight_decay=config["ADAM_WEIGHT_DECAY"])

# Assume your model, train_loader, val_loader, criterion, optimizer, and device are defined

start = timeit.default_timer()
train_losses = []
val_losses = []

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=config["ADAM_BETAS"], lr=config["LEARNING_RATE"], weight_decay=config["ADAM_WEIGHT_DECAY"])

# Assume your model, train_loader, val_loader, criterion, optimizer, and device are defined

start = timeit.default_timer()
train_accuracy = []
val_accuracy = []
train_losses = []
val_losses = []

for epoch in tqdm(range(config["EPOCHS"]), position=0, leave=True):
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0
    
    for idx, (images, labels) in enumerate(tqdm(train_loader, position=0, leave=True)):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)  # labels should be 1D tensor of class indices
        loss.backward()
        optimizer.step()
        

        train_running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_labels.extend(labels.cpu().detach().numpy())
        train_preds.extend(predicted.cpu().detach().numpy())

    train_loss = train_running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(val_loader, position=0, leave=True)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().detach().numpy())
            val_preds.extend(predicted.cpu().detach().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_losses.append(val_loss)

    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    
    train_acc = sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels)
    train_accuracy.append(train_acc)
    print(f"Train Accuracy EPOCH {epoch+1}: {train_acc:.4f}")
    
    val_acc = sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels)
    print(f"Valid Accuracy EPOCH {epoch+1}: {val_acc:.4f}")
    val_accuracy.append(val_acc)
    
    print("-"*30)

stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")



