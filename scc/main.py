import os
import glob
import cv2
import imageio

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from ipywidgets import *
from PIL import Image
from matplotlib.pyplot import figure

from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torchio import AFFINE, DATA, PATH, TYPE, STEM
import torchmetrics
from torchmetrics.functional.detection import intersection_over_union
import torch.optim as optim

import pickle

from model import UNet_ASPP
from Loss import DiceLoss

# Specify the file path
file_path = 'Resize_augmented(350)/dataset.pkl'
# Open the file in read-binary mode and load the dataset
with open(file_path, 'rb') as file:
    augmented_dataset = pickle.load(file)
    

#Training
loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=4, shuffle=True)

def train_model(model, train_loader, num_epochs, optimizer, criterion, device):
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    for batch_idx, batch in enumerate(train_loader):
      inputs, labels = batch['ct']['data'][:, :, :, batch['ct']['data'].shape[3] // 2].to(device), batch['mask']['data'][:, :, :, batch['mask']['data'].shape[3] // 2].to(device)
      optimizer.zero_grad()
      if model.deep_supervision:
        out1, out2, out3, out4 = model(inputs)
        loss = criterion(out1, labels) + criterion(out2, labels) + criterion(out3, labels) + criterion(out4, labels)
      else:
        out = model(inputs)
        loss = criterion(out, labels)
      loss.backward()
      optimizer.step()
      with torch.no_grad():
        iou_val = intersection_over_union(out, labels)
        total_iou += iou_val.item()

      total_loss += loss.item()
      if batch % 100 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, IoU: {:.4f}'.format(epoch+1, num_epochs, batch+1, len(train_loader), loss.item(), iou_val.item()))
    average_loss = total_loss/len(train_loader)
    average_iou = total_iou/len(train_loader)
    print('Epoch [{}/{}], Average Loss: {:.4f}, Average IoU: {:.4f}'
          .format(epoch+1, num_epochs, average_loss, average_iou))
    
#Testing
def test_model(model, val_loader, criterion):
  model.eval()
  total_loss = 0.0
  total_iou = 0.0
  with torch.no_grad():
    for batch, (inputs, labels) in enumerate(val_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      if model.deep_supervision:
        out1, out2, out3, out4 = model(inputs)
        loss = criterion(out1, labels) + criterion(out2, labels) + criterion(out3, labels) + criterion(out4, labels)
      else:
        out = model(inputs)
        loss = criterion(out, labels)
      iou_val = intersection_over_union(out, labels)
      total_iou += iou_val.item()
      total_loss += loss.item()
    average_loss = total_loss/len(val_loader)
    average_iou = total_iou/len(val_loader)
    print('Test Loss: {:.4f}, Test IoU: {:.4f}'.format(average_loss, average_iou))
    
#train the model
input_channels = 1 #change
num_classes = 2 #change
model = UNet_ASPP(num_classes, input_channels, deep_supervision=False) #can change deep_supervision
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #can change optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
num_epochs = 10
train_model(model, loader, num_epochs, optimizer, criterion, device)