#dice loss definition
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module): #just set this as the criterion, like criterion = DiceLoss()
  def __init__(self):
    super(DiceLoss, self).__init__()

  def forward(self, input, target):
    smooth=1e-5 #this is just to avoid divide by 0
    input_flat = input.view(-1) #flatten input
    target_flat = target.view(-1) #flatten true labels
    intersection = (input_flat*target_flat).sum()
    return 1 - ((2.0*intersection+smooth)/(input_flat.sum()+target_flat.sum()+smooth))