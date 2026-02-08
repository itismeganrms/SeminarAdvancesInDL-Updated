import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import time
import datetime
import pylab as pyl
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1) 
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float() 
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()
    

# import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        # Remove channel dim if it's a single channel
        if targets.shape[1] == 1:
            targets = targets[:, 0]  # Shape: (B, D, H, W)

        # Flatten before one-hot encoding
        targets_onehot = F.one_hot(targets.long().view(-1), num_classes)  # Shape: (B*D*H*W, C)

        # Reshape back to (B, C, D, H, W)
        targets_onehot = targets_onehot.view(targets.shape[0], targets.shape[1], targets.shape[2], targets.shape[3], num_classes)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        union = torch.sum(probs + targets_onehot, dims) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou.mean()

