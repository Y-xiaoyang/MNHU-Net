import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def SoftIoULoss( pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1
        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)
        loss = 1 - loss.mean()
        return loss