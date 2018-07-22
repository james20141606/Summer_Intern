from __future__ import print_function, division
from torch.nn.modules.loss import _assert_no_grad, _Loss
import torch.nn.functional as F
import torch
import torch.nn as nn

# define a customized loss function for future development
class WeightedBCELoss(_Loss):

    def __init__(self, size_average=True, reduce=True):
        super(WeightedBCELoss, self).__init__(size_average, reduce)

    def forward(self, input, target, weight):
        _assert_no_grad(target)
        return F.binary_cross_entropy(input, target, weight, self.size_average,
                                      self.reduce)
# criterion = WeightedBCELoss()
# criterion(output, label, class_weight)

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = WeightedBCELoss()

    def soft_dice_coeff(self, input, target):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(target)
            j = torch.sum(input)
            intersection = torch.sum(target * input)
        else:
            i = target.sum(1).sum(1).sum(1)
            j = input.sum(1).sum(1).sum(1)
            intersection = (target * input).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, input, target):
        loss = 1 - self.soft_dice_coeff(input, target)
        return loss

    def __call__(self, input, target, weight):
        a = self.bce_loss(input, target, weight)
        b = self.soft_dice_loss(input, target)
        return a + b


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, input, target):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(target)
            j = torch.sum(input)
            intersection = torch.sum(target * input)
        else:
            i = target.sum(1).sum(1).sum(1)
            j = input.sum(1).sum(1).sum(1)
            intersection = (target * input).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, input, target):
        loss = 1 - self.soft_dice_coeff(input, target)
        return loss

    def __call__(self, input, target):
        return self.soft_dice_loss(input, target)
