from __future__ import print_function, division
from torch.nn.modules.loss import _assert_no_grad, _Loss
import torch.nn.functional as F
import torch

# define a customized loss function for future development
class WeightedBCELoss(_Loss):

    def __init__(self, size_average=True, reduce=True):
        super(WeightedBCELoss, self).__init__(size_average, reduce)

    def forward(self, input, target, weight):
        _assert_no_grad(target)
        return F.binary_cross_entropy(input, target, weight, self.size_average,
                                      self.reduce)

# Weighted binary cross entropy + Dice loss
class BCLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(BCLoss, self).__init__(size_average, reduce)

    def dice_loss(self, input, target):
        smooth = 1.
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        # size_average=True for the dice loss
        return loss / float(input.size()[0])

    def forward(self, input, target, weight):
        _assert_no_grad(target)
        """
        Weighted binary classification loss + Dice coefficient loss
        """
        loss1 = F.binary_cross_entropy(input, target, weight, self.size_average,
                                       self.reduce)
        loss2 = self.dice_loss(input, target)
        return loss1, loss2   

# Focal Loss
class FocalLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super().__init__(size_average, reduce)

    def focal_loss(self, input, target, weight):
        gamma = 2
        eps = 1e-7
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            wflat = weight[index].view(-1)

            iflat = iflat.clamp(eps, 1.0 - eps)
            fc_loss_pos = -1 * tflat * torch.log(iflat) * (1 - iflat) ** gamma
            fc_loss_neg = -1 * (1-tflat) * torch.log(1 - iflat) * (iflat) ** gamma
            fc_loss = fc_loss_pos + fc_loss_neg
            fc_loss = fc_loss * wflat # weighted focal loss

            loss += fc_loss.mean()
        
        return loss / float(input.size()[0])  

    def forward(self, input, target, weight):
        _assert_no_grad(target)
        """
        Weighted Focal Loss
        """
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        loss = self.focal_loss(input, target, weight)
        return loss   


# Focal Loss + Dice Loss
class BCLoss_focal(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super().__init__(size_average, reduce)

    def dice_loss(self, input, target):
        smooth = 1.
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        # size_average=True for the dice loss
        return loss / float(input.size()[0])

    def focal_loss(self, input, target, weight):
        gamma = 2
        eps = 1e-7
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            wflat = weight[index].view(-1)

            iflat = iflat.clamp(eps, 1.0 - eps)
            fc_loss_pos = -1 * tflat * torch.log(iflat) * (1 - iflat) ** gamma
            fc_loss_neg = -1 * (1-tflat) * torch.log(1 - iflat) * (iflat) ** gamma
            fc_loss = fc_loss_pos + fc_loss_neg
            fc_loss = fc_loss * wflat # weighted focal loss

            loss += fc_loss.mean()
        
        return loss / float(input.size()[0])  

    def forward(self, input, target, weight):
        _assert_no_grad(target)
        """
        Weighted binary classification loss + Dice coefficient loss
        """
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        loss1 = self.focal_loss(input, target, weight)
        loss2 = self.dice_loss(input, target)
        return loss1, loss2