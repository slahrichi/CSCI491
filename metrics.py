"""
This file implements various metrics that will be used
during training.

A lot of the code is taken from
https://github.com/bohaohuang/mrs/blob/master/mrs_utils/metric_utils.py
"""

# Pytorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import f1_score
from torchmetrics import AveragePrecision
import numpy as np


def load(metric_name, device):
    # class weights argument is for CrossEntropyLoss, update as we need.
    if metric_name == "softiouloss":
        return SoftIoULoss(device)
    elif metric_name == "crossentropy":
        return nn.CrossEntropyLoss()
    elif metric_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif metric_name == 'MultiLabelSoftMarginLoss':
        return nn.MultiLabelSoftMarginLoss()
    elif metric_name == "iou":
        return IoU(device)
    elif metric_name == "f1":
        return F1(device)
    elif metric_name == "top1precision":
        return Top1Precision(device)
    elif metric_name == 'mean_average_precision':
        return AveragePrecision(task="multiclass", 
                                average='micro')
    
    else:
        return NotImplementedError("This loss/accuracy isn't set up yet.")


# -----------------------------------------------------------------------------------------------------
# loss:


class LossClass(nn.Module):
    """
    The base class of loss metrics, all loss metrics should inherit from this class
    This class contains a function that defines how loss is computed (def forward) and a loss tracker that keeps
    updating the loss within an epoch
    """

    def __init__(self):
        super(LossClass, self).__init__()
        self.loss = 0
        self.cnt = 0

    def forward(self, pred, lbl):
        raise NotImplementedError

    def update(self, loss, size):
        """
        Update the current loss tracker
        :param loss: the computed loss
        :param size: #elements in the batch
        :return:
        """
        self.loss += loss.item() * size
        self.cnt += 1

    def reset(self):
        """
        Reset the loss tracker
        :return:
        """
        self.loss = 0
        self.cnt = 0

    def get_loss(self):
        """
        Get mean loss within this epoch
        :return:
        """
        return self.loss / self.cnt


class SoftIoULoss(LossClass):
    """
    Soft IoU loss that is differentiable
    This code comes from https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
    and https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    Paper: http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
    """

    def __init__(self, device, delta=1e-12):
        super(SoftIoULoss, self).__init__()
        self.name = "softIoU"
        self.device = device
        self.delta = delta

    def forward(self, pred, lbl):
        num_classes = pred.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[lbl.cpu().squeeze(1)].to(self.device)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(pred)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[lbl.squeeze(1)].to(self.device)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(pred, dim=1)
        true_1_hot = true_1_hot.type(pred.type())
        dims = (0,) + tuple(range(2, lbl.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2.0 * intersection / (cardinality + self.delta)).mean()
        return 1 - dice_loss


# https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/jaccard.py
# MUltiClass IOU loss
## class IoU and mean IoU: https://arxiv.org/pdf/1805.06561v1.pdf page 5
class CrossEntropyLoss(LossClass):
    """
    Cross entropy loss function used in training
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.name = "xent"
        # class_weights = torch.tensor([float(a) for a in class_weights])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, lbl):
        if len(lbl.shape) == 4 and lbl.shape[1] == 1:
            lbl = lbl[:, 0, :, :]
        return self.criterion(pred, lbl)


# -----------------------------------------------------------------------------------------------------
# accuracy:


class IoU(nn.Module):
    """
    accuracy evaluation metric for segmentation,
    using  nn.Module
    """

    def __init__(self, device, num_classes=1):
        super(IoU, self).__init__()
        self.name = "IoU"
        self.device = device
        self.num_classes = 1
        # self.delta = thresh

    def forward(self, outputs, labels):
        if self.num_classes == 1:
            outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
            labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
            intersection = (
                (outputs & labels).float().sum((1, 2))
            )  # Will be zero if Truth=0 or Prediction=0
            union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0
            iou = torch.sum(intersection) / torch.sum(
                union
            )  # We smooth our devision to avoid 0/0
            return iou
        else:
            iou_list = list()
            present_iou_list = list()

            pred_labels = outputs.squeeze(1)
            labels = labels.squeeze(1)
            # Note: Following for loop goes from 0 to (num_classes-1)
            # and ignore_index is num_classes, thus ignore_index is
            # not considered in computation of IoU.
            for sem_class in range(self.num_classes):
                pred_inds = pred_labels == sem_class
                target_inds = labels == sem_class
                if target_inds.long().sum().item() == 0:
                    iou_now = 0
                else:
                    intersection_now = (pred_inds[target_inds]).long().sum().item()
                    union_now = (
                        pred_inds.long().sum().item()
                        + target_inds.long().sum().item()
                        - intersection_now
                    )
                    iou_now = float(intersection_now) / float(union_now)
                    present_iou_list.append(iou_now)
                iou_list.append(iou_now)
            return np.mean(present_iou_list)


class F1(nn.Module):
    def __init__(self, device):
        super(F1, self).__init__()
        self.num_classes = 1
        self.name = "F1"
        self.device = device
        self.average_mode = "weighted"

    def forward(self, outputs, labels):
        # input use argmax
        # outputs = torch.argmax(outputs,1).cpu().detach() # check output dimension
        return f1_score(
            labels.cpu().detach(), outputs.cpu().detach(), average=self.average_mode
        )


class Top1Precision(nn.Module):
    def __init__(self, device):
        super(Top1Precision, self).__init__()
        self.name = "Top1Precision"
        self.device = device
        self.num_classes = 1

    def forward(self, outputs, labels):
        from torchmetrics.classification import BinaryPrecision

        metric = BinaryPrecision(num_classes=self.num_classes, top_k=1)
        return metric(outputs.cpu().detach(), labels.cpu().detach())
