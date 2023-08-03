import numpy as np
import torch
import torch.nn as nn

from isegm.utils import misc
import torch.nn.functional as F


class BalancedNormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, varepsilon=0.8, alpha=0.5, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(BalancedNormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._varepsilon = varepsilon
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult

    def forward(self, pred, label):
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))
        beta = (1 - pt) ** self._gamma

        qt = (pred * label)
        lt = ((1 - pred) * (1 - label))
        num_varpi = self._varepsilon * torch.pow(1 - qt, self._gamma + 1) + (1 - self._varepsilon) * torch.pow(1 - lt, self._gamma + 1)

        tol_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        avg_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        dif_kappa = tol_sum / (avg_sum + self._eps)

        if self._detach_delimeter:
            kappa = dif_kappa.detach()
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        fl = - beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = dif_kappa * fl  + num_varpi

        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))
