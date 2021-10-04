# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..builder import LOSSES
from .base import BaseWeightedLoss

@LOSSES.register_module()
class BCELoss(nn.Module):

    def forward(self, pred, target):
        loss = nn.BCELoss(pred, target)
        return loss