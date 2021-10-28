import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class LSTMHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 clip_len: int=24,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.clip_len = clip_len
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool3d = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.avg_pool2d = nn.AdaptiveAvgPool2d((1, None))
        else:
            self.avg_pool3d = None
            self.avg_pool2d = None

        self.lstm = nn.LSTM(input_size=self.in_channels, hidden_size=self.in_channels//2, num_layers=3, dropout=self.dropout_ratio, batch_first=True)
        self.fc_cls = nn.Linear(clip_len*(self.in_channels//2), self.in_channels)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        
        if self.avg_pool3d is not None:
            x = self.avg_pool3d(x) 
        
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x) 
        if self.avg_pool2d is not None:
            x = self.avg_pool2d(x)

        x = x.view(x.shape[0], -1)
#         if self.dropout is not None:
#             x = self.dropout(x)
        
        return x