import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class LSTMHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels=512,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.2,
                 init_std=0.01,
                 clip_len: int=32,
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
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        self.lstm = nn.LSTM(input_size=self.clip_len, hidden_size=self.in_channels, num_layers=3, dropout=self.dropout_ratio, batch_first=True)
        self.fc_cls = nn.Linear(self.in_channels, self.in_channels//2)

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
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        print(x.shape)        
        x = torch.squeeze(x)
        x = x.permute(0, 2, 1)

        print(x.shape)
        
        out, _ = self.lstm(x) 

        if self.dropout is not None:
            x = self.dropout(x)
        print(x.shape)
        x = x.view(x.shape[0], -1)

        cls_score = self.fc_cls(x)
        
        return cls_score