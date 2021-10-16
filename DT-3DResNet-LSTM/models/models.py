import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3DLSTM(nn.Module):
    def __init__(self, 
                backbone: str='slow_r50',
                hidden_dim: int=512, 
                num_clip: int=300,
                freeze: bool=False,
                pretrained: bool=True):
        super(CNN3DLSTM, self).__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.freeze = freeze
        self.pretrained = pretrained
        self.num_clip = num_clip
        self._build_model()

    def _build_model(self):
        resnet = torch.hub.load(
            "facebookresearch/pytorchvideo:main",
            self.backbone,
            pretrained=self.pretrained,
        )
        self.model = nn.Sequential(*list(resnet.children())[0][:-1])
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.adaptivePooling = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.conv_out = nn.Linear(2048, self.hidden_dim)
        self.lstm = nn.LSTM(input_size=self.num_clip, hidden_size=self.hidden_dim, num_layers=3)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim//2)
        self.fc2 = nn.Linear(in_features=self.hidden_dim//2, out_features=self.hidden_dim//4)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):

            x = self.resnet(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         
        
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x