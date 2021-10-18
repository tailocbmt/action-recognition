import torch
from torch.nn.modules import dropout
import pytorch_lightning as pl
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.resnet import create_resnet
import torch.nn.functional as F
from torch import nn

class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, num_subclip: int = 100, lr: float = 2e-4, **kwargs):
        """A classifier for finetuning pretrained video classification backbones from
        torchhub. We use the slow_r50 model here, but you can edit this class to
        use whatever backbone/head you'd like.
        Args:
            num_classes (int, optional): Number of output classes. Defaults to 11.
            lr (float, optional): The learning rate for the Adam optimizer. Defaults to 2e-4.
            freeze_backbone (bool, optional): Whether to freeze the backbone or leave it trainable. Defaults to True.
            pretrained (bool, optional): Use the pretrained model from torchhub. When False, we initialize the
            slow_r50 model from scratch. Defaults to True.
        All extra kwargs will be available via self.hparams.<name-of-arg>. These will also be saved as
        TensorBoard Hparams.
        """
        super().__init__()

        # Saves all kwargs to self.hparams. Use references to self.hparams.<var-name>, not the init args themselves.
        self.save_hyperparameters()

        # Build the model in separate function so its easier to override.
        self.model = self._build_model()

        # Metrics we will keep track of.
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.accuracy = {"train": self.train_acc, "val": self.val_acc}

    def _build_model(self):
        return create_resnet(model_num_class=self.hparams.num_classes)

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        if self.trainer.use_ddp:
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x: torch.Tensor):
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def shared_step(self, batch, mode: str):
        """This shared step handles both the training and validation steps to avoid
        re-writing the same code more than once. The given `mode` will change the name
        of the logged metrics.
        PyTorchVideo batches are dictionaries containing each modality or metadata of
        the batch collated video clips. Kinetics contains the following notable keys:
           {
               'video': <video_tensor>,
               'label': <action_label>,
           }
        - "video" is a Tensor of shape (batch, channels, time, height, Width)
        - "label" is a Tensor of shape (batch, 1)
        The PyTorchVideo models and transforms expect the same input shapes and
        dictionary structure making this function just a matter of unwrapping the dict and
        feeding it through the model/loss.
        Args:
            batch (dict): PyTorchVideo batch dictionary containing a single batch of data.
            mode (str): The type of step. Can be 'train', 'val', or 'test'.
        Returns:
            torch.Tensor: The loss for a single batch step.
        """

        outputs = self(batch["video"])

        loss = self.loss_fn(outputs, batch["label"])
        self.log(f"{mode}_loss", loss)

        proba = outputs.softmax(dim=1)
        preds = proba.argmax(dim=1)

        acc = self.accuracy[mode](preds, batch["label"])
        self.log(f"{mode}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the training epoch. It must
        return a loss that is used for loss.backwards() internally.
        """
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle. For this
        simple example it's mostly the same as the training loop but with a different
        metric name.
        """
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class ResNet3DLSTM(VideoClassificationLightningModule):
    def __init__(self,
            hidden_dim: int=256,
            lstm_dim: int=1, 
            sigmoid: bool=False,
            backbone='slow_r50',
            dropout: float=0.5,
            freeze_backbone: bool=False, 
            pretrained: bool=True, 
            **kwargs):
        super().__init__(
            lstm_dim=lstm_dim,
            hidden_dim=hidden_dim, 
            sigmoid=sigmoid,
            backbone=backbone,
            dropout=dropout,
            freeze_backbone=freeze_backbone, 
            pretrained=pretrained,
            **kwargs,
        )
        
    def _build_model(self):
        # The pretrained resnet model - we strip off its head to get the backbone.
        resnet = torch.hub.load(
            "facebookresearch/pytorchvideo:main",
            self.hparams.backbone,
            pretrained=self.hparams.pretrained,
        )
        self.backbone = nn.Sequential(*list(resnet.children())[0][:-1])

        # Freeze the backbone layers if specified.
        if self.hparams.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Create a new head we will train on top of the backbone.
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv_out = nn.Linear(in_features=2048, out_features=self.hparams.hidden_dim)

        if self.hparams.hidden_dim > 0:
            self.lstm = nn.LSTM(input_size=self.hparams.num_subclip, hidden_size=self.hparams.hidden_dim, num_layers=3, dropout=self.hparams.dropout)
            self.hidden1 = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim//2)
        if self.hparams.sigmoid:
            self.hidden2 = nn.Linear(self.hparams.hidden_dim//2, 1)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = self.conv_out(x)
        if self.hparams.hidden_dim > 0:
            x,_ = self.lstm(x)
            x = self.hidden1(x)
        if self.hparams.sigmoid:
            x = self.hidden2(x)
            x = F.log_softmax(x, dim=1)
        return x