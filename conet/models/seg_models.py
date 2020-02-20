import segmentation_models_pytorch as smp
import torch
from torch import nn


class unet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        model = smp.Unet(encoder_name='resnet34',
                         encoder_weights='imagenet', classes=classes, activation=None)
        self.model = model

    def forward(self, x):
        return self.model(x)


class unet_xception(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        model = smp.Unet(encoder_name='xception',
                         encoder_weights='imagenet', classes=classes, activation=None)
        self.model = model

    def forward(self, x):
        return self.model(x)


class FpnSeResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        model = smp.FPN(encoder_name='se_resnet50',
                        encoder_weights='imagenet', classes=10, activation=None)
        self.model = model

    def forward(self, x):
        return self.model(x)


class Unedensenet121(nn.Module):
    def __init__(self):
        super().__init__()
        model = smp.Unet(encoder_name='densenet121',
                         encoder_weights='imagenet', classes=10, activation=None)
        self.model = model

    def forward(self, x):
        return self.model(x)
