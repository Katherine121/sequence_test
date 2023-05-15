import timm.models
import torchvision.models
from torch import nn


class vit(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        """
        Vision Transformer b-16.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
        """
        super(vit, self).__init__()
        self.backbone = timm.models.vit_small_patch16_224()
        in_features = 768
        self.backbone.head = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of vit.
        :param x: the provided input tensor.
        :return: current position preds.
        """
        x = self.backbone(x)
        return self.head_label(x)


class resnet18(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        """
        ResNet18.
        :param num_classes1: output dimension.
        :param num_classes2: output dimension.
        """
        super(resnet18, self).__init__()
        self.backbone = torchvision.models.resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_label = nn.Linear(in_features, num_classes1)

    def forward(self, x):
        """
        forward pass of resnet18.
        :param x: the provided input tensor.
        :return: current position preds.
        """
        x = self.backbone(x)
        return self.head_label(x)
