from simclr import SimCLR
from simclr.modules import get_resnet

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = nn.Dropout(0.5)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        x = nn.Dropout(0.5)(x)
        return x


class ResNet_18(nn.Module):

    def __init__(self, image_channels, num_classes):

        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

# Pretrained SimCLR model is loaded and only resnet part of it is returned
def load_pretrained_model(model_path, projection_dim=64, freeze=False):
    encoder = ResNet_18(3, 10)
    n_features = encoder.fc.in_features

    simclr_model = SimCLR(encoder, projection_dim, n_features)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    simclr_model.load_state_dict(torch.load(model_path, map_location=device.type))

    resnet_model = simclr_model.encoder

    if freeze:
        for param in resnet_model.parameters():
            param.requires_grad = False

    resnet_model.fc = ResNet_18(3, 10).fc

    resnet_model = resnet_model.to(device)

    return resnet_model


# A regular resnet18 model
def load_model(dropout=True):
    if dropout:
        model = ResNet_18(3, 10)
    else:
        model = get_resnet('resnet18', pretrained=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    return model
