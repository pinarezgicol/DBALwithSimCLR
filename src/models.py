from simclr import SimCLR
from simclr.modules import get_resnet


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):
    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 32,
        img_cols: int = 32,
        maxpool: int = 2,
    ):
        """
        Basic Architecture of CNN
        Attributes:
            num_filters: Number of filters, out channel for 1st and 2nd conv layers,
            kernel_size: Kernel size of convolution,
            dense_layer: Dense layer units,
            img_rows: Height of input image,
            img_cols: Width of input image,
            maxpool: Max pooling size
        """
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(
            num_filters
            * ((img_rows - 2 * kernel_size + 2) // 2)
            * ((img_cols - 2 * kernel_size + 2) // 2),
            dense_layer,
        )
        self.fc2 = nn.Linear(dense_layer, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        out = self.fc2(x)
        return out


# Pretrained SimCLR model is loaded and only resnet part of it is returned
def load_pretrained_model(classifier_model, model_path, projection_dim=64, freeze=False):
    encoder = get_resnet(classifier_model, pretrained=False)
    n_features = encoder.fc.in_features

    simclr_model = SimCLR(encoder, projection_dim, n_features)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    simclr_model.load_state_dict(torch.load(model_path, map_location=device.type))

    resnet_model = simclr_model.encoder

    if freeze:
        for param in resnet_model.parameters():
            param.requires_grad = False

    resnet_model.fc = get_resnet(classifier_model, pretrained=False).fc

    resnet_model = resnet_model.to(device)

    return resnet_model


# A regular resnet18 model
def load_model(classifier_model, image_size=32):
    if classifier_model == "resnet18":
        model = get_resnet(classifier_model, pretrained=False)
    else:
        model = ConvNN(img_rows=image_size, img_cols=image_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    return model