from simclr import SimCLR
from simclr.modules import get_resnet


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):
    def __init__(self, dataset):
        if dataset == "CIFAR10":
            coef = 5
        else:
            coef = 21
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * coef * coef, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = nn.Dropout(0.5)(x)
        x = F.relu(self.fc2(x))
        x = nn.Dropout(0.25)(x)
        x = self.fc3(x)
        return x



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
def load_model(classifier_model, dataset="CIFAR10"):
    if classifier_model == "resnet18":
        model = get_resnet(classifier_model, pretrained=False)
    else:
        model = ConvNN(dataset=dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    return model


def append_dropout(model, rate=0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)
