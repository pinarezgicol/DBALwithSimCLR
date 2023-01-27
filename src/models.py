import torch
from simclr import SimCLR
from simclr.modules import get_resnet


# Pretrained SimCLR model is loaded and only resnet part of it is returned
def load_pretrained_model(resnet, model_path, projection_dim=64, freeze=False):
    encoder = get_resnet(resnet, pretrained=False)
    n_features = encoder.fc.in_features

    simclr_model = SimCLR(encoder, projection_dim, n_features)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    simclr_model.load_state_dict(torch.load(model_path, map_location=device.type))

    resnet_model = simclr_model.encoder

    if freeze:
        for param in resnet_model.parameters():
            param.requires_grad = False

    resnet_model.fc = get_resnet(resnet, pretrained=False).fc

    resnet_model = resnet_model.to(device)

    return resnet_model


# A regular resnet18 model
def load_model(resnet):
    resnet_model = get_resnet(resnet, pretrained=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet_model = resnet_model.to(device)

    return resnet_model