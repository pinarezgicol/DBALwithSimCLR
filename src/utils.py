import os.path

import torchvision
from simclr.modules.transformations import TransformsSimCLR


def load_data(dataset, dataset_dir="./datasets"):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
    elif dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
    else:
        raise NotImplementedError

    return train_dataset, test_dataset
