import os.path

import torchvision
from simclr.modules.transformations import TransformsSimCLR


def load_data(dataset, dataset_dir="./datasets", image_size=32):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=image_size).test_transform,
        )
    elif dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=image_size).test_transform,
        )
    else:
        raise NotImplementedError

    return train_dataset, test_dataset
