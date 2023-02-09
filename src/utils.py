import os.path

import torchvision
from simclr.modules.transformations import TransformsSimCLR


def getResultPath(dataset, network_type, acq_func, exp_iter, freeze=False):
    if freeze:
        return dataset + '_' + network_type + "_freeze" + "_acq_func" + acq_func + "_exp" + str(
            exp_iter) + ".npy"
    return dataset + '_' + network_type + "_acq_func" + acq_func + "_exp" + str(exp_iter) + ".npy"


def load_data(dataset, dataset_dir="./datasets"):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=96).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=96).test_transform,
        )
    elif dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=32).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=32).test_transform,
        )
    else:
        raise NotImplementedError

    return train_dataset, test_dataset
