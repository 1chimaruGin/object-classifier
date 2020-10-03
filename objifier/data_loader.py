import torch
from torchvision import datasets, transforms
import os

transform = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4821, 0.4465], [0.2470, 0.2435, 0.2616]
            ),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4940, 0.4849, 0.4502], [0.2467, 0.2430, 0.2616]
            ),
        ]
    ),
}


def get_loader(root):

    dataset = {
        x: datasets.ImageFolder(os.path.join(root, x), transform=transform[x])
        for x in ["train", "val"]
    }

    data_loader = {
        x: torch.utils.data.DataLoader(
            dataset[x], batch_size=2, shuffle=(x == "train"), num_workers=0
        )
        for x in ["train", "val"]
    }

    dataset_size = {x: len(dataset[x]) for x in ["train", "val"]}

    return data_loader, dataset_size


def CIFAR10(root="data/"):
    dataset = {
        x: datasets.CIFAR10(
            root, train=(x == "train"), download=True, transform=transform[x]
        )
        for x in ["train", "val"]
    }

    data_loader = {
        x: torch.utils.data.DataLoader(
            dataset[x], batch_size=32, shuffle=(x == "train")
        )
        for x in ["train", "val"]
    }

    dataset_size = {x: len(dataset[x]) for x in ["train", "val"]}

    return data_loader, dataset_size
