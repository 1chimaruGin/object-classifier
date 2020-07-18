import torch
from torchvision import datasets, transforms
import os

transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_loader(root):

    dataset = {x: datasets.ImageFolder(os.path.join(root, x), transform=transform[x]) for x in ['train', 'val']}

    data_loader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=4, shuffle=(x=='train'), num_workers=0) for x in ['train', 'val']}

    dataset_size = {x: len(dataset[x]) for x in ['train', 'val']}

    return data_loader, dataset_size