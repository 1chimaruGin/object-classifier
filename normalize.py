import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


dataset = datasets.CIFAR10(
    root="data/", train=False, transform=transforms.ToTensor(), download=True
)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X**2]
    channel_sum, channel_square_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        channel_square_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channel_sum / num_batches
    std = (channel_square_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


mean, std = get_mean_std(data_loader)
print(mean, std)
