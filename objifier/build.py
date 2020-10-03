import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from objifier.data_loader import get_loader, CIFAR10
from objifier.model import Classifier, efft
from objifier.visualize import visualize_single_image
from objifier.train import train_model, load_checkpoint
from objifier.log import setup_default_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, image, class_names, device):
    visualize_single_image(model, image, class_names, device)


def build(config):
    with open(config) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    setup_default_logging(
            log_path=data_dict['logs']
            if data_dict['logs'] else 'log.txt'
        )

    num_classes, class_names = (int(data_dict["nc"]), data_dict["names"])
    assert (
        len(class_names) == num_classes
    ), "Length of class names and number of classes do not match!"

    if data_dict['backbone'] == "efficientNet":
        model = efft(num_classes, weights=data_dict['efftlvl'])
    else:
        model = Classifier(num_classes)

    model = model.to(device)

    if data_dict['mode'] == "train":
        if data_dict['optimizer'] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=1e-2)

        else:
            optimizer = optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
            )

        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        loader, size = get_loader(data_dict['dataset_path']) \
            if data_dict['dataset_path'] else CIFAR10()

        train_model(
            model,
            loader,
            size,
            criterion,
            optimizer,
            scheduler,
            data_dict['epoch'],
            device,
            save_loc=data_dict['output'],
            load_model=data_dict['load'],
        )

    elif data_dict['mode'] == "predict":
        model = load_checkpoint(
            torch.load(
                "{}_best_weights.pth.tar".format(data_dict['output']),
                map_location=torch.device(device),
            ),
            model,
            optimizer,
        )
        predict(model, data_dict['image'], class_names, device)
