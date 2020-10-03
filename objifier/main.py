import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from data_loader import get_loader, CIFAR10
from model import Classifier, efft
from visualize import visualize_single_image
from train import train_model, load_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, image, class_names, device):
    visualize_single_image(model, image, class_names, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default='train', help="Mode")
    parser.add_argument(
        "-im", "--image", type=str, default=None, help="Input Image"
    )
    parser.add_argument(
        "-d", "--root", type=str, default=None, help="Dataset folder"
    )
    parser.add_argument(
        "-opt", "--optimizer", type=str, default="SGD", help="Optimizer"
    )
    parser.add_argument(
        "-epochs", "--epochs", type=int, default=25, help="Number of epochs"
    )
    parser.add_argument(
        "-backbone", "--backbone", type=str, default="resent", help="ConvNet"
    )
    parser.add_argument(
        "-lvl", "--efftlevel", type=int, default=0, help="EfficientNet Level"
    )
    parser.add_argument(
        "-data", "--data", type=str, default="data/data.yaml", help="Yaml file"
    )
    parser.add_argument(
        '-0', '--output', type=str, default='output', help='Output location'
    )

    parser.add_argument("-load", "--load", default=False, type=bool)
    opt = parser.parse_args()

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)

    num_classes, class_names = (int(data_dict["nc"]), data_dict["names"])
    assert (
        len(class_names) == num_classes
    ), "Length of class names and number of classes do not match!"

    if opt.backbone == "efficientNet":
        backbone = opt.output+"/efficientNet"
        model = efft(num_classes, weights=opt.efftlevel)
    else:
        backbone = opt.output+"/resnet"
        model = Classifier(num_classes)
    model = model.to(device)

    if opt.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

    else:
        optimizer = optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    if opt.mode == "train":
        loader, size = get_loader(opt.root) \
            if opt.root else CIFAR10()
        train_model(
            model,
            loader,
            size,
            criterion,
            optimizer,
            scheduler,
            opt.epochs,
            device,
            save_loc=backbone,
            load_model=opt.load,
        )

    elif opt.mode == "predict":
        model = load_checkpoint(
            torch.load(
                "weights/{}_best_weights.pth.tar".format(backbone),
                map_location=torch.device(device),
            ),
            model,
            optimizer,
        )
        predict(model, opt.image, class_names, device)
