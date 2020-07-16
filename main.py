import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data_loader import get_loader
from model import Classifier
from visualize import imshow, visualize_model, visualize_single_image
from train import train_model
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(model, image, device):
    visualize_single_image(model, image, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default=None, help='Mode')
    parser.add_argument('-im', '--image', type=str, default=None, help='Input Image')
    parser.add_argument('-d', '--root', type=str, default='data/hymenoptera_data', help='Dataset folder')
    parser.add_argument('-opt', '--optimizer', type=str, default='SGD', help='Optimizer')
    parser.add_argument('-epochs', '--epochs', type=int, default=25, help='Number of epochs')
    
    opt = parser.parse_args()
    loader, size, class_names = get_loader(opt.root)
    
    num_classes = len(class_names)

    model = Classifier(num_classes)
    model = model.to(device)

    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    else:
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    if opt.mode == 'train':
        train_model(model, loader, size, criterion, optimizer, scheduler, opt.epochs, device)
    
    elif opt.mode == 'predict':
        model = model.load_state_dict(torch.load('weights/best_model'))
        predict(model, opt.image, device)

    