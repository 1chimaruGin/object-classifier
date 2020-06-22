import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from PIL import Image

plt.ion()


class Ant_Bee(object):
    def __init__(self, data_transforms, data_dir):
        super(Ant_Bee, self).__init__()
        self.data_transforms = data_transforms
        self.data_dir = data_dir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataloaders, self.dataset_size, self.class_names = self.load_data()

    def load_data(self):
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                          for x in ['train', 'val']}

        data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size=4, num_workers=4)
                        for x in ['train', 'val']}
        dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        return data_loaders, dataset_size, class_names

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epochs {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 20)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, pred = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(pred == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_size[phase]
                epoch_acc = running_corrects / self.dataset_size[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), 'best_model')
        return model

    @staticmethod
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.waitforbuttonpress()

    def visualize_model(self, model, num_images=6):
        was_training = model.training
        model.eval()
        image_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    image_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, image_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}, actual: {}'.format(self.class_names[preds[j]],
                                                                    self.class_names[labels[j]]))
                    self.imshow(inputs.cpu().data[j])

                    if image_so_far == num_images:
                        model.train(mode=was_training)
                        return model.train(mode=was_training)

    @staticmethod
    def model():
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        exp_optimizer = optim.SGD(model.fc.parameters(), lr=1e-3, momentum=0.9)
        exp_criterion = nn.CrossEntropyLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(exp_optimizer, step_size=7, gamma=0.1)

        return model, exp_criterion, exp_optimizer, exp_lr_scheduler

    def predict(self, model, image):
        loader = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = loader(Image.open(image)).float()
        img = Variable(img, requires_grad=True)
        output = model(img[None, ...])
        _, pred = torch.max(output, 1)

        ax = plt.subplot()
        ax.axis('off')
        ax.set_title('predicted: {}'.format(self.class_names[pred[0]]))
        self.imshow(img.cpu().data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set train or val')
    parser.add_argument('--train', default=False, type=bool, help='Model Training')
    parser.add_argument('--visualize', default=False, type=bool, help='Model Evaluation')
    parser.add_argument('--predict', default=None, type=str, help='Image path to predict')
    args = parser.parse_args()

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

    data_dire = 'data/hymenoptera_data'

    classifer = Ant_Bee(transform, data_dire)

    if args.train:
        model_ft, criterion_ft, optimizer_ft, scheduler_ft = classifer.model()
        model_ft = classifer.train_model(model_ft, criterion_ft, optimizer_ft, scheduler_ft, num_epochs=25)
    elif args.visualize:
        model_ft = classifer.model()[0]
        model_ft.load_state_dict(torch.load('best_model'))
        classifer.visualize_model(model_ft)

    elif args.predict:
        model_ft = classifer.model()[0]
        model_ft.load_state_dict(torch.load('best_model'))
        classifer.predict(model_ft, args.predict)
