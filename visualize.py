import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.waitforbuttonpress()


def visualize_model(model, loader, class_names, device, num_images=6):
    was_training = model.training
    model.eval()
    image_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                image_so_far += 1
                ax = plt.subplot(num_images // 2, 2, image_so_far)
                ax.axis("off")
                ax.set_title(
                    "predicted: {}, actual: {}".format(
                        class_names[preds[j]], class_names[labels[j]]
                    )
                )
                imshow(inputs.cpu().data[j])

                if image_so_far == num_images:
                    model.train(mode=was_training)
                    return model.train(mode=was_training)


def visualize_single_image(model, image, class_names, device):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = transform(Image.open(image)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    ax = plt.subplot()
    ax.axis("off")
    ax.set_title("predicted: {}".format(class_names[pred[0]]))
    imshow(img.squeeze(0).cpu().data)
