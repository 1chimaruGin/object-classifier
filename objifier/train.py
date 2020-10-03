import torch
import tqdm
import time
import copy
import logging
from torch.utils.tensorboard import SummaryWriter
from log import setup_default_logging

writer = SummaryWriter("logs")
setup_default_logging()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    logging.info("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    logging.info("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


def train_model(
    model,
    loader,
    size,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    save_loc,
    load_model,
):
    step = 0
    if load_model:
        step = load_checkpoint(
            torch.load(
                "weights/{}_best_weights.pth.tar".format(save_loc),
                map_location=torch.device(device),
            ),
            model,
            optimizer,
        )

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            mloss = torch.zeros(1, device=device)  # mean losses
            running_loss = 0.0
            running_corrects = 0.0

            pbar = tqdm.tqdm(
                    enumerate(loader[phase]),
                    total=len(loader[phase])
                )
            for i, (inputs, labels) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    step += 1

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)

                mloss = (mloss * i + (loss.item() * inputs.size(0))) / (
                    i + 1
                )  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available() else 0
                )  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 3) % (
                    "Epoch: %g/%g" % (epoch + 1, num_epochs),
                    mem,
                    *mloss,
                    labels.shape[0],
                    inputs.shape[-1],
                )
                pbar.set_description(s)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / size[phase]
            epoch_acc = running_corrects / size[phase]

            if phase == "train":
                writer.add_scalar("loss", epoch_loss, epoch + 1)
                writer.add_scalar("acc", epoch_acc, epoch + 1)
            else:
                writer.add_scalar("val loss", epoch_loss, epoch + 1)
                writer.add_scalar("val acc", epoch_acc, epoch + 1)

            logging.info("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc)
            )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logging.info("Best val Acc: {:4f}".format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    save_checkpoint(
        checkpoint, filename="weights/{}_best_weights.pth.tar".format(save_loc)
    )
    return model
