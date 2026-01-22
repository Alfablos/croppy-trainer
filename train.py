from typing import Optional
import tqdm
import typing
from torch.multiprocessing import cpu_count
from common import Precision
from architecture import Architecture
from data import SmartDocDataset
from enum import Enum
from typing import Callable
import os

import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.nn import Sequential, Sigmoid, Linear, ReLU, Dropout
from torch.optim import Adam, Optimizer
import torchvision.models as visionmodels
from torchvision.transforms import v2 as transformsV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import utils
import data
from common import Device


IMAGE_SIZE: int = 512
DEFAULT_WEIGHTS = visionmodels.ResNet18_Weights.DEFAULT
BATCH_SIZE: int = 64
# BATCH_SIZE: int = 16 took 7h 50min


class CroppyTrainerResnet(nn.Module):
    def __init__(self, resnet_weights, dropout=0.3):
        super().__init__()

        self.model = visionmodels.resnet18(weights=resnet_weights, progress=True)
        self.model.fc = Sequential(
            Dropout(p=dropout),
            # Adding not just one final layer but two, to give the model
            # enough parameters since data will be JPEG with degraded quality
            Linear(in_features=512, out_features=128),  # adds non-linearity
            ReLU(),
            # output = 8 because we have 8 coordinates:
            # the document page has 8 coordinates, not 2 like in bounding boxes of object detection because the camera won't be EXACTLY orthogonal)
            Linear(in_features=128, out_features=8),
            Sigmoid(),  # between 0 and 1 for each coordinate
        )
        self.weights = resnet_weights

    def forward(self, x):
        return self.model(x)


def validation_loss(model, loader, loss_fn, device: Device) -> float:
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device.value), labels.to(device.value)

            preds = model(images)
            val_loss += loss_fn(preds, labels).item()
        return val_loss / len(loader)


s_writer = SummaryWriter()

def train(
    out_file,
    train_dataloader: DataLoader,
    mode_weights,
    dropout: float,
    device: Device,
    loss_function: Callable,
    learning_rate: float,
    epochs: int,
    tensorboard_logdir: str,
    verbose=False,
    progress=False,
    with_tensorboard: bool = False
):
    if mode_weights is None:
        raise ValueError("`model_weights` must be specified!")
    if dropout is None:
        raise ValueError("A `dropout` must be specified!")
    if device is None:
        raise ValueError("A `device` must be specified!")
    if train_dataloader is None:
        raise ValueError("A `train_dataloader` must be specified!")
    if learning_rate is None:
        raise ValueError("A `learning_rate` must be specified!")
    if epochs is None:
        raise ValueError("A value for `epochs` must be specified!")
    if loss_function is None:
        raise ValueError("A `loss_function` must be specified!")
    if not out_file:
        out_file = f"model_{dropout}dropout_{learning_rate}lr_{epochs}epochs.pth"
    if not with_tensorboard and tensorboard_logdir:
        print("Error: A tensorboard log directory was specified but tensorboard is not enabled!")


    try:
        os.stat(out_file)
        raise ValueError(f"{out_file} already exists.")
    except OSError:
        pass

    model = CroppyTrainerResnet(dropout=dropout, resnet_weights=mode_weights).to(device.value)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()
    
    if with_tensorboard:
        url = utils.launch_tensorboard(tensorboard_logdir)
        print(f"Tensorboard is listening at {url}")
    
    if progress:
        epochs_iter = tqdm.trange(epochs, position=0)
    else:
        epochs_iter = range(epochs)
        
    for epoch in epochs_iter:
        epoch_loss = 0.0
        val_loss = 0.0

        sub_bar = tqdm.tqdm(total=len(train_dataloader), leave=True, position=1)
        try:
            for images, labels in train_dataloader:
                images, labels = images.to(device.value), labels.to(device.value)
    
                optimizer.zero_grad()
    
                preds = model(images)
    
                loss = loss_function(preds, labels)
    
                loss.backward()
    
                optimizer.step()
    
                epoch_loss += loss.item()
                if progress:
                    sub_bar.update(1)
        except KeyboardInterrupt:
            print("Aborting due to user interruption...")

        if verbose:
            print(f"Epoch {epoch + 1}: train_loss={epoch_loss}")
        
        s_writer.add_scalar("Train loss", epoch_loss, epoch)

    torch.save(model.state_dict(), out_file)


if __name__ == "__main__":
    weights = visionmodels.ResNet18_Weights.DEFAULT

    t = weights.transforms()
    normalize = transformsV2.Normalize(mean=t.mean, std=t.std)
    # Data augmentation: since the model will deal with smartphone pictures (JPEG)
    # spoiling it with perfect PNGs would harm performamce
    # JPEG(quality=) will make sure the model is robust against
    # less-than-perfect pictures
    train_transform = transformsV2.Compose(
        [
            transformsV2.ToImage(),
            # do NOT add this to preprocessing or the NN will overfit those specific low-quality artifacts and fail
            # to recognize those coming from smartphones
            transformsV2.JPEG(quality=[50, 100]),
            transformsV2.ToDtype(dtype=torch.float32, scale=True),
            normalize,
        ]
    )

    resnet_train_ds = SmartDocDataset(
        lmdb_path="./training_data/data_resnet_Float32.lmdb",
        architecture=Architecture.RESNET,
        precision=Precision.FP32,
        image_transform=train_transform,
        label_transform=None,
        # limit=128
    )

    dataloader = DataLoader(
        pin_memory=True,  # Using CUDA
        dataset=resnet_train_ds,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=14,
    )

    s_writer = SummaryWriter()
    
    train(
        train_dataloader=dataloader,
        mode_weights=weights,
        device=Device.CUDA,
        dropout=0.3,
        loss_function=L1Loss(),
        learning_rate=0.0001,
        # epochs=100,
        epochs=10,
        verbose=True,
        out_file="./model_fp32_batch64.pth",
    )
