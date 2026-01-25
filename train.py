from pandas.core.algorithms import mode
from tensorboard.compat.tensorflow_stub.errors import UnimplementedError
from pathlib import Path
import torch.distributed.optim.post_localSGD_optimizer
import architecture
from typing import Optional
import tqdm
import typing
from torch.multiprocessing import cpu_count
from common import Precision, loss_from_str
from architecture import Architecture
from data import SmartDocDataset
from enum import Enum
from typing import Callable
import os
import json

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
from common import Device, DEFAULT_WEIGHTS




class CroppyNet(
    nn.Module
):  # TODO: make it architecture-agnostic: take in base_model and fully_connected_layers to set self.model.fc
    def __init__(
        self,
        weights,
        architecture: Architecture,
        precision: Precision,
        loss_fn: Callable,
        target_device: Device,
        images_height: int,
        images_width: int,
        learning_rate: float = 0.001,
        dropout=0.3,
    ):
        super().__init__()

        self.weights = weights
        self.architecture = architecture
        self.precision = precision
        self.loss_fn = loss_fn
        self.target_device = target_device
        self.images_height: int = images_height
        self.images_width = images_width
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = visionmodels.resnet18(weights=weights, progress=True)
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
        self.weights = weights

    def forward(self, x):
        return self.model(x)

    def loss_function(self):
        if isinstance(self.loss_fn, L1Loss):
           return "L1Loss"
        else:
            raise UnimplementedError
    
    def from_trained_config(config: dict, device: Device):
        model: CroppyNet = CroppyNet(
            weights=DEFAULT_WEIGHTS,
            loss_fn=loss_from_str(config['loss_fn']),
            architecture=Architecture.from_str(config['architecture']),
            target_device=device,
            images_height=config['images_height'],
            images_width=config['images_width'],
            precision=Precision.from_str(config['precision'])
        )
        model.load_state_dict(torch.load(config['weights_file'], weights_only=True))
        return model.to(device.value) # adds a validation step


@torch.no_grad()
def validation_data(model, loader, loss_fn, epoch: int, device: Device) -> float:
    model.eval()
    val_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device.value), labels.to(device.value)

        preds = model(images)
        val_loss += loss_fn(preds, labels).item()
    return val_loss / len(loader)


def train(
    model: CroppyNet,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader | None,
    epochs: int,
    out_dir: str,
    train_len, # only to append the information to filename and specs
    with_tensorboard: bool = False,
    verbose=False,
    progress=False,
):
    if train_dataloader is None:
        raise ValueError("A `train_dataloader` must be specified!")
    if epochs is None:
        raise ValueError("A value for `epochs` must be specified!")
    
    out_dir = out_dir.rstrip('/')
    if with_tensorboard:
     tensorboard_logdir = out_dir + '/runs'

    if verbose:
        print("Starting training with parameters:")
        for k, v in locals().items():
            print(f"==> {k}: {v}")
            print()


    out_name = (
        f"{model.architecture}_{model.dropout}dropout_{model.learning_rate}lr_{epochs}epochs_{model.precision}_{train_len}x{model.images_height}x{model.images_width}"
    )
    weights_file = out_dir + '/' + out_name + ".pth"
    spec_file = out_dir + '/' + out_name + ".json"
    if os.path.exists(weights_file):
        raise ValueError(f"{weights_file} already exists.")
    if os.path.exists(spec_file):
        raise ValueError(f"{spec_file} already exists.")
    
    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    if with_tensorboard:
        s_writer = SummaryWriter(
            log_dir=tensorboard_logdir
        )
        url = utils.launch_tensorboard(tensorboard_logdir)
        print(f"Tensorboard is listening at {url}")

    if progress:
        epochs_iter = tqdm.trange(epochs, position=0)
    else:
        epochs_iter = range(epochs)
    
    
    # THE MODEL MUST BE MOVED TO cTHE RIGHT DEVICE BEFORE INITIALIZING THE OPTIMIZER
    model = model.to(model.target_device.value)
    optimizer = Adam(model.parameters(), lr=model.learning_rate)

    for epoch in epochs_iter:
        model.train()

        if verbose:
            print(f"Starting epoch {epoch}.")

        cumulative_train_loss = 0.0

        if progress:
            sub_bar = tqdm.tqdm(total=len(train_dataloader), leave=True, position=1)
        try:
            for images, labels in train_dataloader:
                images, labels = images.to(model.target_device.value), labels.to(model.target_device.value)

                optimizer.zero_grad()
                preds = model(images)
                loss = model.loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                cumulative_train_loss += loss.item()

                if progress:
                    sub_bar.update(1)

            if progress:
                sub_bar.close()
        except KeyboardInterrupt:
            print("Aborting due to user interruption...")
            break

        if validation_dataloader:
            try:
                epoch_val_loss = validation_data(
                    model=model,
                    loader=validation_dataloader,
                    loss_fn=model.loss_fn,
                    epoch=epoch,
                    device=model.target_device,
                )
            except KeyboardInterrupt:
                print("Aborting due to user interruption...")
                break

        epoch_train_loss = cumulative_train_loss / len(train_dataloader)
        if verbose:
            if validation_dataloader is not None:
                print(
                    f"Epoch {epoch + 1}: train_loss={epoch_train_loss}, val_loss={epoch_val_loss}"
                )
            else:
                print(f"Epoch {epoch + 1}: train_loss={epoch_train_loss}")

        if with_tensorboard:
            board_payload = {'train': epoch_train_loss}
            if validation_dataloader:
                board_payload['validation'] = epoch_val_loss
            s_writer.add_scalars(
                main_tag="losses",
                tag_scalar_dict=board_payload,
                global_step=epoch + 1
            )
        
        # Saving checkpoint
        checkpoint = {
            'total_epochs': epochs,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss
        }
        torch.save(checkpoint, weights_file)
        
    if with_tensorboard:
        s_writer.close()


    torch.save(model.state_dict(), weights_file)
    with open(spec_file, mode="w") as f:
        f.write(
            json.dumps(
                {
                    "name": out_name,
                    "weights_file": weights_file,
                    "architecture": f"{model.architecture}",
                    "precision": f"{model.precision}",
                    "loss_fn": model.loss_function(),
                    "images_height": model.images_height,
                    "images_width": model.images_width,
                    "dropout": model.dropout,
                    "learning_rate": model.learning_rate,
                    "epochs": epochs,
                    "train_length": train_len
                }
            )
        )
