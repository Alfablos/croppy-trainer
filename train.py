from enum import Enum
from typing import Callable
import os

import cv2
import torch
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.nn import Sequential, Sigmoid, Linear, ReLU, Dropout
from torch.optim import Adam, Optimizer
from torchvision.transforms import v2 as transformsV2
import torchvision.models as visionmodels
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import utils



IMAGE_SIZE: int = 512
DEFAULT_WEIGHTS = visionmodels.ResNet18_Weights.DEFAULT
BATCH_SIZE: int = 16


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class CroppyTrainer(nn.Module):
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


class DocumentSet(Dataset):
    def __init__(self, image_paths, labels, model_weigths):
        if len(image_paths) != len(labels):
            raise ValueError("images and labels have different lengths!")

        super().__init__()

        self.image_paths = image_paths
        self.labels = labels

        t = model_weigths.transforms()
        normalize = transformsV2.Normalize(mean=t.mean, std=t.std)
        # Data augmentation: since the model will deal with smartphone pictures (JPEG)
        # spoiling it with perfect PNGs would harm performamce
        # JPEG(quality=) will make sure the model is robust against
        # less-than-perfect pictures
        self.transform = transformsV2.Compose(
            [
                transformsV2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transformsV2.JPEG(quality=[50, 100]),
                transformsV2.ToImage(),
                transformsV2.ToDtype(dtype=torch.float32, scale=True),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        label = torch.tensor(self.labels[i], dtype=torch.float32)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label


# def pixel_error(preds, labels):
#     """Returns the mean error in calculating coriners positions"""
#     # denormalize coordinates
#     predicted = preds * IMAGE_SIZE
#     actual = labels * IMAGE_SIZE
#
#     # reshaping to (n_batch, 4 corners, )


def train(
    dataset: DocumentSet,
    mode_weights,
    dropout: float,
    device: Device,
    loss_function: Callable,
    learning_rate: float,
    epochs: int,
    verbose=False,
    out_file="./model.pth",
):
    if mode_weights is None:
        raise ValueError("`model_weights` must be specified!")
    if dropout is None:
        raise ValueError("A `dropout` must be specified!")
    if device is None:
        raise ValueError("A `device` must be specified!")
    if dataset is None:
        raise ValueError("A `dataset` must be specified!")
    if learning_rate is None:
        raise ValueError("A `learning_rate` must be specified!")
    if epochs is None:
        raise ValueError("A value for `epochs` must be specified!")
    if loss_function is None:
        raise ValueError("A `loss_function` must be specified!")

    try:
        os.stat(out_file)
        raise ValueError(f"{out_file} already exists.")
    except OSError:
        pass

    model = CroppyTrainer(dropout=dropout, resnet_weights=mode_weights).to(device.value)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_load = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_load:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            preds = model(images)
            loss = loss_function(preds, labels)

            loss.backward()

            optimizer.step()

            epoch_loss += loss

        if verbose:
            print(f"Epoch {epoch + 1}: loss={epoch_loss}")

    torch.save(model.state_dict(), out_file)


if __name__ == "__main__":



    weights = visionmodels.ResNet18_Weights.DEFAULT

    dataset = DocumentSet(image_paths=[], labels=[], model_weigths=weights)

    train(
        dataset=dataset,
        mode_weights=weights,
        device=Device.CUDA,
        dropout=0.3,
        loss_function=L1Loss,
        learning_rate=0.0001,
        epochs=1000,
        verbose=True,
        out_file="./model.pth",
    )
