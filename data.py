from PIL.Image import new
import torchvision.tv_tensors
from numpy.typing import NDArray
from cffi.cparser import lock

import common
from architecture import Architecture
import csv
import json
import pickle
import lmdb
from typing import Any, Optional, List, Callable, Never
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transformsV2
import torchvision.models as visionmodels
from tqdm import tqdm

import utils
from common import Device, Precision, DEFAULT_WEIGHTS
from utils import assert_never
from crawler import crawl
import config


# Smartphone use a 0.75 (3:4) ratio
# ResNet reduces the input by a factor of 32 (12/16)


class SmartDocDataset(Dataset):
    supported_img_formats = ["png"]

    def __init__(
        self,
        lmdb_path: str,
        architecture: Architecture,
        precision: Precision,
        train: bool,
        image_transforms: Optional[Callable] = None,
        label_transforms: Optional[Callable] = None,
        limit: Optional[int] = None,
    ):
        super().__init__()

        self.precision = precision
        self.lmdb_path = lmdb_path
        self.env = None  # opened on first __getitem__
        self.limit = limit
        self.train = train

    def __len__(self):
        if self.limit is not None:
            return self.limit
        else:
            env = self._get_or_init_env()
            with env.begin(write=False) as transaction:
                return int.from_bytes(transaction.get("__len__".encode("ascii"), "big"))

    def __getitem__(self, i):
        img_idx = f"i{i}"
        lbl_idx = f"l{i}"
        # Note: reading back 'corners_recess_percentage', which was stored via struct
        # corners_recess_percentage = struct.unpack('f', transaction.get("my_key".encode("ascii")))[0]


        env = self._get_or_init_env()
        with env.begin(write=False) as transaction:
            image: NDArray = pickle.loads(
                transaction.get(img_idx.encode("ascii"))
            )  # shape = (h, w, 3)
            label: NDArray = pickle.loads(transaction.get(lbl_idx.encode("ascii")))
        h, w, _ = image.shape
        transforms = get_transforms(None, Device.CPU, self.train)
        image_tvtensor = transforms(image)  # shape is now (3, h, w)

        # For labels we need shape (4, 2): [[x1, y1], [x2, y2], ...]
        label_reshaped = label.reshape(-1, 2)

        # Original coordinates: they're not normalized by the image size. More straightforward
        original_coords = torch.from_numpy(label_reshaped).to(dtype=torch.float32)

        # create Keypoints for label
        label_tvtensor = torchvision.tv_tensors.KeyPoints(
            original_coords, canvas_size=(h, w), dtype=torch.float32
        )
        return image_tvtensor, label_tvtensor

    def _get_or_init_env(self):
        # The worker is FORKED by pytorch and if env is open at the time of the fork
        # the handle is copied to. The problem is that at that point
        # another process (forked) will try to access the first worker's memory:
        # segmentation fault!
        if self.env is None or self.env_pid != os.getpid():
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self.env_pid = os.getpid()
        return self.env


def get_transforms(weights, device: Device, train=False):
    if device == Device.CPU:
        if train:
            return config.train_cpu_transforms
        else:
            return config.val_cpu_transforms
    else:
        if weights is None:
            raise ValueError(
                "Weights must be included in the call to `get_transforms` if GPU is involved."
            )
        t = weights.transforms()
        if train:
            return config.train_gpu_transforms(t)
        else:
            return config.val_gpu_transforms(t)


def current_train_transforms(input_path: str | tuple[str, int], output_path: str | None):
    if isinstance(input_path, str):
        img_np = cv2.imread(input_path, cv2.IMREAD_COLOR_BGR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        key = f"i{input_path[1]}".encode("ascii")
        output_path = f"./{input_path[1]}_transformed.jpg"
        env = lmdb.open(
            input_path[0], readonly=True, lock=False, readahead=False, meminit=False
        )
        with env.begin(write=False) as t:
            img_np: NDArray = pickle.loads(t.get(key))  # shape = (h, w, 3)

    ## CPU ##: from data.py
    transforms = get_transforms(None, Device.CPU, train=True)
    image_tvtensor = transforms(img_np)  # shape is now (3, h, w)
    print(f"tensor shape after CPU transforms: {image_tvtensor.shape}")

    ## GPU ## from train.py
    gpu_transforms = get_transforms(common.DEFAULT_WEIGHTS, Device.CUDA, train=True).to(
        "cuda"
    )
    prepared_image = image_tvtensor.unsqueeze(dim=0).to("cuda")
    print(f"GPU: prepared image shape = {prepared_image.shape}")
    image = gpu_transforms(prepared_image)
    print(f"GPU: transformed image shape = {image.shape}")
    image = image.squeeze().to("cpu")  # still shape (3, h, w)
    print(f"GPU: squeezed image shape = {image.shape}")

    # Reverting normalization
    mean = torch.tensor(
        common.DEFAULT_WEIGHTS.transforms().mean
    ).view(
        3, 1, 1
    )  # this shape can be {operator} element-wise with (3, 1, 1) (R, G, B have different means, so we need 3 numbers in the first dimension!)
    std = torch.tensor(common.DEFAULT_WEIGHTS.transforms().std).view(3, 1, 1)
    denormalized = image * std + mean
    denormalized = denormalized.clip(
        0, 1
    )  # data augmentation might have pushed values above 1 or below 0

    # Pytorch speaks (C, H, W), we want (H, W, C)
    result_np = denormalized.permute(1, 2, 0).numpy()
    result_uint8 = (result_np * 255).astype(np.uint8)  # retore values 0-255
    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, result_bgr)


if __name__ == "__main__":
    current_train_transforms(
        ('./hires_compact/training_data/data_resnet_training_1000x1024x768_compacted.lmdb',
            60,
        ),  # input_path='/home/antonio/Downloads/2026-01-24-15-52-49-829.jpg',
        output_path=None,
    )
