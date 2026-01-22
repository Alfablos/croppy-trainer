from numpy.typing import NDArray
from cffi.cparser import lock
from architecture import Architecture
import csv
import json
import pickle
import lmdb
from typing import Any, Optional, List, Callable
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
from common import Device, Precision
from utils import resize_img
from crawler import crawl

# Smartphone use a 0.75 (3:4) ratio
# ResNet reduces the input by a factor of 32 (12/16)
H = 512
W = 384


class SmartDocDataset(Dataset):
    supported_img_formats = ["png"]

    def __init__(
        self,
        lmdb_path: str,
        architecture: Architecture,
        precision: Precision,
        image_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        limit: Optional[int] = None
    ):
        super().__init__()

        self.precision = precision
        self.lmdb_path = lmdb_path
        self.env = None  # opened on first __getitem__
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.limit = limit


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

        # Flow: 1. retrieve (cache or db); 2. transform (optional); 3. to tensor

        env = self._get_or_init_env()
        with env.begin(write=False) as transaction:
            # pickle.loads RETURNS A TUPLE FOR THE IMAGE!
            image: NDArray = pickle.loads(transaction.get(img_idx.encode("ascii")))[0]
            label: NDArray = pickle.loads(transaction.get(lbl_idx.encode("ascii")))
                

        if self.image_transform:
            image_tensor = self.image_transform(image)
        else:
            image_tensor = torch.tensor(image, dtype=self.precision.to_type_gpu())
            # Converts from (h, w, c) into (c, h, w), which pytorch expects.
            # https://docs.pytorch.org/vision/stable/transforms.html
            image_tensor = image_tensor.permute(2, 0, 1)
        
        if self.label_transform:
            label_tensor = self.label_transform(label)
        else:
            label_tensor = torch.tensor(label, dtype=self.precision.to_type_gpu())

        return image_tensor, label_tensor

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


if __name__ == "__main__":
    print(f"Pytorch version: {torch.__version__}")

    # crawl(
    #     root=Path(
    #         "/home/antonio/Downloads/extended_smartdoc_dataset/Extended Smartdoc dataset/train"
    #     ),
    #     images_ext="_in.png",
    #     labels_ext="_gt.png",
    #     output="./datatest.csv",
    #     compute_corners=True,
    #     check_normalization=True,
    #     verbose=True,
    #     precision=Precision.FP32,
    # )

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
            transformsV2.ToPureTensor()
        ]
    )

    resnet_train_ds = SmartDocDataset(
        lmdb_path="./training_data/data_resnet_Float32.lmdb",
        architecture=Architecture.RESNET,
        precision=Precision.FP32,
        image_transform=train_transform,
        label_transform=None
    )

    dataloader = DataLoader(
        pin_memory=True,  # Using CUDA
        dataset=resnet_train_ds,
        shuffle=True,
    )

    img, label = resnet_train_ds[0]
    print(type(img))
    exit(0)
    
    # img is a tensor (3, 512, 384), cv2 wants (512, 384, 3) and BGR
    img_np = img.permute(1, 2, 0).numpy()
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_sample.jpg", (img_bgr * 255).astype(np.uint8))
    print(label)

    # ### U-Net
    # train_transform = transformsV2.Compose(
    #     [
    #         transformsV2.ToImage(),
    #         # do NOT add this to preprocessing or the NN will overfit these low-quality artifacts and fail
    #         # to recognize those coming from smartphones
    #         transformsV2.JPEG(quality=[50, 100]),
    #         transformsV2.ToDtype(dtype=torch.float32, scale=True),
    #         normalize,
    #     ]
    # )
    # train_target_transform = transformsV2.Compose([
    #     transformsV2.ToImage(),
    #     # No scaling! Masks usually need to be 0 or 1 integers/floats, not normalized.
    #     transformsV2.ToDtype(torch.float32, scale=False),
    # ])
    # unet_train_ds = SmartDocDataset(
    #     lmdb_path='./training_data/data_unet_Float32.lmdb',
    #     architecture=Architecture.UNET,
    #     image_transform=train_transform,
    #     label_transform=train_target_transform,
    #     in_memory_cache=True
    # )
