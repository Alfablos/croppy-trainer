from PIL.Image import new
import torchvision.tv_tensors
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
from common import Device, Precision, DEFAULT_WEIGHTS
from utils import resize_img
from crawler import crawl


# Smartphone use a 0.75 (3:4) ratio
# ResNet reduces the input by a factor of 32 (12/16)



class SmartDocDataset(Dataset):
    supported_img_formats = ["png"]

    def __init__(
        self,
        lmdb_path: str,
        architecture: Architecture,
        precision: Precision,
        image_transforms: Optional[Callable] = None,
        label_transforms: Optional[Callable] = None,
        limit: Optional[int] = None,
    ):
        super().__init__()

        self.precision = precision
        self.lmdb_path = lmdb_path
        self.env = None  # opened on first __getitem__
        self.image_transform = image_transforms
        self.label_transform = label_transforms
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
            image: NDArray = pickle.loads(transaction.get(img_idx.encode("ascii")))# shape = (h, w, 3)
            label: NDArray = pickle.loads(transaction.get(lbl_idx.encode("ascii")))

        h, w = image.shape[:2]
        image_tvtensor = transformsV2.JPEG(quality=[50, 100])(image.reshape(3, 1024, 768))
        image_tvtensor = transformsV2.ToImage()(image) # shape is now (3, h, w)
        
        # For labels we need shape (4, 2): [[x1, y1], [x2, y2], ...]
        label_reshaped = label.reshape(-1, 2)
        
        # Original coordinates: TODO NO normalization in coords_from_mask
        original_coords = torch.from_numpy(label_reshaped) * torch.tensor([w, h], dtype=torch.float32) # [w, h] and not [h, w] because w => x, h = y
        
        # create Keypoints for label
        label_tvtensor = torchvision.tv_tensors.KeyPoints(
            original_coords,
            canvas_size=(h, w),
            dtype=torch.float32
        )
        
        return image_tvtensor, label_tvtensor
        
        # The cpu (16 threads) is not enough to preprocess
        # # FINALLY!
        # if self.image_transform:
        #     img_tv, label_tv = self.image_transform(image_tvtensor, label_tvtensor)
        
        # # NOW normalization can happen!
        # # img_tv.shape = (C, H, W)
        # new_h, new_w = img_tv.shape[-2:]
        
        # new_coords_norm = label_tv / torch.tensor([new_w, new_h], dtype=torch.float32) # w = x, h = y
        
        # new_coords_flat = new_coords_norm.flatten()
        
        # # transforms might have brought some corners outside the original size
        # final_label = torch.clamp(new_coords_flat, 0.0, 0.1)


        # return img_tv, final_label

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


def get_transforms(weights, precision: Precision, train=False):
    
    t = weights.transforms()
    pipeline = []
    
    if train:
        pipeline.extend([
            ## geometric ##
            transformsV2.RandomPerspective(distortion_scale=0.5, p=0.5), # p=0.5 => half of the dataset is affected
            # White fill to differ less from the background
            transformsV2.RandomRotation(degrees=(0, 100), fill=255), # let's try but I'm not sure... see https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_rotated_box_transforms.html
            transformsV2.RandomAffine(degrees=(0, 100), fill=255),
            transformsV2.ElasticTransform(alpha=30.0),
            
            ## photometric ##
            transformsV2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transformsV2.GaussianNoise(),
            transformsV2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # transformsV2.JPEG(quality=[50, 100]),  # CPU-bound
        ])
    
    pipeline.extend([
        # All the pipeline must be computed on UINT8, conversion at last
        transformsV2.ToDtype(torch.float32, scale=True),
        transformsV2.Normalize(mean=t.mean, std=t.std)
    ])
    
    return transformsV2.Compose(pipeline)


def current_train_transforms(input_path: str, output_path: str):
    img_np = cv2.imread(input_path, cv2.IMREAD_COLOR_RGB)
    img = transformsV2.ToImage()(img_np)
    transformation_pipeline = get_transforms(
        weights=DEFAULT_WEIGHTS,
        precision=Precision.FP32,
        train=True
    )
    img_tensor = transformation_pipeline(img) # shape = (3, H, W)
    
    # Reverting normalization
    mean = torch.tensor(DEFAULT_WEIGHTS.transforms().mean).view(3, 1, 1) # this shape can be {operator} element-wise with (3, 1, 1) (R, G, B have different means, so we need 3 numbers in the first dimension!)
    std = torch.tensor(DEFAULT_WEIGHTS.transforms().std).view(3, 1, 1)
    denormalized = img_tensor * std + mean
    denormalized = denormalized.clip(0, 1) # data augmentation might have pushed values above 1 or below 0
    
    # Pytorch speaks (C, H, W), we want (H, W, C)
    result_np = denormalized.permute(1, 2, 0).numpy()
    result_uint8 = (result_np * 255).astype(np.uint8) # retore values 0-255
    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, result_bgr)


if __name__ == '__main__':
    current_train_transforms(
        input_path='/home/antonio/Downloads/2026-01-24-15-52-49-829.jpg',
        output_path='./2026-01-24-15-52-49-829_transformed.jpg'
    )