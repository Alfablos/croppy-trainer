import csv
import json
import pickle
import lmdb
from typing import Any
from collections.abc import Callable
from jinja2.nodes import List
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transformsV2
import torchvision.models as visionmodels
from PIL import Image
from tqdm import tqdm

import utils
from common import Device
from utils import Precision, resize_img
from crawler import crawl

# Smartphone use a 0.75 (3:4) ratio
# ResNet reduces the input by a factor of 32 (12/16)
H = 512
W = 384


# class SmartDocDatasetResnet(Dataset):
#     supported_img_formats = ["png"]

#     def __init__(
#         self,
#         image_paths: list[str],
#         labels: np.ndarray,
#         target_h: int = H,
#         target_w: int = W,
#         weights=visionmodels.ResNet18_Weights.DEFAULT,
#         # transform: Callable[[Any], Any] | None = None
#         precompute_to: str | None = None,
#     ):
#         super().__init__()

#         self.image_paths = image_paths
#         self.labels = labels
#         self.target_h = target_h
#         self.target_w = target_w
#         self.weights = weights

#         if precompute_to:
#             # self.transform =
#             self.precompute(precompute_to)
#             self.computed = True
#         else:
#             t = weights.transforms()
#             normalize = transformsV2.Normalize(mean=t.mean, std=t.std)
#             # Data augmentation: since the model will deal with smartphone pictures (JPEG)
#             # spoiling it with perfect PNGs would harm performamce
#             # JPEG(quality=) will make sure the model is robust against
#             # less-than-perfect pictures
#             self.transform = transformsV2.Compose(
#                 [
#                     transformsV2.Resize((self.target_h, self.target_w)),
#                     transformsV2.JPEG(quality=[50, 100]),
#                     transformsV2.ToImage(),
#                     transformsV2.ToDtype(dtype=torch.float32, scale=True),
#                     normalize,
#                 ]
#             )
#             self.computed = False

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitems__(self, i):
#         image_path = self.image_paths[i]
#         label = torch.tensor(self.labels[i], dtype=torch.float32)

#         image = Image.open(image_path).convert("RGB")
#         image = self.transform(image)

#         return image, label


if __name__ == "__main__":
    crawl(
        root=Path(
            "/home/antonio/Downloads/extended_smartdoc_dataset/Extended Smartdoc dataset/train"
        ),
        images_ext="_in.png",
        labels_ext="_gt.png",
        output="./datatest.csv",
        compute_corners=True,
        check_normalization=True,
        verbose=True,
        precision=Precision.FP32,
    )
