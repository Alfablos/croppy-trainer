import pandas as pd
from numpy.typing import NDArray
from sympy import Float
from dataclasses import dataclass
import torchvision.tv_tensors
from abc import ABCMeta, abstractmethod

import common
from architecture import Architecture
from typing import Any, Optional, List, Callable, Never, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from common import Device, Precision, LabelType, DEFAULT_WEIGHTS, DEFAULT_STORAGE_CLASS
import storage
import config


# Smartphone use a 0.75 (3:4) ratio
# ResNet reduces the input by a factor of 32 (12/16)

@dataclass
class DataRow():
    def __init__(
        self,
        image_path: str,
        label_path: str | None,
        tl_x: float | None,
        tl_y: float | None,
        tr_x: float | None,
        tr_y: float | None,
        br_x: float | None,
        br_y: float | None,
        bl_x: float | None,
        bl_y: float | None,
        corner_recess_percentage: float | None
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.tl_x = tl_x
        self.tl_y = tl_y
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.br_x = br_x
        self.br_y = br_y
        self.bl_x = bl_x
        self.bl_y = bl_y
        self.corner_recess_percentage = 0.0 if corner_recess_percentage else corner_recess_percentage
    
    def to_dict(self):
        return {
            'image_path': self.image_path,
            'label_path': self.label_path,
            'x1': self.tl_x,
            'y1': self.tl_y,
            'x2': self.tr_x,
            'y2': self.tr_y,
            'x3': self.br_x,
            'y3': self.br_y,
            'x4': self.bl_x,
            'y4': self.bl_y,
            'corners_recess_percentage': self.corner_recess_percentage
        }



@dataclass
class DataSource(ABCMeta):
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__init__")
            and callable(subclass.__init__)
            and hasattr(subclass, "check")
            and callable(subclass.check)
            and hasattr(subclass, "fetch")
            and callable(subclass.fetch)
            or NotImplemented
        )
    
    @abstractmethod
    def __init__(self, name: str, root_path: str, metadata_file: str | None, corner_recess_percentage):
        pass
    
    @abstractmethod
    def check(self) -> str | None: # return error as a string
        pass
    
    @abstractmethod
    def fetch(self, label_type: LabelType) -> List[DataRow]:
        """
            Transforms the dataset into a canonical data source that can be instructed to
            compute masks or coordinates
            Args:
                label_type: coordinates or mask
            Returns:
                List[RowData]: a list of RowData.
        """
        pass
    
    
        
    


class SmartDocDataset(Dataset):
    supported_img_formats = ["png", "jpg"]

    def __init__(
        self,
        store_path: str,
        architecture: Architecture,
        precision: Precision,
        train: bool,
        image_transforms: Optional[Callable] = None,
        label_transforms: Optional[Callable] = None,
        limit: Optional[int] = None,
    ):
        super().__init__()

        self.precision = precision
        self.store_path = store_path

        with storage.new_store(store_path, write=False) as store:
            self.len = len(store)
        # cannot use self.store to aboid pytorch forking the pointer to an open store
        self.store = None
        self.limit = limit if limit != 0 else None
        self.train = train

    def __len__(self):
        if self.limit is not None:
            # ensures that setting a limit of 100 on a 40 items DB
            # doesn't tell pytorch that there actually are 100 items
            return min(self.limit, self.len)
        else:
            return self.len

    def __getitem__(self, i):
        # self.store will stay around as long as the datastore instance is around
        # __getitem__ is called after pytorch forks (if num_workers > 0)
        if self.store is None:
            self.store = storage.new_store(self.store_path, write=False).__enter__()

        image, label = self.store.get(i)  # shape = (h, w, 3)
        # Tensorflow needs the underlying numpy array to be writable,
        # while data coming directly from arrow and LMDB is memory-mapped and immutable
        # copy creates a writable copy to system RAM
        image, label = image.copy(), label.copy()
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


def get_transforms(weights, device: Device, train=False):
    if device == Device.CPU:
        if train:
            return config.transforms["training"]["cpu"]
        else:
            return config.transforms["validation"]["cpu"]
    else:
        if weights is None:
            raise ValueError(
                "Weights must be included in the call to `get_transforms` if GPU is involved."
            )
        t = weights.transforms()
        if train:
            return config.transforms["training"]["gpu"](t)
        else:
            return config.transforms["validation"]["gpu"](t)


def current_train_transforms(
    input_path: str | tuple[str, int], output_path: str | None
):
    if isinstance(input_path, str):
        img_np = cv2.imread(input_path, cv2.IMREAD_COLOR_BGR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        with storage.new_store(input_path[0], write=False) as store:
            img_np, _ = store.get(input_path[1])
        output_path = f"./{input_path[1]}_transformed.jpg"

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


def get_from_store(idx: int, path: str):
    with storage.new_store(path, write=False) as store:
        image, label = store.get(idx)
    return image, label


def get_store_len(store_path: str):
    with storage.new_store(store_path, write=False) as store:
        data_length = len(store)
    return data_length


def get_store_metadata(store_path: str, key: str):
    with storage.new_store(store_path, write=False) as store:
        return store.get_metadata(key)


if __name__ == "__main__":
    # current_train_transforms(
    #     (
    #         "./hires_compact/training_data/data_resnet_training_1000x1024x768_compacted.lmdb",
    #         60,
    #     ),  # input_path='/home/antonio/Downloads/2026-01-24-15-52-49-829.jpg',
    #     output_path=None,
    # )

    db_path = "croppy_100x512x512_recess0/training_data/data_resnet_training_100x512x512.arrow"
    h, w = get_store_metadata(db_path, "h"), get_store_metadata(db_path, "w")
    print("Images h =", h)
    print("Images w =", w)
    idx = 5
    image, label = get_from_store(idx, db_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{idx}.jpg", image)
    print(label)
    print("Store __len__:", get_store_len(db_path))
