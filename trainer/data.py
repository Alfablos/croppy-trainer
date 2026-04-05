import torchvision.tv_tensors

import common
from architecture import Architecture
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import Precision
import storage
import config


# Smartphone use a 0.75 (3:4) ratio
# ResNet reduces the input by a factor of 32 (12/16)


class SmartDocDataset(Dataset):
    supported_img_formats = ["png", "jpg"]

    def __init__(
        self,
        store_path: str,
        architecture: Architecture,
        precision: Precision,
        cpu_transforms,
        limit: Optional[int] = None,
    ):
        super().__init__()

        self.precision = precision
        self.store_path = store_path
        self.cpu_transforms = cpu_transforms

        with storage.new_store(store_path, write=False) as store:
            self.len = len(store)
        # cannot use self.store to avoid pytorch forking the pointer to an open store
        self.store = None
        self.limit = limit if limit != 0 else None

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
        image_tvtensor = self.cpu_transforms(image)  # shape is now (3, h, w)

        # For labels we need shape (4, 2): [[x1, y1], [x2, y2], ...]
        label_reshaped = label.reshape(-1, 2)

        # Original coordinates: they're not normalized by the image size. More straightforward
        original_coords = torch.from_numpy(label_reshaped).to(dtype=torch.float32)

        # create Keypoints for label
        label_tvtensor = torchvision.tv_tensors.KeyPoints(
            original_coords, canvas_size=(h, w), dtype=torch.float32
        )
        return image_tvtensor, label_tvtensor


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

    ## CPU ##: from data.py — uses Extended transforms as example
    from torchvision.transforms import v2 as transformsV2
    cpu_transforms = transformsV2.Compose([
        transformsV2.ToImage(),
        transformsV2.JPEG(quality=[70, 100]),
        transformsV2.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.4),
        transformsV2.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2)),
    ])
    image_tvtensor = cpu_transforms(img_np)  # shape is now (3, h, w)
    print(f"tensor shape after CPU transforms: {image_tvtensor.shape}")

    ## GPU ## from config.py — use first training source's GPU transforms
    gpu_transforms = config.DATA_SOURCES["training"][0].train_gpu_transforms
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

    db_path = "croppy_512x512/training_data/data_resnet_training_512x512.arrow"
    h, w = get_store_metadata(db_path, "h"), get_store_metadata(db_path, "w")
    print("Images h =", h)
    print("Images w =", w)
    idx = 26000
    image, label = get_from_store(idx, db_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{idx}.jpg", image)
    print(label)
    print("Store __len__:", get_store_len(db_path))
