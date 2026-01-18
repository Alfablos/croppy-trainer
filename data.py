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

# Smartphone use a 0.75 (3:4) ratio
# ResNet reduces the input by a factor of 32 (12/16)
H = 512
W = 384




def precompute(
    self, path: str,
    image_paths: list[str],
    labels: list[str],
    computed_labels: np.ndarray,
    target_h: int,
    target_w: int,
    with_coords: bool = True,
):
    """
    Performs a resize and stores resized images in a LMDB Database at :path
    """
    if not path.endswith(".lmdb"):
        print(f"Warning: saving a LMDB file without the `.lmdb` extension.")
    if os.path.exists(path):
        raise FileExistsError(
            f"The path '{path}' already exists. Refusing to continue."
        )
    single_image_size: int = target_h * target_w * 3  # RGB
    total_map_size: int = int(
        len(self) * single_image_size * 1.2
    )  # 1.2 is a safety margin
    print(f"Allocating {total_map_size / (1024**3)} GB for the lmdb store.")

    # initialize lmdb at `path`
    env = lmdb.open(path, total_map_size)

    # Write each example in the db after converting it to RGB
    print(f"Creating LMDB store at {path}.")
    commit_freq = 100
    csv_f = open(f"{path}_index.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["index", "path"] + [f"c{k}" for k in range(8)]) # index, path, + labels (8 coords of the corners)
    csv_index = 0 # NOT updated when images fail to convert
    have_coords = self.labels.shape[1] == 8
    
    transaction = env.begin(write=True)
    try:
        for i, path in enumerate(
            tqdm(self.image_paths, position=0, desc="Saving precomputed examples")
        ):
            imdata = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
            if imdata is None:
                print(f"Couldn't read image at {path}.")
                continue
            
            resized = resize_img(imdata, self.target_h, self.target_w)
            if resized is None:
                print(f"Couldn't resize image at {path}.")
            # resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            bytes = pickle.dumps(resized)
            key = str(i).encode("ascii")
            transaction.put(key, bytes)
            
            exit(0)
            
            # i=244 => index[44]
            index[i % commit_freq] = os.path.basename(path)

            # Commit every 100 resize operations to save memory
            if ((i + 1) % commit_freq == 0) or (i == len(self) - 1): # every 100 iterations and on the last one
                print(f"Saving checkpoint to {checkpoint}")
                transaction.commit()
                transaction = env.begin(write=True)
                pd.DataFrame(index, columns=["path"]).to_csv(
                    checkpoint,
                    mode='a',
                    header=i + 1 == commit_freq, # only the first time,
                    index=True
                )

        transaction.put(b"__len__", str(len(self)).encode("ascii"))
        transaction.commit()
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        env.close()
    
    self.computed = True
    print("Precomputation complete.")



class SmartDocDatasetResnet(Dataset):
    supported_img_formats = ["png"]

    def __init__(
        self,
        image_paths: list[str],
        labels: np.ndarray,
        target_h: int = H,
        target_w: int = W,
        weights=visionmodels.ResNet18_Weights.DEFAULT,
        # transform: Callable[[Any], Any] | None = None
        precompute_to: str | None = None,
    ):
        super().__init__()

        self.image_paths = image_paths
        self.labels = labels
        self.target_h = target_h
        self.target_w = target_w
        self.weights = weights

        if precompute_to:
            # self.transform =
            self.precompute(precompute_to)
            self.computed = True
        else:
            t = weights.transforms()
            normalize = transformsV2.Normalize(mean=t.mean, std=t.std)
            # Data augmentation: since the model will deal with smartphone pictures (JPEG)
            # spoiling it with perfect PNGs would harm performamce
            # JPEG(quality=) will make sure the model is robust against
            # less-than-perfect pictures
            self.transform = transformsV2.Compose(
                [
                    transformsV2.Resize((self.target_h, self.target_w)),
                    transformsV2.JPEG(quality=[50, 100]),
                    transformsV2.ToImage(),
                    transformsV2.ToDtype(dtype=torch.float32, scale=True),
                    normalize,
                ]
            )
            self.computed = False


    def __len__(self):
        return len(self.image_paths)

    def __getitems__(self, i):
        image_path = self.image_paths[i]
        label = torch.tensor(self.labels[i], dtype=torch.float32)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label


def build_csv(
    root: Path,
    output: str,
    images_ext: str,
    labels_ext: str,
    precision: Precision,
    compute_corners=True,
    check_normalization=True,
    verbose=False,
):
    if images_ext is None:
        raise ValueError(
            "Please, provide the extension for images. For example `.png` or `_img.png`"
        )
    if labels_ext is None:
        raise ValueError(
            "Please, provide the extension for labels. For example `.png` or `_lbl.png`"
        )

    if not os.path.exists(root):
        raise ValueError(f"Root path {root} does not exist.")

    if os.path.exists(output):
        print(f"Output file {output} esists. Refusing to continue.")
        exit(2)

    images = sorted(list(root.glob("**/*" + images_ext)))
    labels = sorted(list(root.glob("**/*" + labels_ext)))
    n_images, n_labels = len(images), len(labels)
    assert n_images == n_labels, "Images and labels differ in number."

    if verbose:
        print(f"Found {len(images)} images with labels.")

    rows = []
    save_chunk_size = 100

    for image, label in tqdm(zip(images, labels, strict=True), total=len(images)):
        try:
            row = {"image_path": image, "label_path": label}

            if compute_corners:
                f = cv2.imread(filename=str(label), flags=cv2.IMREAD_GRAYSCALE)

                if precision != Precision.UINT8:  # only normalize if not on uint8
                    mask = np.divide(f.astype(precision.to_type_cpu()), 255.0)
                else:
                    mask = f.astype(precision.to_type_cpu())

                coords = utils.coords_from_segmentation_mask(
                    mask, precision, device=Device.CPU
                ).numpy()
                fields = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
                for coord_name, value in zip(fields, coords):
                    if (
                        value > 1
                        and precision != Precision.UINT8
                        and check_normalization
                    ):
                        print(
                            f"Warning: label {label} has {coord_name} coordinate with a value > 1 ({value}): {coords}\nYou may want to check your normalization algorithm.",
                            file=sys.stderr,
                        )

                    row[coord_name] = value
            rows.append(row)

            if len(rows) >= save_chunk_size:
                if verbose:
                    print(f"Saving checkpoint to {output}")
                save_to_csv(rows, str(output))
                rows = []

        except KeyboardInterrupt:
            if rows:
                print("Saving remaining rows after user interruption...")
                save_to_csv(rows, str(output))
                print("Done.")
                exit(0)
    if rows:
        save_to_csv(rows, str(output))

    if verbose:
        print(
            f"Done saving labels {'with coordinates' if compute_corners else ''} to {output}"
        )


def save_to_csv(rows: list[dict], dst: str, mode="a"):
    dst_exists = os.path.exists(dst)

    print("Saving to csv")

    if dst_exists and not mode == "a":
        raise ValueError(
            "Refusing to write to exising CSV file when mode is not 'append'."
        )

    df = pd.DataFrame(rows, copy=False)  # !! Do not reset row yet!
    df.to_csv(
        dst,
        mode=mode,
        header=not dst_exists,  # headers only written the first time
        index=False,
    )


if __name__ == "__main__":
    build_csv(
        root="/home/antonio/Downloads/extended_smartdoc_dataset/Extended Smartdoc dataset/train",
        images_ext="_in.png",
        labels_ext="_gt.png",
        output="./datatest.csv",
        compute_corners=True,
        check_normalization=True,
        verbose=True,
        precision=Precision.FP32
    )
