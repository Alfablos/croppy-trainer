from pathlib import Path

import lmdb
import os
from sys import argv

import pickle
from pandas.tests.arrays.masked.test_arrow_compat import pa
import time
from enum import Enum
from typing import List, Any, Literal, Never
from itertools import chain

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import tensorboard

from common import Device, Precision, DEFAULT_WEIGHTS, Purpose


def compact_lmdb(env, dst_path: str):
    env.copy(dst_path, compact=True)


def load_checkpoint(p: str, train: bool = False) -> dict:
    checkpoint = torch.load(p)
    return checkpoint


def assert_never(arg: Never) -> Never:
    raise AssertionError("Expected code to be unreachable")


def resize_img(img, h: int, w: int, interpolation=cv2.INTER_AREA):
    """
    Resizes an image using the CPU to the given shape
    parameters:
        :param img: ndarray
        :param h:   int
        :param w:   int
    """

    return cv2.resize(img, (int(w), int(h)), interpolation=interpolation)


def find_max_dims(paths: List[str]):
    """
    Given a list of image paths it returns the biggest height and biggest width found
    """
    max_h = 0
    max_w = 0
    print("Scanning dataset dimensions...")

    for p in paths:
        with Image.open(p) as img:
            w, h = img.size
            if w > max_w:
                max_w = w
            if h > max_h:
                max_h = h

    return max_h, max_w


def coords_from_segmentation_mask(
    mask: NDArray | torch.Tensor,
    scale_percentage: float,
    device: Device = Device.CPU,
) -> torch.Tensor | NDArray:
    """
    Computes the coordinates of the corner from rectangular
    masks which can also be rotated.
    Parameters:
        :parameter mask: a ndarray/tensor of pixels of shape (h, w, 1) (masks are B/W)

    :returns coords: a torch tensor of 8 NON NORMALIZED points
    """

    gpu = device is not Device.CPU

    if isinstance(mask, np.ndarray) and gpu:
        raise ValueError("if `gpu` is set to `True` you need to pass a Tensor.")
    if isinstance(mask, torch.Tensor) and not gpu:
        raise ValueError("if `gpu` is set to `False` you need to pass a Numpy ndarray.")

    threshold = 127

    ## Compute pixel coordinates ##
    # not using mask > 0 (masks for the dataset only have black or white pixels) because if cv2 applies any filters
    # that make some white pixel go toward "black" (even a bit) they'd be counted as black
    # 127 gives 126 of such filter tolerance without altering the result (losing precision)
    if gpu:
        # select the white pixels
        white_xy = torch.nonzero(
            mask > threshold, as_tuple=False
        )  # [(y, x), ...] NOT [(x, y), ...]
        # this returns a (n_points, 2) matrix, each point has an x and a y column
        white_xy = torch.flip(white_xy, dims=[1]).float()  # now [(x, y), ...]
        if white_xy.shape[0] == 0:
            raise ValueError("Mask is completely black.")
    else:
        ys, xs = np.where(mask > threshold)
        if len(xs) == 0:
            raise ValueError("Mask is completely black.")
        white_xy = np.column_stack((xs, ys))  # I love this!

    # always starting top-left
    if gpu:
        topleft_to_bottoright_diagonal = white_xy.sum(dim=1)
        topright_to_bottomleft_diagonal = white_xy[:, 1] - white_xy[:, 0]  # Ys - Xs
    else:
        # (x + y), returns a (n_points, 1) matrix/vector => the top-left to bottom-right diagonal
        # (y - x) instead returns the orthogonal diagonal: small (very negative) numbers are where y is small and x is big (top-right corner)
        # for each i in points topleft_to_bottoright_diagonal[i] has the sum of its coordinates
        topleft_to_bottoright_diagonal = np.sum(white_xy, axis=1)
        topright_to_bottomleft_diagonal = np.diff(
            white_xy, axis=1
        )  # diff(a, b) = b - a
        # These two diagonal have the min value to the top (y = 0)!

    # Returning RAW PIXEL COORDINATES
    if gpu:
        tl = white_xy[torch.argmin(topleft_to_bottoright_diagonal)]
        tr = white_xy[torch.argmin(topright_to_bottomleft_diagonal)]
        br = white_xy[torch.argmax(topleft_to_bottoright_diagonal)]
        bl = white_xy[torch.argmax(topright_to_bottomleft_diagonal)]
        # return torch.tensor([tl, tr, br, bl], dtype=torch.float32).flatten()
        coords = torch.tensor([tl, tr, br, bl], dtype=torch.float32).flatten()
    else:
        tl = white_xy[np.argmin(topleft_to_bottoright_diagonal)]  # Smallest x + y
        tr = white_xy[np.argmin(topright_to_bottomleft_diagonal)]  # Smallest y - x
        br = white_xy[np.argmax(topleft_to_bottoright_diagonal)]  # Largest x + y
        bl = white_xy[np.argmax(topright_to_bottomleft_diagonal)]  # Largest y - x
        # return np.array([tl, tr, br, bl], dtype=np.float32()).flatten()
        coords = np.array([tl, tr, br, bl], dtype=np.float32()).flatten()

    scaled_coords = scale_to_center(coords, scale_percentage)
    return scaled_coords
    # # normalization
    # if gpu:
    #     corners = torch.stack([tl, tr, br, bl])
    #     w_h = torch.tensor([w, h], device=device.value, dtype=torch.float32)
    #     return (corners / w_h).flatten()
    # else:
    #     # w and h are, for example, 512 and 1024
    #     # tl may be [691, 23]
    #     # norm_tl = [x/w, y/h] = [ 691 / 512, 23 / 1024 ]
    #     norm_tl = [tl[0] / w, tl[1] / h]
    #     norm_tr = [tr[0] / w, tr[1] / h]
    #     norm_br = [br[0] / w, br[1] / h]
    #     norm_bl = [bl[0] / w, bl[1] / h]

    #     return np.array([norm_tl, norm_tr, norm_br, norm_bl]).flatten()


def scale_to_center(coords: torch.Tensor | NDArray, percent: float):
    is_tensor = isinstance(coords, torch.Tensor)
    original_shape = coords.shape
    points = coords.reshape(4, 2)

    center = (
        points.mean(dim=0, keepdim=True)
        if is_tensor
        else points.mean(axis=0, keepdims=True)
    )

    # the corners need to come closer to the center
    # this basically means that the sum of the coordinates of P (point) and C (center) has to stay constant during the operation
    # lerp formula slightly modified: should be instead `points * percent + center * (1 - percent)
    # but I want the user to specify how much they want to SHORTEN, not keep!
    scaled = points * (1 - percent) + center * percent

    if is_tensor:
        return scaled.flatten().to(dtype=coords.dtype).reshape(original_shape)
    else:
        return scaled.flatten().astype(coords.dtype).reshape(original_shape)


def launch_tensorboard(log_dir: str, host: str = "0.0.0.0", port: int = 6006) -> str:
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir, "--host", host, "--port", str(port)])

    if not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    url = tb.launch()
    return url


# AI generated
def dump_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    epoch: int,
    batch_idx: int,
    purpose: Purpose,
    output_dir: str = "./debug_dumps",
):
    """
    Dumps a batch of images with Ground Truth (Green) and Predictions (Red) drawn on them.

    Args:
        images: (B, 3, H, W) Normalized Tensor (on GPU or CPU)
        labels: (B, 8) Normalized coordinates [0-1] (on GPU or CPU)
        preds:  (B, 8) Normalized coordinates [0-1] (on GPU or CPU)
        epoch: Current epoch number
        batch_idx: Current batch index
        output_dir: Directory to save images
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Move everything to CPU and numpy
    images_np = images.detach().cpu()
    labels_np = labels.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()

    # 2. Get Normalization Constants (for Image)
    # ResNet default mean/std
    mean = torch.tensor(DEFAULT_WEIGHTS.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(DEFAULT_WEIGHTS.transforms().std).view(1, 3, 1, 1)

    # 3. Denormalize Images (Batch Operation)
    # (Input - Mean) / Std  ->  Input = (Tensor * Std) + Mean
    images_denorm = images_np * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # Iterate through batch
    batch_size = images.shape[0]
    for i in range(batch_size):
        # A. Setup Image
        # (3, H, W) -> (H, W, 3)
        img = images_denorm[i].permute(1, 2, 0).numpy()
        # Float [0, 1] -> Uint8 [0, 255]
        img_uint8 = (img * 255).astype(np.uint8)
        # RGB -> BGR for OpenCV
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        h, w, _ = img_bgr.shape

        # B. Setup Coordinates
        # Reshape (8,) -> (4, 2)
        gt_coords = labels_np[i].reshape(-1, 2)
        pred_coords = preds_np[i].reshape(-1, 2)

        # Denormalize Coords (0-1 -> Pixels)
        # Multiply x by Width, y by Height
        gt_px = (gt_coords * np.array([w, h])).astype(np.int32)
        pred_px = (pred_coords * np.array([w, h])).astype(np.int32)

        # C. Draw
        # Ground Truth = GREEN
        cv2.polylines(img_bgr, [gt_px], isClosed=True, color=(0, 255, 0), thickness=2)
        # Prediction = RED
        cv2.polylines(img_bgr, [pred_px], isClosed=True, color=(0, 0, 255), thickness=2)

        # D. Save
        fname = f"{purpose}{i}_{batch_idx}_{epoch}.jpg"
        cv2.imwrite(str(Path(output_dir) / fname), img_bgr)


# AI generated
def inspect_dataset(
    lmdb_path: str, output_dir: str, start_idx: int = 0, count: int = 10
):
    """
    Reads raw images and labels from LMDB and draws the stored coordinates.
    """
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening LMDB: {lmdb_path}")
    env = lmdb.open(
        lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
    )

    with env.begin(write=False) as txn:
        # Get dataset length just to be sure
        length_bytes = txn.get("__len__".encode("ascii"))
        if length_bytes:
            total_len = int.from_bytes(length_bytes, "big")
            print(f"Dataset reports length: {total_len}")

        for i in range(start_idx, start_idx + count):
            img_key = f"i{i}".encode("ascii")
            lbl_key = f"l{i}".encode("ascii")

            img_bytes = txn.get(img_key)
            lbl_bytes = txn.get(lbl_key)

            if not img_bytes or not lbl_bytes:
                print(f"Skipping index {i}: Data not found.")
                continue

            # 1. Load Raw Data
            # Image is typically (H, W, 3) BGR/RGB depending on how you saved it
            # Your crawler saves as RGB usually, but OpenCV needs BGR to save.
            image = pickle.loads(img_bytes)

            # Label should be raw coords (8,) or (4, 2)
            label = pickle.loads(lbl_bytes)

            # 2. Process Image for Drawing
            # Ensure it's contiguous and uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Convert RGB -> BGR for OpenCV saving (if stored as RGB)
            # Assuming your preprocessor stored RGB (standard practice in PyTorch land)
            viz_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = viz_image.shape

            # 3. Process Labels
            # Reshape to (4, 2) points
            coords = label.reshape(-1, 2)

            # CAST TO INT: cv2.polylines requires integer coordinates
            # This step will reveal if your data is Normalized (0.0-1.0) or Pixels (0-W)
            # If data is normalized, these ints will all be 0 or 1!

            # CHECK: Are they normalized?
            if np.max(coords) <= 1.5:
                print(
                    f"Warning: Index {i} looks normalized (Max val {np.max(coords)}). Denormalizing..."
                )
                coords = coords * np.array([w, h])

            coords_px = coords.astype(np.int32)

            # 4. Draw
            # Draw contours (Green)
            cv2.polylines(
                viz_image, [coords_px], isClosed=True, color=(0, 255, 0), thickness=2
            )

            # Draw individual points to check ordering (Red circles)
            # Top-Left should be first!
            for idx, point in enumerate(coords_px):
                # Draw heavier circle for first point (TL)
                radius = 8 if idx == 0 else 4
                color = (
                    (0, 0, 255) if idx == 0 else (0, 255, 255)
                )  # Red for TL, Yellow for rest
                cv2.circle(
                    img=viz_image,
                    center=tuple(point),
                    radius=radius,
                    color=color,
                    thickness=3,
                )

            # 5. Save
            out_path = output_dir / f"inspect_{i}.jpg"
            cv2.imwrite(str(out_path), viz_image)
            print(f"Saved {out_path} | Label range: {label.min()} - {label.max()}")

    env.close()


def lmdb_get_int(key: str, lmdb_path: str):
    env = lmdb.open(
        lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
    )

    with env.begin(write=False) as t:
        val = t.get(key.encode("ascii"))
        if val is None:
            print("Not found.")
            exit(1)
        print(int.from_bytes(val, "big"))






if __name__ == "__main__":
    LMDB_PATH = "./hires/training_data/data_resnet_training_22092x1024x768.lmdb"

    # inspect_dataset(
    #     lmdb_path=LMDB_PATH, output_dir="./hires_dump/train", start_idx=0, count=20
    # )

    lmdb_get_int('h', './hires_compact/training_data/data_resnet_training_22092x1024x768_compacted.lmdb')
