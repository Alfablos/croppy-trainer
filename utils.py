from pathlib import Path
import os
from sys import argv
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

from common import Device, Precision, DEFAULT_WEIGHTS


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
        return torch.tensor([tl, tr, br, bl], dtype=torch.uint32).flatten()
    else:
        tl = white_xy[np.argmin(topleft_to_bottoright_diagonal)]  # Smallest x + y
        tr = white_xy[np.argmin(topright_to_bottomleft_diagonal)]  # Smallest y - x
        br = white_xy[np.argmax(topleft_to_bottoright_diagonal)]  # Largest x + y
        bl = white_xy[np.argmax(topright_to_bottomleft_diagonal)]  # Largest y - x
        return np.array([tl, tr, br, bl], dtype=np.uint32()).flatten()

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
        output_dir: str = "./debug_dumps"
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
        fname = f"train{i}_{batch_idx}_{epoch}.jpg"
        cv2.imwrite(str(Path(output_dir) / fname), img_bgr)