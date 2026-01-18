from pandas.tests.arrays.masked.test_arrow_compat import pa
import time
from enum import Enum
from typing import List, Any, Literal

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
from numpy.typing import NDArray
import torch

from common import Device


def resize_img(img, h: int, w: int):
    """
    Resizes an image using the CPU to the given shape
    parameters:
        :param img: ndarray
        :param h:   int
        :param w:   int
    """

    return cv2.resize(img, (int(w), int(h)))


class Precision(Enum):
    FP32 = 4  # 4 bytes
    FP16 = 2
    UINT8 = 1

    def __str__(self):
        if self == Precision.FP32:
            return "Float32"
        elif self == Precision.FP16:
            return "Float16"
        elif self == Precision.UINT8:
            return "UINT8"
        else:
            raise NotImplementedError(
                f"No type associated with {self} for CPU. This is a bug!"
            )

    @staticmethod
    def from_str(s: str):
        l_s = s.lower()
        if l_s in ["float32", "fp32", "f32"]:
            return Precision.FP32
        elif l_s in ["float16", "fp16", "f16"]:
            return Precision.FP16
        elif l_s in ["uint8", "u8", "int8", "i8"]:
            return Precision.UINT8
        else:
            raise NotImplementedError(f"No precision type associated with {s}")

    def to_type_cpu(self) -> np.dtype[Any]:
        if self == Precision.FP32:
            return np.float32()
        elif self == Precision.FP16:
            return np.float16()
        elif self == Precision.UINT8:
            return np.uint8()
        else:
            raise NotImplementedError(
                f"No type associated with {self} for CPU. This is a bug!"
            )

    def to_type_gpu(self) -> torch.dtype:
        if self == Precision.FP32:
            return torch.float32
        elif self == Precision.FP16:
            return torch.float16
        elif self == Precision.UINT8:
            return torch.uint8
        else:
            raise NotImplementedError(
                f"No type associated with {self} for GPU. This is a bug!"
            )




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
    precision: Precision,
    device: Device = Device.CPU,
):
    """
    Computes the coordinates of a PERFECTLY RECTANGULARE/SQUARED
    mask which can also be rotated.
    Parameters:
        :parameter mask: a ndarray/tensor of pixels (normalized, min 0 - max 1), shape = (n, 2)

    :returns coords: a torch tensor of 8 NORMALIZED points
    """
    if not precision:
        raise ValueError("Precision MUST be set.")

    gpu = device is not Device.CPU

    if isinstance(mask, np.ndarray) and gpu:
        raise ValueError("if `gpu` is set to `True` you need to pass a Tensor.")
    if isinstance(mask, torch.Tensor) and not gpu:
        raise ValueError("if `gpu` is set to `False` you need to pass a Numpy ndarray.")

    ## Normalization check (only for FP32 and FP16) ##
    bad_norm = (False, None)
    if (
        precision != Precision.UINT8
    ):  # Normalization of images shouldn't happen for uint8
        # This also covers for integer overflow due to selecting the wrong integer type
        # int8 range from -127 to 127 so a value of 255 overflows to -1
        if gpu:
            if (mask > 1.0).any() or (mask < 0.0).any():
                bad_norm = (True, "gpu")
        else:
            if np.any(mask > 1) or np.any(mask < 0):
                bad_norm = (True, "cpu")

        if bad_norm[0]:
            raise ValueError(
                f"Mask values should be >= 0 and >= 1. You may want to check your normalization algorithm. (device={bad_norm[1]}, dtype={mask.dtype}, precision={precision})"
            )

    threshold = 127 if precision == Precision.UINT8 else 0.5

    h, w = mask.shape

    ## Compute pixel coordinates ##
    # not using mask > 0 (masks for the dataset only have black or white pixels) because if cv2 applies any filters
    # that make some white pixel go toward "black" they'd be counted as black
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

    # Find the indices of the extremes
    if gpu:
        tl = white_xy[torch.argmin(topleft_to_bottoright_diagonal)]
        tr = white_xy[torch.argmin(topright_to_bottomleft_diagonal)]
        br = white_xy[torch.argmax(topleft_to_bottoright_diagonal)]
        bl = white_xy[torch.argmax(topright_to_bottomleft_diagonal)]
    else:
        tl = white_xy[np.argmin(topleft_to_bottoright_diagonal)]  # Smallest x + y
        tr = white_xy[np.argmin(topright_to_bottomleft_diagonal)]  # Smallest y - x
        br = white_xy[np.argmax(topleft_to_bottoright_diagonal)]  # Largest x + y
        bl = white_xy[np.argmax(topright_to_bottomleft_diagonal)]  # Largest y - x

    # normalization
    if gpu:
        corners = torch.stack([tl, tr, br, bl])
        w_h = torch.tensor([w, h], device=device.value, dtype=torch.float32)
        return (corners / w_h).flatten()
    else:
        # w and h are, for example 512 and 1024
        # tl may be [691, 23]
        # norm_tl = [x/w, y/h] = [ 691 / 512, 23 / 1024 ]
        norm_tl = [tl[0] / w, tl[1] / h]
        norm_tr = [tr[0] / w, tr[1] / h]
        norm_br = [br[0] / w, br[1] / h]
        norm_bl = [bl[0] / w, bl[1] / h]

        return torch.tensor(
            np.array([norm_tl, norm_tr, norm_br, norm_bl]).flatten(),
            dtype=torch.float32,
        )


if __name__ == "__main__":
    pass
