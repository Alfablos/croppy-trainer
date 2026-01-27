import torch
from typing import Callable, TypeVar
import cv2
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from common import Precision, Device
from utils import assert_never, resize_img, coords_from_segmentation_mask


class ProcessResult:
    def __init__(self, image: NDArray, label: NDArray):
        self.image = image
        self.label = label


class Architecture(Enum):
    RESNET = "resnet"
    UNET = "unet"

    def __str__(self):
        return self.value

    def get_transform_logic(
        self, coords_scale_percentage: float
    ) -> Callable[[dict, int, int, Precision], ProcessResult]:
        if self == Architecture.RESNET:
            return self._transform_resnet
        elif self == Architecture.UNET:
            return self._transform_unet
        else:
            assert_never(self)

    @staticmethod
    def from_str(s: str):
        l_s = s.lower()
        if l_s == "resnet":
            return Architecture.RESNET
        elif l_s in ["u-net", "unet"]:
            return Architecture.UNET
        else:
            raise NotImplementedError(f"No precision type associated with {s}")

    @staticmethod
    def resize_image(
        path,
        h,
        w,
        color: bool = True,
        resize: bool = True,
        interpolation=cv2.INTER_AREA,
    ):
        imdata = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        if imdata is None:
            raise RuntimeError(f"Could not read image at {path}.")

        original_h, original_w = imdata.shape[:2]
        original_shape = (original_h, original_w)

        if color:  # BRG -> RB
            imdata = cv2.cvtColor(imdata, cv2.COLOR_BGR2RGB)

        if not resize:
            return imdata, original_shape
        img_resized = resize_img(imdata, h, w, interpolation=interpolation)
        if img_resized is None:
            raise RuntimeError(f"Could not resize image at {path}.")

        return img_resized, original_shape

    @staticmethod
    def _transform_resnet(row, h: int, w: int, coords_scale_percentage: float):
        """
        Resize the image and return the coordinates from the mask.
        """
        ipath = row["image_path"]
        img_resized, original_shape = Architecture.resize_image(
            ipath, h, w, resize=True, color=True
        )

        if "x1" in row:  # have coords
            coords = np.array(
                [row[f"{axis}{i}"] for i in range(1, 5) for axis in ("x", "y")]
            )
        elif "label_path" in row:  # compute cords from mask
            mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)
            coords = coords_from_segmentation_mask(
                mask, device=Device.CPU, scale_percentage=coords_scale_percentage
            )
            if isinstance(coords, torch.Tensor):
                coords: NDArray = coords.numpy()
            else:
                coords: NDArray = coords
        else:
            raise ValueError(
                f"Coordinates for ResNet image {[row['image_path']]} were not provided and the data map has no label path to compute them"
            )

        # Scale the coordinates according to the new image size:
        original_h, original_w = original_shape
        x_scale = w / original_w
        y_scale = h / original_h

        coords = coords.reshape(4, 2).astype(
            np.float64
        )  # they were still uint32 and _scale is float
        coords[:, 0] *= x_scale
        coords[:, 1] *= y_scale
        coords = coords.flatten()

        return ProcessResult(img_resized, coords)

    @staticmethod
    def _transform_unet(row, h: int, w: int) -> ProcessResult:
        ipath = row["image_path"]
        mpath = row["label_path"]

        img_resized = Architecture.resize_image(
            row["image_path"], h, w, resize=True, color=True
        )

        mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)

        mask_resized = Architecture.resize_image(
            mpath, h, w, color=False, resize=True, interpolation=cv2.INTER_NEAREST
        )

        return ProcessResult(img_resized, mask_resized)

    def preprocessor_db_map_size(
        self, data_length: int, target_h: int, target_w: int
    ) -> int:
        single_image_size: int = target_h * target_w * 3  # RGB
        total_map_size: int = int(
            data_length * single_image_size * 1.2
        )  # 1.2 is a safety margin

        if self == Architecture.RESNET:
            coord_size = 4 * 8  # (8 uint32, 32Bit each)
            total_coord_size = int(data_length * coord_size * 1.2)
            total_map_size += total_coord_size
            return total_map_size

        elif self == Architecture.UNET:  # U-Net mode, we're storing the masks!
            mask_size = target_h * target_w * 1  # 1 single channel (B/W) for masks
            total_masks_size = int(mask_size * data_length * 1.2)
            total_map_size += total_masks_size
            return total_map_size

        else:
            assert_never(self)

    def get_csv_header(self) -> list[str]:
        if self == Architecture.RESNET:
            return ["index", "path"] + [
                f"{axis}{i}" for i in range(1, 5) for axis in ("x", "y")
            ]  # [f"c{k}" for k in range(8)]
        elif self == Architecture.UNET:
            return ["index", "path"]
        else:
            assert_never(self)

    def find_preprocessor_misconfig(self, config) -> str | None:
        if self == Architecture.RESNET:
            return self._is_valid_resnet_preproc(config)
        elif self == Architecture.UNET:
            return self._is_valid_unet_preproc(config)
        else:
            assert_never(self)

    @staticmethod
    def _is_valid_resnet_preproc(config: dict) -> str | None:
        return None

    @staticmethod
    def _is_valid_unet_preproc(config: dict) -> str | None:
        if config["compute_coords"]:
            return "Found `compute_corners=True` preprocessor's config but U-Net only needs masks."
        else:
            return None
