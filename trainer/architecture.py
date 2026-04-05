import torch
from typing import Callable, Never
import cv2
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from common import Device
from utils import (
    assert_never,
    resize_img,
    coords_from_segmentation_mask,
    get_resize_params,
)


class ProcessResult:
    def __init__(self, image: NDArray, label: NDArray):
        self.image = image
        self.label = label


class Architecture(Enum):
    RESNET = "resnet"
    UNET = "unet"

    def __str__(self):
        return self.value

    @property
    def label_type(self) -> str:
        """Returns the type of label associated with the architecture. 'coordinates' for Resnet and 'mask' for Unet."""
        if self == Architecture.RESNET:
            return "coordinates"
        elif self == Architecture.UNET:
            return "mask"
        else:
            assert_never(self)

    def get_transform_logic(
        self,
    ) -> Callable[[dict, int, int], ProcessResult]:
        if self == Architecture.RESNET:
            return self._transform_resnet
        # elif self == Architecture.UNET:
        #     return self._transform_unet
        else:
            assert_never(Never)

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
        allow_padding: bool,
        color: bool = True,
        interpolation=cv2.INTER_AREA,
    ):
        target_h = h
        target_w = w
        imdata = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        if imdata is None:
            raise RuntimeError(f"Could not read image at {path}.")

        original_h, original_w = imdata.shape[:2]
        original_shape = (original_h, original_w)

        if color:  # BRG -> RB
            imdata = cv2.cvtColor(imdata, cv2.COLOR_BGR2RGB)

        if (
            target_h > original_h or target_w > original_w
        ) and interpolation == cv2.INTER_AREA:
            print(
                "WARNING: the target shape of the image is bigger than the original but INTER_AREA interpolation, which is best for shrinking, is being used. Be sure this is what you intend to do."
            )

        new_h, new_w, pad_h, pad_w, scale = get_resize_params(
            original_h=original_h,
            original_w=original_w,
            target_h=target_h,
            target_w=target_w,
        )

        img_resized = resize_img(imdata, new_h, new_w, interpolation=interpolation)
        if img_resized is None:
            raise RuntimeError(f"Could not resize image at {path}.")

        needs_padding = pad_h != 0 or pad_w != 0
        if needs_padding:
            if needs_padding and not allow_padding:
                raise ValueError(
                    "Cannot resize image: the target shape requires padding but `allow_padding` is set to False."
                )

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            if img_resized.ndim == 2:
                pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right))
            else:
                pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

            img_resized = np.pad(
                array=img_resized,
                pad_width=pad_widths,
                mode="constant",
                constant_values=0,  # black padding, white would confuse the model
            )

        return img_resized, original_shape

    @staticmethod
    def _transform_resnet(row, h: int, w: int):
        """
        Resize the image and return the coordinates from the mask.
        """
        ipath = row["image_path"]
        img_resized, original_shape = Architecture.resize_image(
            ipath, h, w, allow_padding=True, color=True
        )

        original_h, original_w = original_shape
        _, _, pad_h, pad_w, scale = get_resize_params(
            original_h=original_h, original_w=original_w, target_h=h, target_w=w
        )

        if "x1" in row:  # have coords
            coords = np.array(
                [row[f"{axis}{i}"] for i in range(1, 5) for axis in ("x", "y")]
            )
        elif "label_path" in row:  # compute cords from mask
            mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not read label mask at {row['label_path']}")

            coords = coords_from_segmentation_mask(
                mask, device=Device.CPU,
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
        coords = coords.reshape(4, 2).astype(
            np.float64
        )  # they were still uint32 and _scale is float

        coords *= scale
        coords[:, 0] += pad_w // 2
        coords[:, 1] += pad_h // 2

        coords = coords.flatten()

        return ProcessResult(img_resized, coords)

    @staticmethod
    def _transform_unet(row, h: int, w: int) -> ProcessResult:
        ipath = row["image_path"]
        mpath = row["label_path"]

        img_resized, original_shape = Architecture.resize_image(
            row["image_path"], h, w, color=True, allow_padding=True
        )

        mask_resized, original_shape = Architecture.resize_image(
            mpath,
            h,
            w,
            color=False,
            allow_padding=True,
            interpolation=cv2.INTER_NEAREST,
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
