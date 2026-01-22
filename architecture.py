from typing import Callable, TypeVar
import cv2
from enum import Enum
from numpy.typing import NDArray

from common import Precision
from utils import assert_never, resize_img, coords_from_segmentation_mask



class ProcessResult:
    def __init__(self, image: NDArray, label: NDArray):
        self.image = image
        self.label = label


class Architecture(Enum):
    RESNET = "resnet"
    UNET = "unet"

    def get_transform_logic(
        self,
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
    def _process_image(
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
        if color:
            imdata = cv2.cvtColor(imdata, cv2.COLOR_BGR2RGB)

        if not resize:
            return imdata
        img_resized = resize_img(imdata, h, w)
        if img_resized is None:
            raise RuntimeError(f"Could not resize image at {path}.")

        return img_resized

    @staticmethod
    def _transform_resnet(row, h: int, w: int, precision: Precision):
        """
        Resize the image and return the coordinates from the mask.
        """
        img_resized = Architecture._process_image(
            row["image_path"], h, w, resize=True, color=True
        )

        if "x1" in row:  # have coords
            coords = [row[f"{axis}{i}"] for i in range(1, 5) for axis in ("x", "y")]
        elif "label_path" in row:  # compute cords from mask
            mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)
            coords: NDArray = coords_from_segmentation_mask(mask, precision)
        else:
            raise ValueError(
                f"Coordinates for ResNet image {[row['image_path']]} were not provided and the data map has no label path to compute them"
            )

        return ProcessResult(img_resized, np.array(coords))

    @staticmethod
    def _transform_unet(row, h: int, w: int, precision: Precision) -> ProcessResult:
        ipath = row["image_path"]
        mpath = row["label_path"]

        img_resized = Architecture._process_image(
            row["image_path"], h, w, resize=True, color=True
        )

        mask = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)

        mask_resized = Architecture._process_image(
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
            coord_size = 4 * 8  # (8 floats, 32Bit each)
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

    def validate_preprocessor_config(self, config) -> str | None:
        if self == Architecture.RESNET:
            return self._is_valid_resnet_preproc(config)
        elif self == Architecture.UNET:
            return self._is_valid_resnet_preproc(config)
        else:
            assert_never(self)

    @staticmethod
    def _is_valid_resnet_preproc(config: dict) -> str | None:
        return None

    @staticmethod
    def _is_valid_unet_preproc(config: dict) -> str | None:
        if "crawler_config" in config and config["crawler_config"].compute_corners:
            return "Found `compute_corners=True` in crowler's config but U-Net only needs masks."
        elif config["compute_coords"]:
            return "Found `compute_corners=True` preprocessor's config but U-Net only needs masks."
        else:
            return None
