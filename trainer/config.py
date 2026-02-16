from pathlib import Path
from typing import List

import torch
from torchvision.transforms import v2 as transformsV2

import cv2

from common import DataSource, DataRow, Device
import utils


class SmartDocExtendedDataSource(DataSource):
    """
    Handles the SmartDoc Extended dataset.

    Structure:
        - No CSV metadata file
        - Images end in ``_in.png``, labels (masks) end in ``_gt.png``
        - Coordinates are derived from label masks via ``utils.coords_from_segmentation_mask``
    """

    def __init__(self, name: str, root_path: str):
        self.name = name
        self.root_path = Path(root_path)

    def check(self) -> str | None:
        if not self.root_path.exists():
            return f"Data root {self.root_path} does not exist."
        return None

    def fetch(self, architecture) -> List[DataRow]:
        from architecture import Architecture

        images = sorted(self.root_path.glob("**/*_in.png"))
        rows: List[DataRow] = []

        for img_path in images:
            label_path = str(img_path).replace("_in.png", "_gt.png")

            if architecture.label_type == "coordinates":
                # Compute corner coordinates from the segmentation mask
                mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Could not read label mask at {label_path}")

                coords = utils.coords_from_segmentation_mask(mask, device=Device.CPU)
                # coords is a flat array: [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
                rows.append(
                    DataRow(
                        image_path=str(img_path),
                        label_path=label_path,
                        tl_x=float(coords[0]),
                        tl_y=float(coords[1]),
                        tr_x=float(coords[2]),
                        tr_y=float(coords[3]),
                        br_x=float(coords[4]),
                        br_y=float(coords[5]),
                        bl_x=float(coords[6]),
                        bl_y=float(coords[7]),
                    )
                )
            elif architecture.label_type == "mask":
                # For U-Net: just reference the mask file, no coordinates needed
                rows.append(
                    DataRow(
                        image_path=str(img_path),
                        label_path=label_path,
                        tl_x=None,
                        tl_y=None,
                        tr_x=None,
                        tr_y=None,
                        br_x=None,
                        br_y=None,
                        bl_x=None,
                        bl_y=None,
                    )
                )

        return rows


class SmartDocDataSource(DataSource):
    """
    Handles the original SmartDoc 2015 dataset.

    Structure:
        - A CSV file contains label coordinates, one corner per row:
            ``frame_filename,frame_index,name,x,y``
          where ``name`` is one of: tl, tr, br, bl.
        - Four rows per image (one per corner).

    Args:
        split: Optional (start_frac, end_frac) tuple for deterministic
               positional splitting, e.g. (0, 0.8) for first 80%,
               (0.8, 1.0) for last 20%.
    """

    def __init__(
        self,
        name: str,
        root_path: str,
        metadata_file: str,
        split: tuple[float, float] | None = None,
    ):
        self.name = name
        self.root_path = Path(root_path)
        self.metadata_file = Path(metadata_file)
        self.split = split

    def check(self) -> str | None:
        if not self.root_path.exists():
            return f"Data root {self.root_path} does not exist."
        if not self.metadata_file.exists():
            return f"Metadata file {self.metadata_file} does not exist."
        if self.split is not None:
            start, end = self.split
            if not (0.0 <= start < end <= 1.0):
                return f"Invalid split range: ({start}, {end}). Must satisfy 0 <= start < end <= 1."
        return None

    def fetch(self, architecture) -> List[DataRow]:
        import pandas as pd
        from architecture import Architecture

        df = pd.read_csv(self.metadata_file)

        # Group the 4 corner rows per image into a single dict {name: (x, y)}
        grouped = df.groupby("frame_filename")
        rows: List[DataRow] = []

        for filename, group in grouped:
            corners = {row["name"]: (row["x"], row["y"]) for _, row in group.iterrows()}
            image_path = str(self.root_path / filename)

            if architecture.label_type == "coordinates":
                rows.append(
                    DataRow(
                        image_path=image_path,
                        label_path=None,
                        tl_x=float(corners["tl"][0]),
                        tl_y=float(corners["tl"][1]),
                        tr_x=float(corners["tr"][0]),
                        tr_y=float(corners["tr"][1]),
                        br_x=float(corners["br"][0]),
                        br_y=float(corners["br"][1]),
                        bl_x=float(corners["bl"][0]),
                        bl_y=float(corners["bl"][1]),
                    )
                )
            elif architecture.label_type == "mask":
                # SmartDoc original has no mask files — label_path is None
                # The preprocessor will need to generate masks from coordinates
                rows.append(
                    DataRow(
                        image_path=image_path,
                        label_path=None,
                        tl_x=None,
                        tl_y=None,
                        tr_x=None,
                        tr_y=None,
                        br_x=None,
                        br_y=None,
                        bl_x=None,
                        bl_y=None,
                    )
                )

        # Apply deterministic positional split
        if self.split is not None:
            start_frac, end_frac = self.split
            n = len(rows)
            rows = rows[int(n * start_frac):int(n * end_frac)]

        return rows


### Active Data Sources ###
# Keyed by purpose: each precompute pass only crawls the sources it needs.
# For datasets without a train/val directory split, both keys can reference
# the same source — a split mechanism can be added later.

DATA_SOURCES: dict[str, list[DataSource]] = {
    "training": [
        SmartDocExtendedDataSource(
            name="smartdoc_extended_train",
            root_path="/home/antonio/Downloads/smartdoc15/extended_smartdoc_dataset/train",
        ),
        # SmartDocDataSource(
        #     name="smartdoc_original_train",
        #     root_path="/home/antonio/Downloads/smartdoc15/smartdoc2015_extracted_frames/smart_doc_extracted/images",
        #     metadata_file="/home/antonio/Downloads/smartdoc15/smartdoc2015_extracted_frames/frame_data.csv",
        #     split=(0, 0.8),  # first 80%
        # )
    ],
    "validation": [
        SmartDocExtendedDataSource(
            name="smartdoc_extended_validation",
            root_path="/home/antonio/Downloads/smartdoc15/extended_smartdoc_dataset/validation",
        ),
        # SmartDocDataSource(
        #     name="smartdoc_original_val",
        #     root_path="/home/antonio/Downloads/smartdoc15/smartdoc2015_extracted_frames/smart_doc_extracted/images",
        #     metadata_file="/home/antonio/Downloads/smartdoc15/smartdoc2015_extracted_frames/frame_data.csv",
        #     split=(0.8, 1.0),  # last 20%
        # )
    ]
}

### Transforms ###

transforms = {
    "training": {
        "cpu": transformsV2.Compose(
            [
                transformsV2.ToImage(),
                # transformsV2.JPEG(quality=[70, 100]),  # CPU-bound, cannot run on GPU
                # transformsV2.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.4),
                # # transformsV2.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2)),
            ]
        ),
        "gpu": lambda t: transformsV2.Compose(
            [
                # transformsV2.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2)),
                # transformsV2.ElasticTransform(alpha=15.0),
                # transformsV2.RandomPerspective(
                #     distortion_scale=0.25, p=0.3, fill=(255, 255, 255)
                # ),  # p=0.5 => half of the dataset is affected
                # # All the pipeline must be computed on UINT8, conversion at last
                # transformsV2.ToDtype(torch.float32, scale=True),
                # transformsV2.GaussianNoise(),  # needs float input or turns uint8 into floats!
                transformsV2.ToDtype(torch.float32, scale=True),
                transformsV2.Normalize(mean=t.mean, std=t.std),
            ]
        ),
    },
    "validation": {
        "cpu": transformsV2.Compose([transformsV2.ToImage()]),
        "gpu": lambda t: transformsV2.Compose(
            [
                # All the pipeline must be computed on UINT8, conversion at last
                transformsV2.ToDtype(torch.float32, scale=True),
                transformsV2.Normalize(mean=t.mean, std=t.std),
            ]
        ),
    },
}

## Discarded:
# White fill to differ less from the background
# transformsV2.RandomRotation(degrees=(0, 100), fill=255), # let's try but I'm not sure... see https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_rotated_box_transforms.html
# transformsV2.RandomAffine(degrees=(0, 100), fill=255),
