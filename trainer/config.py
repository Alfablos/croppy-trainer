from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as visionmodels
from torchvision.transforms import v2 as transformsV2

import cv2

from common import DataSource, DataRow, Device
import utils

### Model Architecture ###
backbone_model_fn = visionmodels.resnet18
backbone_weights = visionmodels.ResNet18_Weights.DEFAULT
freeze_backbone = True          # layers before layer2
freeze_backbone_layer2 = False  # unfreeze layer2 so it learns corner-relevant features
backbone_output_channels = 128  # layer2 output: resnet18/34 → 128
backbone_downsample_factor = 8  # conv1(×2) · maxpool(×2) · layer2(×2) = 8×

# Precomputed from backbone_weights — used by DataSource GPU transforms
_t = backbone_weights.transforms()
_normalize = transformsV2.Normalize(mean=_t.mean, std=_t.std)


class SmartDocExtendedDataSource(DataSource):
    """
    Handles the SmartDoc Extended dataset.

    Structure:
        - No CSV metadata file
        - Images end in ``_in.png``, labels (masks) end in ``_gt.png``
        - Coordinates are derived from label masks via ``utils.coords_from_segmentation_mask``

    Synthetic images benefit from heavy augmentation to simulate real camera artifacts.
    """

    def __init__(
        self,
        name: str,
        root_path: str,
        crawl_output_path: str | None = None,
        precompute_base_dir: str | None = None,
        ):
        self.name = name
        self.root_path = Path(root_path)
        self._crawl_output_path = crawl_output_path
        self._precompute_base_dir = precompute_base_dir
        # CPU: source-specific augmentations (run per-sample in DataLoader workers)
        self.train_cpu_transforms = transformsV2.Compose([
            transformsV2.ToImage(),
            transformsV2.JPEG(quality=[70, 100]),
            transformsV2.ColorJitter(brightness=0.5, contrast=0.8, saturation=0.4),
            transformsV2.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2)),
            transformsV2.ToDtype(torch.float32, scale=True),
            transformsV2.GaussianNoise(),
        ])
        self.val_cpu_transforms = transformsV2.Compose([
            transformsV2.ToImage(),
            transformsV2.ToDtype(torch.float32, scale=True),
        ])
        # GPU: backbone normalization (run per-batch in training loop)
        self.train_gpu_transforms = transformsV2.Compose([_normalize])
        self.val_gpu_transforms = transformsV2.Compose([_normalize])

    def check(self) -> str | None:
        if not self.root_path.exists():
            return f"Data root {self.root_path} does not exist."
        return None

    def fetch(self, architecture) -> List[DataRow]:

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
        crawl_output_path: str | None = None,
        precompute_base_dir: str | None = None,
    ):
        self.name = name
        self.root_path = Path(root_path)
        self.metadata_file = Path(metadata_file)
        self.split = split
        self._crawl_output_path = crawl_output_path
        self._precompute_base_dir = precompute_base_dir
        # Real video frames already have natural variation — no augmentation needed
        self.train_cpu_transforms = transformsV2.Compose([
            transformsV2.ToImage(),
            transformsV2.ToDtype(torch.float32, scale=True),
        ])
        self.val_cpu_transforms = transformsV2.Compose([
            transformsV2.ToImage(),
            transformsV2.ToDtype(torch.float32, scale=True),
        ])
        # GPU: backbone normalization only
        self.train_gpu_transforms = transformsV2.Compose([_normalize])
        self.val_gpu_transforms = transformsV2.Compose([_normalize])

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

# Path configuration section
# Provides default paths for each purpose
# Can be overridden per-DataSource via precompute_base_dir parameter
PATHS: dict[str, dict[str, str]] = {
    "training": {
        "precompute_output_dir": "./data",
    },
    "validation": {
        "precompute_output_dir": "./data",
    },
    "test": {
        "precompute_output_dir": "./data",
    },
}

DATA_SOURCES: dict[str, list[DataSource]] = {
    "training": [
        SmartDocExtendedDataSource(
            name="smartdoc_extended_train",
            root_path="/home/antonio/Downloads/smartdoc15/extended_smartdoc_dataset/train",
            crawl_output_path="./data/crawl_smartdoc_extended_train.csv",
            # precompute_base_dir not specified, will use PATHS["training"]["precompute_output_dir"]
        ),
        SmartDocDataSource(
            name="smartdoc_original_train",
            root_path="/home/antonio/Downloads/smartdoc15/smartdoc2015_extracted_frames/smart_doc_extracted/images",
            metadata_file="/home/antonio/Downloads/smartdoc15/smartdoc2015_extracted_frames/frame_data.csv",
            split=(0, 0.8),  # first 80%
            crawl_output_path="./data/crawl_smartdoc_original_train.csv",
        ),
    ],
    "validation": [
        SmartDocExtendedDataSource(
            name="smartdoc_extended_validation",
            root_path="/home/antonio/Downloads/smartdoc15/extended_smartdoc_dataset/validation",
            crawl_output_path="./data/crawl_smartdoc_extended_validation.csv",
        ),
        SmartDocDataSource(
            name="smartdoc_original_val",
            root_path="/home/antonio/Downloads/smartdoc15/smartdoc2015_extracted_frames/smart_doc_extracted/images",
            metadata_file="/home/antonio/Downloads/smartdoc15/smartdoc2015_extracted_frames/frame_data.csv",
            split=(0.8, 1.0),  # last 20%
            crawl_output_path="./data/crawl_smartdoc_original_val.csv",
        ),
    ],
}


### CoordConv ###
# Coordinate grids are injected AFTER the backbone, not before conv1.
# The backbone detects visual features; the grids tell the head WHERE those
# features are. The head sees raw 0-1 position values alongside visual features.
coord_conv = True


class AddCoordChannels(nn.Module):
    """Appends normalized x, y coordinate meshgrids as extra channels (CoordConv)."""
    def forward(self, x):
        batch, _, h, w = x.shape
        y_coords = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype)
        y_grid = y_coords.view(1, 1, h, 1).expand(batch, 1, h, w)
        x_grid = x_coords.view(1, 1, 1, w).expand(batch, 1, h, w)
        return torch.cat([x, x_grid, y_grid], dim=1)

### Soft-Argmax ###
# Converts heatmaps to coordinates differentiably. See HEATMAPS_APPROACH.md for details.
initial_temperature = 1.0


class SoftArgmax2D(nn.Module):
    """Convert (B, K, H, W) heatmaps to (B, K, 2) normalized coordinates via soft-argmax.

    Learnable temperature scales logits before softmax: higher temperature produces a
    sharper probability distribution, yielding more precise coordinate extraction.
    """
    def __init__(self, temperature=initial_temperature):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, heatmaps):
        B, K, H, W = heatmaps.shape
        flat = heatmaps.view(B, K, -1) * self.temperature
        probs = F.softmax(flat, dim=-1).view(B, K, H, W)
        x_coords = torch.linspace(0, 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
        y_coords = torch.linspace(0, 1, H, device=heatmaps.device, dtype=heatmaps.dtype)
        x = (probs.sum(dim=-2) * x_coords).sum(dim=-1)
        y = (probs.sum(dim=-1) * y_coords).sum(dim=-1)
        return torch.stack([x, y], dim=-1)


### Heatmap Head ###
head_input_channels = backbone_output_channels + (2 if coord_conv else 0)  # 130

head = nn.Sequential(
    nn.Conv2d(head_input_channels, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64×64 → 128×128
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 4, kernel_size=1),  # 4 heatmaps, one per corner
)

### Optimizer ###
weight_decay = 1e-4
grad_clip_max_norm = 1.0  # max L2 norm for gradient clipping; None to disable

### LR Scheduler ###
scheduler_factor = 0.5
scheduler_patience = 4
scheduler_mode = "min"
scheduler_threshold = 1e-3
