from __future__ import annotations

from typing import Any, List, TYPE_CHECKING
from enum import Enum
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

if TYPE_CHECKING:
    from architecture import Architecture

import numpy as np
import torch
import torchvision.models as visionmodels

# DEFAULT_WEIGHTS = visionmodels.ResNet34_Weights.DEFAULT
DEFAULT_WEIGHTS = visionmodels.ResNet18_Weights.DEFAULT
DEFAULT_STORAGE_CLASS = "arrow"


def device_from_obj(x: torch.Tensor | np.ndarray):
    return x.device


class Purpose(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(s: str):
        s = s.lower()
        if s in ["train", "training", "tr"]:
            return Purpose.TRAINING
        elif s in ["validation", "val"]:
            return Purpose.VALIDATION
        elif s == "test":
            return Purpose.TEST
        else:
            raise NotImplementedError(f"No purpose associated with {s}")


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    def __str__(self):
        return self.value

    def from_tensor(self, t: torch.Tensor):
        if t.device == -1:
            return self.CPU
        else:
            return self.CUDA

    @staticmethod
    def from_str(s: str):
        s = s.lower()
        if s in ["cuda", "gpu"]:
            return Device.CUDA
        elif s == "cpu":
            return Device.CPU
        elif s in ["mps", "metal"]:
            return Device.MPS
        else:
            raise NotImplementedError(f"No device type associated with {s}")


class Precision(Enum):
    FP32 = 32  # 4 bytes
    FP16 = 16
    UINT8 = 8

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
            return np.float32
        elif self == Precision.FP16:
            return np.float16
        elif self == Precision.UINT8:
            return np.uint8
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


@dataclass
class DataRow:
    def __init__(
        self,
        image_path: str,
        label_path: str | None,
        tl_x: float | None,
        tl_y: float | None,
        tr_x: float | None,
        tr_y: float | None,
        br_x: float | None,
        br_y: float | None,
        bl_x: float | None,
        bl_y: float | None,
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.tl_x = tl_x
        self.tl_y = tl_y
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.br_x = br_x
        self.br_y = br_y
        self.bl_x = bl_x
        self.bl_y = bl_y

    def to_dict(self):
        return {
            "image_path": self.image_path,
            "label_path": self.label_path,
            "x1": self.tl_x,
            "y1": self.tl_y,
            "x2": self.tr_x,
            "y2": self.tr_y,
            "x3": self.br_x,
            "y3": self.br_y,
            "x4": self.bl_x,
            "y4": self.bl_y,
        }


class DataSource(metaclass=ABCMeta):
    """
    Subclasses must set:
        name: str
        train_cpu_transforms: torchvision Compose — CPU transforms for training (per-sample, in DataLoader workers)
        val_cpu_transforms: torchvision Compose — CPU transforms for validation
        train_gpu_transforms: torchvision Compose — GPU transforms for training (per-batch, in training loop)
        val_gpu_transforms: torchvision Compose — GPU transforms for validation
    """

    @abstractmethod
    def __init__(
        self,
        name: str,
        root_path: str,
        crawl_output_path: str | None = None,
        precompute_base_dir: str | None = None,
    ):
        """
        Args:
            name: Unique identifier for this data source
            root_path: Path to the root directory containing images/labels
            crawl_output_path: Path where crawl output CSV should be saved
            precompute_base_dir: Base directory for precompute outputs
                             (defaults to PATHS config if not specified)
        """
        pass

    @property
    def crawl_output_path(self) -> str:
        """Get the crawl output path for this data source."""
        if self._crawl_output_path is None:
            return f"./data/crawl_{self.name}.csv"
        return self._crawl_output_path

    def get_precompute_base_dir(self, purpose: Purpose, paths_config: dict | None = None) -> str:
        """
        Get the base directory for precompute output for this data source.

        Priority:
        1. DataSource's precompute_base_dir (if set)
        2. paths_config[str(purpose)]["precompute_output_dir"] (if provided)
        3. Default: "{purpose}_data"

        Args:
            purpose: The purpose (training/validation/test)
            paths_config: The PATHS configuration from config.py

        Returns:
            The base directory for precompute output
        """
        # First, check if this source has an override
        if self._precompute_base_dir is not None:
            return self._precompute_base_dir

        # Second, check the global config
        if paths_config is not None:
            purpose_str = str(purpose)
            if purpose_str in paths_config:
                return paths_config[purpose_str]["precompute_output_dir"]

        # Fallback to default
        return f"{purpose_str}_data"

    def get_precompute_output_path(
        self,
        architecture: "Architecture",
        purpose: Purpose,
        target_h: int,
        target_w: int,
        paths_config: dict | None = None,
        compacted: bool = False,
    ) -> str:
        """
        Get the full path for a precompute store.

        Path format: {base_dir}/{purpose}_data/data_{arch}_{source_name}_{h}x{w}[._compacted].{ext}

        Args:
            architecture: The model architecture
            purpose: The purpose (training/validation/test)
            target_h: Target height
            target_w: Target width
            paths_config: The PATHS configuration from config.py
            compacted: Whether this is a compacted store

        Returns:
            Full path to the store file
        """
        base_dir = self.get_precompute_base_dir(purpose, paths_config)
        ext = DEFAULT_STORAGE_CLASS

        compacted_str = "_compacted" if compacted else ""
        return f"{base_dir}/{purpose}_data/data_{architecture.value}_{self.name}_{target_h}x{target_w}{compacted_str}.{ext}"

    @abstractmethod
    def check(self) -> str | None:  # return error as a string
        pass

    @abstractmethod
    def fetch(self, architecture: Architecture) -> List[DataRow]:
        """
        Transforms the dataset into a canonical data source.
        The architecture determines what fields are required:
        - 'coordinates' (resnet): x1..y4 must be present
        - 'mask' (unet): label_path must be present

        Returns:
            List[DataRow]: a list of DataRow.
        """
        pass
