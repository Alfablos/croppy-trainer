from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING
from enum import Enum
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

if TYPE_CHECKING:
    from architecture import Architecture

import numpy as np
import torch
import torchvision.models as visionmodels

DEFAULT_WEIGHTS = visionmodels.ResNet34_Weights.DEFAULT
# DEFAULT_WEIGHTS = visionmodels.ResNet18_Weights.DEFAULT
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
    @abstractmethod
    def __init__(
        self,
        name: str,
        root_path: str,
    ):
        pass

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
