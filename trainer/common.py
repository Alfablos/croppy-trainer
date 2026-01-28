from tensorboard.compat.tensorflow_stub.errors import UnimplementedError
from markdown.test_tools import Kwargs
from torch.nn import L1Loss, MSELoss
from typing import Any
from enum import Enum

import numpy as np
import torch
import torchvision.models as visionmodels

DEFAULT_WEIGHTS = visionmodels.ResNet18_Weights.DEFAULT


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


def loss_from_str(s: str, **loss_opts):
    s = s.lower()
    if s in ["l1", "l1loss", "l1_loss"]:
        return L1Loss(loss_opts)
    elif s in ["mse", "mseloss", "mse_loss"]:
        return MSELoss(loss_opts)
    else:
        raise UnimplementedError
