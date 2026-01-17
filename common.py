from enum import Enum

import numpy as np
import torch

def device_from_obj(x: torch.Tensor | np.ndarray):
    return x.device

class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    def from_tensor(self, t: torch.Tensor):
        if t.device == -1:
            return self.CPU
        else:
            return self.CUDA