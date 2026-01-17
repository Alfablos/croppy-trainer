import subprocess
from pathlib import Path
from sys import stderr
from enum import Enum

import numpy as np
import torch


class Precision(Enum):
    FP32  = 4 # 4 bytes
    FP16  = 2
    UINT8 = 1

    def __str__(self):
        if self == Precision.FP32:
            return "Float32"
        elif self == Precision.FP16:
            return "Float16"
        elif self == Precision.UINT8:
            return "UINT8"
        else:
            raise NotImplementedError(f"No type associated with {self} for CPU. This is a bug!")


    def to_type_cpu(self) -> np.dtype:
        if self == Precision.FP32:
            return np.float32
        elif self == Precision.FP16:
            return np.float16
        elif self == Precision.UINT8:
            return np.uint8
        else:
            raise NotImplementedError(f"No type associated with {self} for CPU. This is a bug!")

    def to_type_gpu(self) -> torch.dtype:
        if self == Precision.FP32:
            return torch.float32
        elif self == Precision.FP16:
            return torch.float16
        elif self == Precision.UINT8:
            return torch.uint8
        else:
            raise NotImplementedError(f"No type associated with {self} for GPU. This is a bug!")



def create_path_batch(root: str | Path, n):
    root = root if isinstance(root, Path) else Path(root)
    root = root.glob("**/*_gt.png")

    paths = []
    for i, path in enumerate(root):
        if i < n:
            paths.append(str(path))
        else:
            break

    n_paths = len(paths)
    if n_paths < n:
        raise ValueError(f"Can't create a batch of {n} paths from a root containing only {n_paths} of them")
    return paths

def drop_disk_cache(p: str, verbose=False):
    if verbose:
        print("Invalidating disk caches, this may take a while...")
    cmd = ["sudo", "-S", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
    proc = subprocess.Popen( # not handling exception
        args=cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = proc.communicate(input=p + "\n")
    rcode = proc.returncode
    if rcode == 0 and verbose:
        print("Disk cache invalidated successfully.")
    else:
        print(err, file=stderr)
    return rcode
