import subprocess
from pathlib import Path
from sys import stderr
from enum import Enum

class Precision(Enum):
    FP32  = 4 # 4 bytes
    FP16  = 2
    UINT8 = 1


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
