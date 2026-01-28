"""
This module analyzes the performance of running an algorithm
for retrieving corners coordinates given a mask.
The function that implements the algorithm
1. Pure CPU multithreading (no transfer needed)
2. CPU (Disk -> PCIe transfer -> CPU) + GPU
3. TODO: CPU + GPU
4. TODO: All of the above with lower precision (FP16, UINT8)
5. TODO: Federated with a cloud instance
"""

import logging
from logging import info, debug
from datetime import datetime
import os
import math
import threading
from asyncio import CancelledError
from sys import stderr
import subprocess
import time
import getpass
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, NoReturn, Any

import pandas as pd
import cv2
import numpy as np
import torch
from tqdm import tqdm
from triton.language import dtype

from common import Device, Precision
from utils import coords_from_segmentation_mask, find_max_dims
from performance_utils import drop_disk_cache, create_path_batch


class TqdmHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("performance.log", mode="a")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d | %H:%M:%S")
)
logger.addHandler(file_handler)
logger.addHandler(TqdmHandler())

logger.info(f"{datetime.now()}")


cuda_device = 0
dateset_path = Path(
    "/home/antonio/Downloads/extended_smartdoc_dataset/Extended Smartdoc dataset/train"
)


def batch_to_gpu(
    paths: List[str | np.ndarray],
    max_h: int,
    max_w: int,
    progress_bar,
    precision: Precision,
):
    if not precision:
        raise ValueError("Precision MUST be specified")

    n = len(paths)
    tensor = torch.zeros(size=(n, max_h, max_w), dtype=precision.to_type_gpu())

    for i, p in enumerate(paths):
        # handle mixed dataset
        if isinstance(p, str):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        else:
            img = p

        assert img.dtype == np.uint8(), (
            "Should have a np.ndarray of np.uint8() at this point..."
        )

        h, w = img.shape

        # While READING the image, since each pixel is in range 0-255 np.int8() (CPU first)
        img = img.astype(np.uint8())

        if precision != Precision.UINT8:
            img = img.astype(precision.to_type_cpu()) / 255.0

        # NOT tensor[i] = torch.from_numpy(img_norm) !! but
        tensor[i, :h, :w] = torch.from_numpy(img)
        # because the image will be at most as big as max_h and max_w but can be smaller!
    # tensor = tensor.to(Device.CUDA.value)
    # transferring to GPU
    torch.cuda.synchronize()
    progress_bar.update(n)
    return tensor.to(Device.CUDA.value)


def images_for_vram(
    vram_commitment, max_h, max_w, precision: Precision = Precision.FP32
):
    """
    Computes the number of images that can be stored in the given amount of memory.
    The resuslt is rounded by excess in case all the images have max_h and max_w
    :param vram_commitment: amount of memory the user is willing to commit
    :param max_h:
    :param max_w:
    :param precision: How many bytes to use to define pixel intensity
    :return:
    """
    # n x h x w x P (If P = float32 => 4 bytes) = ram_commitment
    # n = ram_commitment / (h * w * 4)
    img_size = max_h * max_w * precision.value
    return int(vram_commitment // img_size)  # integer division floors the value


def compute_gpu(batch: List[str], vram_commitment, precision):
    if not precision:
        raise ValueError("Precision MUST be set.")
    max_h, max_w = find_max_dims(batch)
    max_images = images_for_vram(vram_commitment, max_h, max_w, precision)
    total_images = len(batch)
    start = time.time()

    with tqdm(
        total=total_images,
        leave=True,
        position=0,
        desc=f"Uploading {total_images} images to GPU",
    ) as uploadbar:
        with tqdm(
            total=total_images,
            leave=True,
            position=1,
            desc="Computing mask corner coordinates",
        ) as workbar:
            for i in range(0, total_images, max_images):
                tensor = batch_to_gpu(
                    batch[i : i + max_images], max_h, max_w, uploadbar, precision
                )
                minibatch_size = tensor.shape[0]
                for img in range(minibatch_size):
                    _ = coords_from_segmentation_mask(
                        tensor[img], scale_percentage=1.0, device=Device.CUDA
                    )
                    workbar.update(1)
                del tensor
    return time.time() - start


def compute_cpu(path: str, precision: Precision, verbose=False):
    if not precision:
        raise ValueError("Precision must be specified.")
    try:
        start_imload = time.time()
        image = cv2.imread(
            filename=path,
            flags=cv2.IMREAD_GRAYSCALE,
        )  # RETURNS (H, W), NOT (W, H)!
        # While READING the image, since each pixel is in range 0-255 np.int8() (CPU first)
        image.astype(np.uint8())
        time_imload = time.time() - start_imload
        if verbose:
            logger.info(f"Loading an image from disk took {time_imload}")

        if precision != Precision.UINT8:
            image = np.divide(image, 255.0)

        if verbose:
            logger.info(f"The image stays on CPU")

        start_compute_time = time.time()
        whites = coords_from_segmentation_mask(
            image, scale_percentage=1.0, device=Device.CPU
        )
        compute_time = time.time() - start_compute_time
        if verbose:
            logger.info(f"Coords extraction took {compute_time}")

        return time.time() - start_imload
    except KeyboardInterrupt:
        return None


def _multiprocess_wrapper(args):
    return compute_cpu(*args)


def compute_cpu_multiprocess(
    paths, cores, precision: Precision, progress_bar=None, verbose=False
):
    if not cores:
        raise ValueError("`cores` is required for `compute_cpu_multiprocess`")

    start_t = time.time()
    paths = [str(p) for p in paths]
    args = map(lambda p: (p, precision, verbose), paths)

    with Pool(processes=cores) as pool:
        for _ in pool.imap_unordered(_multiprocess_wrapper, args):
            if progress_bar:
                progress_bar.update(1)
    return time.time() - start_t


def benchmark_cpu(
    batches: List[List[str]],
    batch_sizes: List[int],
    precision: Precision,
    csv_file: str | None = None,
    n_cores=cpu_count(),
    verbose=False,
):
    f_args = locals()
    f_args["batches"] = "<Redacted: too long>"
    logger.debug("Starting CPU benchmark with parameters:")
    logger.debug(f_args)

    if not precision:
        raise ValueError("Precision MUST be specified.")
    # password = getpass.getpass("Enter your password to invalidate disk cache: ") # if stdin is not accessible it throws an exception
    if csv_file:
        try:
            durations = pd.read_csv(csv_file, index_col=False)
        except FileNotFoundError:
            durations = pd.DataFrame([], columns=["batch_size", "duration"], index=None)
    else:
        durations = pd.DataFrame([], columns=["batch_size", "duration"], index=None)

    try:
        for i, b in enumerate(batches):
            try:
                size = batch_sizes[i]
                if size in durations["batch_size"].values:
                    if verbose:
                        logger.debug(
                            f"Batch {i + 1} ({size} items) already computed. Skipping..."
                        )
                    continue

                if verbose:
                    logger.info(f"Computing batch {i + 1}, {size} images.")
                with tqdm(
                    total=size, leave=False, desc=f"Batch {i + 1}: "
                ) as progress_bar:
                    t = compute_cpu_multiprocess(b, n_cores, progress_bar, precision)
                if verbose:
                    logger.debug(f"Took {t} seconds.")
                # drop_disk_cache(password, verbose=True)
                durations = pd.concat(
                    objs=[
                        durations,
                        pd.DataFrame({"batch_size": [size], "duration": [t]}),
                    ],
                    ignore_index=True,
                )
                if verbose:
                    logger.debug(f"Batch {i + 1} done.")
            except KeyboardInterrupt:
                break
    finally:
        # This is python :D
        if csv_file:
            durations.sort_values(by=["batch_size"], inplace=True)
            durations.to_csv(csv_file, mode="w", header=True, index=False)
            return None
        else:
            return durations


def benchmark_gpu(
    vram_commitment,
    batches: List[List[str]],
    batch_sizes: List[int],
    precision: Precision,
    csv_file: str | None = None,
):
    f_args = locals()
    f_args["batches"] = "<Redacted: too long>"
    logger.debug("Starting GPU benchmark with parameters:")
    logger.debug(f_args)
    if csv_file:
        try:
            durations = pd.read_csv(csv_file, index_col=False)
        except FileNotFoundError:
            durations = pd.DataFrame(
                [], columns=["batch_size", "duration", "vram_usage"], index=None
            )
    else:
        durations = pd.DataFrame(
            [], columns=["batch_size", "duration", "vram_usage"], index=None
        )

    # nvmlInit()
    # cuda_d = nvmlDeviceGetHandleByIndex(cuda_device)
    try:
        for i, b in enumerate(batches):
            try:
                size = batch_sizes[i]
                if size in durations["batch_size"].values:
                    logger.debug(
                        f"Batch {i + 1} ({size} items) already computed. Skipping..."
                    )
                    continue

                logger.info(f"Computing batch {i + 1}, {size} images.")
                t = compute_gpu(b, vram_commitment)
                logger.debug(f"Took {t} seconds.")
                # drop_disk_cache(password, verbose=True)
                # cuda_info = nvmlDeviceGetMemoryInfo(cuda_d)
                durations = pd.concat(
                    objs=[
                        durations,
                        pd.DataFrame(
                            {
                                "batch_size": [size],
                                "duration": [t],
                                # cannot use vram_usage due to del tensor in compute_gpu()
                                # 'vram_usage': cuda_info.used
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                logger.debug(f"Batch {i + 1} done.")
            except KeyboardInterrupt:
                break
    finally:
        if csv_file:
            durations.sort_values(by=["batch_size"], inplace=True)
            durations.to_csv(csv_file, mode="w", header=True, index=False)
        else:
            return durations


def compute_hybrid(
    cpu_split: float,
    n_cores: int,
    vram_commitment: float,
    precision: Precision,
    batches: List[List[str]],
    batch_sizes: List[int],
    csv_file: str | None = None,
):
    f_args = locals()
    f_args["batches"] = "<Redacted: too long>"
    logger.debug("Starting HYBRID benchmark with parameters:")
    logger.debug(f_args)

    if not precision:
        raise ValueError("Precision MUST be specified.")

    if csv_file:
        try:
            durations = pd.read_csv(csv_file, index_col=False)
        except FileNotFoundError:
            durations = pd.DataFrame([], columns=["batch_size", "duration"], index=None)
    else:
        durations = pd.DataFrame([], columns=["batch_size", "duration"], index=None)

    try:
        for i, b in enumerate(batches):
            size = len(b)
            if size == 1:
                raise ValueError(
                    "Cannot split batches of length 1, they'd only be computed by the GPU."
                )
            try:
                split_idx = int(size * cpu_split)  # floor
                cpu_batch = b[:split_idx]
                gpu_batch = b[split_idx:]

                if size in durations["batch_size"].values:
                    logger.debug(
                        f"Batch {i + 1} ({size} items) already computed. Skipping..."
                    )
                    continue

                logger.info(f"Computing batch {i + 1}, {size} images.")

                start_time = time.time()
                if cpu_batch:

                    def _run_cpu(cores, batch, precision, f):
                        return f(batch, cores, precision)

                    cpu_thread = threading.Thread(
                        target=_run_cpu,
                        args=(n_cores, cpu_batch, precision, compute_cpu_multiprocess),
                    )
                    cpu_thread.start()
                if gpu_batch:
                    _ = compute_gpu(gpu_batch, vram_commitment, precision)
                if cpu_batch:
                    cpu_thread.join()
                t = time.time() - start_time
                logger.debug(f"Took {t} seconds.")
                # drop_disk_cache(password, verbose=True)
                # cuda_info = nvmlDeviceGetMemoryInfo(cuda_d)
                durations = pd.concat(
                    objs=[
                        durations,
                        pd.DataFrame(
                            {
                                "batch_size": [size],
                                "duration": [t],
                                # cannot use vram_usage due to del tensor in compute_gpu()
                                # 'vram_usage': cuda_info.used
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                print(f"Batch {i + 1} done.")
                if csv_file:
                    durations.sort_values(by=["batch_size"], inplace=True)
                    durations.to_csv(csv_file, mode="w", header=True, index=False)

            except KeyboardInterrupt:
                break
    finally:
        if csv_file:
            durations.sort_values(by=["batch_size"], inplace=True)
            durations.to_csv(csv_file, mode="w", header=True, index=False)
        else:
            return durations


if __name__ == "__main__":
    batch_sizes = [
        # 1,
        10,
        20,
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        700,
        1000,
        1200,
        1700,
        2500,
        3000,
        3500,
        4000,
        5000,
        10000,
        # 15000,
        # 22000,
    ]
    batches = list(
        map(lambda size: create_path_batch(dateset_path, size), batch_sizes)
    )  # eager
    all_paths = [
        path for path_group in batches for path in path_group
    ]  # total_paths = [element for sublist in nested_list for element in sublist]
    paths_len = len(batches)
    logger.debug("Starting Benchmark.")

    # val = benchmark_cpu(
    #     # csv_file="./compute_durations_cpu_multiprocess.csv",
    #     csv_file='./test.csv',
    #     batches=batches,
    #     batch_sizes=batch_sizes,
    #     n_cores=cpu_count() -  1
    # )
    # val = benchmark_gpu(
    #     batches=batches,
    #     batch_sizes=batch_sizes,
    #     # csv_file = "compute_durations_gpu_iter.csv",
    #     vram_commitment=14 * (1024**3)
    # )

    n_cores = 11
    cpu_split = 0.7
    vram_commitment = 14.5 * (1024**3)
    precision = Precision.FP16
    csv_file = f"./compute_durations_hybrid_{n_cores}c_{cpu_split}cpu_{vram_commitment / (1024**3)}GVRAM_{precision}.csv"

    val = compute_hybrid(
        cpu_split=cpu_split,
        n_cores=n_cores,
        vram_commitment=vram_commitment,
        csv_file=csv_file,
        batches=batches,
        batch_sizes=batch_sizes,
        precision=precision,
    )

    print(val)
