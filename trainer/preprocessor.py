import sys

import shutil
from time import sleep
from numpy.typing import NDArray
from torch.multiprocessing import cpu_count
import multiprocessing
from functools import partial
from typing import Callable

import storage
from common import Purpose, DEFAULT_STORAGE_CLASS
from pathlib import Path
import os
import csv
from tqdm import tqdm

import pandas as pd


from architecture import Architecture, ProcessResult


def worker(
    row: dict,
    transform_fn: Callable[[dict, int, int, float], ProcessResult],
    coords_scale_percentage: float,
    target_h: int,
    target_w: int,
    strict: bool,
) -> tuple[dict, NDArray, NDArray] | None:
    try:
        result = transform_fn(row, target_h, target_w, coords_scale_percentage)
        img, label = result.image, result.label
        return row, img, label
    except Exception as e:
        if strict:
            raise e
        print(f"Skipping {row['image_path']} due to an error:", e)
        return None


def precompute(
    architecture: Architecture,
    purpose: Purpose,
    output_dir: str,
    # heights of the training images
    target_h: int,
    # weight of the training images
    target_w: int,
    dataset_map_csv: str,
    coords_scale_percentage: float,
    # Every how many iterations data is written to disk
    commit_freq: int = 100,
    # No actual computation
    dry_run: bool = False,
    verbose: bool = False,
    progress: bool = False,
    # Whether the output should contain corners coords
    # If true they'll be computed if the csv does not contain them alread
    # If csv_path is none so the file will be computed and will
    # contain the coordinates depending on this value
    compute_corners: bool = True,
    strict: bool = True,
    n_workers: int = int(cpu_count() / 2),
    compact_store: bool = False,
):
    """
    Performs a resize and stores resized images in a LMDB Database at :path
    """

    args = locals()

    err = architecture.find_preprocessor_misconfig(args)
    if err:
        raise ValueError(f"Invalid configuration for a {architecture.value} architecture: {err}")

    if output_dir in [".", "./"]:
        output_dir = f"{purpose}_data"
    else:
        output_dir = output_dir.rstrip("/") + "/" + f"{purpose}_data"

    rows = pd.read_csv(dataset_map_csv).to_dict("records")
    has_coords = "x1" in rows[0]

    output_dir: Path = Path(output_dir)

    ## Preflight checks
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Destination {output_dir} exists but is a file.")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_length = len(rows)

    total_map_size = architecture.preprocessor_db_map_size(
        data_length, target_h, target_w
    )

    db_path_noext = (
        str(output_dir)
        + f"/data_{architecture.value}_{str(purpose)}_{data_length}x{target_h}x{target_w}"
    )
    db_path = db_path_noext + "." + DEFAULT_STORAGE_CLASS

    # if the user wants to compact the store and the
    # compacted store path is not empty, throw an error.
    # It's ok to error if even if the db path does not exist,
    # compacted data must not be overwritten
    if compact_store:
        compacted_db_path = db_path_noext + "_compacted." + DEFAULT_STORAGE_CLASS
        if (
            os.path.exists(compacted_db_path)
            and not len(os.listdir(compacted_db_path)) == 0
        ):
            raise FileExistsError(
                f"Directory {compacted_db_path} exists and is not empty. Refusing to continue."
            )
    # if the uncompacted store path exists,
    # we either compact it or error
    if os.path.exists(db_path):
        if compact_store:
            cp = Path(compacted_db_path)
            print(f"Path {db_path} already exists, skipping store creation.")
            print(f"Compacting existing store {db_path} to {cp}.")
            if cp.exists():
                raise FileExistsError(
                    f"File {compacted_db_path} exists. Refusing to continue."
                )
            cp.mkdir(parents=True)
            with storage.new_store(compacted_db_path, write=True) as store:
                store.compact(compacted_db_path)
            shutil.rmtree(db_path)
            return
        raise FileExistsError(db_path)

    index_path = (
        str(output_dir)
        + f"/index_{architecture.value}_{str(purpose)}_{data_length}x{target_h}x{target_w}.csv"
    )
    if os.path.exists(index_path):
        raise FileExistsError(index_path)

    print(f"Allocating {total_map_size / (1024**3):.2f} GB for the {DEFAULT_STORAGE_CLASS} store.")

    if dry_run:
        return
    else:
        if verbose:
            print("Waiting 5 seconds before starting, press Ctrl + c to interrupt...")
        sleep(5)


    # Write each example in the db after converting it to RGB
    if verbose:
        print(f"Creating LMDB store at {db_path}.")
    csv_index_file = open(f"{index_path}", mode="w", newline="")
    csv_writer = csv.writer(csv_index_file)
    csv_header = architecture.get_csv_header()
    db_index = 0  # NOT updated when images fail to convert (if not strict)
    # transaction = env.begin(write=True)

    transform = architecture.get_transform_logic(coords_scale_percentage)

    worker_f = partial(
        worker,
        transform_fn=transform,
        coords_scale_percentage=coords_scale_percentage,
        target_h=target_h,
        target_w=target_w,
        strict=strict,
    )

    store = storage.new_store(db_path, write=True)
    store.set_metadata('corners_recess_percentage', str(coords_scale_percentage))
    store.set_metadata('size', str(total_map_size))
    store.set_metadata('h', str(target_h))
    store.set_metadata('w', str(target_w))

    with multiprocessing.Pool(n_workers) as pool:
        result_iter = pool.imap(
            worker_f, rows, chunksize=10
        )  # trying to preserve order, not using imap_unordered

        if progress:
            bar = tqdm(total=len(rows), bar_format="{bar}{l_bar}{r_bar}’")

        with store as s:

            try:
                for result in result_iter:
                    if not result:
                        if verbose:
                            bar.update(1)
                        continue

                    row, img, label = result
                    ipath = row["image_path"]
                    lpath = row["label_path"]

                    if architecture == Architecture.RESNET:
                        csv_writer.writerow([db_index, ipath, *label])
                    elif architecture == Architecture.UNET:
                        csv_writer.writerow([db_index, ipath, lpath])

                    s.append(img, label)

                    if progress:
                        bar.update(1)
                    db_index += 1
                
                if compact_store:
                    print(f"Compacting LMDB store {db_path} to {compacted_db_path}")
                    Path(compacted_db_path).mkdir(parents=True, exist_ok=True)
                    s.compact(compacted_db_path)

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                raise e

    if compact_store:
        shutil.rmtree(db_path)

    print("Precomputation complete.")


if __name__ == "__main__":
    pass
