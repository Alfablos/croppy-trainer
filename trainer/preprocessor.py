import sys

import shutil
from time import sleep
from numpy.typing import NDArray
from torch.multiprocessing import cpu_count
import multiprocessing
from functools import partial
from typing import Callable

import storage
from common import Purpose, DEFAULT_STORAGE_CLASS, DataSource
from pathlib import Path
import os
import csv
from tqdm import tqdm

import pandas as pd


from architecture import Architecture, ProcessResult


def worker(
    row: dict,
    transform_fn: Callable[[dict, int, int], ProcessResult],
    target_h: int,
    target_w: int,
    strict: bool,
) -> tuple[dict, NDArray, NDArray] | None:
    try:
        result = transform_fn(row, target_h, target_w)
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
    target_h: int,
    # weight of the training images
    target_w: int,
    dataset_map_csv: str,
    output_dir: str | None = None,
    source_name: str | None = None,
    source: DataSource | None = None,
    # Every how many iterations data is written to disk
    commit_freq: int = 100,
    # No actual computation
    dry_run: bool = False,
    verbose: bool = False,
    progress: bool = False,
    strict: bool = True,
    n_workers: int = int(cpu_count() / 2),
    compact_store: bool = False,
    paths_config: dict | None = None,
):
    """
    Performs a resize and stores resized images in a LMDB Database at :path

    Args:
        architecture: The model architecture
        purpose: The purpose (training/validation/test)
        target_h: Target height
        target_w: Target width
        dataset_map_csv: Path to the crawl output CSV
        output_dir: Output directory (for internal use; normally derived from source)
        source_name: Name of the data source (legacy)
        source: DataSource object (preferred over source_name)
        commit_freq: Commit frequency
        dry_run: Dry run
        verbose: Verbose output
        progress: Show progress
        strict: Strict mode
        n_workers: Number of workers
        compact_store: Compact the store
        paths_config: Path configuration from config.py
    """
    # Determine output directory
    if output_dir is None:
        if source is None and source_name is None:
            raise ValueError(
                "Either 'source' or 'source_name' must be provided."
            )

        if source is not None:
            output_dir = source.get_precompute_base_dir(purpose, paths_config)
        else:
            # Fallback for legacy behavior
            output_dir = f"{purpose}_data"

    args = locals()

    err = architecture.find_preprocessor_misconfig(args)
    if err:
        raise ValueError(
            f"Invalid configuration for a {architecture.value} architecture: {err}"
        )

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

    # Update store path generation
    if source is not None:
        db_path_noext = str(output_dir) + f"/data_{architecture.value}_{source.name}_{target_h}x{target_w}"
    else:
        name_part = f"_{source_name}" if source_name else f"_{str(purpose)}"
        db_path_noext = str(output_dir) + f"/data_{architecture.value}{name_part}_{target_h}x{target_w}"
    db_path = db_path_noext + "." + DEFAULT_STORAGE_CLASS

    # if the user wants to compact the store and the
    # compacted store path is not empty, throw an error.
    # It's ok to error if even if the db path does not exist,
    # compacted data must not be overwritten
    if compact_store:
        compacted_db_path = db_path_noext + "_compacted" + "." + DEFAULT_STORAGE_CLASS
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

    index_name = source.name if source is not None else (source_name or str(purpose))
    index_path = (
        str(output_dir)
        + f"/index_{architecture.value}_{index_name}_{data_length}x{target_h}x{target_w}.csv"
    )
    if os.path.exists(index_path):
        raise FileExistsError(index_path)

    print(
        f"Allocating {total_map_size / (1024**3):.2f} GB for the {DEFAULT_STORAGE_CLASS} store."
    )

    if dry_run:
        return
    else:
        if verbose:
            print("Waiting 5 seconds before starting, press Ctrl + c to interrupt...")
        sleep(5)

    # Write each example in the db after converting it to RGB
    if verbose:
        print(f"Creating {DEFAULT_STORAGE_CLASS} store at {db_path}.")
    csv_index_file = open(f"{index_path}", mode="w", newline="")
    csv_writer = csv.writer(csv_index_file)
    csv_header = architecture.get_csv_header()
    db_index = 0  # NOT updated when images fail to convert (if not strict)
    # transaction = env.begin(write=True)

    transform = architecture.get_transform_logic()

    worker_f = partial(
        worker,
        transform_fn=transform,
        target_h=target_h,
        target_w=target_w,
        strict=strict,
    )

    store = storage.new_store(db_path, write=True)
    store.set_metadata("size", str(total_map_size))
    store.set_metadata("h", str(target_h))
    store.set_metadata("w", str(target_w))

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
                        if progress:
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
